using System.Threading.Tasks;
using OpenCvSharp;
using OpenCvSharp.Detail;

namespace PalmVein
{
    public class StreamShow
    {
        private Mat _frame;
        public Mat Frame
        {
            get => _frame;
            set
            {
                _updated = true;
                _frame = value;
            }
        }

        private readonly Window _window;
        private readonly Window _roiWindow;
        
        private bool _updated = false;
        private readonly Preprocess _preprocess;
        
        public StreamShow()
        {
            _window = new Window("Source");
            _roiWindow = new Window("ROI");
            _preprocess = new Preprocess();
        }

        public void Start()
        {
            Task.Factory.StartNew(() =>
            {
                while (true)
                {
                    if(!_updated)
                        continue;

                    var output = _preprocess.FrameTick(_frame);
                    _window.ShowImage(output.Frame);
                
                    output.Frame.Dispose();

                    if(output.Roi == null || output.Roi.IsDisposed)
                        continue;

                    if (output.Roi.Height == Preprocess.DefaultRoiSize)
                    {
                        _roiWindow.ShowImage(output.Roi);

                        _window.ShowImage(_frame);
                        _updated = false;
                    }

                    output.Roi.Dispose();
                }
            });
        }
    }
}