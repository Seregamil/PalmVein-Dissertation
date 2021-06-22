using System;
using System.Threading;
using System.Threading.Tasks;
using OpenCvSharp;

namespace BioVein
{
    public class Stream
    {
        public class Getter
        {
            private readonly VideoCapture _capture;
            public Mat Frame;

            public bool DoesEnabled = true;
        
            public Getter(string url)
            {
                _capture = new VideoCapture(url);
                Frame = new Mat();
            
                if (!_capture.Read(Frame))
                    throw new Exception($"Err with getting frame");
            }

            public void Start()
            {
                Task.Factory.StartNew(() =>
                {
                    while (DoesEnabled && _capture.Read(Frame)) {}
                });
            }

            public void Stop()
            {
                DoesEnabled = false;
            }
        }

        public class Updater
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
            private Predictor _predictor;
        
            public Updater()
            {
                _window = new Window("Source");
                _roiWindow = new Window("ROI");

                if (Program.DoesPredictorEnabled)
                {
                    _predictor = new Predictor(
                        pathToModel: @"C:\Users\CXXY\Desktop\ml\resnet-my.onnx",
                        labelsPath: @"C:\Users\CXXY\Desktop\ml\datasets\train",
                        threshold: 0.06191919191919192f);
                }
                //0.06191919191919192
                //3.2015202045440674

            }

            public void Start()
            {
                Task.Factory.StartNew(() =>
                {
                    while (true)
                    {
                        if(!_updated)
                            continue;

                        var (output, roi) = Preprocess.FrameTick(_frame);
                        _window.ShowImage(output);
                
                        output.Dispose();

                        if(roi == null || roi.IsDisposed)
                            continue;

                        if (roi.Height == Preprocess.DefaultRoiSize)
                        {
                            _roiWindow.ShowImage(roi);
                            if(Program.DoesPredictorEnabled)
                                _predictor.Predict(roi);
                            // Task.Factory.StartNew(() =>
                            // {
                            //     _predictor.Predict(roi);
                            // });
                            // _window.ShowImage(_frame);
                            _updated = false;
                        }

                        roi.Dispose();
                    }
                });
            }
        }
    }
}