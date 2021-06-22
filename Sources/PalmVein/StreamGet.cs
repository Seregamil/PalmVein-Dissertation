using System;
using System.Threading;
using System.Threading.Tasks;
using OpenCvSharp;

namespace PalmVein
{
    public class StreamGet
    {
        private readonly VideoCapture _capture;
        public Mat Frame;

        public bool DoesEnabled = true;
        
        public StreamGet(string url)
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
}