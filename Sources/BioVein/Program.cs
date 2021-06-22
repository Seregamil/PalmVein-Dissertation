using System;
using System.Diagnostics;
using OpenCvSharp;

namespace BioVein
{
    class Program
    {
        public const string ConnectionUrl = "rtsp://v-sensor:root@192.168.1.20:554/live/ch00_0";

        public const bool DoesOrbEnabled = false;
        public const bool DoesPredictorEnabled = false;
        
        static void Main(string[] args)
        { 
            var videoGetter = new Stream.Getter(ConnectionUrl);
            var videoSetter = new Stream.Updater();
           
            videoGetter.Start();
            videoSetter.Start();

            while (videoGetter.DoesEnabled)
            {
                var key = Cv2.WaitKey(1);
                if (key == 27) // esc
                    break;

                // var sw = new Stopwatch();
                // sw.Start();
               
                if(videoGetter.Frame.Height == 0)
                    continue;

                // window.ShowImage(videoGetter.Frame);
                videoSetter.Frame = videoGetter.Frame;
               
                // Console.WriteLine($"Elapsed {sw.Elapsed}");
                // sw.Stop();
            }
        }
    }
}