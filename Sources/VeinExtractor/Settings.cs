using System;
using OpenCvSharp;

namespace VeinExtractor
{
    public class Settings
    {
        public static float Alpha = 0.1f;
        public static float K = 0.1f;
        public static int Iters = 1;
        public Settings(Window window)
        {
            window.CreateTrackbar("Alpha", 1, 100, pos =>
            {
                if(pos == 0)
                    return;
                Alpha = pos / 100f;
            });
            
            window.CreateTrackbar("K", 1, 1000, pos =>
            {
                if(pos == 0)
                    return;
                K = pos / 100f;
            });
            
            window.CreateTrackbar("Iters", Iters, 100, pos =>
            {
                Iters = pos;
            });
        }
    }
}