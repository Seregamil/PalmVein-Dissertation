using OpenCvSharp;

namespace PalmVein
{
    public class Settings
    {
        public static int DeNoiseAreaValue = 10;
        private Window _window;
        public Settings()
        {
            _window = new Window("Settings");
            _window.CreateTrackbar("AREA", DeNoiseAreaValue, 255, pos =>
            {
                DeNoiseAreaValue = pos;
            });
        }
    }
}