using OpenCvSharp;

namespace VeinExtractor
{
    public class SensorContext
    {
        public Mat Source { get; set; }
        public Mat[] Spectres { get; set; }
        
        public Mat Green { get; set; }
        public Mat Blue { get; set; }
        public Mat Red { get; set; }
        
        public Mat Threshold { get; set; }
        public Mat Roi { get; set; }

        public SensorContext(Mat frame)
        {
            Source = frame.Clone();

            Spectres = Source.Split();

            Red = Spectres[2];
            Green = Spectres[1]; // used for drawing veins ( see DrawVeins with this reference )
            Blue = Spectres[0]; // used for finding contours

            // Cv2.ImWrite("Source.jpg", Source);
            // Cv2.ImWrite("Red.jpg", Red);
            // Cv2.ImWrite("Green.jpg", Green);
            // Cv2.ImWrite("Blue.jpg", Blue);

            Roi = new Mat();
            Threshold = new Mat();
        }

        public void Dispose()
        {
            Source.Dispose();
            
            foreach (var spectre in Spectres)
                spectre.Dispose();
            
            Red.Dispose();
            Green.Dispose();
            Blue.Dispose();

            Threshold.Dispose();
            Roi.Dispose();
        }
    }
}