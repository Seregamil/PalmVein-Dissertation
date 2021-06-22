using System.Collections.Generic;
using OpenCvSharp;

namespace BioVein
{
    public class ORBMatcher
    {
        public static (IEnumerable<KeyPoint> points, Mat descriptors) CalculateKeyPoints(ORB orb, Mat frame)
        {
            var descriptors = new Mat();
            orb.DetectAndCompute(frame, null, out var points, descriptors);
            return (points, descriptors);
        }
    }
}