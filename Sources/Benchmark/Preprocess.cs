using System;
using System.Linq;
using OpenCvSharp;
using OpenCvSharp.XImgProc;

namespace Benchmark
{
    public class Preprocess
    {
        public const int DefaultRoiSize = 200;
        private const int DefaultWindowHeight = 720;
        private const int DefaultWindowWidth = 1280;
        
        public static Mat FrameTick(Mat frame)
        {
            var merge = new Mat();
            
            var context = new SensorContext(frame);
            Cv2.Threshold(context.Blue, context.Threshold, 0, 255, ThresholdTypes.Otsu | ThresholdTypes.Binary);

            context.Green = DrawVeins(context.Green);
            context.Red = DrawVeins(context.Red);
            
            Cv2.Add(context.Green, context.Red, merge);
            
            var roi = GetHandRoi(context.Threshold);
            var outRoi = new Mat();
            // CvXImgProc.Thinning(merge, merge);
            // merge = RemoveNoiseByConnectedComponents(merge, DenoiseArea);
            
            if (roi.Width != 0 && roi.Height != 0)
            {
                using (var roiSource = new Mat(context.Green, roi))
                {
                    Cv2.Resize(roiSource, roiSource, new Size(DefaultRoiSize, DefaultRoiSize)); // resize image 
                    // output.Roi = RemoveNoiseByConnectedComponents(roiSource.Clone(), 15);

                    if (roiSource.Height > 0 && roiSource.Width > 0 && roiSource.Height == DefaultRoiSize && roiSource.Width == DefaultRoiSize)
                    {
                        var deNoised = RemoveNoiseByConnectedComponents(roiSource, 10); // rm dots and small components
                        // var (points, descriptors) = _features.CalculateKeyPoints(deNoised); // get features keypoints

                        CvXImgProc.Thinning(deNoised, outRoi);
                        // outRoi = deNoised.Clone();
                        // Cv2.DrawKeypoints(output.Roi, points, output.Roi, Scalar.Red);
                        
                        // _features.GetBestSimilarityModel(descriptors.Clone());
                        // _features.GetBestSimilarityKnnModel(descriptors.Clone());
                    }
                }
            }
            context.Dispose();

            return outRoi;
        }
        
        private static readonly CLAHE[] ClaheArr =
        {
            Cv2.CreateCLAHE(2.0, new Size(3 + 2, 3 + 2)),
            Cv2.CreateCLAHE(2.0, new Size(3 + 4, 3 + 4))
        };
        
        private static Point GetHandCenter(Mat frame, out double radius)
        {
            using var distanceTransform = new Mat();
            Cv2.DistanceTransform(frame, distanceTransform, DistanceTypes.L2, DistanceTransformMasks.Mask3);

            var maxIdx = new int[2];
            var minIdx = new int[2];
            Cv2.MinMaxIdx(distanceTransform, out _, out radius, minIdx, maxIdx);

            return new Point(maxIdx[1], maxIdx[0]);
        }
        
        private static Rect GetHandRoi(Mat frame)
        {
            var center = GetHandCenter(frame, out var centerRadius);
        
            var a = (2 * centerRadius) / Math.Sqrt (2);
        
            var xx = center.X - centerRadius * Math.Cos (45 * Math.PI / 180);
            var yy = center.Y - centerRadius * Math.Sin (45 * Math.PI / 180);
        
            if (xx < 0 || yy < 0 || (xx + a) > DefaultWindowWidth || (yy + a) > DefaultWindowHeight)
                return new Rect(0, 0, 0, 0);
            
            return new Rect(new Point(xx, yy), new Size(a, a));
        }
        
        private static Mat RemoveNoiseByConnectedComponents(Mat frame, int maxArea)
        {
            var cc = Cv2.ConnectedComponentsEx(frame, PixelConnectivity.Connectivity8,
                ConnectedComponentsAlgorithmsTypes.WU);
                         
            var output = new Mat();
            if (cc.LabelCount > 1)
                cc.FilterByBlobs(frame, output,
                    cc.Blobs
                        .Where(blob => blob.Area > maxArea));
            else
                return frame;
            
            return output;
        }
        
        private static readonly Mat Structure = Cv2.GetStructuringElement(MorphShapes.Cross, new Size(2, 2));

        private static Mat MultiClahe(Mat source)
        {
            foreach (var clahe in ClaheArr)
                clahe.Apply(source, source);
            return source;
        }
        
        private static Mat DrawVeins(Mat channel)
        {
            channel = MultiClahe(channel);
            Cv2.GaussianBlur (channel, channel, new Size (0, 0), sigmaX : 5);
            
            Cv2.MorphologyEx(channel, channel, MorphTypes.Dilate, Structure, iterations: 20);
            Cv2.AdaptiveThreshold(channel, channel, 255, AdaptiveThresholdTypes.MeanC, ThresholdTypes.BinaryInv, 9, 2);
            return channel;
        }

        private static Mat DefaultDrawing(Mat channel)
        {
            channel = MultiClahe(channel);
            
            Cv2.GaussianBlur (channel, channel, new Size (7, 7), sigmaX : 5);
            Cv2.MorphologyEx(channel, channel, MorphTypes.Erode, Structure, iterations: 1);
            Cv2.MorphologyEx(channel, channel, MorphTypes.Dilate, Structure, iterations: 9);
            
            Cv2.AdaptiveThreshold(channel, channel, 255, AdaptiveThresholdTypes.GaussianC, ThresholdTypes.BinaryInv, 9, 2);
            // Cv2.MorphologyEx(channel, channel, MorphTypes.Gradient, Structure); // maby
            return channel;
        }
        
        private static Mat HideBackground(Mat frame, Mat threshold)
        { // https://stackoverflow.com/questions/64295209/removing-background-around-contour
            var masked = new Mat();
            Cv2.BitwiseAnd(frame, frame, masked, threshold);
            return masked;
        }
    }
}