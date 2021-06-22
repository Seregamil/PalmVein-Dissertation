using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Diagnostics.CodeAnalysis;
using System.Linq;
using OpenCvSharp;
using OpenCvSharp.XImgProc;

namespace PalmVein
{
    public class Preprocess
    {
        private FeaturesHandler _features;
        
        private const int DefaultWindowHeight = 720;
        private const int DefaultWindowWidth = 1280;
        public const int DefaultRoiSize = 200;
        

        private static readonly CLAHE[] ClaheArr =
        {
            Cv2.CreateCLAHE(2.0, new Size(3 + 2, 3 + 2)),
            Cv2.CreateCLAHE(2.0, new Size(3 + 4, 3 + 4))
        };

        private static readonly Mat Structure = Cv2.GetStructuringElement(MorphShapes.Cross, new Size(2, 2));
        
        public Preprocess()
        {
            _features = new FeaturesHandler(Program.Store);
            // new Settings();
        }

        public OutputScheme FrameTick(Mat frame, bool doesSave = false)
        {
            var sw = new Stopwatch();
            sw.Start();
            
            var output = new OutputScheme()
            {
                Frame = frame.Clone(),
            };
            
            var context = new SensorContext(output.Frame);
            
            // using var context.Threshold = new Mat();
            Cv2.Threshold(context.Blue, context.Threshold, 0, 255, ThresholdTypes.Otsu | ThresholdTypes.Binary);
            
            // Hide background near palm
            Cv2.BitwiseAnd(context.Green, context.Green, context.Green, context.Threshold);
            
            // Draw veins on hand
            context.Green = DrawVeinsV2(context.Green, context.Red);
            
            output.Roi = context.Green.Clone();
            // Find ROI contours
            var roi = GetHandRoi(context.Threshold);
        
            // Check does roi finded
            if (roi.Width != 0 && roi.Height != 0)
            {
                using (var roiSource = new Mat(context.Green, roi))
                {
                    Cv2.Resize(roiSource, roiSource, new Size(DefaultRoiSize, DefaultRoiSize)); // resize image 
                    // output.Roi = RemoveNoiseByConnectedComponents(roiSource.Clone(), 15);

                    if (roiSource.Height > 0 && roiSource.Width > 0 && roiSource.Height == DefaultRoiSize && roiSource.Width == DefaultRoiSize)
                    {
                        var deNoised = RemoveNoiseByConnectedComponents(roiSource, Settings.DeNoiseAreaValue); // rm dots and small components
                        var (points, descriptors) = _features.CalculateKeyPoints(deNoised); // get features keypoints

                        output.Roi = deNoised.Clone();
                        // Cv2.DrawKeypoints(output.Roi, points, output.Roi, Scalar.Red);
                        
                        // _features.GetBestSimilarityModel(descriptors.Clone());
                        _features.GetBestSimilarityKnnModel(descriptors.Clone());
                    }
                }
            }
            
            context.Dispose();
            // Console.WriteLine($"Frame elapsed by {sw.Elapsed}");
            sw.Stop();
        
            return output;
        }

        [SuppressMessage("ReSharper.DPA", "DPA0003: Excessive memory allocations in LOH", MessageId = "type: System.Int32[,]")]
        private Mat RemoveNoiseByConnectedComponents(Mat frame, int maxArea)
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

        private Rect GetHandRoi(Mat frame)
        {
            var center = GetHandCenter(frame, out var centerRadius);
        
            var a = (2 * centerRadius) / Math.Sqrt (2);
        
            var xx = center.X - centerRadius * Math.Cos (45 * Math.PI / 180);
            var yy = center.Y - centerRadius * Math.Sin (45 * Math.PI / 180);
        
            if (xx < 0 || yy < 0 || (xx + a) > DefaultWindowWidth || (yy + a) > DefaultWindowHeight)
                return new Rect(0, 0, 0, 0);
            
            return new Rect(new Point(xx, yy), new Size(a, a));
        }
        
        private Mat DrawVeins(Mat channel)
        {
            channel = MultiClahe(channel);
            
            Cv2.GaussianBlur (channel, channel, new Size (7, 7), sigmaX : 5);
            Cv2.MorphologyEx(channel, channel, MorphTypes.Erode, Structure, iterations: 1);
            Cv2.MorphologyEx(channel, channel, MorphTypes.Dilate, Structure, iterations: 9);
            
            Cv2.AdaptiveThreshold(channel, channel, 255, AdaptiveThresholdTypes.GaussianC, ThresholdTypes.BinaryInv, 9, 2);
            Cv2.MorphologyEx(channel, channel, MorphTypes.Gradient, Structure); // maby
            
            //Cv2.MorphologyEx(channel, channel, MorphTypes.Open, _Structure, iterations: 1); // maby
            // Cv2.MorphologyEx(channel, channel, MorphTypes.Close, _Structure, iterations: 1); // maby
            
            return channel;
        }
        
        private Mat DrawVeinsV2(Mat green, Mat red)
        {
            green = MultiClahe(green);
            Cv2.GaussianBlur (green, green, new Size (0, 0), sigmaX : 5);
            Cv2.MorphologyEx(green, green, MorphTypes.Dilate, Structure, iterations: 20);
            Cv2.AdaptiveThreshold(green, green, 255, AdaptiveThresholdTypes.MeanC, ThresholdTypes.BinaryInv, 9, 2);

            red = MultiClahe(red);
            Cv2.GaussianBlur (red, red, new Size (0, 0), sigmaX : 5);
            Cv2.MorphologyEx(red, red, MorphTypes.Dilate, Structure, iterations: 20);
            Cv2.AdaptiveThreshold(red, red, 255, AdaptiveThresholdTypes.MeanC, ThresholdTypes.BinaryInv, 9, 2);
            
            var output = new Mat();
            Cv2.Add(green, red, output);
            CvXImgProc.Thinning(output, output);

            return output;
        }

        private Mat MultiClahe(Mat source)
        {
            foreach (var clahe in ClaheArr)
                clahe.Apply(source, source);
            return source;
        }

        /// <summary>
        /// Вовзращает позицию центра ладони
        /// </summary>
        /// <param name="frame"></param>
        /// <param name="radius"></param>
        /// <returns></returns>
        private Point GetHandCenter(Mat frame, out double radius)
        {
            using var distanceTransform = new Mat();
            Cv2.DistanceTransform(frame, distanceTransform, DistanceTypes.L2, DistanceTransformMasks.Mask3);

            var maxIdx = new int[2];
            var minIdx = new int[2];
            Cv2.MinMaxIdx(distanceTransform, out _, out radius, minIdx, maxIdx);

            return new Point(maxIdx[1], maxIdx[0]);
        }
        
        // private  Mat HideBackground(Mat frame, Mat context.Threshold)
        // { // https://stackoverflow.com/questions/64295209/removing-background-around-contour
        //     var masked = new Mat();
        //     Cv2.BitwiseAnd(frame, frame, masked, context.Threshold);
        //     return masked;
        // }
    }
}