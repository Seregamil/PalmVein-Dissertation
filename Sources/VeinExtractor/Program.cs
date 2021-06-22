using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Threading.Tasks;
using OpenCvSharp;
using OpenCvSharp.XImgProc;
using VeinExtractor.Ml;

namespace VeinExtractor
{
    public class Program
    {
        public const string ConnectionUrl = "rtsp://v-sensor:root@192.168.1.20:554/live/ch00_0";
        
        // private const int DefaultWindowHeight = 720;
        // private const int DefaultWindowWidth = 1280;
        private const int DefaultWindowHeight = 600;
        private const int DefaultWindowWidth = 800;
        
        public const int DefaultRoiSize = 224;

        public static bool DoesRegistrationEnabled = false;
        public static int RegistrationFramesElapsed = 0;
        public const int FramesForRegister = 100;
        public static readonly List<Mat> RegistrationFrames = new List<Mat>();

        private const string TrainedModelPath = @"C:\Users\CXXY\Desktop\ml\dcc7de15-e05d-40b1-b791-27eab6f768a2.onnx";
        private const string LabelsPath = @"C:\Users\CXXY\Desktop\ml\datasets_my\train";
        
        private static Predictor _predictor;

        static void Main(string[] args)
        {
            // _predictor = new Predictor(LabelsPath, TrainedModelPath);
            
            var windowMerged = new Window("Merge");
            var windowRoi = new Window("Roi");
            
            var capture = new VideoCapture(ConnectionUrl);
            var frame = new Mat();
            
            if (!capture.Read(frame))
                throw new Exception($"Err with getting frame");

            while (capture.Read(frame))
            {
                var key = Cv2.WaitKey(1);
                if (key == 27) // esc
                    break;

                if (key == 101) // E?
                {
                    if (DoesRegistrationEnabled)
                        ResetRegister();
                    else
                        StartRegister();
                }

                var (output, roi) = FrameTick(frame);
                windowMerged.ShowImage(output);
                output.Dispose();

                if (roi == null || roi.IsDisposed) {
                    if(DoesRegistrationEnabled)
                        ResetRegister();
                    continue;
                }

                if (roi.Height == DefaultRoiSize)
                {
                    if (DoesRegistrationEnabled)
                    {
                        using (var rgb = roi.Clone())
                        {
                            Cv2.CvtColor(rgb, rgb, ColorConversionCodes.GRAY2RGB);
                            RegistrationFrames.Add(roi.Clone());

                        }
                        // RegistrationFrames.Add(roi.Clone());
                        RegistrationFramesElapsed++;
                        
                        if(RegistrationFramesElapsed == FramesForRegister)
                            FinishRegister();
                    }
                    else
                    {
                        // _predictor.Predict(roi);
                    }

                    windowRoi.ShowImage(roi);
                }

                roi.Dispose();
            }
        }

        public static (Mat output, Mat roi) FrameTick(Mat frame)
        {
            var merge = new Mat();
            
            var context = new SensorContext(frame);
            Cv2.Threshold(context.Blue, context.Threshold, 0, 255, ThresholdTypes.Otsu | ThresholdTypes.Binary);
            // Cv2.ImWrite("Threshold.jpg", context.Threshold);
            
            context.Green = DrawVeins(context.Green);
            context.Red = DrawVeins(context.Red);
            
            // Cv2.ImWrite("G_Veins.jpg", context.Green);
            // Cv2.ImWrite("R_Veins.jpg", context.Red);

            Cv2.Add(context.Green, context.Red, merge);
            // Cv2.ImWrite("Merge.jpg", merge);

            var roi = GetHandRoi(context.Threshold);
            // Console.WriteLine(roi);
            var outRoi = new Mat();
            // CvXImgProc.Thinning(merge, merge);
            // merge = RemoveNoiseByConnectedComponents(merge, DenoiseArea);
            
            if (roi.Width != 0 && roi.Height != 0)
            {
                using (var sourceImage = new Mat(context.Source, roi))
                using (var roiSource = new Mat(context.Green, roi))
                {
                    Cv2.Resize(roiSource, roiSource, new Size(DefaultRoiSize, DefaultRoiSize)); // resize image 
                    Cv2.Resize(sourceImage, sourceImage, new Size(DefaultRoiSize, DefaultRoiSize)); // resize image 
                    
                    // output.Roi = RemoveNoiseByConnectedComponents(roiSource.Clone(), 15);
                    // Cv2.ImWrite("RoiResized.jpg", roiSource);
                    if (roiSource.Height > 0 && roiSource.Width > 0 && roiSource.Height == DefaultRoiSize && roiSource.Width == DefaultRoiSize)
                    {
                        var deNoised = RemoveNoiseByConnectedComponents(roiSource, 10); // rm dots and small components
                        // var (points, descriptors) = _features.CalculateKeyPoints(deNoised); // get features keypoints
                        // Cv2.ImWrite("Denoise.jpg", deNoised);

                        CvXImgProc.Thinning(deNoised, outRoi);
                        // Cv2.BitwiseAnd(sourceImage, outRoi, outRoi);
                        // Cv2.ImWrite("Thinning.jpg", outRoi);

                        // outRoi = deNoised.Clone();
                        // Cv2.DrawKeypoints(output.Roi, points, output.Roi, Scalar.Red);
                        
                        // _features.GetBestSimilarityModel(descriptors.Clone());
                        // _features.GetBestSimilarityKnnModel(descriptors.Clone());
                    }
                }
            }
            context.Dispose();
            
            // Cv2.Merge(new [] {
            //     outRoi.Clone(),
            //     outRoi.Clone(),
            //     outRoi.Clone()
            // }, outRoi);
            
            return (merge, outRoi);
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
            // Cv2.ImWrite("DistanceTransform.jpg", distanceTransform);
            
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
            // Cv2.ImWrite("Clahe.jpg", channel);
            Cv2.GaussianBlur (channel, channel, new Size (0, 0), sigmaX : 5);
            // Cv2.ImWrite("Blur.jpg", channel);

            Cv2.MorphologyEx(channel, channel, MorphTypes.Dilate, Structure, iterations: 20);
            // Cv2.ImWrite("Dilate.jpg", channel);
            Cv2.AdaptiveThreshold(channel, channel, 255, AdaptiveThresholdTypes.MeanC, ThresholdTypes.BinaryInv, 9, 2);
            // Cv2.ImWrite("AdaptiveThreshold.jpg", channel);
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

        public static void StartRegister()
        {
            RegistrationFramesElapsed = 0;
            RegistrationFrames.Clear();
            
            DoesRegistrationEnabled = true;
            
            Console.WriteLine("Registration started!");
        }

        public static void FinishRegister()
        {
            DoesRegistrationEnabled = false;
            
            var dirName = $"{DateTime.Now.Ticks}/";
            Directory.CreateDirectory(dirName);
            for (var i = 0; i < RegistrationFrames.Count; i++)
            {
                // Cv2.ImWrite($"{dirName}/{i}.jpg", RegistrationFrames[i]);
                RegistrationFrames[i].Dispose();
            }
            RegistrationFrames.Clear();
            Console.WriteLine($"Success registered {dirName}");
        }

        public static void ResetRegister()
        {
            DoesRegistrationEnabled = false;
            RegistrationFrames.Clear();
            Console.WriteLine($"Register Err");
        }
    }
}