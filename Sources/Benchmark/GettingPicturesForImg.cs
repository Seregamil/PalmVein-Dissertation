using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using OpenCvSharp;
using OpenCvSharp.Features2D;
using OpenCvSharp.XFeatures2D;

namespace Benchmark
{
    public class GettingPicturesForImg
    {
        // private const int TotalAttempt = 1000;
        private const string CasiaPath = @"W:\CXXY\CasiaPrepare\images";
        private const string OutputPath = @"C:\Users\CXXY\Desktop\diploma-images\third-part-images";
        
        private static List<Mat> PreloadCasiaFiles;
        
        private static BRISK _brisk;
        private static ORB _orb;
        private static SIFT _fSift;
        private static SURF _surf;
        private static AKAZE _akaze;

        public static void Start()
        {
            // load files
            PreloadCasiaFiles = Directory.GetFiles(CasiaPath, "*.jpg")
                .Take(6)
                .Select(x => Cv2.ImRead(x))
                .ToList();

            Console.WriteLine($"Loaded {PreloadCasiaFiles.Count}");
            var path = $"{OutputPath}\\original-keys";
            if(Directory.Exists(path))
                Directory.Delete(path, true);

            Directory.CreateDirectory(path);

            _brisk = BRISK.Create(1);
            _orb = ORB.Create(1500, edgeThreshold: 1);
            _fSift = SIFT.Create(1500);
            _surf = SURF.Create(1);
            _akaze = AKAZE.Create();

            var mapPath = $"{path}\\BRISK";
            Directory.CreateDirectory(mapPath);
            
            for (var i = 0; i < PreloadCasiaFiles.Count; i++)
            {
                using (var descriptors = new Mat())
                using (var mat = new Mat())
                {
                    _brisk.DetectAndCompute(PreloadCasiaFiles[i], null, out var keypoints, descriptors);
                    Cv2.DrawKeypoints(PreloadCasiaFiles[i], keypoints, mat, Scalar.Red);
                    // Cv2.ImWrite($"{mapPath}\\{i}.jpg", mat);
                }
            }

            mapPath = $"{path}\\ORB";
            Directory.CreateDirectory(mapPath);
            
            for (var i = 0; i < PreloadCasiaFiles.Count; i++)
            {
                using (var descriptors = new Mat())
                using (var mat = new Mat())
                {
                    _orb.DetectAndCompute(PreloadCasiaFiles[i], null, out var keypoints, descriptors);
                    Cv2.DrawKeypoints(PreloadCasiaFiles[i], keypoints, mat, Scalar.Red);
                    // Cv2.ImWrite($"{mapPath}\\{i}.jpg", mat);
                }
            }
            
            mapPath = $"{path}\\SIFT";
            Directory.CreateDirectory(mapPath);
            
            for (var i = 0; i < PreloadCasiaFiles.Count; i++)
            {
                using (var descriptors = new Mat())
                using (var mat = new Mat())
                {
                    _fSift.DetectAndCompute(PreloadCasiaFiles[i], null, out var keypoints, descriptors);
                    Cv2.DrawKeypoints(PreloadCasiaFiles[i], keypoints, mat, Scalar.Red);
                    // Cv2.ImWrite($"{mapPath}\\{i}.jpg", mat);
                }
            }
            
            mapPath = $"{path}\\SURF";
            Directory.CreateDirectory(mapPath);
            
            for (var i = 0; i < PreloadCasiaFiles.Count; i++)
            {
                using (var descriptors = new Mat())
                using (var mat = new Mat())
                {
                    _surf.DetectAndCompute(PreloadCasiaFiles[i], null, out var keypoints, descriptors);
                    Cv2.DrawKeypoints(PreloadCasiaFiles[i], keypoints, mat, Scalar.Red);
                    // Cv2.ImWrite($"{mapPath}\\{i}.jpg", mat);
                }
            }
            
            mapPath = $"{path}\\AKAZE";
            Directory.CreateDirectory(mapPath);
            
            for (var i = 0; i < PreloadCasiaFiles.Count; i++)
            {
                using (var descriptors = new Mat())
                using (var mat = new Mat())
                {
                    _akaze.DetectAndCompute(PreloadCasiaFiles[i], null, out var keypoints, descriptors);
                    Cv2.DrawKeypoints(PreloadCasiaFiles[i], keypoints, mat, Scalar.Red);
                    // Cv2.ImWrite($"{mapPath}\\{i}.jpg", mat);
                }
            }
        }
    }
}