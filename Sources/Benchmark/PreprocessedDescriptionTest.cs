using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using OpenCvSharp;
using OpenCvSharp.Features2D;
using OpenCvSharp.XFeatures2D;

namespace Benchmark
{
    public class PreprocessedDescriptionTest
    {
        // private const int TotalAttempt = 1000;
        private const string CasiaPath = @"W:\CXXY\CasiaPrepare\images";

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
                // .Take(TotalAttempt)
                .Select(x => Preprocess.FrameTick(Cv2.ImRead(x).Clone()).Clone())
                .ToList();

            // output preloaded files
            Console.WriteLine($"Total loaded: {PreloadCasiaFiles.Count}");

            Console.WriteLine("BRISK benchmark");

            _brisk = BRISK.Create(1);
            _orb = ORB.Create(1500);
            _fSift = SIFT.Create(1500);
            _surf = SURF.Create(30);
            _akaze = AKAZE.Create();

            var timesPool = new List<int>();
            var outputsCountPool = new List<int>();
            var sw = new Stopwatch();
            var swGlobal = new Stopwatch();

            var descriptors = new Mat();

            // ---------------------------------------------------------------------------------------------------------

            swGlobal.Start();
            PreloadCasiaFiles.ForEach(x =>
            {
                sw.Start();
                _brisk.DetectAndCompute(x, null, out var keypoints, descriptors);
                outputsCountPool.Add(keypoints.Length);

                sw.Stop();
                timesPool.Add(sw.Elapsed.Milliseconds);
                sw.Reset();
            });
            swGlobal.Stop();

            Console.WriteLine($"BRISK bench was cleared;\n\t" +
                              $"Elapsed: {swGlobal.Elapsed}\n\t" +
                              $"Average time: {timesPool.Average()}ms for one\n\t" +
                              $"Average keypoints: {outputsCountPool.Average()} for one");

            outputsCountPool.Clear();
            timesPool.Clear();

            // ---------------------------------------------------------------------------------------------------------

            Console.WriteLine("ORB benchmark");
            swGlobal.Start();
            PreloadCasiaFiles.ForEach(x =>
            {
                sw.Start();
                _orb.DetectAndCompute(x, null, out var keypoints, descriptors);
                outputsCountPool.Add(keypoints.Length);
                sw.Stop();
                timesPool.Add(sw.Elapsed.Milliseconds);
                sw.Reset();
            });
            swGlobal.Stop();

            Console.WriteLine($"ORB bench was cleared;\n\t" +
                              $"Elapsed: {swGlobal.Elapsed}\n\t" +
                              $"Average time: {timesPool.Average()}ms for one\n\t" +
                              $"Average keypoints: {outputsCountPool.Average()} for one");

            outputsCountPool.Clear();
            timesPool.Clear();

            // ---------------------------------------------------------------------------------------------------------

            Console.WriteLine("SIFT benchmark");
            swGlobal.Start();
            PreloadCasiaFiles.ForEach(x =>
            {
                sw.Start();
                _fSift.DetectAndCompute(x, null, out var keypoints, descriptors);
                outputsCountPool.Add(keypoints.Length);
                sw.Stop();
                timesPool.Add(sw.Elapsed.Milliseconds);
                sw.Reset();
            });
            swGlobal.Stop();

            Console.WriteLine($"SIFT bench was cleared;\n\t" +
                              $"Elapsed: {swGlobal.Elapsed}\n\t" +
                              $"Average time: {timesPool.Average()}ms for one\n\t" +
                              $"Average keypoints: {outputsCountPool.Average()} for one");

            outputsCountPool.Clear();
            timesPool.Clear();

            // ---------------------------------------------------------------------------------------------------------

            Console.WriteLine("SURF benchmark");

            swGlobal.Start();
            PreloadCasiaFiles.ForEach(x =>
            {
                sw.Start();
                _surf.DetectAndCompute(x, null, out var keypoints, descriptors);
                outputsCountPool.Add(keypoints.Length);
                sw.Stop();
                timesPool.Add(sw.Elapsed.Milliseconds);
                sw.Reset();
            });
            swGlobal.Stop();

            Console.WriteLine($"SURF bench was cleared;\n\t" +
                              $"Elapsed: {swGlobal.Elapsed}\n\t" +
                              $"Average time: {timesPool.Average()}ms for one\n\t" +
                              $"Average keypoints: {outputsCountPool.Average()} for one");

            outputsCountPool.Clear();
            timesPool.Clear();

            // ---------------------------------------------------------------------------------------------------------

            Console.WriteLine("AKAZE benchmark");
            swGlobal.Start();
            PreloadCasiaFiles.ForEach(x =>
            {
                sw.Start();
                _akaze.DetectAndCompute(x, null, out var keypoints, descriptors);
                outputsCountPool.Add(keypoints.Length);
                sw.Stop();
                timesPool.Add(sw.Elapsed.Milliseconds);
                sw.Reset();
            });
            swGlobal.Stop();

            Console.WriteLine($"AKAZE bench was cleared;\n\t" +
                              $"Elapsed: {swGlobal.Elapsed}\n\t" +
                              $"Average time: {timesPool.Average()}ms for one\n\t" +
                              $"Average keypoints: {outputsCountPool.Average()} for one");

            outputsCountPool.Clear();
            timesPool.Clear();
            // dispose collection
            PreloadCasiaFiles.ForEach(x => x.Dispose());
        }
    }
}