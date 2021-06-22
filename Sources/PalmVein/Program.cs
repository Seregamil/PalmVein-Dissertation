using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using Newtonsoft.Json;
using OpenCvSharp;
using OpenCvSharp.ImgHash;

namespace PalmVein
{
    internal static class Program
    {
        public static Store Store;
        public const string ConnectionUrl = "rtsp://v-sensor:root@192.168.1.20:554/live/ch00_0";
        // public const string ConnectionUrl = "rtsp://ubnt:ubnt@192.168.1.6:554/live/ch00_0";
        private const string StorePath = @"C:\Users\CXXY\Desktop\Store";
        
        private static void Main()
        {
            Store = new Store(StorePath, "*.*"); // load storage
            
            /*var videoGetter = new StreamGet(ConnectionUrl);
            var videoSetter = new StreamShow();
            
            videoGetter.Start();
            videoSetter.Start();

            while (videoGetter.DoesEnabled)
            {
                var key = Cv2.WaitKey(1);
                if (key == 27) // esc
                    break;

                var sw = new Stopwatch();
                sw.Start();
                
                if(videoGetter.Frame.Height == 0)
                    continue;

                // window.ShowImage(videoGetter.Frame);
                videoSetter.Frame = videoGetter.Frame;
                
                // Console.WriteLine($"Elapsed {sw.Elapsed}");
                sw.Stop();
            }*/
            PrepareCasiaTrainData();
        }

        public static void PrepareCasiaTrainData()
        {
            const string casiaPath = @"W:\Disertation\CASIA\images";
            var files = Directory.GetFiles(casiaPath, "*.jpg")
                .Where(x => x.Contains("l_940"))
                .ToList();
            
            if (files.Count == 0)
                throw new Exception("No one file for train");

            const string trainPath = @"W:\Disertation\CASIA\train";
            if(Directory.Exists(trainPath))
                Directory.Delete(trainPath, true);

            Directory.CreateDirectory(trainPath);

            var preprocess = new Preprocess();
            foreach (var file in files)
            {
                // var fileName = file.Replace(casiaPath, trainPath);
                // fileName = fileName.Replace("jpg", "png");
                
                var humanName = file.Remove(0, casiaPath.Length + 1);
                humanName = humanName.Remove(humanName.IndexOf('.'));
                humanName = humanName.Replace("_l_940", "");
                var humanId = Convert.ToInt32(humanName.Substring(0, humanName.IndexOf('_')));

                var directory = $"{trainPath}/{humanId}";
                if (!Directory.Exists(directory))
                    Directory.CreateDirectory(directory);

                var total = Directory.GetFiles(directory).Length;
                
                var src = Cv2.ImRead(file);
                
                Console.WriteLine(humanName);
                var outputScheme = preprocess.FrameTick(src);
                
                if (!Cv2.ImWrite($"{directory}/{total}.jpg", outputScheme.Roi))
                    throw new Exception($"Cant save {directory}/{total}.jpg");
            }
        }
        
        public static void PrepareTrainData()
        {
            const string casiaPath = @"C:\Users\CXXY\Desktop\SimpleStore";
            var files = Directory.GetFiles(casiaPath, "*.png")
                .ToList();
            
            if (files.Count == 0)
                throw new Exception("No one file for train");

            var pHash = PHash.Create();
            var preprocess = new Preprocess();
            foreach (var file in files)
            {
                var fileName = file.Replace("jpg", "png");
                
                var humanName = file.Remove(0, casiaPath.Length + 1);
                humanName = humanName.Remove(humanName.IndexOf('.'));
                
                var src = Cv2.ImRead(file);

                Console.WriteLine($"Equal {humanName}");
                foreach (var humanModel in Store.Humans)
                {
                    var hash = new Mat();
                    pHash.Compute(src, hash);
                
                    Console.WriteLine($"Hash equal with {humanModel.Name}. Val: {pHash.Compare(humanModel.Hash, hash)}");
                }
            }
        }
    }
}