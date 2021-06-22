using System;
using System.IO;
using OpenCvSharp;

namespace Tongji
{
    class Program
    {
        // 10l x 10r images for human, 20 by series
        private const string RootDirectory = @"D:\Palmvein\source";
        private const string OutputDirectory = @"D:\Palmvein\output";
        private const string TrainDirectory = @"C:\Users\CXXY\Desktop\ml\datasets_my\train";
        private const string ValidationDirectory = @"C:\Users\CXXY\Desktop\ml\datasets_my\val";
        
        static void Main(string[] args)
        {
            /*if(Directory.Exists(OutputDirectory))
                Directory.Delete(OutputDirectory, true);

            Directory.CreateDirectory(OutputDirectory);*/
            
            if(Directory.Exists(TrainDirectory))
                Directory.Delete(TrainDirectory, true);

            Directory.CreateDirectory(TrainDirectory);
            
            if(Directory.Exists(ValidationDirectory))
                Directory.Delete(ValidationDirectory, true);

            Directory.CreateDirectory(ValidationDirectory);

            var dirs = Directory.GetDirectories(RootDirectory);
            /*foreach (var dir in dirs)
            {
                var files = Directory.GetFiles(dir, "*.tiff");

                var save = true;
                var itemId = 0;
                
                for (int i = 0, index = 0; i < files.Length; i++)
                {
                    if (i % 20 == 0)
                        index++;
                    
                    if (++itemId == 11)
                    {
                        save = !save;
                        itemId = 1;
                    }
                    
                    if(!save) continue;
                    
                    var path = $"{OutputDirectory}/{index}";
                    if (!Directory.Exists(path))
                        Directory.CreateDirectory(path);

                    var filesInDir = Directory.GetFiles(path).Length;
                    var handle = Cv2.ImRead(files[i]);
                    Cv2.ImWrite($"{path}/{filesInDir}.jpg", handle);
                }
            }*/

            dirs = Directory.GetDirectories(OutputDirectory);
            for (var i = 0; i < dirs.Length; i++)
            {
                var files = Directory.GetFiles(dirs[i]);
                var index = 0;
                foreach (var file in files)
                {
                    var path = ++index > 2 
                        ? $"{TrainDirectory}/{i}" 
                        : $"{ValidationDirectory}/{i}";

                    // var path = $"{TrainDirectory}/{i}";
                    // var roiPath = $"{ValidationDirectory}/{i}";
                    
                    if (!Directory.Exists(path))
                        Directory.CreateDirectory(path);

                    var filesInDir = Directory.GetFiles(path).Length;
                    using (var frame = Cv2.ImRead(file))
                    {
                        var output = VeinExtractor.Program.FrameTick(frame);
                        if(output.roi == null || output.roi.IsDisposed || output.roi.Empty())
                            continue;
                        
                        Cv2.CvtColor(output.roi, frame, ColorConversionCodes.GRAY2RGB);
                        Cv2.ImWrite($"{path}/{filesInDir}.jpg", frame);
                        // Cv2.ImWrite($"{roiPath}/{filesInDir}.jpg", frame);
                    }
                }
            }
            // Console.WriteLine("Hello World!");
        }
    }
} // 218, 295, 72, 223, 162, 180