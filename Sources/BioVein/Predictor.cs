using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using OpenCvSharp;
using OpenCvSharp.Dnn;

namespace BioVein
{
    public class Predictor
    {
        private float _threshold = 9999f;
        private string _pathToModel;
        private string _labelsPath;
        
        private List<string> _classes;
        private Net _net;
        
        public Predictor(string pathToModel, string labelsPath, float threshold)
        {
            _pathToModel = pathToModel;
            _labelsPath = labelsPath;
            _threshold = threshold;
            
            _classes = new List<string>();
            Directory
                .GetDirectories(labelsPath)
                .ToList()
                .ForEach(x =>
                {
                    _classes.Add(new DirectoryInfo(x).Name);
                });
            
            Console.WriteLine($"Total classes loaded: {_classes.Count}");
            
            _net = Net.ReadNetFromONNX(_pathToModel);
        }
        
        public void Predict(Mat frame)
        {
            using (var rgb = new Mat()) 
            {
                Cv2.CvtColor(frame, rgb, ColorConversionCodes.GRAY2RGB);

                using (var blob = CvDnn.BlobFromImage(rgb,
                    size: frame.Size(),
                    crop: false,
                    scaleFactor: 1.0 / 255.0))
                {
                    _net.SetInput(blob);
                }
            }

            using (var detection = _net.Forward())
            {
                // Console.WriteLine(detection); // 1*5*CV_32FC1 ->  1 row 5 classes

                detection.GetArray(out float[] values);
                
                // // Finding max
                var max = values.Max();

                // // Positioning max
                var position = Array.IndexOf(values, max);

                // Console.WriteLine($"Predicted: {_labels[position]}");
                for (var i = 0; i != values.Length; i++)
                {
                    // var confidence = detection.At<float>(0, i);
                    Console.Write($"{values[i]}, ");
                }

                Console.WriteLine(max >= _threshold ? $"; Max: {max}; Predicted: {_classes[position]}" : ";");
            }
        }
    }
}