using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using OpenCvSharp;
using OpenCvSharp.Dnn;

namespace VeinExtractor.Ml
{
    public class Predictor
    {
        private const float Threshold = 0.30f;
        
        private string _modelPath;
        private List<string> _classes;
        private Net _net;
        
        public Predictor(string labelsPath, string modelPath)
        {
            _modelPath = modelPath;

            _classes = new List<string>();
            Directory
                .GetDirectories(labelsPath)
                .ToList()
                .ForEach(x =>
                {
                    _classes.Add(new DirectoryInfo(x).Name);
                });
            
            _net = Net.ReadNetFromONNX(modelPath);
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

                Console.WriteLine(max >= Threshold ? $"; Max: {max}; Predicted: {_classes[position]}" : ";");
            }
        }
        
        public float[] GetPredictionResult(Mat frame)
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

            float[] values;
            using (var detection = _net.Forward())
            {
                // Console.WriteLine(detection); // 1*5*CV_32FC1 ->  1 row 5 classes

                detection.GetArray(out values);
            }

            return values;
        }
    }
}