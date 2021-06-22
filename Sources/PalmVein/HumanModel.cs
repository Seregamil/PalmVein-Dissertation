using System;
using System.Collections;
using System.Collections.Generic;
using OpenCvSharp;
using OpenCvSharp.ImgHash;

namespace PalmVein
{
    public class HumanModel
    {
        public int Id { get; set; }
        public Guid Guid { get; set; }
        public string Name { get; set; }
        public Mat Source { get; set; }
        public Mat Hash { get; set; }
        
        public IEnumerable<KeyPoint> KeyPoints { get; set; }
        
        public IEnumerable<KeyPoint> KeyPointsObject { get; set; }
        public IEnumerable<KeyPoint> KeyPointsScene { get; set; }
        
        public Mat Descriptors { get; set; }

    }
}