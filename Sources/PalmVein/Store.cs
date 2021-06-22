using System;
using System.Collections;
using System.Collections.Generic;
using System.IO;
using OpenCvSharp;
using OpenCvSharp.ImgHash;

namespace PalmVein
{
    public class Store : IEnumerable
    {
        private string _directory;
        private PHash _pHash;
        public List<HumanModel> Humans;
        
        public Store(string directory, string searchOption)
        {
            _directory = directory;
            _pHash = PHash.Create();
            
            Humans = new List<HumanModel>();

            var files = Directory.GetFiles(_directory, searchOption);
            if (files.Length == 0)
                throw new Exception($"No one source data in storage");

            foreach (var file in files)
            {
                var humanName = file.Remove(0, _directory.Length + 1);
                humanName = humanName.Remove(humanName.IndexOf('.'));

                var source = new Mat(file, ImreadModes.Grayscale);
                
                var hash = new Mat();
                _pHash.Compute(source, hash);

                var model = new HumanModel
                {
                    // Guid = Guid.NewGuid(),
                    Name = humanName,
                    Source = source,
                    Hash = hash
                };
                
                Humans.Add(model);
                
                Console.WriteLine($"Human {humanName} was loaded {model.Hash}");
            }
        }

        public IEnumerator GetEnumerator()
        {
            throw new NotImplementedException();
        }
    }
}