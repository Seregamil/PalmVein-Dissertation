using System;
using System.Collections.Generic;
using System.Diagnostics.CodeAnalysis;
using System.IO;
using System.Linq;
using System.Net;
using System.Threading.Tasks;
using Newtonsoft.Json;
using OpenCvSharp;
using OpenCvSharp.Features2D;
using OpenCvSharp.Flann;
using OpenCvSharp.ImgHash;

namespace PalmVein
{
    public class FeaturesHandler
    {
        private ORB _orb;
        
        private BFMatcher _bfMatcher;

        private readonly Store _store;
        private const double LowRatio = 0.95;

        private PHash _pHash;
        
        public FeaturesHandler(Store store)
        {
            _pHash = PHash.Create();
            _store = store;
            
            /*  https://habr.com/ru/post/414459/
             *  nfeatures – максимальное число особых точек
                scaleFactor – множитель для пирамиды изображений, больше единицы. Значение 2 реализует классическую пирамиду.
                nlevels – число уровней в пирамиде изображений.
                edgeThreshold – число пикселов у границы изображения, где особые точки не детектируются.
                firstLevel – оставить нулём.
                WTA_K – число точек, которое требуется для одного элемента дескриптора. Если равно 2, то сравнивается яркость двух случайно выбранных пикселов. Это и нужно.
                scoreType – если 0, то в качестве меры особенности используется харрис, иначе – мера FAST (на основе суммы модулей разностей яркостей в точках окружности). Мера FAST чуть менее стабильная, но быстрее работает.
                patchSize – размер окрестности, из которой выбираются случайные пикселы для сравнения. Код производит поиск и сравнение особых точек на двух картинках, «templ.bmp» и «img.bmp»
             */
            
            // TODO: Work with nFeatures and edge threshold
            _orb = ORB.Create(scoreType: ORBScoreType.Harris, nFeatures: 1500, edgeThreshold: 0);
            
            // _bfMatcher = new BFMatcher(NormTypes.Hamming | NormTypes.L2, true); // for stable 'match' 
            _bfMatcher = new BFMatcher(NormTypes.Hamming | NormTypes.L2 | NormTypes.L2SQR); // for knn match

            foreach (var human in store.Humans)
            { // prepare human data
                var (points, descriptors) = CalculateKeyPoints(human.Source);
                
                human.KeyPoints = points;
                human.Descriptors = descriptors;
            }
        }

        public (IEnumerable<KeyPoint> points, Mat descriptors) CalculateKeyPoints(Mat frame)
        {
            var descriptors = new Mat();
            _orb.DetectAndCompute(frame, null, out var points, descriptors);
            return (points, descriptors);
        }

        public void GetBestSimilarityModel(Mat query)
        {
            const int matchesPoolCount = 10;
            
            Task.Factory.StartNew(() =>
            {
                HumanModel goodHuman = null;
                var average = 9999f;
                var matchesCount = 0;
                // var matchesPool = new List<DMatch>();

                foreach (var storeHuman in _store.Humans)
                {
                    /*
                        DMatch.distance - Distance between descriptors. The lower, the better it is.
                        DMatch.trainIdx - Index of the descriptor in train descriptors
                        DMatch.queryIdx - Index of the descriptor in query descriptors
                        DMatch.imgIdx - Index of the train image.
                     */
                    var result = _bfMatcher.Match(query, storeHuman.Descriptors);
                    var goodMatches = result
                        // .Where(x => x.Distance > 10)
                        .OrderBy(x => x.Distance)
                        // .Take(matchesPoolCount)
                        .ToArray();

                    // if(goodMatches.Length < matchesPoolCount)
                    //     continue;
                    
                    // Console.Write($"{storeHuman.Name} -> ");
                    // foreach (var goodMatch in goodMatches)
                    // {
                    //     Console.Write($"{goodMatch.Distance}, ");
                    // }
                    // Console.WriteLine();
                    
                    // var distance = goodMatches
                    //     .Average(x => x.Distance);
                    
                    // if(distance > average)
                    //     continue;
                    
                    if(matchesCount > goodMatches.Length)
                        continue;
                    
                    goodHuman = storeHuman;
                    matchesCount = goodMatches.Length;
                    // average = distance;

                }

                if (goodHuman != null)
                {
                    // var score = (double) average / goodHuman.Descriptors.Rows < 33;
                    // if(average / matchesPoolCount < 50)
                        Console.WriteLine($"OUT: {goodHuman.Name} Total matches: {matchesCount} Average: {average}");
                    // else 
                    //     Console.WriteLine("No one best matches");
                }
                // else
                // {
                //     Console.WriteLine("No one best matches");
                // }
            });
        }
        
        [SuppressMessage("ReSharper.DPA", "DPA0002: Excessive memory allocations in SOH", MessageId = "type: OpenCvSharp.DMatch[]")]
        public void GetBestSimilarityKnnModel(Mat query)
        {
            const int bestMatchesThreshold = 600;
            Task.Factory.StartNew(() =>
            {
                HumanModel goodHuman = null;
                var matchesCount = 0;
                var distanceValue = 0;
                var pHashValue = 30.0;

                foreach (var storeHuman in _store.Humans)
                {
                    using (var hash = new Mat())
                    {
                        _pHash.Compute(query, hash);

                        var localPHash = _pHash.Compare(storeHuman.Hash, hash);
                        if (localPHash <= 5)
                        {
                            Console.WriteLine($"PHASH COMPARED SUCCESS");

                            if (localPHash == 0)
                            {
                                // if equals
                                matchesCount = int.MaxValue;
                                goodHuman = storeHuman;
                                break;
                            }

                            if (localPHash < pHashValue)
                            {
                                matchesCount = int.MaxValue;
                                goodHuman = storeHuman;
                                pHashValue = localPHash;
                                continue;
                            }
                        }
                    }

                    /*
                        DMatch.distance - Distance between descriptors. The lower, the better it is.
                        DMatch.trainIdx - Index of the descriptor in train descriptors
                        DMatch.queryIdx - Index of the descriptor in query descriptors
                        DMatch.imgIdx - Index of the train image.
                     */
                    var result = _bfMatcher.KnnMatch(query, storeHuman.Descriptors, 2);
                    var best = new List<DMatch>();

                    foreach (var dMatches in result)
                    {
                        var m = dMatches[0];
                        var n = dMatches[1];

                        if (m.Distance < LowRatio * n.Distance)
                            best.Add(m);
                    }
                    
                    if(best.Count < bestMatchesThreshold)
                        continue;
                    
                    var sumOfDistance = best.Sum(x => x.Distance);
                    // Console.Write($"{storeHuman.Name} ({best.Count}) - {sumOfDistance}; ");

                    if (best.Count <= matchesCount || !(sumOfDistance > distanceValue)) 
                        continue;
                    
                    matchesCount = best.Count;
                    goodHuman = storeHuman;
                }

                if (goodHuman != null)
                {
                    
                    Console.WriteLine($"OUT: {goodHuman.Name} Total best matches: {matchesCount}");
                }
            });
        }
    }
}