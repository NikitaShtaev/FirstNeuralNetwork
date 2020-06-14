using Microsoft.VisualStudio.TestTools.UnitTesting;
using NeuralNetworkClasses;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;

namespace NeuralNetworkClasses.Tests
{
    [TestClass()]
    public class NeuralNetWorkTests
    {
        [TestMethod()]
        public void FeedForwardTest()
        {
            var outputs = new double[] { 0, 0, 1, 0, 0, 0, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1 };
            var inputs = new double[,]
            {
                // Result - sick = 1
                //          healthy = 0

                //Wrong Temperature T
                //Good age A
                //Smoking S
                //Good food F

                //T  A  S  F
                { 0, 0, 0, 0 },
                { 0, 0, 0, 1 },
                { 0, 0, 1, 0 },
                { 0, 0, 1, 1 },
                { 0, 1, 0, 0 },
                { 0, 1, 0, 1 },
                { 0, 1, 1, 0 },
                { 0, 1, 1, 1 },
                { 1, 0, 0, 0 },
                { 1, 0, 0, 1 },
                { 1, 0, 1, 0 },
                { 1, 0, 1, 1 },
                { 1, 1, 0, 0 },
                { 1, 1, 0, 1 },
                { 1, 1, 1, 0 },
                { 1, 1, 1, 1 }
            };

            //arrange
            var topology = new Topology(4, 1, 0.1, 2);
            var neuralNetwork = new NeuralNetWork(topology);
            var results = new List<double>();

            //act
            var difference = neuralNetwork.Learn(outputs, inputs, 10000);
            for (int i = 0; i < outputs.Length; i++)
            {
                var row = NeuralNetWork.GetRow(inputs, i);
                results.Add(neuralNetwork.FeedForward(row).Output);
            }


            //assert
            for (int i = 0; i < results.Count; i++)
            {
                var expected = Math.Round(outputs[i], 2);
                var actual = Math.Round(results[i], 2);
                Assert.AreEqual(expected, actual);
            }
        }
        [TestMethod()]
        public void DataSetTest()
        {
            //Read
            var inputs = new List<double[]>();
            var outputs = new List<double>();
            using (var sr = new StreamReader("datasets_33180_43520_heart.csv"))
            {
                var header = sr.ReadLine();
                while (!sr.EndOfStream)
                {
                    var row = sr.ReadLine();
                    var temp = row.Split(',');
                    var values = row.Split(',').Select(v => Convert.ToDouble(v)).ToList();
                    var output = values.Last();
                    var input = values.Take(values.Count - 1).ToArray();
                    outputs.Add(output);
                    inputs.Add(input);
                }
            }
            var inputSignals = new double[inputs.Count, inputs[0].Length];
            for (int i = 0; i < inputSignals.GetLength(0); i++)
            {
                for (int j = 0; j < inputSignals.GetLength(1); j++)
                {
                    inputSignals[i, j] = inputs[i][j];
                }
            }

            //arrange
            var topology = new Topology(outputs.Count, 1, 0.1, outputs.Count / 2);
            var neuralNetwork = new NeuralNetWork(topology);
            var results = new List<double>();

            //Act
            var difference = neuralNetwork.Learn(outputs.ToArray(), inputSignals, 10);
            for (int i = 0; i < outputs.Count; i++)
            {
                results.Add(neuralNetwork.FeedForward(inputs[i]).Output);
            }


            //Assert
            for (int i = 0; i < results.Count; i++)
            {
                var expected = Math.Round(outputs[i], 2);
                var actual = Math.Round(results[i], 2);
                Assert.AreEqual(expected, actual);
            }
        }
        [TestMethod()]
        public void RecognizeImageTest()
        {
            //Arrange
            var size = 1000;
            var parasitizedPath = @"E:\Downloads_chrome\cell_images\Parasitized\";
            var unparasitizedPath = @"E:\Downloads_chrome\cell_images\Uninfected\";
            var converter = new PictureConverter();
            var testparasitizedImageInput = converter.Convert(@"E:\Projects C#\Project1 - Lessons\FirstNeuralNetwork\NeuralNetworkClassesTests\Images\Parasitized.png");
            var testunparasitizedImageInput = converter.Convert(@"E:\Projects C#\Project1 - Lessons\FirstNeuralNetwork\NeuralNetworkClassesTests\Images\Unparasitized.png");
            var topology = new Topology(testparasitizedImageInput.Count, 1, 0.1, testparasitizedImageInput.Count / 2);
            var nuralNetwork = new NeuralNetWork(topology);
            double[,] parasitizedInputs = GetData(parasitizedPath, converter, testparasitizedImageInput, size);
            double[,] unparasitizedInputs = GetData(unparasitizedPath, converter, testunparasitizedImageInput, size);
            //Act
            nuralNetwork.Learn(new double[] { 1 }, parasitizedInputs, 2);
            nuralNetwork.Learn(new double[] { 0 }, unparasitizedInputs, 2);
            var par = nuralNetwork.FeedForward(testparasitizedImageInput.Select(t => (double)t).ToArray());
            var unpar = nuralNetwork.FeedForward(testunparasitizedImageInput.Select(t => (double)t).ToArray());
            //Assert
            Assert.AreEqual(1, Math.Round(par.Output, 2));
            Assert.AreEqual(0, Math.Round(unpar.Output, 2));
        }

        private static double[,] GetData(string parasitizedPath, PictureConverter converter, List<double> testImageInput, int size)
        {
            var images = Directory.GetFiles(parasitizedPath);
            var result = new double[size, testImageInput.Count];
            for (int i = 0; i < size; i++)
            {
                var image = converter.Convert(images[i]);
                for (int j = 0; j < image.Count; j++)
                {
                    result[i, j] = image[j];
                }
            }

            return result;
        }
    }
}