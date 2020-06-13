using Microsoft.VisualStudio.TestTools.UnitTesting;
using NeuralNetworkClasses;
using System;
using System.Collections.Generic;

namespace NeuralNetworkClasses.Tests
{
    [TestClass()]
    public class NeuralNetWorkTests
    {
        [TestMethod()]
        public void FeedForwardTest()
        {
            var dataSet = new List<Tuple<double, double[]>>
            {
                // Result - sick = 1
                //          healthy = 0

                //Wrong Temperature T
                //Good age A
                //Smoking S
                //Good food F
                //                                             T  A  S  F
                new Tuple<double, double[]> (0, new double[] { 0, 0, 0, 0 }),
                new Tuple<double, double[]> (0, new double[] { 0, 0, 0, 1 }),
                new Tuple<double, double[]> (1, new double[] { 0, 0, 1, 0 }),
                new Tuple<double, double[]> (0, new double[] { 0, 0, 1, 1 }),
                new Tuple<double, double[]> (0, new double[] { 0, 1, 0, 0 }),
                new Tuple<double, double[]> (0, new double[] { 0, 1, 0, 1 }),
                new Tuple<double, double[]> (1, new double[] { 0, 1, 1, 0 }),
                new Tuple<double, double[]> (0, new double[] { 0, 1, 1, 1 }),
                new Tuple<double, double[]> (1, new double[] { 1, 0, 0, 0 }),
                new Tuple<double, double[]> (1, new double[] { 1, 0, 0, 1 }),
                new Tuple<double, double[]> (1, new double[] { 1, 0, 1, 0 }),
                new Tuple<double, double[]> (1, new double[] { 1, 0, 1, 1 }),
                new Tuple<double, double[]> (1, new double[] { 1, 1, 0, 0 }),
                new Tuple<double, double[]> (0, new double[] { 1, 1, 0, 1 }),
                new Tuple<double, double[]> (1, new double[] { 1, 1, 1, 0 }),
                new Tuple<double, double[]> (1, new double[] { 1, 1, 1, 1 }),
            };

            //arrange
            var topology = new Topology(4, 1, 0.05, 2);
            var neuralNetwork = new NeuralNetWork(topology);
            var results = new List<double>();

            //act
            var difference = neuralNetwork.Learn(dataSet, 100000);
            foreach (var data in dataSet)
            {
                results.Add(neuralNetwork.FeedForward(data.Item2).Output);
            }


            //assert
            for (int i = 0; i < results.Count; i++)
            {
                var expected = Math.Round(dataSet[i].Item1, 3);
                var actual = Math.Round(results[i], 3);
                Assert.AreEqual(expected, actual);
            }
            
            //Assert.IsTrue(result.Output > trueValue);
        }
    }
}