using Microsoft.VisualStudio.TestTools.UnitTesting;
using NeuralNetworkClasses;
using System.Collections.Generic;

namespace NeuralNetworkClasses.Tests
{
    [TestClass()]
    public class NeuralNetWorkTests
    {
        [TestMethod()]
        public void FeedForwardTest()
        {
            //arrange
            var topology = new Topology(4, 1, 2);
            var neuralNetwork = new NeuralNetWork(topology);
            neuralNetwork.Layers[1].Neurons[0].SetWeights(0.5, -0.1, 0.3, -0.1);
            neuralNetwork.Layers[1].Neurons[1].SetWeights(0.5, -0.3, 0.7, -0.3);
            neuralNetwork.Layers[1].Neurons[0].SetWeights(1.2, 0.8);
            //act
            var result = neuralNetwork.FeedForward(new List<double> { 1, 0, 0, 0 });
            var trueValue = 0.5;
            //assert
            Assert.IsTrue(result.Output > trueValue);
        }
    }
}