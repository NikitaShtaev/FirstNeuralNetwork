

using System;
using System.Collections.Generic;
using System.Linq;

namespace NeuralNetworkClasses
{
    public class NeuralNetWork
    {
        public List<Layer> Layers { get; }
        public Topology Topology { get; }
        public NeuralNetWork(Topology topology)
        {
            //TODO: check data in NeuralNetwork class.
            Topology = topology;
            Layers = new List<Layer>();
            CreateInputLayer();
            CreateHiddenLayers();
            CreateOutputLayers();
        }
        public Neuron FeedForward(params double[] inputSignals)
        {
            //TODO: chech data in method FeedForward for class NeuralNetwork (inputSignals.Count = Topology.InputCount).
            SendSignalsToInputNeurons(inputSignals);
            FeedForwardAllLayersAfeterInput();
            if (Topology.OutputCount == 1)
            {
                return Layers.Last().Neurons[0];
            }
            else
            {
                return Layers.Last().Neurons.OrderByDescending(n => n.Output).First();
            }

        }
        public double Learn(List<Tuple<double, double[]>> dataSet, int age)
        {
            var error = 0.0;
            for (int i = 0; i < age; i++)
            {
                foreach (var data in dataSet)
                {
                    error += BackPropagation(data.Item1, data.Item2);
                }
            }
            return error / age;
        }
        private double BackPropagation(double expected, params double[] inputs)
        {
            var actual = FeedForward(inputs).Output;
            var difference = actual - expected;
            foreach (var neuron in Layers.Last().Neurons)
            {
                neuron.Learn(difference, Topology.LearningRate);
            }
            for (int j = Layers.Count-2; j >=0 ; j--)
            {
                var layer = Layers[j];
                var previousLayer = Layers[j + 1];
                for (int i = 0; i < layer.NeuronCount; i++)
                {
                    var neuron = layer.Neurons[i];
                    for (int k = 0; k < previousLayer.Neurons.Count; k++)
                    {
                        var previousNeuron = previousLayer.Neurons[k];
                        var error = previousNeuron.Weights[i] * previousNeuron.Delta;
                        neuron.Learn(error, Topology.LearningRate);
                    }
                }
            }
            return difference * difference;
        }
        private void FeedForwardAllLayersAfeterInput()
        {
            for (int i = 1; i < Layers.Count; i++)
            {
                var previousLayerSignals = Layers[i - 1].GetSignals();
                var layer = Layers[i];
                foreach (var neuron in layer.Neurons)
                {
                    neuron.FeedForward(previousLayerSignals);
                }

            }
        }

        private void SendSignalsToInputNeurons(params double[] inputSignals)
        {
            for (int i = 0; i < inputSignals.Length; i++)
            {
                var signal = new List<double>() { inputSignals[i] };
                var neuron = Layers[0].Neurons[i]; //Changed from Neurons[0]
                neuron.FeedForward(signal);
            }
        }

        private void CreateInputLayer()
        {
            var inputNeurons = new List<Neuron>();
            for (int i = 0; i < Topology.InputCount; i++)
            {
                inputNeurons.Add(new Neuron(1, NeuronType.Input));
            }
            Layers.Add(new Layer(inputNeurons, NeuronType.Input));
        }

        private void CreateHiddenLayers()
        {
            for (int j = 0; j < Topology.HiddenLayers.Count; j++)
            {
                var hiddenNeurons = new List<Neuron>();
                var lastLayer = Layers.Last();
                for (int i = 0; i < Topology.HiddenLayers[j]; i++)
                {
                    hiddenNeurons.Add(new Neuron(lastLayer.NeuronCount));
                }
                Layers.Add(new Layer(hiddenNeurons));
            }
        }

        private void CreateOutputLayers()
        {
            var outputNeurons = new List<Neuron>();
            var lastLayer = Layers.Last();
            for (int i = 0; i < Topology.OutputCount; i++)
            {
                outputNeurons.Add(new Neuron(lastLayer.NeuronCount, NeuronType.Output));
            }
            Layers.Add(new Layer(outputNeurons, NeuronType.Output));
        }
    }
}
