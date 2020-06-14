

using System;
using System.Collections.Generic;
using System.Drawing;
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
                var result = Layers.Last().Neurons[0];
                return result;
            }
            else
            {
                var result = Layers.Last().Neurons.OrderByDescending(n => n.Output).First();
                return result;
            }

        }
        public double Learn(double[] expected, double[,] inputs, int age)
        {
            var signals = Normalization(inputs);
            var error = 0.0;
            for (int i = 0; i < age; i++)
            {
                for (int j = 0; j < expected.Length; j++)
                {
                    var output = expected[j];
                    var input = GetRow(signals, j);
                    error += BackPropagation(output, input);
                }
            }
            var result = error / age;
            return result;
        }
        public static double[] GetRow(double[,] matrix, int row)
        {
            var colomns = matrix.GetLength(1);
            var array = new double[colomns];
            for (int i = 0; i < colomns; i++)
            {
                array[i] = matrix[row, i];
            }
            return array;
        }
        private double[,] Scalling(double [,] inputs)
        {
            var result = new double[inputs.GetLength(0), inputs.GetLength(1)];
            for (int column = 0; column < inputs.GetLength(1); column++)
            {
                var min = inputs[0, column];
                var max = inputs[0, column];
                for (int row = 1; row < inputs.GetLength(0); row++)
                {
                    var item = inputs[row, column];
                    if (item< min)
                    {
                        min = item;
                    }
                    if (item>max)
                    {
                        max = item;
                    }
                }
                var divider = max - min;
                for (int row = 1; row < inputs.GetLength(0); row++)
                {
                    result[row, column] = (inputs[row, column] - min) / divider;
                }
            }
            return result;
        }
        private double[,] Normalization(double[,] inputs)
        {
            var result = new double[inputs.GetLength(0), inputs.GetLength(1)];
            for (int column = 0; column < inputs.GetLength(1); column++)
            {
                //Average
                var sum = 0.0;
                for (int row = 0; row < inputs.GetLength(0); row++)
                {
                    sum += inputs[row, column];
                }
                var average = sum / inputs.GetLength(0);

                //Standart Square Error
                var error = 0.0;
                for (int row = 0; row < inputs.GetLength(0); row++)
                {
                    error += Math.Pow((inputs[row, column] - average), 2);
                }
                var standartError = Math.Sqrt(error / inputs.GetLength(0));

                //Change inputs
                for (int row = 0; row < inputs.GetLength(0); row++)
                {
                    result[row, column] = (inputs[row, column] - average) / standartError;
                }
            }
            return result;
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
            var result = difference * difference;
            return result;
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
