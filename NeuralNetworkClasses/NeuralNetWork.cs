

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
        public Neuron FeedForward(List<double> inputSignals)
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

        private void SendSignalsToInputNeurons(List<double> inputSignals)
        {
            for (int i = 0; i < inputSignals.Count; i++)
            {
                var signal = new List<double>() { inputSignals[i] };
                var neuron = Layers[0].Neurons[0];
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
                    hiddenNeurons.Add(new Neuron(lastLayer.Count));
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
                outputNeurons.Add(new Neuron(lastLayer.Count, NeuronType.Output));
            }
            Layers.Add(new Layer(outputNeurons, NeuronType.Output));
        }
    }
}
