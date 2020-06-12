
using System;
using System.Collections.Generic;

namespace NeuralNetworkClasses
{
    public class Neuron
    {
        public List<double> Weights { get; }
        public NeuronType NeuronType { get; }
        public double Output { get; private set; }
        public Neuron(int inpuCount, NeuronType type = NeuronType.Normal)
        {
            //TODO: check input data in Neuron class.
            NeuronType = type;
            Weights = new List<double>();

            for (int i = 0; i < inpuCount; i++)
            {
                Weights.Add(1);
            }
        }
        public double FeedForward (List<double> inputs)
        {
            //TODO: check input data in method FeedForward.
            var sum = 0.0;
            for (int i = 0; i < inputs.Count; i++)
            {
                sum += inputs[i] * Weights[i];
            }
            if (NeuronType != NeuronType.Input)
            {
                Output = Sigmoid(sum);
            }
            else
            {
                Output = sum;
            }

            return Output;
        }
        private double Sigmoid(double x)
        {
            return 1.0 / (1.0 + Math.Pow(Math.E, -x));
        }
        public void SetWeights(params double[] weights)
        {
            //TODO: delete after teaching on neural network.
            for (int i = 0; i < weights.Length; i++)
            {
                Weights[i] = weights[i];
            }
        }
        public override string ToString()
        {
            return Output.ToString();
        }
    }
}
