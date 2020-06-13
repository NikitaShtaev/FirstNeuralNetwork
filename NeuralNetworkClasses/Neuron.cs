
using System;
using System.Collections.Generic;

namespace NeuralNetworkClasses
{
    public class Neuron
    {
        public List<double> Weights { get; }
        public List<double> Inputs { get; }
        public NeuronType NeuronType { get; }
        public double Output { get; private set; }
        public double Delta { get; private set; }
        public Neuron(int inputCount, NeuronType type = NeuronType.Normal)
        {
            //TODO: check input data in Neuron class.
            NeuronType = type;
            Weights = new List<double>();
            Inputs = new List<double>();

            InitRandomWeights(inputCount);
        }

        private void InitRandomWeights(int inputCount)
        {
            var rnd = new Random();
            for (int i = 0; i < inputCount; i++)
            {
                if(NeuronType == NeuronType.Input)
                {
                    Weights.Add(1);
                }
                else
                {
                    Weights.Add(rnd.NextDouble());
                }
                Inputs.Add(0);
            }
        }

        public double FeedForward (List<double> inputs)
        {
            //TODO: check input data in method FeedForward.
            for (int i = 0; i < inputs.Count; i++)
            {
                Inputs[i] = inputs[i];
            }
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
        private double SigmoidDx(double x)
        {
            var sigmoid = Sigmoid(x);
            return sigmoid / (1 - sigmoid);
        }
        public void Learn(double error, double learningRate)
        {
            if (NeuronType == NeuronType.Normal)
            {
                return;
            }
            Delta = error * SigmoidDx(Output);
            for (int i = 0; i < Weights.Count; i++)
            {
                Weights[i] = Weights[i] - Inputs[i] * Delta * learningRate;
            }
        }
        public override string ToString()
        {
            return Output.ToString();
        }
    }
}
