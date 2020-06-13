
using System.Collections.Generic;

namespace NeuralNetworkClasses
{
    public class Layer
    {
        public List<Neuron> Neurons { get; }
        public int NeuronCount => Neurons?.Count ?? 0;
        public NeuronType Type;
        public Layer(List<Neuron> neurons, NeuronType type = NeuronType.Normal)
        {
            //TODO: check data in Layer class.
            Neurons = neurons;
            Type = type;
        }
        public List<double> GetSignals()
        {
            var result = new List<double>();
            foreach (var neuron in Neurons)
            {
                result.Add(neuron.Output);
            }
            return result;
        }
        public override string ToString()
        {
            return Type.ToString();
        }
    }
}
