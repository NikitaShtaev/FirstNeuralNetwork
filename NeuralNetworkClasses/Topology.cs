

using System.Collections.Generic;

namespace NeuralNetworkClasses
{
    public class Topology
    {
        public int InputCount { get;  }
        public int OutputCount { get; }
        public List<int> HiddenLayers { get; }
        public Topology(int inputCount, int outputCount, params int[] layers)
        {
            //TODO: check data in class TOPOLOGY.
            InputCount = inputCount;
            OutputCount = outputCount;
            HiddenLayers = new List<int>();
            HiddenLayers.AddRange(layers);
        }
    }
}
