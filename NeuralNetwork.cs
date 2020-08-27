/*
 * Author: Kevin Barkevich
 */

using System;

namespace CSharp_Neural_Network
{

    struct TrainingInput
    {
        public bool[] ins;
        public bool[] outs;
    }
    struct TrainingSet
    {
        public TrainingInput[] set;
    }
    class NeuralNetwork
    {
        public double LearningRate { get; set; }

        private InputPerceptron[] Inputs { get; set; }
        private Perceptron[,] HiddenLayers { get; set; }
        private Perceptron[] Outputs { get; set; }
        private Link[] Links { get; set; }
        private bool Extension { get; set; }

        /// <summary>
        /// Constructor for a single-layer perceptron-based neural network.
        /// </summary>
        /// <param name="_LearningRate">Rate at which the weights of Links are adjusted.</param>
        /// <param name="inputs">Number of inputs to expect for this neural network.</param>
        /// <param name="outputs">Number of outputs to expect for this neural network.</param>
        /// <param name="extension">Whether or not to create an additional constant input set to 1. This is required for some functions, such as AND.</param>
        public NeuralNetwork(double _LearningRate, uint inputs, uint outputs, uint hiddenLayers, uint layersWidth, bool extension)
        {
            if (extension)
                inputs++;
            LearningRate = _LearningRate;
            Inputs = new InputPerceptron[inputs];
            HiddenLayers = new Perceptron[hiddenLayers, layersWidth];
            Outputs = new Perceptron[outputs];
            uint links_count = 0;
            if (hiddenLayers > 0)
            {
                links_count += inputs * layersWidth;
                for(int i = 0; i < hiddenLayers-1; i++)
                {
                    links_count += layersWidth * layersWidth;
                }
                links_count += layersWidth * outputs;
                Links = new Link[inputs * outputs];
            }
            else
            {
                Links = new Link[inputs * outputs];
            }
            Extension = extension;

            uint idCount = 0;
            for (int i = 0; i < inputs; i++)
            {
                InputPerceptron perceptron = new InputPerceptron(idCount);
                Inputs[i] = perceptron;
                idCount++;
            }

            links_count = 0;
            Perceptron[] last_layer = Inputs;
            if (hiddenLayers > 0 && layersWidth > 0)
            {
                for (int i = 0; i < hiddenLayers; i++)
                {
                    Perceptron[] current_layer = new Perceptron[layersWidth];
                    for (int j = 0; j < layersWidth; j++)
                    {
                        Perceptron currentPerceptron = new Perceptron(idCount);
                        HiddenLayers[i, j] = currentPerceptron;
                        current_layer[j] = currentPerceptron;
                        idCount++;
                        for (int k = 0; k < last_layer.Length; k++)
                        {
                            Link newLink = new Link(last_layer[k], currentPerceptron);
                            Links[links_count] = newLink;
                            links_count++;
                        }
                    }
                    last_layer = new Perceptron[layersWidth];
                    current_layer.CopyTo(last_layer, 0);
                }
            }

            for (int i = 0; i < outputs; i++)
            {
                Perceptron perceptron = new Perceptron(idCount);
                Outputs[i] = perceptron;
                idCount++;

                foreach (Perceptron previous in last_layer)
                {
                    Link link = new Link(previous, perceptron);
                    Links[links_count] = link;
                    links_count++;
                }
            }
        }

        /// <summary>
        /// Train the neural network off of a training set.
        /// </summary>
        /// <param name="trainingSet">The training set to train the neural network off of. Must represent a LINEARLY SEPERABLE function.</param>
        public void Train(TrainingSet trainingSet)
        {
            bool done = false;
            uint epochCount = 0;
            while (!done)
            {
                // ---------------------- BEGIN EPOCH ----------------------
                epochCount++;
                Console.WriteLine();
                Console.WriteLine("---------------------- EPOCH: " + epochCount + "----------------------");
                done = true;
                foreach (TrainingInput trainingInput in trainingSet.set)
                {
                    if (((!Extension && trainingInput.ins.Length == Inputs.Length) || (Extension && trainingInput.ins.Length == Inputs.Length - 1)) && trainingInput.outs.Length == Outputs.Length)
                    {
                        // ---------------------- TRAINING RUN PART 1: SET INPUTS & EXPECTED OUTPUTS ----------------------
                        double[] expected = new double[trainingInput.outs.Length];
                        string expectedStr = "[";
                        uint count = 0;
                        foreach (bool currOut in trainingInput.outs)
                        {
                            double ex = 0;
                            if (currOut)
                                ex = 1;
                            expectedStr += ex.ToString() + ",";
                            expected[count] = ex;
                            count++;
                        }
                        expectedStr += "]";
                        for (int i = 0; i < trainingInput.ins.Length; i++)
                        {
                            double input = 0;
                            if (trainingInput.ins[i])
                                input = 1;
                            Inputs[i].Set(input);
                        }
                        if (Extension)
                            Inputs[Inputs.Length - 1].Set(1);
                        // ---------------------- TRAINING RUN PART 2: RUN AND PRINT OUTPUTS + EXPECTED OUTPUTS ----------------------
                        Run();
                        Console.WriteLine("Expected: " + expectedStr);
                        Console.WriteLine();
                        // ---------------------- TRAINING RUN PART 3: BACKPROPAGATE ---------------------- 
                        for (int i = 0; i < expected.Length; i++)
                        {
                            double error = expected[i] - Outputs[i].Read();
                            if (error != 0.0)
                            {
                                done = false;
                                Outputs[i].BackPropogate(LearningRate, error);
                            }
                        }
                    }
                    else
                    {
                        throw new Exception("Error: Training input input or output sizes do not match that of the network!");
                    }
                }
                // ---------------------- END EPOCH ----------------------
            }
            Console.WriteLine();
            Console.WriteLine("Neural Network successfully trained!");
        }

        /// <summary>
        /// Run the neural network with the current inputs stored in the input perceptrons.
        /// </summary>
        private void Run()
        {
            Console.WriteLine("Input: " + Input());
            foreach (Perceptron perceptron in Outputs)
            {
                perceptron.Set();
            }
            Console.WriteLine("Output: " + Output());
        }

        /// <summary>
        /// Obtain a string representing the inputs stored in the input perceptrons.
        /// </summary>
        /// <returns>String representing the inputs stored in the input perceptrons</returns>
        public string Input()
        {
            string res = "[";
            foreach (Perceptron perceptron in Inputs)
            {
                res += perceptron.Read() + ",";
            }
            res += "]";
            return res;
        }

        /// <summary>
        /// Obtain a string representing the outputs stored in the output perceptrons.
        /// </summary>
        /// <returns>String representing the outputs stored in the output perceptrons.</returns>
        public string Output()
        {
            string res = "[";
            foreach (Perceptron perceptron in Outputs)
            {
                res += perceptron.Read() + ",";
            }
            res += "]";
            return res;
        }

        /// <summary>
        /// Override function to convert the Neural Network to a string.
        /// </summary>
        /// <returns>String representing a rough outline of the neural network.</returns>
        public override string ToString()
        {
            string res = "NeuralNetwork{Links:";
            foreach (Link link in Links)
            {
                res += link.ToString() + ",\n";
            }
            res += "}\n\n";

            return res;
        }
    }
}
