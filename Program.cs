/*
 * Author: Kevin Barkevich
 */

using System;

namespace CSharp_Neural_Network
{
    class Program
    {
        static void Main(string[] args)
        {
            // Set to train the neural network on the OR function.
            TrainingInput[] OR_SET = new TrainingInput[4];
            OR_SET[0].ins = new bool[] { false, false };
            OR_SET[0].outs = new bool[] { false };

            OR_SET[1].ins = new bool[] { false, true };
            OR_SET[1].outs = new bool[] { true };

            OR_SET[2].ins = new bool[] { true, false };
            OR_SET[2].outs = new bool[] { true };

            OR_SET[3].ins = new bool[] { true, true };
            OR_SET[3].outs = new bool[] { true };

            // Set to train the neural network on the AND function.
            TrainingInput[] AND_SET = new TrainingInput[4];
            AND_SET[0].ins = new bool[] { false, false };
            AND_SET[0].outs = new bool[] { false };

            AND_SET[1].ins = new bool[] { false, true };
            AND_SET[1].outs = new bool[] { false };

            AND_SET[2].ins = new bool[] { true, false };
            AND_SET[2].outs = new bool[] { false };

            AND_SET[3].ins = new bool[] { true, true };
            AND_SET[3].outs = new bool[] { true };

            // Set to train the neural network on the NOT function.
            TrainingInput[] NOT_SET = new TrainingInput[2];
            NOT_SET[0].ins = new bool[] { false };
            NOT_SET[0].outs = new bool[] { true };

            NOT_SET[1].ins = new bool[] { true };
            NOT_SET[1].outs = new bool[] { false };

            // Set to train the neural network on the XOR function.
            TrainingInput[] XOR_SET = new TrainingInput[4];
            XOR_SET[0].ins = new bool[] { false, false };
            XOR_SET[0].outs = new bool[] { false };

            XOR_SET[1].ins = new bool[] { false, true };
            XOR_SET[1].outs = new bool[] { true };

            XOR_SET[2].ins = new bool[] { true, false };
            XOR_SET[2].outs = new bool[] { true };

            XOR_SET[3].ins = new bool[] { true, true };
            XOR_SET[3].outs = new bool[] { false };


            TrainingSet TRAINING_SET = new TrainingSet();
            TRAINING_SET.set = AND_SET;  // Choose set to train on, or make your own function set

            Console.WriteLine("Generating Neural Network...");
            NeuralNetwork network = new NeuralNetwork(0.2, 2, 1, 0, 0, Perceptron.FUNCTION_TYPE.SIGMOID, true);
            Console.WriteLine("Generated!");
            network.Train(TRAINING_SET, 0.95, false);
        }
    }
}
