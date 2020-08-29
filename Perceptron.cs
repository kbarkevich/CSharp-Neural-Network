/*
 * Author: Kevin Barkevich
 */

using System;
using System.Collections.Generic;
using System.Numerics;
using System.Threading.Tasks;

namespace CSharp_Neural_Network
{
    class Perceptron
    {
        public enum FUNCTION_TYPE { SIGMOID };
        readonly static Func<double, double> SIGMOID_FUNCTION = (double input) => { return 1 / (1 + Math.Pow(Math.E, -input)); };
        readonly static Func<double, double> SIGMOID_PRIME_FUNCTION = (double input) => { return SIGMOID_FUNCTION(input) * (1 - SIGMOID_FUNCTION(input)); };

        protected double Z { get; set; }
        protected double Value { get; set; }
        protected List<Link> InputLinks { get; set; }
        protected List<Link> OutputLinks { get; set; }
        public uint Id { get; set; }
        private Func<double, double> Sigma { get; set; }
        protected Func<double, double> SigmaPrime { get; set; }
        public double Error { get; protected set; }

        /// <summary>
        /// Constructor for a perceptron, an artificial neuron.
        /// </summary>
        /// <param name="id">Id for this perceptron, decorative only.</param>
        /// <param name="functionType">Activation function to use.</param>
        public Perceptron(uint id, FUNCTION_TYPE functionType)
        {
            InputLinks = new List<Link>();
            OutputLinks = new List<Link>();
            Z = 0;
            Value = 0;
            Id = id;
            Sigma = SIGMOID_FUNCTION;
            SigmaPrime = SIGMOID_PRIME_FUNCTION;
        }

        /// <summary>
        /// Add to this perceptron an associated link that feeds this perceptron with input.
        /// </summary>
        /// <param name="link">Link to associate.</param>
        public void AddInputLink(Link link)
        {
            InputLinks.Add(link);
        }

        /// <summary>
        /// Add to this perceptron an associated link that this perceptron feeds with output.
        /// </summary>
        /// <param name="link">Link to associate.</param>
        public void AddOutputLink(Link link)
        {
            OutputLinks.Add(link);
        }

        /// <summary>
        /// Tell this perceptron to read its associated input links and store the resulting value after a Step Function.
        /// </summary>
        public void Set()
        {
            Z = 0;
            foreach (Link link in InputLinks)
            {
                Z += link.Pass();
            }
            Value = Sigma(Z);
            
        }

        /// <summary>
        /// Back-propogate from this perceptron, adjusting weights to its associated input links.
        /// </summary>
        /// <param name="learningRate">The learning rate to adjust the input weights by.</param>
        /// <param name="error">The error size.</param>
        /// <param name="enhancedOutput">Print enhanced debugging output.</param>
        public Task BackPropogate(double learningRate, bool enhancedOutput)
        {
            CalculateError();
            foreach (Link inputLink in InputLinks)
            {
                inputLink.BackPropogate(learningRate, Error, enhancedOutput);
            }
            return Task.CompletedTask;
        }

        /// <summary>
        /// Calculate and store the error based on the stored valus and the Sigma Prime function.
        /// </summary>
        public void CalculateError()
        {
            double sum = 0.0;
            foreach (Link link in OutputLinks)
            {
                sum += (link.Weight * link.End.Error);
            }
            Error = sum * SigmaPrime(Z);
        }

        /// <summary>
        /// Read the stored value of this perceptron.
        /// </summary>
        /// <returns>The stored value of this perceptron.</returns>
        public double Read()
        {
            return Value;
        }

        /// <summary>
        /// Override function to convert the perceptron to a string.
        /// </summary>
        /// <returns>The perceptron represented as a string.</returns>
        public override string ToString()
        {
            return "Perceptron{Input Links: " + InputLinks.Count.ToString() + ", Output Links: " + OutputLinks.Count.ToString() + "}";
        }
    }

    class InputPerceptron : Perceptron
    {
        /// <summary>
        /// Constructor for a perceptron that accepts input from externally rather than from other perceptrons' links.
        /// </summary>
        /// <param name="id">Id for this perceptron, decorative only.</param>
        public InputPerceptron(uint id, FUNCTION_TYPE functionType) : base(id, functionType)
        {
            OutputLinks = new List<Link>();
            Z = 0;
            Value = 0;
        }
        
        /// <summary>
        /// Set the input for this external-input perceptron, after applying a Step Function.
        /// </summary>
        /// <param name="value">Value to set this perceptron to.</param>
        public void Set(double value)
        {
            Value = Math.Ceiling(value);
        }
    }

    class OutputPerceptron : Perceptron
    {
        /// <summary>
        /// Constructor for a perceptron that accepts input from externally rather than from other perceptrons' links.
        /// </summary>
        /// <param name="id">Id for this perceptron, decorative only.</param>
        /// <param name="functionType">Activation function to use.</param>
        public OutputPerceptron(uint id, FUNCTION_TYPE functionType) : base(id, functionType)
        {
            InputLinks = new List<Link>();
            Z = 0;
            Value = 0;
        }

        /// <summary>
        /// Back-propogate from this perceptron, adjusting weights to its associated input links.
        /// </summary>
        /// <param name="learningRate">The learning rate to adjust the input weights by.</param>
        /// <param name="error">The error size.</param>
        /// <param name="enhancedOutput">Print enhanced debugging output.</param>
        public Task BackPropogate(double learningRate, double expected, bool enhancedOutput)
        {
            CalculateError(expected);
            foreach (Link inputLink in InputLinks)
            {
                inputLink.BackPropogate(learningRate, Error, enhancedOutput);
            }
            return Task.CompletedTask;
        }

        /// <summary>
        /// Calculate and store the error based on the stored values, Sigma Prime function, and the provided expected input.
        /// </summary>
        /// <param name="expected"></param>
        public void CalculateError(double expected)
        {
            Error = (Value - expected) * SigmaPrime(Z);
        }

    }
}
