/*
 * Author: Kevin Barkevich
 */

using System;
using System.Collections.Generic;

namespace CSharp_Neural_Network
{
    class Perceptron
    {
        protected double Value { get; set; }
        private List<Link> InputLinks { get; set; }
        protected List<Link> OutputLinks { get; set; }
        public uint Id { get; set; }

        /// <summary>
        /// Constructor for a perceptron, an artificial neuron.
        /// </summary>
        /// <param name="id">Id for this perceptron, decorative only.</param>
        public Perceptron(uint id)
        {
            InputLinks = new List<Link>();
            OutputLinks = new List<Link>();
            Value = 0;
            Id = id;
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
            double value = 0.0;
            foreach (Link link in InputLinks)
            {
                value += link.Pass();
            }
            Value = Math.Ceiling(value);
        }

        /// <summary>
        /// Back-propogate from this perceptron, adjusting weights to its associated input links.
        /// </summary>
        /// <param name="learningRate">The learning rate to adjust the input weights by.</param>
        /// <param name="error">The error size.</param>
        public void BackPropogate(double learningRate, double error)
        {
            foreach (Link inputLink in InputLinks)
            {
                inputLink.BackPropogate(learningRate, error);
            }
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
        public InputPerceptron(uint id) : base(id)
        {
            OutputLinks = new List<Link>();
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
}
