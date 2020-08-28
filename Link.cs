/*
 * Author: Kevin Barkevich
 */

using System;

namespace CSharp_Neural_Network
{
    class Link
    {
        public double Weight { get; private set; }
        public Perceptron Start { get; set; }
        public Perceptron End { get; set; }

        /// <summary>
        /// Constructor for a link between perceptrons.
        /// </summary>
        /// <param name="_Start">Perceptron this link originates from.</param>
        /// <param name="_End">Perceptron this link passes a value to.</param>
        public Link(Perceptron _Start, Perceptron _End)
        {
            Start = _Start;
            End = _End;
            var rand = new Random();
            int modifier = rand.Next(2);
            if (modifier == 0)
                modifier = -1;
            else
                modifier = 1;
            Weight = rand.NextDouble() * modifier;
            Start.AddOutputLink(this);
            End.AddInputLink(this);
        }

        /// <summary>
        /// Obtain the weighted value to pass on to the End perceptron.
        /// </summary>
        /// <returns></returns>
        public double Pass()
        {
            return Start.Read() * Weight;
        }

        /// <summary>
        /// Propogate an error to the weight of this link and adjust.
        /// </summary>
        /// <param name="learningRate">Rate at which the weight is adjusted.</param>
        /// <param name="error">The size of the error.</param>
        public void BackPropogate(double learningRate, double error, bool enhancedOutput)
        {
            if (enhancedOutput)
            {
                Console.WriteLine("UPDATING " + ToString());
                Console.WriteLine("Rate: " + learningRate.ToString() + ", xi:" + Start.Read().ToString() + ", error: " + error.ToString());
                Console.WriteLine("From " + Weight.ToString());
            }
            Weight = Weight - (learningRate * Start.Read() * error);
            if (enhancedOutput)
                Console.WriteLine("To   " + Weight.ToString());
        }

        /// <summary>
        /// Override function to convert the link to a string.
        /// </summary>
        /// <returns>The link represented by a string.</returns>
        public override string ToString()
        {
            return "Link{from: " + Start.Id + ", to: " + End.Id + "}";
        }
    }
}
