# CSharp-Neural-Network
A perceptron-based Neural Network in C#.

This project uses the .NET Core framework.

## Function Types
This is currently a single-layer perceptron-based neural network. That means this network is able to learn linearly seperable functions such as:
* AND
* OR
* NOT
* NAND

Non-linearly seperable functions are not able to be learned by this neural network. Example:
* XOR

## Execution
This program should be run via Program.cs. No parameters are needed to train this neural network.

To train this network on a new function or with new data, add new datasets mirroring OR_SET, AND_SET, etc in Program.cs.
