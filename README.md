# CSharp-Neural-Network
A perceptron-based Neural Network in C#.

This project uses the .NET Core framework.

## Function Types
This is a multi-layer perceptron-based neural network. It is able to learn, with no hidden layers, linearly seperable functions such as:
* AND
* OR
* NOT
* NAND

With appropriately numbered and sized hidden layers, the neural network is also able to learn non-linearly seperable functions such as:
* XOR

## Execution
This program should be run via Program.cs. No parameters are needed to train this neural network.

To train this network on a new function or with new data, add new datasets mirroring OR_SET, AND_SET, XOR_SET, etc in Program.cs.
