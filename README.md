# MNIST NEURAL NETWORK (From scratch, in C++)
## By Porter Thomson, 97.32% Accurracy!
A feedforward neural network for MNIST Digit classification. Implemented from scratch in C++, using the Eigen linear algebra library. No high level ML frameworks are used. Forward Propagation, Backward Propagation, Gradient Descent, and Serialization are all hand written. (See earlier git versions where I use Eigen::VectorXf for forward propagation instead of the faster Eigen::MatrxXf) When testing with  training-60k image dataset had 98.2% accuracy, but thats generalization, the testing-10k dataset had 97.32% accuracy

## Architecture
- Input: 784 (28x28 grayscale pixel matrix)
- Hidden Layer: 128 neurons, ReLU activation
- Output Layer: 10 neurons, Softmax activation
- Weight Intialization: He Initialization
- Cost Function: Cross Entropy

## Project Layout
DataLoader.h / DataLoader.cpp - MNIST idx file parsing and batching
Layer.h / Layer.cpp - a single layer object
Network.h / Network.cpp - connects all the layers together, performs training
main.cpp - CLI interface
unit_tests/ - standalone unit tests for each class
training_data/ - full 60k-image MNIST training set (idx format)
testing_data/ - full 10k-image MNIST testing set (idx format) plus small 10/50/100-image subsets used by the unit tests
network_bin

## Checkpoint format
After 10 epochs the network is automatically saved to a binary file incase of a crash. After a network finishes training, i.e. Accuracy doesn't improve by .05% for 6 epochs or 100 epochs have been ran the network is saved to a binary file and the program exits.


#
Under construction... First time ever writing a README so mind the terrible formatting
