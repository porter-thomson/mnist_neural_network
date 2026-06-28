# MNIST NEURAL NETWORK (From scratch, in C++)
## By Porter Thomson, 97.32% Accuracy!
A feedforward neural network for MNIST Digit classification. Implemented from scratch in C++, using the Eigen linear algebra library. No high level ML frameworks are used. Forward Propagation, Backward Propagation, Gradient Descent, and Serialization are all hand written. (See earlier git versions where I use `Eigen::VectorXf` for forward propagation instead of `Eigen::MatrixXf`. Later changed so that multiple images are processed at the same time.) When testing with  training-60k image dataset had 98.2% accuracy and the testing-10k dataset had 97.32% accuracy.



## Architecture
- Input: 784 neurons (28x28 grayscale pixel matrix)
- Hidden Layer: 128 neurons, ReLU activation
- Output Layer: 10 neurons, Softmax activation
- Weight Intialization: He Initialization
- Cost Function: Cross Entropy

## Project Layout
```
DataLoader.h / DataLoader.cpp - MNIST idx file parsing and batching
Layer.h / Layer.cpp           - a single layer object
Network.h / Network.cpp       - connects all the layers together, performs training
main.cpp                      - CLI interface
unit_tests/                   - standalone unit tests for each class
training_data/                - full 60k-image MNIST training set (idx format)
testing_data/                 - full 10k-image MNIST testing set (idx format) plus small 10/50/100-image subsets used by the unit tests
network_bin/                  - directory of networks trained by the model, including the model with 97.32% accuracy.
```
## Installation
```bash

git clone --recurse-submodules https://github.com/porter-thomson/mnist_neural_network.git
```

## Building
```bash
make
```
Creates an executable named `nn` in the main directory.

## Usage
| Flag | Effect |
|------|--------|
| `--train` | Trains the network |
| `--test` | Evaluates the network on the test set |
| `--confusion` | Prints out a confusion matrix after each test set |
| `--eta <float>` | Overrides the learning rate |
| `--file <path>` | Loads a checkpoint instead of starting fresh |


## Code
The Network architecture is hardcoded into `main.cpp`.
```cpp
Network network({784, 128, 10}, {Activation::RELU, Activation::SOFTMAX});
```

In order to change the architecture of the network `main.cpp` needs to be modified.

To change the minimum improvement between epochs, the patience, or the max number of epochs run during a single execution adjust these values in `TrainNetwork` in `main.cpp` accordingly then recompile.
```cpp
  const float min_improvement = 0.0005f;
  const int patience = 6;
  const int max_epochs = 100;
```

## Checkpoint format
After 10 epochs the network is automatically saved to a binary file incase of a crash. After a network finishes training, i.e. Accuracy doesn't improve by .05% for 6 epochs or after 100 epochs the network is saved to a binary file and the program exits.
```
[uint8_t layer count]
[float   eta]
for each layer:
  [uint16_t rows]
  [uint16_t cols]
  [rows * cols] weights
  [rows float]  bias
  [uint8_t activation]
```
I chose not to add a checksum or magic number because of the extra logic.

## Results
After training the network for many epochs it converged to 97.32%
```bash
./nn --test --file network_bin/97.32-accuracy.bin --confusion
Welcome to Neural Network, built by Porter Thomson
Loading Testing Data
Testing Data Loaded!
Test accuracy: 97.32%
Confusion Matrix (rows = true label, cols = predicted label):
           0     1     2     3     4     5     6     7     8     9
   0 |   970     0     1     2     1     2     1     2     1     0
   1 |     0  1123     3     1     0     1     3     0     4     0
   2 |     5     1  1001     6     2     0     2     8     7     0
   3 |     0     0     4   992     0     1     0     6     5     2
   4 |     0     0     2     1   960     0     6     2     2     9
   5 |     5     0     0    15     1   858     6     0     5     2
   6 |     5     3     1     1     7     5   933     1     2     0
   7 |     0     7    15     1     1     0     0   997     1     6
   8 |     5     1     3     8     3     4     5     6   937     2
   9 |     5     5     2    10    14     4     0     6     2   961

Per-Digit Precision / Recall:
   Digit   Precision      Recall
       0    97.4874%    98.9796%
       1    98.5088%    98.9427%
       2    96.9961%    96.9961%
       3    95.6606%    98.2178%
       4    97.0677%    97.7597%
       5    98.0571%    96.1883%
       6    97.5941%    97.3904%
       7    96.9844%    96.9844%
       8    96.9979%    96.2012%
       9    97.8615%    95.2428%
```
**But what is Precision and Recall?**
Precision (False Positive) - Of everything predicted to be a given digit, what fraction was actually correct. In confusion matrix terms it's the diagonal divided by the sum of the column.

Recall (False Negative) - Of everything that's actually a given digit, what fraction did the network successfully catch. i.e. diagonal divided by the sum of the row.

**Interpretation:**
The handwritten digit 3 had the lowest precision at 95.7% meaning it had the highest proportion of false positives. Looking at the confusion matrix we can see that the network commonly mistook the digit 5 and 9 (15 and 10 false positive respectively) for a 3.

Digit 9 had the lowest recall at 95.2% meaning it had the highest proportion of false negatives. Drawing from the confusion matrix we can see that the network commonly mistook the digit 9 for a 3 or 4 (10 and 14 respectively).
## References

When learning about neural network architecture I found these helpful.
- http://neuralnetworksanddeeplearning.com/chap2.html
Great for understanding the notation and how the formulas and variables come together
- https://www.youtube.com/playlist?list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi
Very interesting series with great definitions supported by visuals. Chapters 2, 3, and 4 are excellent to understand back propagation.
- https://www.geeksforgeeks.org/deep-learning/kaiming-initialization-in-deep-learning/
For He/Kaiming Initialization
- https://libeigen.gitlab.io/
Eigen library source code 
- https://en.wikipedia.org/wiki/Backpropagation
Skimmed through wiki article 

---

## Walkthrough

There is some important notation needed to understand the math behind my neural network. 
$$
\begin{align*}
x &\in \mathbb{R}^{n}          &&\text{input vector $n$ neurons} \\
W &\in \mathbb{R}^{m \times n} &&\text{weight matrix } (m \text{ neurons}, n \text{ inputs}) \\
b &\in \mathbb{R}^{m}          &&\text{bias vector} \\
z &= W \times x + b &&\text{weighted input (pre-activation)} \\
a &= \sigma(z)        &&\text{layer output (post-activation)}
\end{align*}
$$

This notation refers to a single layer. Since networks are made up of multiple layers, to differentiate between layers the superscript $l$ refers to the $l$th layer in the network. Thus:
$$
\begin{align*}
&x^{l}, W^l, b^l, etc. &&\text{correspond to $l$th layer}  \\
\end{align*}
$$

**Network Initialization**
When creating a neural network it is important to initialize the weights to give the model a starting point for the optimization process. There are many different techniques for weight initalization but I settled on the He initialization since it was works well with the ReLU activation function.
$$
\begin{align*}
W \rightarrow \mathcal{N} (0, \sqrt\frac{2}{n})
\end{align*}
$$
Here is the associated formula, I linked a article explaining the process in the References section.

**Activation Function**
A non-linear function applied to a layer's weighted output, determining what the layer actually outputs allowing the network to learn actual patterns. For this neural network I used two activation functions, ReLU and Softmax.

ReLU - Rectified Linear Unit. This activation function squashes all negative output neurons to 0. This is used in the hidden layers to help solve the vanishing gradient problem. Represented by the formula:
$$
f(z_i) = max(0, z_i)
$$

Softmax - Creates a probability distribution from the weighted output. Used in the output layer, where the digit with the highest probability becomes the prediction. Represented by the formula:
$$
softmax(z_i) = \frac{e^{z_i}}{\sum_{j}^{K}e^{z_j}}
$$


**Forward Propagation**
The process of computing a network's output by passing data sequentially through the layers.

When given an input vector $x$, compute the weighted output for the 1st hidden layer.
$$
\begin{align*}
z^1 = W^1 \times x + b^1
\end{align*}
$$
Then apply the given activation function.
$$
\begin{align*}
a^1 = \sigma(z^1)
\end{align*}
$$

Continue this process for each layer until the output layer has been calculated, with equivalent formula:
$$
\begin{align*}
a^l = \sigma(W^l \times a^{l-1} + b^l)
\end{align*}
$$


**Backward Propagation**
This is actually how the neural network "learns" and the hardest for me to understand. 

... Under contruction.
