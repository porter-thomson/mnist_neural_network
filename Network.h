/* Copyright @2026 Porter Thomson
 * porterrt@uw.edu
 * The Network class represents a full feedforward neural network.
 * built as a vector of Layer objects.
*/
#pragma once
#include <cstdint>
#include <vector>
#include <Eigen/Dense>
#include <utility>
#include <fstream>
#include <string>

#include "Layer.h"

using std::vector;

class Network {
  public:
    // Constructs a Network from a list of layer_sizes and activations.
    // layer_sizes[0] represents the number of nodes in the input layer, 
    // layer_sizes[n-1] represents the number of nodes in the output layer,
    // layer_sizes[1...n-2] represents the number of nodes in each hidden
    // layer respectively.
    // acts[n-1] represents the output layer activation SOFTMAX.
    // acts[...n-2] represents hidden layer activation RELU.
    Network(const vector<uint32_t>& layer_sizes, const vector<Activation>& acts);

    // Reconstructs a Network from a Binary file.
    //  filename - filename of binary file in current directory
    // Reads meta data,
    //  layers   - number of layers in Network
    //  eta      - learning rate of the Network
    // Constructs each layer through Layer(ifstream) constructor.
    Network(const std::string& filename);
    ~Network() = default;
    Network(const Network& other) = delete;
    Network& operator=(const Network& other) = delete;

    // Sets the learning rate of the network.
    //  eta - new learning rate.
    void SetEta(float eta) { eta_ = eta; }

    // Runs one full forward + backward pass through the network on a
    // mini batch.
    //  batch.first  - input matrix, (input_size x batch_size)
    //  batch.second - digit label for each example in the batch
    // Returns the average cross-entropy cost over this batch.
    float Train(std::pair<Eigen::MatrixXf, vector<uint8_t>> batch);

    // Runs input through network and returns the final layer's (softmax).
    //  input - input matrix, (input_size x batch_size)
    Eigen::MatrixXf ForwardProp(const Eigen::MatrixXf& input);

    // Runs input matrix through network and returns the estimate for
    // each example based on the softmax.
    vector<uint8_t> Predict(const Eigen::MatrixXf& input);

    // Serializes the entire network -- layer count, learning rate,
    // then each layer -- to a new binary file at the given path.
    void Save(const std::string& filename);

  private:
    vector<Layer> layers_;  // Number of layers in network.
    float eta_;  // Learning rate
};
