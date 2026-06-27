/* Copyright @2026 Porter Thomson
 * porterrt@uw.edu
 * The Layer class represents a single Layer in a neural network
*/

#pragma once
#include <cstdint>
#include <Eigen/Dense>
#include <fstream>

// The activation function applied to a Layer's output.
// RELU is intended for hidden layer, SOFTMAX is intended for the
// output layer.
enum class Activation {
  RELU,
  SOFTMAX
};

class Layer {
  public:
    // Construct a randomly initialized Layer.
    //  input_size  - Columns of weights_ (Nodes in l-1 layer)
    //  output_size - Rows of weights_ (Nodes in l layer)
    //  act         - activation function applied during Forward()
    // Weights are He-initialized, bias starts at zero.
    Layer(int input_size, int output_size, Activation act) :
      act_(act) {
        weights_.resize(output_size, input_size);
        HeInitialization();
        bias_ = Eigen::VectorXf::Zero(output_size);
    }

    // Reconstructs a Layer from a Binary file.
    // Expects the exact byte layout written by Save():
    // rows, cols, weight data, bias data, 1-byte activation
    // tag.
    Layer(std::ifstream& file);
    ~Layer() = default;
    Layer(const Layer& other) = default;
    Layer& operator=(const Layer& other) = default;

    // Runs one forward pass: activation_ = act(weights_ * input + bias_).
    // input must be (input_size x batch_size), else runtime_error is thrown.
    // Result is cached in activation_
    void Forward(const Eigen::MatrixXf& input);

    // Runs one backward pass for this layer given the error signal
    // propagated from the next layer (or, for the output layer, the
    // (a - y) softmax cross-entropy gradient computed by Network).
    //   error           - dC/dz for this layer's output; may be modified
    //                      in place (e.g. masked by the ReLU derivative)
    //   prev_activation - this layer's input during the matching
    //                      Forward() call (i.e. the previous layer's
    //                      activation, or the raw network input for
    //                      layer 0)
    //   eta             - learning rate for this update
    // Updates weights_ and bias_ in place, and returns the error to
    // propagate to the previous layer (weights_^T * error), computed
    // BEFORE weights_ is overwritten.
    Eigen::MatrixXf Backward(Eigen::MatrixXf& error,
      const Eigen::MatrixXf& prev_activation, const float& eta);

    // Returns the output of the most recent Forward() call.
    const Eigen::MatrixXf& GetActivation() const { return activation_; }

    // Serializes this layer's shape, weights, bias, and activation type
    // to an already-open binary file stream, in the exact order the
    // Layer(std::ifstream&) constructor expects to read them back.
    void Save(std::ofstream& file);


  private:
    Eigen::MatrixXf weights_;  // (output_size x input_size)
    Eigen::MatrixXf activation_;
    Eigen::VectorXf bias_;  // (output_size)
    Activation act_;

    // Randomizes weights_ using He initialization.
    void HeInitialization();
};