/* Copyright @2026 Porter Thomson
 * porterrt@uw.edu
 * The Layer class represents a single Layer in a neural network
 *
*/

#pragma once
#include <cstdint>
#include <Eigen/Dense>
#include <fstream>

enum class Activation {
  RELU,
  SOFTMAX
};

class Layer {
  public:
    Layer(int input_size, int output_size, Activation act) :
      act_(act) {
        weights_.resize(output_size, input_size);
        HeInitialization();
        bias_ = Eigen::VectorXf::Zero(output_size);
    }
    Layer(std::ifstream& file);
    ~Layer() = default;
    Layer(const Layer& other) = default;
    Layer& operator=(const Layer& other) = default;

    void Forward(const Eigen::MatrixXf& input);

    Eigen::MatrixXf Backward(Eigen::MatrixXf& error,
      const Eigen::MatrixXf& prev_activation, const float& eta);

    const Eigen::MatrixXf& GetActivation() const { return activation_; }

    void Save(std::ofstream& file);


  private:
    Eigen::MatrixXf weights_;
    Eigen::MatrixXf activation_;
    Eigen::VectorXf bias_;
    Activation act_;
    void HeInitialization();
};