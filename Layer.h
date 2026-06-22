

#pragma once
#include <cstdint>
#include <Eigen/Dense>

enum class Activation {
  RELU,
  SOFTMAX
};

class Layer {
  public:
  Layer(int input_size, int output_size, Activation act) :
    input_size_(input_size), 
    output_size_(output_size), 
    act_(act) {
      weights_.resize(output_size_, input_size_);
      HeInitialization();
      bias_ = Eigen::VectorXf::Zero(output_size_);
  }
  ~Layer() = default;
  Layer(const Layer& other) = delete;
  Layer& operator=(const Layer& other) = delete;

  void Forward(const Eigen::MatrixXf& input);

  Eigen::MatrixXf Backward(Eigen::MatrixXf& error,
    const Eigen::MatrixXf& prev_activation);


  private:
  uint32_t input_size_;
  uint32_t output_size_;
  float eta_;
  Eigen::MatrixXf weights_;
  Eigen::MatrixXf activation_;
  Eigen::VectorXf bias_;
  Activation act_;
  void HeInitialization();
};