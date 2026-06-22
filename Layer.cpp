/*
 *
*/
#include "Layer.h"
#include <random>

using namespace Eigen;

void Forward(const MatrixXf& input) {
  if (input.row() != input_size_) {
    throw std::runtime_error("Incorrect Vector size of: "
      + std::to_string(input.rows()) + " Expected: " + std::to_string(input_size_));
  }
  activation_ = weights_ * input;
  activation_.colwise() += bias_;
  switch (act_) {
    case Activation::RELU:
      activation_ = activation_.array().max(0.0f).matrix();
      break;
    case Activation::SOFTMAX: {
      ArrayXXf exp_vals = activation_.array().exp();
      activation_ = (exp_vals.rowwise() / exp_vals.colwise().sum()).matrix();
      break;
    }
  }
}

MatrixXf Backward(MatrixXf& error, const MatrixXf& prev_activation) {
  switch (act_) {
    case Activation::RELU: {
      MatrixXf d_a = (activation_.array() > 0.0f).cast<float>();
      error = error.array() * d_a.array();
      break;
    }
    case Activation::SOFTMAX:
      break;
  }

  MatrixXf prev_error = weights_.transpose() * error;

  weights_ = weights_ - ((eta_/error.cols()) * error * prev_activation.transpose());
  bias_ = bias_ - ((eta_/error.cols()) * error.rowwise().sum());

  return prev_error;
}


void Layer::HeInitialization() {
  float sd = std::sqrt(2.0 / weights_.cols());
  std::mt19937 rand(std::random_device{}());
  std::normal_distribution<float> dist(0.0, sd);
  weights_ = weights_.unaryExpr([&](float) {
    return (float) dist(rand);
  });
}


