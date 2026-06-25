/*
 *
*/
#include <random>
#include <Eigen/Dense>

#include "Layer.h"

using namespace Eigen;

Layer::Layer(std::ifstream& file) {
  if (!file.is_open()) {
    throw std::runtime_error("File is not open");
  }
  uint16_t rows, cols;
  file.read(reinterpret_cast<char*>(&rows), sizeof(rows));
  file.read(reinterpret_cast<char*>(&cols), sizeof(cols));

  weights_.resize(rows, cols);
  file.read(reinterpret_cast<char*>(weights_.data()), rows * cols * sizeof(float));

  bias_.resize(rows);
  file.read(reinterpret_cast<char*>(bias_.data()), rows * sizeof(float));

  uint8_t act;
  file.read(reinterpret_cast<char*>(&act), sizeof(act));
  act_ = static_cast<Activation>(act);
}

void Layer::Forward(const MatrixXf& input) {
  if (input.rows() != weights_.cols()) {
    throw std::runtime_error("Incorrect Vector size of: "
      + std::to_string(input.rows()) + " Expected: " + std::to_string(weights_.cols()));
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

MatrixXf Layer::Backward(MatrixXf& error, const MatrixXf& prev_activation, const float& eta) {
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

  weights_ = weights_ - ((eta/error.cols()) * error * prev_activation.transpose());
  bias_ = bias_ - ((eta/error.cols()) * error.rowwise().sum());

  return prev_error;
}

void Layer::Save(std::ofstream& file) {
  if (!file.is_open()) {
    throw std::runtime_error("File is not open");
  }
  uint16_t rows = weights_.rows();
  uint16_t cols = weights_.cols();

  file.write(reinterpret_cast<char*>(&rows), sizeof(rows));
  file.write(reinterpret_cast<char*>(&cols), sizeof(cols));
  file.write(reinterpret_cast<char*>(weights_.data()), rows * cols * sizeof(float));

  file.write(reinterpret_cast<char*>(bias_.data()), rows * sizeof(float));

  uint8_t act = static_cast<uint8_t>(act_);
  file.write(reinterpret_cast<char*>(&act), sizeof(act));
}

void Layer::HeInitialization() {
  float sd = std::sqrt(2.0 / weights_.cols());
  std::mt19937 rand(std::random_device{}());
  std::normal_distribution<float> dist(0.0, sd);
  weights_ = weights_.unaryExpr([&](float) {
    return (float) dist(rand);
  });
}
