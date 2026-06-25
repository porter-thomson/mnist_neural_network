#include <iostream>
#include <fstream>
#include <cstdio>
#include <stdexcept>
#include <Eigen/Dense>

#include "../Layer.h"

using Eigen::MatrixXf;
using Eigen::VectorXf;

static int failures = 0;

#define CHECK(cond, msg) \
  do { \
    if (!(cond)) { \
      std::cout << "FAIL: " << msg << " (line " << __LINE__ << ")\n"; \
      failures++; \
    } else { \
      std::cout << "PASS: " << msg << "\n"; \
    } \
  } while (0)

void TestForwardShape() {
  Layer layer(784, 128, Activation::RELU);
  MatrixXf input = MatrixXf::Random(784, 5);
  layer.Forward(input);
  CHECK(layer.GetActivation().rows() == 128, "Forward output rows == output_size");
  CHECK(layer.GetActivation().cols() == 5, "Forward output cols == batch_size");
}

void TestForwardMismatchThrows() {
  Layer layer(784, 128, Activation::RELU);
  MatrixXf bad_input = MatrixXf::Random(10, 5);
  bool threw = false;
  try {
    layer.Forward(bad_input);
  } catch (const std::runtime_error&) {
    threw = true;
  }
  CHECK(threw, "Forward throws std::runtime_error on mismatched input rows");
}

void TestReluNonNegative() {
  Layer layer(784, 64, Activation::RELU);
  MatrixXf input = MatrixXf::Random(784, 8);
  layer.Forward(input);
  CHECK(layer.GetActivation().minCoeff() >= 0.0f, "RELU output has no negative values");
}

void TestSoftmaxColumnsSumToOne() {
  Layer layer(64, 10, Activation::SOFTMAX);
  MatrixXf input = MatrixXf::Random(64, 6);
  layer.Forward(input);

  VectorXf col_sums = layer.GetActivation().colwise().sum();
  bool all_close = true;
  for (int i = 0; i < col_sums.size(); i++) {
    if (std::abs(col_sums(i) - 1.0f) > 1e-4f) all_close = false;
  }
  CHECK(all_close, "Softmax output columns each sum to 1");
  CHECK(layer.GetActivation().minCoeff() >= 0.0f, "Softmax output is non-negative");
}

void TestBackwardShape() {
  Layer layer(20, 10, Activation::RELU);
  MatrixXf input = MatrixXf::Random(20, 4);
  layer.Forward(input);

  MatrixXf error = MatrixXf::Random(10, 4);
  float eta = 0.01f;
  MatrixXf prev_error = layer.Backward(error, input, eta);

  CHECK(prev_error.rows() == 20, "Backward returns propagated error with rows == input_size");
  CHECK(prev_error.cols() == 4, "Backward returns propagated error with cols == batch_size");
}

void TestSaveLoadRoundTrip() {
  Layer layer(15, 8, Activation::RELU);
  MatrixXf input = MatrixXf::Random(15, 3);
  layer.Forward(input);
  MatrixXf before = layer.GetActivation();

  const std::string path = "layer_test_checkpoint.bin";
  {
    std::ofstream out(path, std::ios::binary);
    layer.Save(out);
  }

  std::ifstream in(path, std::ios::binary);
  Layer loaded(in);
  loaded.Forward(input);
  MatrixXf after = loaded.GetActivation();

  float max_diff = (before - after).cwiseAbs().maxCoeff();
  CHECK(max_diff < 1e-6f, "Layer Save/Load round trip reproduces identical forward output");

  std::remove(path.c_str());
}

int main() {
  TestForwardShape();
  TestForwardMismatchThrows();
  TestReluNonNegative();
  TestSoftmaxColumnsSumToOne();
  TestBackwardShape();
  TestSaveLoadRoundTrip();

  std::cout << "\n" << failures << " failure(s).\n";
  return failures == 0 ? 0 : 1;
}