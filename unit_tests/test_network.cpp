#include <iostream>
#include <cstdio>
#include <string>
#include <vector>
#include <utility>
#include <Eigen/Dense>

#include "../Network.h"

using Eigen::MatrixXf;

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

void TestForwardPropShape() {
  Network net({784, 128, 10}, {Activation::RELU, Activation::SOFTMAX});
  MatrixXf input = MatrixXf::Random(784, 6);
  MatrixXf output = net.ForwardProp(input);
  CHECK(output.rows() == 10, "ForwardProp output rows == final layer size");
  CHECK(output.cols() == 6, "ForwardProp output cols == batch_size");
}

void TestPredictReturnsValidLabels() {
  Network net({784, 32, 10}, {Activation::RELU, Activation::SOFTMAX});
  MatrixXf input = MatrixXf::Random(784, 5);
  std::vector<uint8_t> preds = net.Predict(input);

  CHECK(preds.size() == 5, "Predict returns exactly one label per example");
  bool all_valid = true;
  for (auto p : preds) {
    if (p > 9) all_valid = false;
  }
  CHECK(all_valid, "Predict labels all fall within [0, 9]");
}

void TestTrainReducesCostOnFixedBatch() {
  Network net({20, 16, 4}, {Activation::RELU, Activation::SOFTMAX});
  net.SetEta(0.1f);

  MatrixXf input = MatrixXf::Random(20, 8);
  std::vector<uint8_t> labels = {0, 1, 2, 3, 0, 1, 2, 3};
  std::pair<MatrixXf, std::vector<uint8_t>> batch = {input, labels};

  float first_cost = net.Train(batch);
  float last_cost = first_cost;
  for (int i = 0; i < 50; i++) {
    last_cost = net.Train(batch);
  }

  CHECK(last_cost < first_cost,
    "Repeated training on a fixed batch decreases cost (first=" +
    std::to_string(first_cost) + ", last=" + std::to_string(last_cost) + ")");
}

void TestSaveLoadRoundTrip() {
  Network net({784, 128, 10}, {Activation::RELU, Activation::SOFTMAX});
  MatrixXf input = MatrixXf::Random(784, 4);
  MatrixXf before = net.ForwardProp(input);

  const std::string path = "network_test_checkpoint.bin";
  net.Save(path);

  Network loaded(path);
  MatrixXf after = loaded.ForwardProp(input);

  float max_diff = (before - after).cwiseAbs().maxCoeff();
  CHECK(max_diff < 1e-6f, "Network Save/Load round trip reproduces identical ForwardProp output");

  std::remove(path.c_str());
}

int main() {
  TestForwardPropShape();
  TestPredictReturnsValidLabels();
  TestTrainReducesCostOnFixedBatch();
  TestSaveLoadRoundTrip();

  std::cout << "\n" << failures << " failure(s).\n";
  return failures == 0 ? 0 : 1;
}