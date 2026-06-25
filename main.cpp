#include <iostream>
#include <Eigen/Dense>

#include "Network.h"

using namespace Eigen;

void TestSaveLoad() {
  Network net({784, 128, 10}, {Activation::RELU, Activation::SOFTMAX});

  MatrixXf dummy_input = MatrixXf::Random(784, 4);  // 4 fake examples
  MatrixXf before = net.ForwardProp(dummy_input);

  std::string path = "test_checkpoint.bin";
  net.Save(path);

  Network loaded(path);
  MatrixXf after = loaded.ForwardProp(dummy_input);

  float max_diff = (before - after).cwiseAbs().maxCoeff();
  std::cout << "Max difference after save/load: " << max_diff << std::endl;

  if (max_diff < 1e-6f) {
    std::cout << "PASS: Save/Load round-trip matches.\n";
  } else {
    std::cout << "FAIL: outputs diverged.\n";
  }
}

int main() {
  TestSaveLoad();
  return 0;
}