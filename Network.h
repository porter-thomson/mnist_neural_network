
#pragma once
#include <cstdint>
#include <vector>
#include <Eigen/Dense>
#include <utility>

#include "Layer.h"

using namespace std::vector;

class Network {
  public:
    Network(const vector<uint32_t>& layer_sizes, const vector<Activation>& acts);
    ~Network() = default;
    Network(const Network& other) = delete;
    Network& operator=(const Network& other) = delete;

    void SetEta(float eta) { eta_ = eta; }
    void Train(std::pair<Eigen::MatrixXf, vector<uint8_t>> batch);

  private:
    vector<Layer> layers_;
    Eigen::MatrixXf input_act_;
    float eta_;
}