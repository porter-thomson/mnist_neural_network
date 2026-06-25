
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
    Network(const vector<uint32_t>& layer_sizes, const vector<Activation>& acts);
    Network(const std::string& filename);
    ~Network() = default;
    Network(const Network& other) = delete;
    Network& operator=(const Network& other) = delete;

    void SetEta(float eta) { eta_ = eta; }
    float Train(std::pair<Eigen::MatrixXf, vector<uint8_t>> batch);
    Eigen::MatrixXf ForwardProp(const Eigen::MatrixXf& input);
    vector<uint8_t> Predict(const Eigen::MatrixXf& input);
    void Save(const std::string& filename);

  private:
    vector<Layer> layers_;
    float eta_;
};