/*
 *
*/
#include <vector>
#include <utility>
#include <Eigen/Dense>
#include <string>

#include "Network.h"
#include "Layer.h"

using namespace Eigen;
using std::pair;
using std::vector;
using std::ofstream;
using std::ifstream;

Network::Network(const vector<uint32_t>& layer_sizes, const vector<Activation>& acts) {
  if (layer_sizes.size() - 1 != acts.size()) {
    throw std::runtime_error("Invalid sizes of vector");
  }
  for (uint16_t i = 0; i < acts.size(); i++) {
    layers_.push_back(Layer(layer_sizes[i], layer_sizes[i + 1], acts[i]));
  }
  eta_ = 0.01f;
}

Network::Network(const std::string& filename) {
  ifstream file(filename, std::ios::binary);
  if (!file.is_open()) {
    throw std::runtime_error("Couldn't open file");
  }
  uint8_t num_layer;
  file.read(reinterpret_cast<char*>(&num_layer) ,sizeof(num_layer));
  file.read(reinterpret_cast<char*>(&eta_), sizeof(float));
  for (uint8_t i = 0; i < num_layer; i++) {
    layers_.push_back(Layer(file));
  }
}

float Network::Train(pair<MatrixXf, vector<uint8_t>> batch) {
  MatrixXf error = ForwardProp(batch.first);
  float cost = 0.0f;
  for (uint32_t i = 0; i < batch.second.size(); i++) {
    cost -= std::log(error(batch.second[i], i));
    error(batch.second[i], i) -= 1.0f;
  }
  cost /= batch.second.size();
  for (int i = static_cast<int>(layers_.size()) - 1; i >= 0; i--) {
    if (i != 0) {
      error = layers_[i].Backward(error, layers_[i - 1].GetActivation(), eta_);
    } else {
      error = layers_[i].Backward(error, batch.first, eta_);
    }
  }
  return cost;
}

MatrixXf Network::ForwardProp(const MatrixXf& input) {
  for (uint16_t i = 0; i < layers_.size(); i++) {
    if (i == 0) {
      layers_[i].Forward(input);
    } else {
      layers_[i].Forward(layers_[i - 1].GetActivation());
    }
  }
  return layers_.back().GetActivation();
}

vector<uint8_t> Network::Predict(const MatrixXf& input) {
  MatrixXf prediction = ForwardProp(input);
  vector<uint8_t> ret(prediction.cols());
  Index temp;
  for (int i = 0; i < prediction.cols(); i++) {
    prediction.col(i).maxCoeff(&temp);
    ret[i] = static_cast<uint8_t>(temp);
  }
  return ret;
}

void Network::Save(const std::string& filename) {
  ofstream file(filename, std::ios::binary);
  if (!file.is_open()) {
    throw std::runtime_error("File couldn't be opened.");
  }
  uint8_t size = layers_.size();
  file.write(reinterpret_cast<char*>(&size), sizeof(size));
  file.write(reinterpret_cast<char*>(&eta_), sizeof(eta_));
  for (uint8_t i = 0; i < size; i++) {
    layers_[i].Save(file);
  }
  file.close();
}
