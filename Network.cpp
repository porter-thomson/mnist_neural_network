/*
 *
*/
#include "Network.h"
#include "Layer.h"

using namespace Eigen;

Network::Network(const vector<uint32_t>& layer_sizes, const vector<Activation>& acts) {
  if (layer_sizes.size() - 1 != acts.size()) {
    throw std::runtime_error("Invalid sizes of vector");
  }
  for (int i = 0; i < acts.size(); i++) {
    layers_.push_back(Layer(layer_sizes[i], layer_sizes[i + 1], acts[i]));
  }
  eta_ = 0.01f;
}

Network::Train(pair<MatrixXf, vector<uint8_t>> batch) {
  for (int i = 0; i < layers_.size(); i++) {
    if (i == 0) {
      layers_[i].Forward(batch.first);
    } else {
      layers_[i].Forward(layers_[i - 1].GetActivation());
    }
  }
  MatrixXf error = layers_.back().GetActivation();
  for (int i = 0; i < batch.second.size(); i++) {
    error(batch.second[i], i) -= 1.0f;
  }
  for (int i = layers_.size() - 1; i >= 0; i--) {
    if (i != 0) {
      error = layers_[i].Backward(error, layers_[i - 1].GetActivation(), eta_);
    } else {
      error = layers_[i].Backward(error, batch.first, eta_);
    }
  }
}

float Network::CostFunction() {
  
}



