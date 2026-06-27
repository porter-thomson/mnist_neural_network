/* Copyright @2026 Porter Thomson
 * porterrt@uw.edu
 * Command-line entry point for training/testing the neural network
 * against MNIST. Supports loading and continuing training on a
 * previously-saved checkpoint.
*/
#include <iostream>
#include <Eigen/Dense>
#include <string>
#include <utility>
#include <vector>
#include <cstdlib>
#include <stdexcept>

#include "Network.h"
#include "Layer.h"
#include "DataLoader.h"

using namespace Eigen;
using std::string;

// Number of examples per mini-batch during both training and evaluation.
#define BATCH_SIZE 500

// Writes the current state of network to disk as
// "checkpoint-epoch-<epoch>.bin", using Network::Save's binary format.
void SaveNetwork(Network& network, int epoch) {
  string filename = "checkpoint-epoch-" + std::to_string(epoch) + ".bin";
  network.Save(filename);
}

// Runs network's predictions against every example in test_loader and
// returns the fraction correct (0.0 - 1.0). Leaves test_loader
// reshuffled and reset, ready for a subsequent call.
float TestNetwork(Network& network, DataLoader& test_loader) {
  int count = 0, correct = 0;
  while (test_loader.hasNext()) {
    pair<MatrixXf, vector<uint8_t>> batch = test_loader.nextBatch();
    vector<uint8_t> pred = network.Predict(batch.first);
    for (uint16_t i = 0; i < pred.size(); i++) {
      if (pred[i] == batch.second[i]) correct++;
    }
    count += pred.size();
  }
  test_loader.shuffle();
  return static_cast<float>(correct) / static_cast<float>(count);
}

// Trains network on train_loader for up to max_epochs epochs,
// stopping early once test-set accuracy fails to improve by at least
// min_improvement for `patience` consecutive epochs. Prints per-epoch
// progress, periodically checkpoints every 10 epochs, and always
// writes a final checkpoint when training stops.
void TrainNetwork(Network& network, DataLoader& train_loader, DataLoader& test_loader) {
  const float min_improvement = 0.0005f;
  const int patience = 6;
  const int max_epochs = 100;
  int patience_counter = 0, epoch = 0;
  float prev_accuracy = 0.0f;
  while (patience > patience_counter && epoch < max_epochs) {
    float cost = 0.0f;
    int num_batches = 0;
    while (train_loader.hasNext()) {
      cost += network.Train(train_loader.nextBatch());
      num_batches++;
    }
    float ave_cost = cost / static_cast<float>(num_batches);
    float accuracy = TestNetwork(network, test_loader);
    if (accuracy - prev_accuracy < min_improvement) patience_counter++;
    else patience_counter = 0;
    prev_accuracy = accuracy;
    train_loader.shuffle();
    epoch++;
    std::cout << "Epoch: " << epoch << std::endl;
    std::cout << "\tAverage Cost: " << ave_cost << std::endl;
    std::cout << "\tAccuracy: " << accuracy * 100.0f << "%" << std::endl;
    if (epoch % 10 == 0) {
      SaveNetwork(network, epoch);
    }
  }
  if (max_epochs != epoch) {
  std::cout << "Accuracy converged after " << epoch << " epochs. Final accuracy: "
            << prev_accuracy * 100.0f << "%" << std::endl;
  } else {
    std::cout << "Reached max epoch cap (" << max_epochs << ") without converging. "
              << "Final accuracy: " << prev_accuracy * 100.0f << "%" << std::endl;
  }
  SaveNetwork(network, epoch);
}

int main(int argc, char* argv[]) {
  std::cout << "Welcome to Neural Network, built by Porter Thomson" << std::endl;
  bool train = false, test = false, eta = false;
  string filename;
  float etaf;
  
  for (int i = 1; i < argc; i++) {
    string arg = argv[i];

    // ---- Argument parsing ----
    // --train          : train the network for this run
    // --test           : evaluate the network on the test set
    // --eta <float>    : override the learning rate (default set by Network)
    // --file <path>    : load an existing checkpoint instead of starting fresh
    if (arg == "--test") test = true;
    else if (arg == "--train") train = true;
    else if (arg == "--eta") {
      if (i + 1 >= argc) {
        std::cerr << "Error: --eta requires a value" << std::endl;
        return EXIT_FAILURE;
      }
      eta = true;
      try {
        etaf = std::stof(argv[++i]);
      } catch (const std::invalid_argument&) {
        std::cerr << "Error: invalid eta value '" << argv[i] << "'" << std::endl;
        return EXIT_FAILURE;
      }
    }
    else if (arg == "--file") {
      if (i + 1 >= argc) {
        std::cerr << "Error: --file requires a value" << std::endl;
        return EXIT_FAILURE;
      }
      filename = argv[++i];
    } else {
      std::cerr << "Unknown argument " << arg << std::endl;
      return EXIT_FAILURE;
    }
  }
  
  if (!train && !test) {
    std::cout << "Nothing to be done." << std::endl;
    return EXIT_SUCCESS;
  }
  Network* network;
  if (filename != "") {
    network = new Network(filename);
  } else {
    vector<uint32_t> layer = {784, 128, 10};
    vector<Activation> act = {Activation::RELU, Activation::SOFTMAX};
    network = new Network(layer, act);
  }
  if (eta == true) network->SetEta(etaf);
  std::cout << "Loading Testing Data" << std::endl;
  DataLoader test_loader("testing_data/t10k-images.idx3-ubyte", "testing_data/t10k-labels.idx1-ubyte", BATCH_SIZE);
  std::cout << "Testing Data Loaded!" << std::endl;
  if (train == true) {
    std::cout << "Loading Training Data" << std::endl;
    DataLoader train_loader("training_data/train-images.idx3-ubyte", "training_data/train-labels.idx1-ubyte", BATCH_SIZE);
    std::cout << "Training Data Loaded!" << std::endl;
    TrainNetwork(*network, train_loader, test_loader);
  }
  if (test == true) {
    test_loader.shuffle();
    float res = TestNetwork(*network, test_loader);
    std::cout << "Test accuracy: " << res * 100.0f << "%" << std::endl;
  }

  delete network;
  return EXIT_SUCCESS;
}