
// #include the .h
#include "DataLoader.h"

#include <numeric>
#include <algorithm>
#include <random>


bool DataLoader::hasNext() const {
  return (current_index_ < indices_.size());
}
// change this to Eigen::MatrixXf
pair<vector<Eigen::VectorXf>, vector<uint8_t>> DataLoader::nextBatch() {
  size_t end = std::min(current_index_ + (size_t)batch_size_, indices_.size());
  uint32_t iter = (uint32_t)(end - current_index_);
  vector<Eigen::VectorXf> images(iter);
  vector<uint8_t> labels(iter);
  for (uint32_t i = 0; i < iter; i++) {
    images[i] = images_[indices_[current_index_ + i]];
    labels[i] = labels_[indices_[current_index_ + i]];
  }
  current_index_ = end;
  return std::make_pair(images, labels);
}

void DataLoader::reset() {
  current_index_ = 0;
}

void DataLoader::shuffle() {
  std::mt19937 rand(std::random_device{}());
  std::shuffle(indices_.begin(), indices_.end(), rand);
  reset();
}

void DataLoader::loadImages(const string& path) {
  std::ifstream image_file(path, std::ios::binary);
  if (!image_file.is_open()) {
    throw std::runtime_error("Could not open file: " + path);
  }

  uint32_t magic_number = readBigEndianUint32(image_file);
  if (magic_number != 0x00000803) {
    throw std::runtime_error("Invalid magic number for images file");
  }

  num_images_ = readBigEndianUint32(image_file);
  uint32_t rows = readBigEndianUint32(image_file);
  uint32_t cols = readBigEndianUint32(image_file);

  indices_.resize(num_images_);
  std::iota(indices_.begin(), indices_.end(), 0);

  images_.resize(num_images_);
  uint32_t num_pixels = rows * cols;
  vector<uint8_t> buf(num_pixels);
  for (uint32_t i = 0; i < num_images_; i++) {
    image_file.read(reinterpret_cast<char*>(buf.data()), num_pixels);

    Eigen::VectorXf image(num_pixels);
    for (uint32_t j = 0; j < num_pixels; j++) {
      image(j) = buf[j] / 255.0f;
    }
    images_[i] = image;
  }
}

void DataLoader::loadLabels(const string& path) {
  std::ifstream label_file(path, std::ios::binary);
  if (!label_file.is_open()) {
    throw std::runtime_error("Could not open file: " + path);
  }

  uint32_t magic_number = readBigEndianUint32(label_file);
  if (magic_number != 0x00000801) {
    throw std::runtime_error("Invalid magic number for label file");
  }
  uint32_t num_labels = readBigEndianUint32(label_file);
  if (num_labels != num_images_) {
    throw std::runtime_error("Number of images (" + std::to_string(num_images_) +
                         ") and labels (" + std::to_string(num_labels) + ") do not match");
  }

  labels_.resize(num_labels);
  label_file.read(reinterpret_cast<char*>(labels_.data()), num_labels);
}

uint32_t DataLoader::readBigEndianUint32(std::ifstream& f) {
    uint8_t bytes[4];
    f.read(reinterpret_cast<char*>(bytes), 4);
    return (bytes[0] << 24) | (bytes[1] << 16) | (bytes[2] << 8) | bytes[3];
}
