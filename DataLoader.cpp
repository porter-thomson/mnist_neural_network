
// #include the .h
#include "DataLoader.h"
#include <fstream>

DataLoader::DataLoader(const string& images_file_name, const string& labels_file_name,
  uint32_t batch_size) {
  if (images_file_name) {
  
  }
  if (labels_file_name) {

  }
  batch_size_ = batch_size;
  loadImages(images_file_name);
  loadLabels(labels_file_name);
}

bool DataLoader::hasNext() const {

}

pair<vector<Eigen::VectorXf>, vector<uint8_t>> DataLoader::nextBatch() {

}

void DataLoader::reset() {

}

void DataLoader::shuffle() {

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

  uint32_t num_images = readBigEndianUint32(image_file);
  uint32_t rows = readBigEndianUint32(image_file);
  uint32_t cols = readBigEndianUint32(image_file);

  images_.resize(num_images);
  uint32_t num_pixels = rows * cols;
  uint8_t buf[num_pixels];
  for (uint32_t i = 0; i < num_images; i++) {
    image_file.read(reinterpret_cast<char*>(buf), num_pixels);

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
  labels_.resize(num_labels);
  label_file.read(reinterpret_cast<char*>(labels_.data()), num_labels);
}

uint32_t DataLoader::readBigEndianUint32(std::ifstream& f) {
    uint8_t bytes[4];
    f.read(reinterpret_cast<char*>(bytes), 4);
    return (bytes[0] << 24) | (bytes[1] << 16) | (bytes[2] << 8) | bytes[3];
}
