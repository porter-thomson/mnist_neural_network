
/* need to write all the #include
*/
#pragma once
#include <cstdint>
#include <string>
#include <vector>
#include <utility>
#include <Eigen/Dense>
#include <fstream>


using std::string;
using std::vector;
using std::pair;

class DataLoader {
  public:
  DataLoader(const string& images_file_name, const string& labels_file_name,
    uint32_t batch_size) : batch_size_(batch_size), current_index_(0) {
      loadImages(images_file_name);
      loadLabels(labels_file_name);
    }
  ~DataLoader() = default;
  DataLoader(const DataLoader& other) = delete;
  DataLoader& operator=(const DataLoader& other) = delete;

  bool hasNext() const;

  pair<Eigen::MatrixXf, vector<uint8_t>> nextBatch();

  void reset();

  void shuffle();

  private:
  size_t current_index_;
  uint32_t batch_size_;
  uint32_t num_images_;
  vector<Eigen::VectorXf> images_;
  vector<uint8_t> labels_;
  vector<uint32_t> indices_;
  void loadImages(const string& path);
  void loadLabels(const string& path);
  uint32_t readBigEndianUint32(std::ifstream& f);
};
