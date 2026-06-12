
/* need to write all the #include
*/
#pragma once
#include <cstdint>
#include <string>
#include <vector>
#include <utility>
#include <Eigen/Dense>


using std::string;
using std::vector;
using std::pair;

class DataLoader {
  public:
  DataLoader(const string& images_file_name, const string& labels_file_name,
    uint32_t batch_size);
  ~DataLoader();
  DataLoader(const DataLoader& other) = delete;
  DataLoader& operator=(const DataLoader& other) = delete;

  bool hasNext() const;
  pair<vector<Eigen::VectorXf>, vector<uint8_t>> nextBatch();
  void reset();
  void shuffle();



  private:
  size_t current_index_;
  uint32_t batch_size_;
  vector<Eigen::VectorXf> images_;
  vector<uint8_t> labels_;
  void loadImages(const string& path);
  void loadLabels(const string& path);
  uint32_t readBigEndianUint32(std::ifstream& f);
};
