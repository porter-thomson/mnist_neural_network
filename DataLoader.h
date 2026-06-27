/* Copyright @2026 Porter Thomson
 * porterrt@uw.edu
 * The DataLoader class reads MNIST-format image/label files and 
 * serves them up as shuffled mini-batches for training and evaluation.
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

// Loads a paired MNIST images/labels file and hands out mini-batches
// of a fixed size. Supports re-shuffling between epochs without
// re-reading from disk.
class DataLoader {
  public:
  // Loads every image/bael from disk into memory immediately.
  //    images_file_name - path to an MNIST idx3 image file
  //    labels_file_name - path to the matching MNIST idx1 label file
  //    batch_size       - number of examples returned per nextBatch() call
  // Throws std::runtime_error on a bad magic number, missing file, or a 
  // mismatched image/label count.
  DataLoader(const string& images_file_name, const string& labels_file_name,
    uint32_t batch_size) : current_index_(0), batch_size_(batch_size) {
      loadImages(images_file_name);
      loadLabels(labels_file_name);
    }
  ~DataLoader() = default;
  DataLoader(const DataLoader& other) = delete;
  DataLoader& operator=(const DataLoader& other) = delete;

  // Returns True as long as there's at least one more example left
  // before EOF, (Then shuffle() should be called).
  bool hasNext() const;

  // Returns the next mini-batch of (at_most) batch_size_ examples,
  // advancing the internal read positon. The final batch of a pass
  // may be smaller than batch_size_ if the dataset doesn't divide evenly.
  //  .first  - image batch, (784 x batch_size)
  //  .second - digit label for each example. (batch_size)
  pair<Eigen::MatrixXf, vector<uint8_t>> nextBatch();

  // Sets current_index_ back to 0.
  void reset();

  // Radomly sorts examples via indices_ array, leaving underlying
  // data unchanged and calls reset(). represents "creating a new epoch"
  void shuffle();

  private:
  size_t current_index_;  // internal position within indices_
  uint32_t batch_size_;
  uint32_t num_images_;
  vector<Eigen::VectorXf> images_;  // every loaded image, (784 x num_images_) 
                                    // pixel values between [0, 1]
  vector<uint8_t> labels_;  // every loaded label, parallel to images_
  vector<uint32_t> indices_;  // current (possible shuffled) read order

  // Parses an MNIST idx3 image file into images_ and num_images_
  void loadImages(const string& path);

  // Parses an MNIST idx1 label file into labels_.
  void loadLabels(const string& path);

  // Reads 4 bytes from f and interprets them as a big-endian uint32_t
  // (the byte order used throughout the MNIST file format).
  uint32_t readBigEndianUint32(std::ifstream& f);
};
