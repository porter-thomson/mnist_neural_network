#include <iostream>
#include <algorithm>
#include <vector>
#include <string>
#include <Eigen/Dense>

#include "../DataLoader.h"

static int failures = 0;

#define CHECK(cond, msg) \
  do { \
    if (!(cond)) { \
      std::cout << "FAIL: " << msg << " (line " << __LINE__ << ")\n"; \
      failures++; \
    } else { \
      std::cout << "PASS: " << msg << "\n"; \
    } \
  } while (0)

const std::string kImages = "../testing_data/small_mnist_dataset/test10-images-idx3-ubyte";
const std::string kLabels = "../testing_data/small_mnist_dataset/test10-labels-idx1-ubyte";
const int kTotalExamples = 10;

void TestBatchShape() {
  DataLoader loader(kImages, kLabels, 10);
  CHECK(loader.hasNext(), "hasNext() is true before any batches are taken");

  auto batch = loader.nextBatch();
  CHECK(batch.first.rows() == 784, "Image batch has 784 pixel rows per example");
  CHECK(batch.first.cols() == static_cast<long>(batch.second.size()),
    "Image batch column count matches label count");
}

void TestFullPassCoversAllExamples() {
  // batch_size=4 doesn't evenly divide 10; this checks total coverage
  // regardless of how the last partial batch is handled internally.
  DataLoader loader(kImages, kLabels, 4);
  int total = 0;
  while (loader.hasNext()) {
    auto batch = loader.nextBatch();
    total += static_cast<int>(batch.second.size());
  }
  CHECK(total == kTotalExamples, "Full pass with batch_size=4 covers all 10 examples");
}

void TestResetAllowsSecondPass() {
  DataLoader loader(kImages, kLabels, 3);
  while (loader.hasNext()) loader.nextBatch();
  CHECK(!loader.hasNext(), "hasNext() is false after exhausting all batches");

  loader.reset();
  CHECK(loader.hasNext(), "hasNext() is true again after reset()");

  int total = 0;
  while (loader.hasNext()) {
    auto batch = loader.nextBatch();
    total += static_cast<int>(batch.second.size());
  }
  CHECK(total == kTotalExamples, "Full pass after reset() still covers all 10 examples");
}

void TestShufflePreservesData() {
  DataLoader loader(kImages, kLabels, 10);

  std::vector<uint8_t> unshuffled_labels;
  while (loader.hasNext()) {
    auto batch = loader.nextBatch();
    for (auto l : batch.second) unshuffled_labels.push_back(l);
  }

  loader.reset();
  loader.shuffle();

  std::vector<uint8_t> shuffled_labels;
  while (loader.hasNext()) {
    auto batch = loader.nextBatch();
    for (auto l : batch.second) shuffled_labels.push_back(l);
  }

  CHECK(shuffled_labels.size() == unshuffled_labels.size(),
    "shuffle() preserves total example count");

  std::vector<uint8_t> sorted_a = unshuffled_labels;
  std::vector<uint8_t> sorted_b = shuffled_labels;
  std::sort(sorted_a.begin(), sorted_a.end());
  std::sort(sorted_b.begin(), sorted_b.end());
  CHECK(sorted_a == sorted_b,
    "shuffle() preserves the multiset of labels (no data lost or duplicated)");

  // Informational only -- not a failure either way, since a shuffle could
  // coincidentally reproduce the original order, especially with only 10 items.
  if (unshuffled_labels != shuffled_labels) {
    std::cout << "INFO: shuffle() changed example order this run.\n";
  } else {
    std::cout << "INFO: shuffle() produced the same order this run "
                 "(possible by chance with n=10, not necessarily a bug).\n";
  }
}

int main() {
  TestBatchShape();
  TestFullPassCoversAllExamples();
  TestResetAllowsSecondPass();
  TestShufflePreservesData();

  std::cout << "\n" << failures << " failure(s).\n";
  return failures == 0 ? 0 : 1;
}