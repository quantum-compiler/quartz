#pragma once

#include "../context/context.h"
#include "../dag/dag.h"

#include <unordered_set>

class Dataset {
 public:
  bool save_json(Context *ctx, const std::string &file_name) const;

  // This function runs in O(1).
  [[nodiscard]] int num_hash_values() const;

  // This function runs in O(num_hash_values()).
  [[nodiscard]] int num_total_dags() const;

  auto &operator[](const DAGHashType &val) {
    return dataset[val];
  }

  // Returns true iff the hash value is new to the |dataset|.
  bool insert(Context *ctx, std::unique_ptr<DAG> dag);

  void clear();

  std::unordered_map<DAGHashType,
                     std::vector<std::unique_ptr<DAG>>> dataset;
};
