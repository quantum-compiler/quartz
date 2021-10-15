#pragma once

#include "../context/context.h"
#include "../dag/dag.h"

#include <unordered_set>

class Dataset {
 public:
  bool save_json(const std::string &file_name) const;

  // This function runs in O(1).
  [[nodiscard]] int num_hash_values() const;

  // This function runs in O(num_hash_values()).
  [[nodiscard]] int num_total_dags() const;

  auto &operator[](const DAGHashType &val) {
    return dataset[val];
  }

  // Returns true iff the hash value is new to the dataset.
  bool insert(Context *ctx, std::unique_ptr<DAG> dag) {
    const auto hash_value = dag->hash(ctx);
    bool ret = dataset.count(hash_value) == 0;
    dataset[hash_value].push_back(std::move(dag));
    return ret;
  }

  std::unordered_map<DAGHashType,
                     std::vector<std::unique_ptr<DAG>>> dataset;
};
