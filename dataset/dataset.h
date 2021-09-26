#pragma once

#include "../context/context.h"
#include "../dag/dag.h"

#include <unordered_set>

class Dataset {
 public:
  void save_json(const std::string &file_name);

  auto &operator[](const DAGHashType &val) {
    return dataset[val];
  }

  void insert(Context *ctx, std::unique_ptr<DAG> dag) {
    const auto hash_value = dag->hash(ctx);
    dataset[hash_value].insert(std::move(dag));
  }

  std::unordered_map<DAGHashType,
                     std::unordered_set<std::unique_ptr<DAG>>> dataset;
};
