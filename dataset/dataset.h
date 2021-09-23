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

  std::unordered_map<DAGHashType,
                     std::unordered_set<DAG *> > dataset;
};
