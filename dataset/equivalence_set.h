#pragma once

#include "../context/context.h"
#include "../dag/dag.h"

#include <unordered_set>

class EquivalenceSet {
 public:
  bool load_json(Context *ctx, const std::string &file_name);

  std::unordered_map<EquivalenceHashType,
                     std::unordered_set<std::unique_ptr<DAG>>,
                     PairHash> dataset;
};
