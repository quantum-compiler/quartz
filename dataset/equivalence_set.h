#pragma once

#include "../context/context.h"
#include "../dag/dag.h"

#include <list>
#include <unordered_set>
#include <unordered_map>

class EquivalenceSet {
 public:
  bool load_json(Context *ctx, const std::string &file_name);

  // We cannot use std::vector here because that would need
  // std::unordered_set<std::unique_ptr<DAG>> to be copy-constructable.
  std::unordered_map<DAGHashType,
                     std::list<std::unordered_set<std::unique_ptr<DAG>>>>
      dataset;
};
