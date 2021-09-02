#pragma once

#include "dagnode.h"
#include "daghyperedge.h"

class DAG {
 public:
  std::vector<std::unique_ptr<DAGNode>> nodes;
  std::vector<std::unique_ptr<DAGHyperEdge>> edges;
  // The gates' information is owned by edges.
};
