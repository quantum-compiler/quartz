#pragma once

#include "utils.h"
#include "../gate/gate.h"

#include <vector>

class DAGNode;

// A hyperedge in DAG corresponds to a gate in the circuit.
class DAGHyperEdge {
 public:
  std::vector<DAGNode *> input_nodes;  // Nodes including parameters!
  std::vector<DAGNode *> output_nodes;

  const Gate* gate;
};
