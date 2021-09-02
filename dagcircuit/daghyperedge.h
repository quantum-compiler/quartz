#pragma once

#include "utils.h"

#include <vector>

class DAGNode;

// A hyperedge in DAG corresponds to a gate in the circuit.
class DAGHyperEdge {
 public:
  std::vector<DAGNode *> input_node;  // Nodes including parameters!
  std::vector<DAGNode *> output_node;

  std::unique_ptr<Gate> gate;
};
