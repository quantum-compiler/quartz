#pragma once

#include <vector>

class DAGHyperEdge;

// A node in DAG corresponds to a wire in the circuit.
class DAGNode {
 public:
  enum Type {
    internal_qubit,
    input_qubit,
    output_qubit,
    input_param,
    internal_param
  };
  Type type;
  std::vector<DAGHyperEdge *> input_edges;
  std::vector<DAGHyperEdge *> output_edges;
};
