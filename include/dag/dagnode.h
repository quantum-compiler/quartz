#pragma once

#include <string>
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

  [[nodiscard]] bool is_qubit() const;
  [[nodiscard]] bool is_parameter() const;
  [[nodiscard]] std::string to_string() const;

  Type type;
  // If this node is a qubit, |index| is the qubit id it correspond to,
  // ranging [0, get_num_qubits()).
  // If this node is a parameter, |index| is the parameter id,
  // ranging [0, get_num_total_parameters()).
  int index;
  std::vector<DAGHyperEdge *> input_edges;
  std::vector<DAGHyperEdge *> output_edges;
};
