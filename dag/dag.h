#pragma once

#include "dagnode.h"
#include "daghyperedge.h"
#include "../utils/utils.h"
#include "../gate/gate.h"
#include "../math/vector.h"

class DAG {
 public:
  DAG(int _num_qubits, int _num_parameters);
  std::vector<std::unique_ptr<DAGNode>> nodes;
  std::vector<std::unique_ptr<DAGHyperEdge>> edges;
  // The gates' information is owned by edges.
  std::vector<DAGNode *> outputs;
  std::vector<DAGNode *> parameters;
  bool add_gate(const std::vector<int> &qubit_indices,
                const std::vector<int> &parameter_indices,
                Gate *gate,
                int *output_para_index);
  bool evaluate(const Vector &input_dis,
                const std::vector<ParamType> &input_parameters,
                Vector &output_dis) const;
  [[nodiscard]] int get_num_qubits() const;
  [[nodiscard]] int get_num_input_parameters() const;
  [[nodiscard]] int get_num_total_parameters() const;
 private:
  int num_qubits, num_input_parameters;
};
