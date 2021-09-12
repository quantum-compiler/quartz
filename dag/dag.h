#pragma once

#include "dagnode.h"
#include "daghyperedge.h"
#include "../utils/utils.h"
#include "../gate/gate.h"

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
                const Gate *gate,
                int *output_para_index);
  bool evaluate(const std::vector<ComplexType> &input_dis,
                const std::vector<ParamType> &parameters,
                std::vector<ComplexType> &output_dis) const;
  int get_num_qubits() const;
  int num_qubits, num_parameters;
};
