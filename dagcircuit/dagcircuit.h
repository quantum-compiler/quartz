#pragma once

#include "../gate/gate.h"
#include "dagnode.h"
#include "dag.h"
#include "utils.h"

#include <vector>

class DAGCircuit {
 public:
  int num_qubits;
  int num_params;
  int num_gates;

  std::vector<ParamType> params;

  // (num_qubits * 2 + num_params) wires
  std::vector<DAGNode *> input_wires;
  std::vector<DAGNode *> output_wires;
  std::vector<DAGNode *> param_wires;

  std::unique_ptr<DAG> dag;
  // The gates' information is owned by edges in the DAG.
};
