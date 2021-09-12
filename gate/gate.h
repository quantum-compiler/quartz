#pragma once

#include "../math/matrix.h"
#include "gate_utils.h"

#include <iostream>
#include <memory>

class Gate {
 public:
  // TODO: Return a raw pointer in this function
  virtual std::unique_ptr<MatrixBase> to_matrix() const {
    std::cerr << "Gate::to_matrix() called." << std::endl;
    return std::make_unique<MatrixBase>();
  }
  Gate(GateType tp, int num_qubits, int num_parameters);
  virtual bool is_parameter_gate() const = 0;
  virtual bool is_quantum_gate() const = 0;
  int get_num_qubits() const;
  int get_num_parameters() const;

  GateType tp;
  int num_qubits, num_parameters;
};

// Include all gate names here.
#define PER_GATE(x, XGate) class XGate;

#include "gates.inc.h"

#undef PER_GATE
