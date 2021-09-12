#pragma once

#include "../math/matrix.h"
#include "gate_utils.h"

#include <iostream>
#include <memory>

class Gate {
 public:
  virtual MatrixBase *get_matrix() = 0;
  Gate(GateType tp, int num_qubits, int num_parameters);
  [[nodiscard]] virtual bool is_parameter_gate() const = 0;
  [[nodiscard]] virtual bool is_quantum_gate() const = 0;
  [[nodiscard]] int get_num_qubits() const;
  [[nodiscard]] int get_num_parameters() const;
  virtual ~Gate() = default;

  GateType tp;
  int num_qubits, num_parameters;
};

// Include all gate names here.
#define PER_GATE(x, XGate) class XGate;

#include "gates.inc.h"

#undef PER_GATE
