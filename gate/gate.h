#pragma once

#include "../math/matrix.h"
#include "gate_utils.h"

#include <iostream>
#include <memory>
#include <unordered_map>
#include <vector>

class Gate {
public:
  Gate(GateType tp, int num_qubits, int num_parameters);
  virtual MatrixBase *get_matrix();
  virtual MatrixBase *get_matrix(const std::vector<ParamType> &params);
  virtual ParamType compute(const std::vector<ParamType> &input_params);
  [[nodiscard]] virtual bool is_commutative() const; // for traditional gates
  [[nodiscard]] int get_num_qubits() const;
  [[nodiscard]] int get_num_parameters() const;
  [[nodiscard]] bool is_parameter_gate() const;
  [[nodiscard]] bool is_quantum_gate() const;
  [[nodiscard]] bool is_parametrized_gate() const;
  [[nodiscard]] bool is_toffoli_gate() const;
  virtual ~Gate() = default;

  GateType tp;
  int num_qubits, num_parameters;
};

// Include all gate names here.
#define PER_GATE(x, XGate) class XGate;

#include "gates.inc.h"

#undef PER_GATE
