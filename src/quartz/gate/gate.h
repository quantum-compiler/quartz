#pragma once

#include "../math/matrix.h"
#include "gate_utils.h"

#include <iostream>
#include <memory>
#include <unordered_map>
#include <vector>

namespace quartz {
class Gate {
public:
  Gate(GateType tp, int num_qubits, int num_parameters);
  virtual MatrixBase *get_matrix();
  virtual MatrixBase *get_matrix(const std::vector<ParamType> &params);
  virtual ParamType compute(const std::vector<ParamType> &input_params);
  [[nodiscard]] virtual bool is_commutative() const; // for traditional gates
  [[nodiscard]] virtual bool
  is_symmetric() const; // for 2-qubit gates; currently unused (always false)
  // Returns true iff the number of non-zero elements in the matrix
  // representation of the quantum gate is exactly 2 to the power of num_qubits.
  [[nodiscard]] virtual bool is_sparse() const;
  // Returns the number of control qubits for controlled gates; or 0 if it is
  // not a controlled gate.
  [[nodiscard]] virtual int get_num_control_qubits() const;
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

} // namespace quartz
