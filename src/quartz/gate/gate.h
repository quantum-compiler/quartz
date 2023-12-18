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
  /**
   * For arithmetic computation "gates", compute the result parameter.
   */
  virtual ParamType compute(const std::vector<ParamType> &input_params);
  /**
   * Only for arithmetic computation "gates".
   */
  [[nodiscard]] virtual bool is_commutative() const;
  /**
   * @return True if the gate is a multi-qubit gate and the gate remains
   * the same under any qubit permutation (e.g., CZ, CP, CCZ, SWAP).
   * Default value is false.
   */
  [[nodiscard]] virtual bool is_symmetric() const;
  /**
   * @return True if the number of non-zero elements in the matrix
   * representation of the quantum gate is exactly 2 to the power of num_qubits.
   * Default value is false.
   */
  [[nodiscard]] virtual bool is_sparse() const;
  /**
   * @return True if the matrix representation of the quantum gate is a
   * diagonal matrix (e.g., Z, CZ, P, CP).
   * Default value is false.
   */
  [[nodiscard]] virtual bool is_diagonal() const;
  /**
   * @return The number of control qubits for controlled gates; or 0 if it is
   * not a controlled gate.
   * Default value is 0.
   */
  [[nodiscard]] virtual int get_num_control_qubits() const;
  /**
   * @return The control state for controlled gates. Only overridden by
   * GeneralControlledGate.
   * Default value is a vector of |get_num_control_qubits()| true's.
   */
  [[nodiscard]] virtual std::vector<bool> get_control_state() const;

  [[nodiscard]] int get_num_qubits() const;
  [[nodiscard]] int get_num_parameters() const;
  /**
   * @return True if the gate is an arithmetic computation "gate".
   * Implemented as having 0 qubits and not input qubit/param "gate".
   */
  [[nodiscard]] bool is_parameter_gate() const;
  [[nodiscard]] bool is_quantum_gate() const;
  [[nodiscard]] bool is_parametrized_gate() const;
  /**
   * @return True if the gate is a 3-qubit gate with 2 control qubits.
   */
  [[nodiscard]] bool is_toffoli_gate() const;
  virtual ~Gate() = default;

  GateType tp;
  int num_qubits, num_parameters;
};

// Include all gate names here.
#define PER_GATE(x, XGate) class XGate;

#include "gates.inc.h"

#undef PER_GATE

}  // namespace quartz
