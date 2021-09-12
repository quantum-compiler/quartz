#pragma once

#include "gate.h"

class XGate : public Gate {
 public:
  XGate(): Gate(GateType::x, 1/*num_qubits*/, 0/*num_parameters*/) {}
  std::unique_ptr<MatrixBase> to_matrix() const {
    return std::make_unique<Matrix<2>>(Matrix<2>({{0, 1}, {1, 0}}));
  }
  bool is_parameter_gate() const {
    return false;
  }
  bool is_quantum_gate() const {
    return true;
  }

};
