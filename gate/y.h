#pragma once

#include "gate.h"

class YGate : public Gate {
public:
  YGate() : Gate(GateType::y, 1/*num_qubits*/, 0/*num_parameters*/) {}
  std::unique_ptr<MatrixBase> to_matrix() const {
    using namespace std::complex_literals;
    return std::make_unique<Matrix<2>>(Matrix<2>({{0, -1.0i}, {1.0i, 0}}));
  }
  bool is_parameter_gate() const {
    return false;
  }
  bool is_quantum_gate() const {
    return true;
  }
};
