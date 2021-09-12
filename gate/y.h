#pragma once

#include "gate.h"

class YGate : public Gate {
 public:
  YGate() : Gate(GateType::y, 1/*num_qubits*/, 0/*num_parameters*/) {
    using namespace std::complex_literals;
    mat = Matrix<2>({{0, -1.0i}, {1.0i, 0}});
  }
  MatrixBase *get_matrix() override {
    return &mat;
  }
  bool is_parameter_gate() const override {
    return false;
  }
  bool is_quantum_gate() const override {
    return true;
  }
  Matrix<2> mat;
};
