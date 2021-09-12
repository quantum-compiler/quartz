#pragma once

#include "gate.h"

class XGate : public Gate {
 public:
  XGate() : Gate(GateType::x, 1/*num_qubits*/, 0/*num_parameters*/),
            mat({{0, 1}, {1, 0}}) {}

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
