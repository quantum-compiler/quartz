#pragma once

#include "gate.h"

class XGate : public Gate {
 public:
  XGate() : Gate(GateType::x, 1/*num_qubits*/, 0/*num_parameters*/),
            mat({{0, 1}, {1, 0}}) {}

  MatrixBase *get_matrix() override {
    return &mat;
  }
  Matrix<2> mat;
};
