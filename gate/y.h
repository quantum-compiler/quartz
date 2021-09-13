#pragma once

#include "gate.h"

class YGate : public Gate {
 public:
  YGate() : Gate(GateType::y, 1/*num_qubits*/, 0/*num_parameters*/),
            mat({{0, -1.0i}, {1.0i, 0}}) {}
  MatrixBase *get_matrix() override {
    return &mat;
  }
  Matrix<2> mat;
};
