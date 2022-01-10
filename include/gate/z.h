#pragma once

#include "gate.h"

class ZGate : public Gate {
 public:
  ZGate() : Gate(GateType::z, 1/*num_qubits*/, 0/*num_parameters*/),
            mat({{1, 0}, {0, -1}}) {}

  MatrixBase *get_matrix() override {
    return &mat;
  }
  Matrix<2> mat;
};
