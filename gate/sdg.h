#pragma once

#include "gate.h"

class SDGGate : public Gate {
 public:
  SDGGate() : Gate(GateType::s, 1/*num_qubits*/, 0/*num_parameters*/),
            mat({{1, 0}, {0, -1.0i}}) {}

  MatrixBase *get_matrix() override {
    return &mat;
  }
  Matrix<2> mat;
};
