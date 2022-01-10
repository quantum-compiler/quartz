#pragma once

#include "gate.h"

class TGate : public Gate {
 public:
  TGate() : Gate(GateType::t, 1/*num_qubits*/, 0/*num_parameters*/),
            mat({{1, 0}, {0, std::sqrt(2)/2 * (1.0 + 1.0i)}}) {}

  MatrixBase *get_matrix() override {
    return &mat;
  }
  Matrix<2> mat;
};
