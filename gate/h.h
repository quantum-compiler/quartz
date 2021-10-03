#pragma once

#include "gate.h"

class HGate : public Gate {
 public:
  HGate() : Gate(GateType::h, 1/*num_qubits*/, 0/*num_parameters*/),
            mat({{1/std::sqrt(2), 1/std::sqrt(2)}, {1/std::sqrt(2), -1/std::sqrt(2)}}) {}

  MatrixBase *get_matrix() override {
    return &mat;
  }
  Matrix<2> mat;
};
