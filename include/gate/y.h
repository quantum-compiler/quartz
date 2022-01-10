#pragma once

#include "gate.h"

class YGate : public Gate {
 public:
  YGate() : Gate(GateType::y, 1/*num_qubits*/, 0/*num_parameters*/),
            mat({{ComplexType(0), ComplexType(-1.0i)},
                 {ComplexType(1.0i), ComplexType(0)}}) {}
  MatrixBase *get_matrix() override {
    return &mat;
  }
  Matrix<2> mat;
};
