#pragma once

#include "gate.h"

namespace quartz {
class RY1Gate : public Gate {
 public:
  RY1Gate()
      : Gate(GateType::ry1, 1 /*num_qubits*/, 0 /*num_parameters*/),
        mat({{ComplexType(1 / std::sqrt(2)), ComplexType(-1 / std::sqrt(2))},
             {ComplexType(1 / std::sqrt(2)), ComplexType(1 / std::sqrt(2))}}) {}

  MatrixBase *get_matrix() override { return &mat; }
  Matrix<2> mat;
};

}  // namespace quartz
