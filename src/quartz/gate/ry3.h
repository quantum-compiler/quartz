#pragma once

#include "gate.h"

namespace quartz {
class RY3Gate : public Gate {
 public:
  RY3Gate()
      : Gate(GateType::ry3, 1 /*num_qubits*/, 0 /*num_parameters*/),
        mat({{ComplexType(-1 / std::sqrt(2)), ComplexType(-1 / std::sqrt(2))},
             {ComplexType(1 / std::sqrt(2)), ComplexType(-1 / std::sqrt(2))}}) {
  }

  MatrixBase *get_matrix() override { return &mat; }
  Matrix<2> mat;
};

}  // namespace quartz
