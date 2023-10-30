#pragma once

#include "gate.h"

namespace quartz {
class RX3Gate : public Gate {
 public:
  RX3Gate()
      : Gate(GateType::rx3, 1 /*num_qubits*/, 0 /*num_parameters*/),
        mat({{ComplexType(1 / std::sqrt(2)), ComplexType(1.0i / std::sqrt(2))},
             {ComplexType(1.0i / std::sqrt(2)),
              ComplexType(1 / std::sqrt(2))}}) {}

  MatrixBase *get_matrix() override { return &mat; }
  Matrix<2> mat;
};

}  // namespace quartz
