#pragma once

#include "gate.h"

namespace quartz {
class HGate : public Gate {
 public:
  HGate()
      : Gate(GateType::h, 1 /*num_qubits*/, 0 /*num_parameters*/),
        mat({{ComplexType(1 / std::sqrt(2)), ComplexType(1 / std::sqrt(2))},
             {ComplexType(1 / std::sqrt(2)), ComplexType(-1 / std::sqrt(2))}}) {
  }

  MatrixBase *get_matrix() override { return &mat; }
  Matrix<2> mat;
};

}  // namespace quartz
