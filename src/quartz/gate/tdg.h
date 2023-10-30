#pragma once

#include "gate.h"

namespace quartz {
class TDGGate : public Gate {
 public:
  TDGGate()
      : Gate(GateType::tdg, 1 /*num_qubits*/, 0 /*num_parameters*/),
        mat({{1, 0}, {0, std::sqrt(2) / 2 * (1.0 - 1.0i)}}) {}

  MatrixBase *get_matrix() override { return &mat; }
  bool is_sparse() const override { return true; }
  bool is_diagonal() const override { return true; }
  Matrix<2> mat;
};

}  // namespace quartz
