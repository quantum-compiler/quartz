#pragma once

#include "../math/matrix.h"
#include "gate.h"

#include <assert.h>

namespace quartz {
class SWAPGate : public Gate {
 public:
  SWAPGate()
      : Gate(GateType::swap, 2 /*num_qubits*/, 0 /*num_parameters*/),
        mat({{1, 0, 0, 0}, {0, 0, 1, 0}, {0, 1, 0, 0}, {0, 0, 0, 1}}) {}
  MatrixBase *get_matrix() override { return &mat; }
  bool is_symmetric() const override { return true; }
  bool is_sparse() const override { return true; }
  Matrix<4> mat;
};
}  // namespace quartz
