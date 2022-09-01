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
  Matrix<4> mat;
};
} // namespace quartz
