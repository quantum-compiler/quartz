#pragma once

#include "../math/matrix.h"
#include "gate.h"
#include <assert.h>

namespace quartz {
class CCZGate : public Gate {
public:
  CCZGate()
      : Gate(GateType::ccz, 3 /*num_qubits*/, 0 /*num_parameters*/),
        mat({{1, 0, 0, 0, 0, 0, 0, 0},
             {0, 1, 0, 0, 0, 0, 0, 0},
             {0, 0, 1, 0, 0, 0, 0, 0},
             {0, 0, 0, 1, 0, 0, 0, 0},
             {0, 0, 0, 0, 1, 0, 0, 0},
             {0, 0, 0, 0, 0, 1, 0, 0},
             {0, 0, 0, 0, 0, 0, 1, 0},
             {0, 0, 0, 0, 0, 0, 0, -1}}) {}
  MatrixBase *get_matrix() override { return &mat; }
  bool is_sparse() const override { return true; }
  Matrix<8> mat;
};

} // namespace quartz
