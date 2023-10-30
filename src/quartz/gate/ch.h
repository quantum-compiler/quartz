#pragma once

#include "../math/matrix.h"
#include "gate.h"

#include <assert.h>

namespace quartz {
class CHGate : public Gate {
 public:
  CHGate()
      : Gate(GateType::ch, 2 /*num_qubits*/, 0 /*num_parameters*/),
        mat({{1, 0, 0, 0},
             {0, 1 / std::sqrt(2), 0, 1 / std::sqrt(2)},
             {0, 0, 1, 0},
             {0, 1 / std::sqrt(2), 0, -1 / std::sqrt(2)}}) {}
  MatrixBase *get_matrix() override { return &mat; }
  int get_num_control_qubits() const override { return 1; }
  Matrix<4> mat;
};

}  // namespace quartz
