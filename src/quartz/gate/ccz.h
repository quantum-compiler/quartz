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
  bool is_symmetric() const override { return true; }
  bool is_sparse() const override { return true; }
  bool is_diagonal() const override { return true; }
  int get_num_control_qubits() const override { return 2; }
  Matrix<8> mat;
};

}  // namespace quartz
