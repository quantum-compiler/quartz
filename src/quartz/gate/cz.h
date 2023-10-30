#pragma once

#include "../math/matrix.h"
#include "gate.h"

#include <assert.h>

namespace quartz {
class CZGate : public Gate {
 public:
  CZGate()
      : Gate(GateType::cz, 2 /*num_qubits*/, 0 /*num_parameters*/),
        mat({{1, 0, 0, 0}, {0, 1, 0, 0}, {0, 0, 1, 0}, {0, 0, 0, -1}}) {}
  MatrixBase *get_matrix() override { return &mat; }
  bool is_symmetric() const override { return true; }
  bool is_sparse() const override { return true; }
  bool is_diagonal() const override { return true; }
  int get_num_control_qubits() const override { return 1; }
  Matrix<4> mat;
};

}  // namespace quartz
