#pragma once

#include "../math/matrix.h"
#include "gate.h"

#include <assert.h>

namespace quartz {
class B2Gate : public Gate {
 public:
  B2Gate() : Gate(GateType::b2, 2 /*num_qubits*/, 1 /*num_parameters*/) {}
  MatrixBase *get_matrix() override {
    assert(false);
    auto mat = std::make_unique<Matrix<4>>(
        Matrix<4>({{0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}}));

    return mat.get();
  }
  bool is_symmetric() const override { return true; }
  bool is_sparse() const override { return true; }
  bool is_diagonal() const override { return true; }
  int get_num_control_qubits() const override { return 1; }
};

}  // namespace quartz
