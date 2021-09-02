#pragma once

#include "gate.h"

class XGate : public Gate {
 public:
  std::unique_ptr<MatrixBase> to_matrix() const {
    return std::make_unique<Matrix<2>>(Matrix<2>({{0, 1}, {1, 0}}));
  }
};
