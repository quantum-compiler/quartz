#pragma once

#include "gate.h"

class YGate : public Gate {
 public:
  std::unique_ptr<MatrixBase> to_matrix() const {
    using namespace std::complex_literals;
    return std::make_unique<Matrix<2>>(Matrix<2>({{0, -1.0i}, {1.0i, 0}}));
  }
};
