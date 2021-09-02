#pragma once

#include "../math/matrix.h"

#include <iostream>

class Gate {
 public:
  virtual std::unique_ptr<MatrixBase> to_matrix() const {
    std::cerr << "Gate::to_matrix() called." << std::endl;
    return std::make_unique<MatrixBase>();
  }
};
