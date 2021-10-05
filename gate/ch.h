#pragma once

#include "gate.h"
#include "../math/matrix.h"
#include <assert.h>

class CHGate : public Gate {
 public:
  CHGate() : Gate(GateType::ch, 2/*num_qubits*/, 0/*num_parameters*/),
             mat({{1, 0, 0, 0}, {0, 1/std::sqrt(2), 0, 1/std::sqrt(2)}, {0, 0, 1, 0}, {0, 1/std::sqrt(2), 0, -1/std::sqrt(2)}}) {}
  MatrixBase *get_matrix() override {
    return &mat;
  }
  Matrix<4> mat;
};
