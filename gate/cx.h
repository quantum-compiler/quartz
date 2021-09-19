#pragma once

#include "gate.h"
#include "../math/matrix.h"
#include <assert.h>

class CXGate : public Gate {
 public:
  CXGate() : Gate(GateType::cx, 2/*num_qubits*/, 0/*num_parameters*/),
             mat({{1, 0, 0, 0}, {0, 0, 0, 1}, {0, 0, 1, 0}, {0, 1, 0, 0}}) {}
  MatrixBase *get_matrix() override {
    return &mat;
  }
  Matrix<4> mat;
};
