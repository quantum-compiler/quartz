#pragma once

#include "../math/matrix.h"
#include "gate.h"

#include <assert.h>

namespace quartz {
class CXGate : public Gate {
 public:
  CXGate()
      : Gate(GateType::cx, 2 /*num_qubits*/, 0 /*num_parameters*/),
        mat({{ComplexType(1), ComplexType(0), ComplexType(0), ComplexType(0)},
             {ComplexType(0), ComplexType(0), ComplexType(0), ComplexType(1)},
             {ComplexType(0), ComplexType(0), ComplexType(1), ComplexType(0)},
             {ComplexType(0), ComplexType(1), ComplexType(0),
              ComplexType(0)}}) {}
  MatrixBase *get_matrix() override { return &mat; }
  bool is_sparse() const override { return true; }
  int get_num_control_qubits() const override { return 1; }
  Matrix<4> mat;
};

}  // namespace quartz
