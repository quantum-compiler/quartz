#pragma once

#include "gate.h"
#include "../math/matrix.h"
#include <assert.h>

class CXGate : public Gate {
 public:
  CXGate() : Gate(GateType::cx, 2/*num_qubits*/, 0/*num_parameters*/),
             mat({{ComplexType(1), ComplexType(0), ComplexType(0),
                   ComplexType(0)},
                  {ComplexType(0), ComplexType(0), ComplexType(0),
                   ComplexType(1)},
                  {ComplexType(0), ComplexType(0), ComplexType(1),
                   ComplexType(0)},
                  {ComplexType(0), ComplexType(1), ComplexType(0),
                   ComplexType(0)}}) {}
  MatrixBase *get_matrix() override {
    return &mat;
  }
  Matrix<4> mat;
};