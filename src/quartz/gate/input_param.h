#pragma once

#include "../math/matrix.h"
#include "gate.h"
#include <assert.h>

namespace quartz {
// Only used as a wrapper of input qubit in TASO graph
// TODO
class InputParamGate : public Gate {
public:
  InputParamGate()
      : Gate(GateType::input_param, 0 /*num_qubits*/, 0 /*num_parameters*/),
        mat() {}
  MatrixBase *get_matrix() override { return &mat; }
  Matrix<1> mat;
};
} // namespace quartz
