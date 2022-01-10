#pragma once

#include "gate.h"
#include "../math/matrix.h"
#include <assert.h>

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
