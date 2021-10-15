#pragma once

#include "gate.h"
#include "../math/matrix.h"
#include <assert.h>

// Only used as a wrapper of input qubit in TASO graph
// TODO
class InputQubitGate : public Gate {
public:
  InputQubitGate()
      : Gate(GateType::input_qubit, 0 /*num_qubits*/, 0 /*num_parameters*/),
        mat() {}
  MatrixBase *get_matrix() override { return &mat; }
  Matrix<0> mat;
};
