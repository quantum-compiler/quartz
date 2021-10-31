#pragma once

#include "gate.h"
#include "../math/matrix.h"
#include <assert.h>

class CCZGate : public Gate {
public:
  CCZGate()
      : Gate(GateType::ccz, 3 /*num_qubits*/, 0 /*num_parameters*/),
        mat({{1, 0, 0, 0, 0, 0, 0, 0},
             {0, 1, 0, 0, 0, 0, 0, 0},
             {0, 0, 1, 0, 0, 0, 0, 0},
             {0, 0, 0, 1, 0, 0, 0, 0},
             {0, 0, 0, 0, 1, 0, 0, 0},
             {0, 0, 0, 0, 0, 1, 0, 0},
             {0, 0, 0, 0, 0, 0, 1, 0},
             {0, 0, 0, 0, 0, 0, 0, -1}}) {}
  MatrixBase *get_matrix() override { return &mat; }
  Matrix<8> mat;
};
