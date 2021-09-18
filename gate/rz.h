#pragma once

#include "gate.h"
#include <assert.h>

class RZGate : public Gate {
public:
  RZGate() : Gate(GateType::rz, 1/*num_qubits*/, 1/*num_parameters*/) {}
  MatrixBase *get_matrix(const std::vector<ParamType> &params) override {
    assert(params.size() == 1);
    ParamType theta = params[0];
    if (cached_matrices.find(theta) == cached_matrices.end()) {
      // e ^ {i * theta} = cos theta + i sin theta
      Matrix<2>* mat = new Matrix<2>({{cos(theta) - 1.0i * sin(theta), 0}, {0, cos(theta) + 1.0i * sin(theta)}});
      cached_matrices[theta] = mat;
    }
    return cached_matrices[theta];
  }
  std::unordered_map<float, Matrix<2>* > cached_matrices;
};
