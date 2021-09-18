#pragma once

#include "gate.h"
#include <assert.h>

class RYGate : public Gate {
 public:
  RYGate() : Gate(GateType::ry, 1/*num_qubits*/, 1/*num_parameters*/) {}
  MatrixBase *get_matrix(const std::vector<ParamType> &params) override {
    assert(params.size() == 1);
    ParamType theta = params[0];
    if (cached_matrices.find(theta) == cached_matrices.end()) {
      auto mat =
          std::make_unique<Matrix<2>>(Matrix<2>({{cos(theta), -sin(theta)},
                                                 {sin(theta), cos(theta)}}));
      cached_matrices[theta] = std::move(mat);
    }
    return cached_matrices[theta].get();
  }
  std::unordered_map<float, std::unique_ptr<Matrix<2>>> cached_matrices;
};
