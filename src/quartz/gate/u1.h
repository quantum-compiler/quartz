#pragma once

#include "../math/matrix.h"
#include "gate.h"

#include <cassert>

namespace quartz {
class U1Gate : public Gate {
 public:
  U1Gate() : Gate(GateType::u1, 1 /*num_qubits*/, 1 /*num_parameters*/) {}
  MatrixBase *get_matrix(const std::vector<ParamType> &params) override {
    assert(params.size() == 1);
    ParamType lambda = params[0];
    if (cached_matrices.find(lambda) == cached_matrices.end()) {
      auto mat = std::make_unique<Matrix<2>>(
          Matrix<2>({{1, 0}, {0, cos(lambda) + 1.0i * sin(lambda)}}));
      cached_matrices[lambda] = std::move(mat);
    }
    return cached_matrices[lambda].get();
  }
  bool is_sparse() const override { return true; }
  bool is_diagonal() const override { return true; }
  std::unordered_map<ParamType, std::unique_ptr<Matrix<2>>> cached_matrices;
};

}  // namespace quartz
