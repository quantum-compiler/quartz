#pragma once

#include "../math/matrix.h"
#include "gate.h"

#include <assert.h>

namespace quartz {
class PDGGate : public Gate {
 public:
  PDGGate() : Gate(GateType::pdg, 1 /*num_qubits*/, 1 /*num_parameters*/) {}
  MatrixBase *get_matrix(const std::vector<ParamType> &params) override {
    assert(params.size() == 1);
    ParamType phi = params[0];
    if (cached_matrices.find(phi) == cached_matrices.end()) {
      auto mat = std::make_unique<Matrix<2>>(
          Matrix<2>({{1, 0}, {0, cos(phi) - 1.0i * sin(phi)}}));
      cached_matrices[phi] = std::move(mat);
    }
    return cached_matrices[phi].get();
  }
  bool is_sparse() const override { return true; }
  bool is_diagonal() const override { return true; }
  std::unordered_map<float, std::unique_ptr<Matrix<2>>> cached_matrices;
};

}  // namespace quartz
