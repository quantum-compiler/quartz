#pragma once

#include "../math/matrix.h"
#include "gate.h"

#include <assert.h>

namespace quartz {
class U2Gate : public Gate {
 public:
  U2Gate() : Gate(GateType::u2, 1 /*num_qubits*/, 2 /*num_parameters*/) {}
  MatrixBase *get_matrix(const std::vector<ParamType> &params) override {
    assert(params.size() == 2);
    ParamType phi = params[0];
    ParamType lambda = params[1];
    if (cached_matrices.find(phi) == cached_matrices.end() ||
        cached_matrices[phi].find(lambda) == cached_matrices[phi].end()) {
      auto mat = std::make_unique<Matrix<2>>(
          Matrix<2>({{1 / std::sqrt(2),
                      -1 / std::sqrt(2) * (cos(lambda) + 1.0i * sin(lambda))},
                     {1 / std::sqrt(2) * (cos(phi) + 1.0i * sin(phi)),
                      1 / std::sqrt(2) *
                          (cos(phi + lambda) + 1.0i * sin(phi + lambda))}}));
      cached_matrices[phi][lambda] = std::move(mat);
    }
    return cached_matrices[phi][lambda].get();
  }
  std::unordered_map<float,
                     std::unordered_map<float, std::unique_ptr<Matrix<2>>>>
      cached_matrices;
};

}  // namespace quartz
