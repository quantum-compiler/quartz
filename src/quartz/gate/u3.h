#pragma once

#include "../math/matrix.h"
#include "gate.h"

#include <assert.h>

namespace quartz {
class U3Gate : public Gate {
 public:
  U3Gate() : Gate(GateType::u3, 1 /*num_qubits*/, 3 /*num_parameters*/) {}
  MatrixBase *get_matrix(const std::vector<ParamType> &params) override {
    assert(params.size() == 3);
    const ParamType &theta = params[0];
    const ParamType &phi = params[1];
    const ParamType &lambda = params[2];
    if (cached_matrices.find(theta) == cached_matrices.end() ||
        cached_matrices[theta].find(phi) == cached_matrices[theta].end() ||
        cached_matrices[theta][phi].find(lambda) ==
            cached_matrices[theta][phi].end()) {
      auto mat = std::make_unique<Matrix<2>>(Matrix<2>(
          {{cos_param(theta),
            -sin_param(theta) * (cos_param(lambda) + 1.0i * sin_param(lambda))},
           {sin_param(theta) * (cos_param(phi) + 1.0i * sin_param(phi)),
            cos_param(theta) *
                (cos_param(phi + lambda) + 1.0i * sin_param(phi + lambda))}}));
      cached_matrices[theta][phi][lambda] = std::move(mat);
    }
    return cached_matrices[theta][phi][lambda].get();
  }
  bool is_param_halved(int i) const override { return i == 0; }
  std::unordered_map<
      ParamType,
      std::unordered_map<
          ParamType,
          std::unordered_map<ParamType, std::unique_ptr<Matrix<2>>, ParamHash>,
          ParamHash>,
      ParamHash>
      cached_matrices;
};

}  // namespace quartz
