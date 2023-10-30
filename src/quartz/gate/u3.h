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
    ParamType theta = params[0];
    ParamType phi = params[1];
    ParamType lambda = params[2];
    if (cached_matrices.find(theta) == cached_matrices.end() ||
        cached_matrices[theta].find(phi) == cached_matrices[theta].end() ||
        cached_matrices[theta][phi].find(lambda) ==
            cached_matrices[theta][phi].end()) {
      auto mat = std::make_unique<Matrix<2>>(Matrix<2>(
          {{cos(theta), -sin(theta) * (cos(lambda) + 1.0i * sin(lambda))},
           {sin(theta) * (cos(phi) + 1.0i * sin(phi)),
            cos(theta) * (cos(phi + lambda) + 1.0i * sin(phi + lambda))}}));
      cached_matrices[theta][phi][lambda] = std::move(mat);
    }
    return cached_matrices[theta][phi][lambda].get();
  }
  std::unordered_map<
      float, std::unordered_map<
                 float, std::unordered_map<float, std::unique_ptr<Matrix<2>>>>>
      cached_matrices;
};

}  // namespace quartz
