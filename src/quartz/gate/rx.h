#pragma once

#include "../math/matrix.h"
#include "gate.h"

#include <assert.h>

namespace quartz {
class RXGate : public Gate {
 public:
  RXGate() : Gate(GateType::rx, 1 /*num_qubits*/, 1 /*num_parameters*/) {}
  MatrixBase *get_matrix(const std::vector<ParamType> &params) override {
    assert(params.size() == 1);
    const ParamType &theta = params[0];
    if (cached_matrices.find(theta) == cached_matrices.end()) {
      auto mat = std::make_unique<Matrix<2>>(
          Matrix<2>({{ComplexType(cos_param(theta)),
                      ComplexType(-1.0i * sin_param(theta))},
                     {ComplexType(-1.0i * sin_param(theta)),
                      ComplexType(cos_param(theta))}}));
      cached_matrices[theta] = std::move(mat);
    }
    return cached_matrices[theta].get();
  }
  bool is_param_halved(int i) const override { return true; }
  std::unordered_map<ParamType, std::unique_ptr<Matrix<2>>, ParamHash>
      cached_matrices;
};

}  // namespace quartz
