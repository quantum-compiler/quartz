#pragma once

#include "../math/matrix.h"
#include "gate.h"

#include <cassert>

namespace quartz {
class CU1Gate : public Gate {
 public:
  CU1Gate() : Gate(GateType::cu1, 2 /*num_qubits*/, 1 /*num_parameters*/) {}
  MatrixBase *get_matrix(const std::vector<ParamType> &params) override {
    assert(params.size() == 1);
    ParamType phi = params[0];
    if (cached_matrices.find(phi) == cached_matrices.end()) {
      auto mat = std::make_unique<Matrix<4>>(
          Matrix<4>({{1, 0, 0, 0},
                     {0, 1, 0, 0},
                     {0, 0, 1, 0},
                     {0, 0, 0, cos(phi) + 1.0i * sin(phi)}}));
      cached_matrices[phi] = std::move(mat);
    }
    return cached_matrices[phi].get();
  }
  bool is_sparse() const override { return true; }
  bool is_diagonal() const override { return true; }
  int get_num_control_qubits() const override { return 1; }
  std::unordered_map<ParamType, std::unique_ptr<Matrix<4>>> cached_matrices;
};

}  // namespace quartz
