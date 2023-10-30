#pragma once

#include "../math/matrix.h"
#include "gate.h"

#include <assert.h>

namespace quartz {
class AddGate : public Gate {
 public:
  AddGate() : Gate(GateType::add, 0 /*num_qubits*/, 2 /*num_parameters*/) {}
  ParamType compute(const std::vector<ParamType> &input_params) override {
    assert(input_params.size() == 2);
    return input_params[0] + input_params[1];
  }
  bool is_commutative() const override { return true; }
};

}  // namespace quartz
