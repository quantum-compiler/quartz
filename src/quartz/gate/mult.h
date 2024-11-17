#pragma once

#include "gate.h"

#include <assert.h>

namespace quartz {
class MultGate : public Gate {
 public:
  MultGate() : Gate(GateType::mult, 0 /*num_qubits*/, 2 /*num_parameters*/) {}
  ParamType compute(const std::vector<ParamType> &input_params) override {
    assert(input_params.size() == 2);
    return input_params[0] * input_params[1];
  }
  bool is_commutative() const override { return true; }
};

}  // namespace quartz
