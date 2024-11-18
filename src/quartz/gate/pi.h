#pragma once

#include "../math/matrix.h"
#include "gate.h"

namespace quartz {
class PiGate : public Gate {
 public:
  PiGate() : Gate(GateType::neg, 0 /*num_qubits*/, 1 /*num_parameters*/) {}
  ParamType compute(const std::vector<ParamType> &input_params) override {
    assert(input_params.size() == 1);
    return PI / input_params[0];
  }
};
}  // namespace quartz
