#include "gate_utils.h"

#include <cassert>

namespace quartz {
std::string gate_type_name(GateType gt) {
  switch (gt) {
#define PER_GATE(x, XGate)                                                     \
  case GateType::x:                                                            \
    return #x;

#include "quartz/gate/gates.inc.h"

#undef PER_GATE
  }
  return "undefined";
}

GateType to_gate_type(const std::string &name) {
  if (false) {
  }
#define PER_GATE(x, XGate) else if (name == std::string(#x)) return GateType::x;

#include "quartz/gate/gates.inc.h"

#undef PER_GATE

  assert(false && "unknown gate type");
  return GateType();
}

std::vector<GateType> all_supported_gates() {
#define PER_GATE(x, XGate) GateType::x,
  std::vector<GateType> result = {
#include "quartz/gate/gates.inc.h"
  };
#undef PER_GATE
  return result;
}

std::vector<GateType> ibmq_gate_set() {
  return {GateType::u1, GateType::u2, GateType::u3, GateType::cx};
}

std::vector<GateType> voqc_gate_set() {
  return {GateType::h, GateType::x, GateType::rz, GateType::cx};
}

}  // namespace quartz
