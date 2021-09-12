#include "context.h"
#include "../gate/all_gates.h"

Context::Context(const std::vector<GateType> &supported_gates) {
  gates_.reserve(supported_gates.size());
  for (const auto &gate : supported_gates) {
    insert_gate(gate);
  }
}

Gate *Context::get_gate(GateType tp) {
  return gates_[tp].get();
}

bool Context::insert_gate(GateType tp) {
  if (gates_.count(tp) > 0) {
    return false;
  }
  std::unique_ptr<Gate> new_gate;

#define PER_GATE(x, XGate)                  \
    case GateType::x:                       \
      new_gate = std::make_unique<XGate>(); \
      break;

  switch (tp) {
#include "../gate/gates.inc.h"
  }

#undef PER_GATE

  gates_[tp] = std::move(new_gate);
  return true;
}
