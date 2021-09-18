#include "gate_utils.h"

std::string gate_type_name(GateType gt) {
  switch (gt) {
#define PER_GATE(x, XGate) \
    case GateType::x:      \
      return #x;

#include "gates.inc.h"

#undef PER_GATE
  }
  return "undefined";
}
