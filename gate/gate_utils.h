#pragma once

#include <string>
#include <vector>

enum class GateType {
// Definition of gate types. This will expand to "x, y, z, rx, ry, ..."
#define PER_GATE(x, XGate) x,

#include "gates.inc.h"

#undef PER_GATE
};

std::string gate_type_name(GateType gt);

GateType to_gate_type(const std::string &name);

std::vector<GateType> all_supported_gates();

class Gate;
