#pragma once

#include "../gate/gate.h"
#include "../utils/utils.h"

#include <vector>

namespace quartz {

class CircuitWire;

/**
 * A gate in the circuit.
 * Stores the gate type, input and output information.
 */
class CircuitGate {
public:
  int get_min_qubit_index() const;
  std::vector<CircuitWire *> input_wires; // Include parameters!
  std::vector<CircuitWire *> output_wires;

  Gate *gate;
};
} // namespace quartz
