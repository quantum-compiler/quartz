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
  // Get the minimum qubit index of the gate.
  [[nodiscard]] int get_min_qubit_index() const;
  // Get the qubit indices of the gate.
  [[nodiscard]] std::vector<int> get_qubit_indices() const;
  // Get the control qubit indices of the gate if it is a controlled gate.
  [[nodiscard]] std::vector<int> get_control_qubit_indices() const;
  std::vector<CircuitWire *> input_wires; // Include parameters!
  std::vector<CircuitWire *> output_wires;

  Gate *gate;
};
} // namespace quartz
