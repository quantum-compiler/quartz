#pragma once

#include <string>
#include <vector>

namespace quartz {
class CircuitGate;

/**
 * A wire in the circuit.
 * Can be a qubit or a parameter (see comment below).
 */
class CircuitWire {
 public:
  enum Type {
    internal_qubit,
    input_qubit,
    output_qubit,
    input_param,
    internal_param
  };

  [[nodiscard]] bool is_qubit() const;
  [[nodiscard]] bool is_parameter() const;
  [[nodiscard]] std::string to_string() const;

  Type type;
  // If this wire is a qubit, |index| is the qubit id it corresponds to,
  // ranging [0, get_num_qubits()).
  // If this wire is a parameter, |index| is the parameter id.
  int index;
  std::vector<CircuitGate *> input_gates;
  std::vector<CircuitGate *> output_gates;
};

}  // namespace quartz
