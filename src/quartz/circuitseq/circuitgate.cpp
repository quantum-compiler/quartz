#include "circuitgate.h"
#include "circuitwire.h"

namespace quartz {
int CircuitGate::get_min_qubit_index() const {
  int result = -1;
  for (auto &input_node : input_wires) {
    if (input_node->is_qubit() &&
        (result == -1 || input_node->index < result)) {
      result = input_node->index;
    }
  }
  return result;
}

std::vector<int> CircuitGate::get_qubit_indices() const {
  std::vector<int> result;
  result.reserve(gate->get_num_qubits());
  for (auto &input_node : input_wires) {
    if (input_node->is_qubit()) {
      result.push_back(input_node->index);
    }
  }
  return result;
}

std::vector<int> CircuitGate::get_control_qubit_indices() const {
  const int num_control_qubits = gate->get_num_control_qubits();
  if (num_control_qubits == 0) {
    return std::vector<int>();
  }
  std::vector<int> result;
  result.reserve(num_control_qubits);
  for (auto &input_node : input_wires) {
    // Control qubits always appear first.
    if (input_node->is_qubit()) {
      result.push_back(input_node->index);
      if (result.size() == num_control_qubits) {
        break;
      }
    }
  }
  return result;
}

} // namespace quartz
