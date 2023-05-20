#include "circuitgate.h"
#include "circuitwire.h"
#include "context/context.h"

#include <algorithm>
#include <cassert>
#include <cctype>

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

std::vector<int> CircuitGate::get_insular_qubit_indices() const {
  if (gate->get_num_qubits() == 1) {
    if (gate->is_sparse()) {
      // Diagonal or anti-diagonal.
      for (auto &wire : input_wires) {
        if (wire->is_qubit()) {
          return {wire->index};
        }
      }
      assert(false);
    } else {
      return {};
    }
  }
  if (gate->is_diagonal()) {
    // All qubits are insular for diagonal gates.
    return get_qubit_indices();
  }
  int remaining_control_qubits = gate->get_num_control_qubits();
  if (remaining_control_qubits == 0) {
    // No qubit are insular for multi-qubit non-controlled gates.
    return {};
  }
  std::vector<int> result;
  result.reserve(remaining_control_qubits);
  for (auto &input_node : input_wires) {
    // Control qubits always appear first.
    if (input_node->is_qubit()) {
      remaining_control_qubits--;
      result.push_back(input_node->index);
      if (remaining_control_qubits == 0) {
        break;
      }
    }
  }
  return result;
}

std::vector<int> CircuitGate::get_non_insular_qubit_indices() const {
  if (gate->get_num_qubits() == 1) {
    if (gate->is_sparse()) {
      // Diagonal or anti-diagonal.
      return {};
    } else {
      for (auto &wire : input_wires) {
        if (wire->is_qubit()) {
          return {wire->index};
        }
      }
      assert(false);
    }
  }
  if (gate->is_diagonal()) {
    // All qubits are insular for diagonal gates.
    return {};
  }
  int remaining_control_qubits = gate->get_num_control_qubits();
  if (remaining_control_qubits == 0) {
    // No qubit are insular for multi-qubit non-controlled gates.
    return get_qubit_indices();
  }
  std::vector<int> result;
  result.reserve(gate->get_num_qubits() - remaining_control_qubits);
  for (auto &input_node : input_wires) {
    // Control qubits always appear first.
    if (input_node->is_qubit()) {
      remaining_control_qubits--;
      if (remaining_control_qubits < 0) {
        result.push_back(input_node->index);
      }
    }
  }
  return result;
}

std::string CircuitGate::to_string() const {
  std::string result;
  if (output_wires.size() == 1) {
    result += output_wires[0]->to_string();
  } else if (output_wires.size() == 2) {
    result += "[" + output_wires[0]->to_string();
    result += ", " + output_wires[1]->to_string();
    result += "]";
  } else {
    assert(false && "A circuit gate should have 1 or 2 outputs.");
  }
  result += " = ";
  result += gate_type_name(gate->tp);
  if (gate->get_num_control_qubits() > 0) {
    auto control_state = gate->get_control_state();
    if (!std::all_of(control_state.begin(), control_state.end(),
                     [](bool v) { return v; })) {
      // Not a simple controlled gate
      result += "[";
      for (const auto &value : control_state) {
        result += (int)value + '0';
      }
      result += "]";
    }
  }
  result += "(";
  for (int j = 0; j < (int)input_wires.size(); j++) {
    result += input_wires[j]->to_string();
    if (j != (int)input_wires.size() - 1) {
      result += ", ";
    }
  }
  result += ")";
  return result;
}

std::string CircuitGate::to_qasm_style_string(Context *ctx,
                                              int param_precision) const {
  assert(gate->is_quantum_gate());
  std::string result;
  if (gate->get_num_control_qubits() > 0) {
    auto control_state = gate->get_control_state();
    if (!std::all_of(control_state.begin(), control_state.end(),
                     [](bool v) { return v; })) {
      // Not a simple controlled gate
      auto control_qubits = get_control_qubit_indices();
      for (int i = 0; i < (int)control_state.size(); i++) {
        if (!control_state[i]) {
          result += "x q[" + std::to_string(control_qubits[i]) + "];\n";
        }
      }
    }
  }

  auto gate_name = gate_type_name(gate->tp);
  std::transform(gate_name.begin(), gate_name.end(), gate_name.begin(),
                 [](unsigned char c) { return std::tolower(c); });
  result += gate_name;
  if (gate->get_num_parameters() > 0) {
    int num_remaining_parameters = gate->get_num_parameters();
    result += "(";
    for (int j = 0; j < (int)input_wires.size(); j++) {
      if (input_wires[j]->is_parameter()) {
        assert(ctx->param_has_value(input_wires[j]->index));
        std::ostringstream out;
        out.precision(param_precision);
        out << std::fixed << ctx->get_param_value(input_wires[j]->index);
        result += std::move(out).str();
        num_remaining_parameters--;
        if (num_remaining_parameters != 0) {
          result += ",";
        }
      }
    }
    result += ")";
  }
  result += " ";
  bool first_qubit = true;
  for (int j = 0; j < (int)input_wires.size(); j++) {
    if (input_wires[j]->is_qubit()) {
      if (first_qubit) {
        first_qubit = false;
      } else {
        result += ",";
      }
      result += "q[" + std::to_string(input_wires[j]->index) + "]";
    }
  }
  result += ";\n";

  if (gate->get_num_control_qubits() > 0) {
    auto control_state = gate->get_control_state();
    if (!std::all_of(control_state.begin(), control_state.end(),
                     [](bool v) { return v; })) {
      // Not a simple controlled gate
      auto control_qubits = get_control_qubit_indices();
      for (int i = 0; i < (int)control_state.size(); i++) {
        if (!control_state[i]) {
          result += "x q[" + std::to_string(control_qubits[i]) + "];\n";
        }
      }
    }
  }
  return result;
}

} // namespace quartz
