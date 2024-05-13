#include "circuitgate.h"

#include "quartz/circuitseq/circuitwire.h"
#include "quartz/context/context.h"
#include "quartz/utils/string_utils.h"

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
  } else {
    result += "[";
    for (int i = 0; i < (int)output_wires.size(); i++) {
      result += output_wires[i]->to_string();
      if (i != (int)output_wires.size() - 1) {
        result += ", ";
      }
    }
    result += "]";
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

std::string CircuitGate::to_json() const {
  std::string result;
  result += "[";
  result += "\"" + gate_type_name(gate->tp) + "\", ";
  result += "[";
  for (int j = 0; j < (int)output_wires.size(); j++) {
    result += "\"" + output_wires[j]->to_string() + "\"";
    if (j != (int)output_wires.size() - 1) {
      result += ", ";
    }
  }
  result += "], ";

  result += "[";
  for (int j = 0; j < (int)input_wires.size(); j++) {
    result += "\"" + input_wires[j]->to_string() + "\"";
    if (j != (int)input_wires.size() - 1) {
      result += ", ";
    }
  }
  result += "]]";
  return result;
}

void CircuitGate::read_json(std::istream &fin, Context *ctx,
                            std::vector<int> &input_qubits,
                            std::vector<int> &input_params,
                            std::vector<int> &output_qubits,
                            std::vector<int> &output_params, Gate *&gate) {
  fin.ignore(std::numeric_limits<std::streamsize>::max(), '\"');
  std::string name;
  std::getline(fin, name, '\"');
  auto gate_type = to_gate_type(name);
  gate = ctx->get_gate(gate_type);

  auto read_indices = [&fin](std::vector<int> &qubit_indices,
                             std::vector<int> &param_indices) {
    fin.ignore(std::numeric_limits<std::streamsize>::max(), '[');
    while (true) {
      char ch;
      fin.get(ch);
      while (ch != '\"' && ch != ']') {
        fin.get(ch);
      }
      if (ch == ']') {
        break;
      }

      // New index
      fin.get(ch);
      assert(ch == 'P' || ch == 'Q');
      int index;
      fin >> index;
      fin.ignore();  // '\"'
      if (ch == 'Q') {
        qubit_indices.push_back(index);
      } else {
        param_indices.push_back(index);
      }
    }
  };
  read_indices(output_qubits, output_params);
  read_indices(input_qubits, input_params);
  fin.ignore(std::numeric_limits<std::streamsize>::max(), ']');
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
      result += "//ctrl\n";
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
    for (auto input_wire : input_wires) {
      if (input_wire->is_parameter()) {
        assert(ctx->param_has_value(input_wire->index));
        std::ostringstream out;
        out.precision(param_precision);
        const auto &param_value = ctx->get_param_value(input_wire->index);
        if (param_value == 0) {
          // optimization: if a parameter is 0, do not output that many digits
          out << "0";
        } else {
          out << std::fixed << param_value;
        }
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
  for (auto input_wire : input_wires) {
    if (input_wire->is_qubit()) {
      if (first_qubit) {
        first_qubit = false;
      } else {
        result += ",";
      }
      result += "q[" + std::to_string(input_wire->index) + "]";
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

bool CircuitGate::equivalent(
    const CircuitGate *this_gate, const CircuitGate *other_gate,
    std::unordered_map<CircuitWire *, CircuitWire *> &wires_mapping,
    bool update_mapping, std::queue<CircuitWire *> *wires_to_search,
    bool backward) {
  if (this_gate->gate->tp != other_gate->gate->tp) {
    return false;
  }
  if (this_gate->input_wires.size() != other_gate->input_wires.size() ||
      this_gate->output_wires.size() != other_gate->output_wires.size()) {
    return false;
  }
  if (backward) {
    for (int j = 0; j < (int)this_gate->output_wires.size(); j++) {
      assert(this_gate->output_wires[j]->is_qubit());
      // The output wire must have been mapped.
      assert(wires_mapping.count(this_gate->output_wires[j]) != 0);
      if (wires_mapping[this_gate->output_wires[j]] !=
          other_gate->output_wires[j]) {
        return false;
      }
    }
    if (update_mapping) {
      // Map input wires
      for (int j = 0; j < (int)this_gate->input_wires.size(); j++) {
        assert(wires_mapping.count(this_gate->input_wires[j]) == 0);
        wires_mapping[this_gate->input_wires[j]] = other_gate->input_wires[j];
        wires_to_search->push(this_gate->input_wires[j]);
      }
    } else {
      // Verify mapping
      for (int j = 0; j < (int)this_gate->input_wires.size(); j++) {
        if (wires_mapping[this_gate->input_wires[j]] !=
            other_gate->input_wires[j]) {
          return false;
        }
      }
    }
  } else {
    for (int j = 0; j < (int)this_gate->input_wires.size(); j++) {
      if (this_gate->input_wires[j]->is_qubit()) {
        // The input wire must have been mapped.
        assert(wires_mapping.count(this_gate->input_wires[j]) != 0);
        if (wires_mapping[this_gate->input_wires[j]] !=
            other_gate->input_wires[j]) {
          return false;
        }
      } else {
        // parameters should not be mapped
        if (other_gate->input_wires[j] != this_gate->input_wires[j]) {
          return false;
        }
      }
    }
    if (update_mapping) {
      // Map output wires
      for (int j = 0; j < (int)this_gate->output_wires.size(); j++) {
        assert(wires_mapping.count(this_gate->output_wires[j]) == 0);
        wires_mapping[this_gate->output_wires[j]] = other_gate->output_wires[j];
        wires_to_search->push(this_gate->output_wires[j]);
      }
    } else {
      // Verify mapping
      for (int j = 0; j < (int)this_gate->output_wires.size(); j++) {
        if (wires_mapping[this_gate->output_wires[j]] !=
            other_gate->output_wires[j]) {
          return false;
        }
      }
    }
  }
  // Equivalent
  return true;
}

}  // namespace quartz
