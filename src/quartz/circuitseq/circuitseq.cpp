#include "circuitseq.h"
#include "../context/context.h"
#include "../gate/gate.h"
#include "../parser/qasm_parser.h"

#include <algorithm>
#include <cassert>
#include <charconv>
#include <fstream>
#include <queue>
#include <unordered_set>
#include <utility>

namespace quartz {
CircuitSeq::CircuitSeq(int num_qubits, int num_input_parameters)
    : num_qubits(num_qubits), num_input_parameters(num_input_parameters),
      hash_value_(0), hash_value_valid_(false) {
  wires.reserve(num_qubits + num_input_parameters);
  outputs.reserve(num_qubits);
  parameters.reserve(num_input_parameters);
  // Initialize num_qubits qubits
  for (int i = 0; i < num_qubits; i++) {
    auto wire = std::make_unique<CircuitWire>();
    wire->type = CircuitWire::input_qubit;
    wire->index = i;
    outputs.push_back(wire.get());
    wires.push_back(std::move(wire));
  }
  // Initialize num_input_parameters parameters
  for (int i = 0; i < num_input_parameters; i++) {
    auto wire = std::make_unique<CircuitWire>();
    wire->type = CircuitWire::input_param;
    wire->index = i;
    parameters.push_back(wire.get());
    wires.push_back(std::move(wire));
  }
}

CircuitSeq::CircuitSeq(const CircuitSeq &other) { clone_from(other, {}, {}); }

std::unique_ptr<CircuitSeq> CircuitSeq::clone() const {
  return std::make_unique<CircuitSeq>(*this);
}

bool CircuitSeq::fully_equivalent(const CircuitSeq &other) const {
  if (this == &other) {
    return true;
  }
  // Do not check the hash value because of floating point errors
  // and it is possible that one of the two sequences may have not calculated
  // the hash value.
  if (num_qubits != other.num_qubits ||
      num_input_parameters != other.num_input_parameters) {
    return false;
  }
  if (wires.size() != other.wires.size() ||
      gates.size() != other.gates.size()) {
    return false;
  }
  std::unordered_map<CircuitWire *, CircuitWire *> wires_mapping;
  for (int i = 0; i < (int)wires.size(); i++) {
    wires_mapping[other.wires[i].get()] = wires[i].get();
  }
  for (int i = 0; i < (int)gates.size(); i++) {
    if (gates[i]->gate->tp != other.gates[i]->gate->tp) {
      return false;
    }
    if (gates[i]->input_wires.size() != other.gates[i]->input_wires.size() ||
        gates[i]->output_wires.size() != other.gates[i]->output_wires.size()) {
      return false;
    }
    for (int j = 0; j < (int)gates[i]->input_wires.size(); j++) {
      if (wires_mapping[other.gates[i]->input_wires[j]] !=
          gates[i]->input_wires[j]) {
        return false;
      }
    }
    for (int j = 0; j < (int)gates[i]->output_wires.size(); j++) {
      if (wires_mapping[other.gates[i]->output_wires[j]] !=
          gates[i]->output_wires[j]) {
        return false;
      }
    }
  }
  return true;
}

bool CircuitSeq::fully_equivalent(Context *ctx, CircuitSeq &other) {
  if (hash(ctx) != other.hash(ctx)) {
    return false;
  }
  return fully_equivalent(other);
}

bool CircuitSeq::less_than(const CircuitSeq &other) const {
  if (this == &other) {
    return false;
  }
  if (num_qubits != other.num_qubits) {
    return num_qubits < other.num_qubits;
  }
  if (get_num_gates() != other.get_num_gates()) {
    return get_num_gates() < other.get_num_gates();
  }
  if (get_num_input_parameters() != other.get_num_input_parameters()) {
    return get_num_input_parameters() < other.get_num_input_parameters();
  }
  if (get_num_total_parameters() != other.get_num_total_parameters()) {
    // We want fewer quantum gates, i.e., more traditional parameters.
    return get_num_total_parameters() > other.get_num_total_parameters();
  }
  for (int i = 0; i < (int)gates.size(); i++) {
    if (gates[i]->gate->tp != other.gates[i]->gate->tp) {
      return gates[i]->gate->tp < other.gates[i]->gate->tp;
    }
    assert(gates[i]->input_wires.size() == other.gates[i]->input_wires.size());
    assert(gates[i]->output_wires.size() ==
           other.gates[i]->output_wires.size());
    for (int j = 0; j < (int)gates[i]->input_wires.size(); j++) {
      if (gates[i]->input_wires[j]->is_qubit() !=
          other.gates[i]->input_wires[j]->is_qubit()) {
        return gates[i]->input_wires[j]->is_qubit();
      }
      if (gates[i]->input_wires[j]->index !=
          other.gates[i]->input_wires[j]->index) {
        return gates[i]->input_wires[j]->index <
               other.gates[i]->input_wires[j]->index;
      }
    }
    for (int j = 0; j < (int)gates[i]->output_wires.size(); j++) {
      if (gates[i]->output_wires[j]->is_qubit() !=
          other.gates[i]->output_wires[j]->is_qubit()) {
        return gates[i]->output_wires[j]->is_qubit();
      }
      if (gates[i]->output_wires[j]->index !=
          other.gates[i]->output_wires[j]->index) {
        return gates[i]->output_wires[j]->index <
               other.gates[i]->output_wires[j]->index;
      }
    }
  }
  return false; // fully equivalent
}

bool CircuitSeq::add_gate(const std::vector<int> &qubit_indices,
                          const std::vector<int> &parameter_indices, Gate *gate,
                          int *output_para_index) {
  if (gate->get_num_qubits() != qubit_indices.size())
    return false;
  if (gate->get_num_parameters() != parameter_indices.size())
    return false;
  if (gate->is_parameter_gate() && output_para_index == nullptr)
    return false;
  // qubit indices must stay in range
  for (auto qubit_idx : qubit_indices)
    if ((qubit_idx < 0) || (qubit_idx >= get_num_qubits()))
      return false;
  // parameter indices must stay in range
  for (auto para_idx : parameter_indices)
    if ((para_idx < 0) || (para_idx >= parameters.size()))
      return false;
  auto circuit_gate = std::make_unique<CircuitGate>();
  circuit_gate->gate = gate;
  for (auto qubit_idx : qubit_indices) {
    circuit_gate->input_wires.push_back(outputs[qubit_idx]);
    outputs[qubit_idx]->output_gates.push_back(circuit_gate.get());
  }
  for (auto para_idx : parameter_indices) {
    circuit_gate->input_wires.push_back(parameters[para_idx]);
    parameters[para_idx]->output_gates.push_back(circuit_gate.get());
  }
  if (gate->is_parameter_gate()) {
    auto wire = std::make_unique<CircuitWire>();
    wire->type = CircuitWire::internal_param;
    wire->index = *output_para_index = (int)parameters.size();
    wire->input_gates.push_back(circuit_gate.get());
    circuit_gate->output_wires.push_back(wire.get());
    parameters.push_back(wire.get());
    wires.push_back(std::move(wire));
  } else {
    assert(gate->is_quantum_gate());
    for (auto qubit_idx : qubit_indices) {
      auto wire = std::make_unique<CircuitWire>();
      wire->type = CircuitWire::internal_qubit;
      wire->index = qubit_idx;
      wire->input_gates.push_back(circuit_gate.get());
      circuit_gate->output_wires.push_back(wire.get());
      outputs[qubit_idx] = wire.get(); // Update outputs
      wires.push_back(std::move(wire));
    }
  }
  gates.push_back(std::move(circuit_gate));
  hash_value_valid_ = false;
  return true;
}

bool CircuitSeq::add_gate(CircuitGate *gate) {
  std::vector<int> qubit_indices;
  std::vector<int> parameter_indices;
  int output_para_index;
  for (auto &wire : gate->input_wires) {
    if (wire->is_qubit()) {
      qubit_indices.push_back(wire->index);
    } else {
      parameter_indices.push_back(wire->index);
    }
  }
  return add_gate(qubit_indices, parameter_indices, gate->gate,
                  &output_para_index);
}

bool CircuitSeq::insert_gate(int insert_position,
                             const std::vector<int> &qubit_indices,
                             const std::vector<int> &parameter_indices,
                             Gate *gate, int *output_para_index) {
  if (insert_position < 0 || insert_position > (int)gates.size())
    return false;
  if (gate->get_num_qubits() != qubit_indices.size())
    return false;
  if (gate->get_num_parameters() != parameter_indices.size())
    return false;
  if (gate->is_parameter_gate() && output_para_index == nullptr)
    return false;
  // qubit indices must stay in range
  for (auto qubit_idx : qubit_indices)
    if ((qubit_idx < 0) || (qubit_idx >= get_num_qubits()))
      return false;
  // parameter indices must stay in range
  for (auto para_idx : parameter_indices)
    if ((para_idx < 0) || (para_idx >= parameters.size()))
      return false;
  // Find the location to insert.
  std::vector<CircuitWire *> previous_wires(get_num_qubits());
  for (int i = 0; i < get_num_qubits(); i++) {
    previous_wires[i] = wires[i].get();
  }
  for (int i = 0; i < insert_position; i++) {
    for (auto &output_wire : gates[i]->output_wires) {
      if (output_wire->is_qubit()) {
        previous_wires[output_wire->index] = output_wire;
      }
    }
  }
  auto circuit_gate = std::make_unique<CircuitGate>();
  circuit_gate->gate = gate;
  for (auto qubit_idx : qubit_indices) {
    circuit_gate->input_wires.push_back(previous_wires[qubit_idx]);
    if (gate->is_quantum_gate()) {
      auto wire = std::make_unique<CircuitWire>();
      wire->type = CircuitWire::internal_qubit;
      wire->index = qubit_idx;
      wire->input_gates.push_back(circuit_gate.get());
      circuit_gate->output_wires.push_back(wire.get());
      if (outputs[qubit_idx] == previous_wires[qubit_idx]) {
        outputs[qubit_idx] = wire.get(); // Update outputs
      } else {
        for (auto &output_gate : previous_wires[qubit_idx]->output_gates) {
          // Should have exactly one |output_gate|
          for (auto &input_wire : output_gate->input_wires) {
            if (input_wire == previous_wires[qubit_idx]) {
              input_wire = wire.get();
              break;
            }
          }
        }
      }
      // XXX: the wires are placed at the end, so it will be not compatible
      // with remove_last_gate().
      wires.push_back(std::move(wire));
    }
    previous_wires[qubit_idx]->output_gates.push_back(circuit_gate.get());
  }
  for (auto para_idx : parameter_indices) {
    circuit_gate->input_wires.push_back(parameters[para_idx]);
    parameters[para_idx]->output_gates.push_back(circuit_gate.get());
  }
  if (gate->is_parameter_gate()) {
    auto wire = std::make_unique<CircuitWire>();
    wire->type = CircuitWire::internal_param;
    wire->index = *output_para_index = (int)parameters.size();
    wire->input_gates.push_back(circuit_gate.get());
    circuit_gate->output_wires.push_back(wire.get());
    parameters.push_back(wire.get());
    // XXX: the wires are placed at the end, so it will be not compatible
    // with remove_last_gate().
    wires.push_back(std::move(wire));
  }
  gates.insert(gates.begin() + insert_position, std::move(circuit_gate));
  hash_value_valid_ = false;
  return true;
}

bool CircuitSeq::insert_gate(int insert_position, CircuitGate *gate) {
  std::vector<int> qubit_indices;
  std::vector<int> parameter_indices;
  int output_para_index;
  for (auto &wire : gate->input_wires) {
    if (wire->is_qubit()) {
      qubit_indices.push_back(wire->index);
    } else {
      parameter_indices.push_back(wire->index);
    }
  }
  return insert_gate(insert_position, qubit_indices, parameter_indices,
                     gate->gate, &output_para_index);
}

void CircuitSeq::add_input_parameter() {
  auto wire = std::make_unique<CircuitWire>();
  wire->type = CircuitWire::input_param;
  wire->index = num_input_parameters;
  parameters.insert(parameters.begin() + num_input_parameters, wire.get());
  wires.insert(wires.begin() + num_qubits + num_input_parameters,
               std::move(wire));

  num_input_parameters++;

  // Update internal parameters' indices
  for (auto &it : wires) {
    if (it->type == CircuitWire::internal_param) {
      it->index++;
    }
  }

  // This function should not modify the hash value.
}

bool CircuitSeq::remove_last_gate() {
  if (gates.empty()) {
    return false;
  }

  auto *circuit_gate = gates.back().get();
  auto *gate = circuit_gate->gate;
  // Remove gates from input wires.
  for (auto *input_wire : circuit_gate->input_wires) {
    assert(!input_wire->output_gates.empty());
    assert(input_wire->output_gates.back() == circuit_gate);
    input_wire->output_gates.pop_back();
  }

  if (gate->is_parameter_gate()) {
    // Remove the parameter.
    assert(!wires.empty());
    assert(wires.back()->type == CircuitWire::internal_param);
    assert(wires.back()->index == (int)parameters.size() - 1);
    parameters.pop_back();
    wires.pop_back();
  } else {
    assert(gate->is_quantum_gate());
    // Restore the outputs.
    for (auto *input_wire : circuit_gate->input_wires) {
      if (input_wire->is_qubit()) {
        outputs[input_wire->index] = input_wire;
      }
    }
    // Remove the qubit wires.
    while (!wires.empty() && !wires.back()->input_gates.empty() &&
           wires.back()->input_gates.back() == circuit_gate) {
      wires.pop_back();
    }
  }

  // Remove the circuit_gate.
  gates.pop_back();

  hash_value_valid_ = false;
  return true;
}

void CircuitSeq::generate_parameter_gates(Context *ctx,
                                          int max_recursion_depth) {
  assert(max_recursion_depth == 1);
  for (const auto &idx : ctx->get_supported_parameter_gates()) {
    Gate *gate = ctx->get_gate(idx);
    if (gate->get_num_parameters() == 1) {
      std::vector<int> param_indices(1);
      for (param_indices[0] = 0; param_indices[0] < get_num_input_parameters();
           param_indices[0]++) {
        int output_param_index;
        bool ret = add_gate({}, param_indices, gate, &output_param_index);
        assert(ret);
      }
    } else if (gate->get_num_parameters() == 2) {
      // Case: 0-qubit operators with 2 parameters
      std::vector<int> param_indices(2);
      for (param_indices[0] = 0; param_indices[0] < get_num_input_parameters();
           param_indices[0]++) {
        for (param_indices[1] = 0;
             param_indices[1] < get_num_input_parameters();
             param_indices[1]++) {
          if (gate->is_commutative() && param_indices[0] > param_indices[1]) {
            // For commutative gates, enforce param_indices[0]
            // <= param_indices[1]
            continue;
          }
          int output_param_index;
          bool ret = add_gate({}, param_indices, gate, &output_param_index);
          assert(ret);
        }
      }
    } else {
      assert(false && "Unsupported gate type");
    }
  }
}

int CircuitSeq::remove_gate(CircuitGate *circuit_gate) {
  auto gate_pos = std::find_if(
      gates.begin(), gates.end(),
      [&](std::unique_ptr<CircuitGate> &p) { return p.get() == circuit_gate; });
  if (gate_pos == gates.end()) {
    return 0;
  }

  auto *gate = circuit_gate->gate;
  // Remove gates from input wires.
  for (auto *input_wire : circuit_gate->input_wires) {
    assert(!input_wire->output_gates.empty());
    auto it = std::find(input_wire->output_gates.begin(),
                        input_wire->output_gates.end(), circuit_gate);
    assert(it != input_wire->output_gates.end());
    input_wire->output_gates.erase(it);
  }

  int ret = 1;

  if (gate->is_parameter_gate()) {
    // Remove the parameter.
    assert(circuit_gate->output_wires.size() == 1);
    auto wire = circuit_gate->output_wires[0];
    assert(wire->type == CircuitWire::internal_param);
    while (!wire->output_gates.empty()) {
      // Remove gates using the parameter at first.
      // Note: we can't use a for loop with iterators because they
      // will be invalidated.
      ret += remove_gate(wire->output_gates[0]);
    }
    auto it = std::find_if(
        wires.begin(), wires.end(),
        [&](std::unique_ptr<CircuitWire> &p) { return p.get() == wire; });
    assert(it != wires.end());
    auto idx = wire->index;
    assert(idx >= get_num_input_parameters());
    wires.erase(it);
    parameters.erase(parameters.begin() + idx);
    // Update the parameter indices.
    for (auto &j : wires) {
      if (j->is_parameter() && j->index > idx) {
        j->index--;
      }
    }
  } else {
    assert(gate->is_quantum_gate());
    int num_outputs = (int)circuit_gate->output_wires.size();
    int j = 0;
    for (int i = 0; i < num_outputs; i++, j++) {
      // Match the input qubits and the output qubits.
      while (j < (int)circuit_gate->input_wires.size() &&
             !circuit_gate->input_wires[j]->is_qubit()) {
        j++;
      }
      assert(j < (int)circuit_gate->input_wires.size());
      assert(circuit_gate->input_wires[j]->index ==
             circuit_gate->output_wires[i]->index);
      if (outputs[circuit_gate->output_wires[i]->index] ==
          circuit_gate->output_wires[i]) {
        // Restore the outputs.
        outputs[circuit_gate->output_wires[i]->index] =
            circuit_gate->input_wires[j];
      }
      if (circuit_gate->output_wires[i]->output_gates.empty()) {
        // Remove the qubit wires.
        auto it = std::find_if(
            wires.begin(), wires.end(), [&](std::unique_ptr<CircuitWire> &p) {
              return p.get() == circuit_gate->output_wires[i];
            });
        assert(it != wires.end());
        wires.erase(it);
      } else {
        // Merge the adjacent qubit wires.
        for (auto &e : circuit_gate->output_wires[i]->output_gates) {
          auto it = std::find(e->input_wires.begin(), e->input_wires.end(),
                              circuit_gate->output_wires[i]);
          assert(it != e->input_wires.end());
          *it = circuit_gate->input_wires[j];
          circuit_gate->input_wires[j]->output_gates.push_back(e);
        }
        // And then remove the disconnected qubit wire.
        auto it = std::find_if(
            wires.begin(), wires.end(), [&](std::unique_ptr<CircuitWire> &p) {
              return p.get() == circuit_gate->output_wires[i];
            });
        assert(it != wires.end());
        wires.erase(it);
      }
    }
  }

  // Remove the gate.
  gate_pos = std::find_if(
      gates.begin(), gates.end(),
      [&](std::unique_ptr<CircuitGate> &p) { return p.get() == circuit_gate; });
  assert(gate_pos != gates.end());
  gates.erase(gate_pos);

  hash_value_valid_ = false;
  return ret;
}

int CircuitSeq::remove_first_quantum_gate() {
  for (auto &circuit_gate : gates) {
    if (circuit_gate->gate->is_quantum_gate()) {
      return remove_gate(circuit_gate.get());
    }
  }
  return 0; // nothing removed
}

bool CircuitSeq::evaluate(const Vector &input_dis,
                          const std::vector<ParamType> &input_parameters,
                          Vector &output_dis,
                          std::vector<ParamType> *parameter_values) const {
  // We should have 2**n entries for the distribution
  if (input_dis.size() != (1 << get_num_qubits()))
    return false;
  if (input_parameters.size() != get_num_input_parameters())
    return false;
  assert(get_num_input_parameters() <= get_num_total_parameters());
  bool output_parameter_values = true;
  if (!parameter_values) {
    parameter_values = new std::vector<ParamType>();
    output_parameter_values = false;
  }
  *parameter_values = input_parameters;
  parameter_values->resize(get_num_total_parameters());

  output_dis = input_dis;

  // Assume the gates are already sorted in the topological order.
  const int num_gates = (int)gates.size();
  for (int i = 0; i < num_gates; i++) {
    std::vector<int> qubit_indices;
    std::vector<ParamType> params;
    for (const auto &input_wire : gates[i]->input_wires) {
      if (input_wire->is_qubit()) {
        qubit_indices.push_back(input_wire->index);
      } else {
        params.push_back((*parameter_values)[input_wire->index]);
      }
    }
    if (gates[i]->gate->is_parameter_gate()) {
      // A parameter gate. Compute the new parameter.
      assert(gates[i]->output_wires.size() == 1);
      const auto &output_wire = gates[i]->output_wires[0];
      (*parameter_values)[output_wire->index] = gates[i]->gate->compute(params);
    } else {
      // A quantum gate. Update the distribution.
      assert(gates[i]->gate->is_quantum_gate());
      auto *mat = gates[i]->gate->get_matrix(params);
      output_dis.apply_matrix(mat, qubit_indices);
    }
  }
  if (!output_parameter_values) {
    // Delete the temporary variable newed in this function.
    delete parameter_values;
  }
  return true;
}

int CircuitSeq::get_num_qubits() const { return num_qubits; }

int CircuitSeq::get_num_input_parameters() const {
  return num_input_parameters;
}

int CircuitSeq::get_num_total_parameters() const {
  return (int)parameters.size();
}

int CircuitSeq::get_num_internal_parameters() const {
  return (int)parameters.size() - num_input_parameters;
}

int CircuitSeq::get_num_gates() const { return (int)gates.size(); }

int CircuitSeq::get_circuit_depth() const {
  std::vector<int> depth(get_num_qubits(), 0);
  for (auto &circuit_gate : gates) {
    if (circuit_gate->gate->is_quantum_gate()) {
      int max_previous_depth = 0;
      for (auto &input_wire : circuit_gate->input_wires) {
        max_previous_depth =
            std::max(max_previous_depth, depth[input_wire->index]);
      }
      for (auto &input_wire : circuit_gate->input_wires) {
        depth[input_wire->index] = max_previous_depth + 1;
      }
    }
  }
  return *std::max_element(depth.begin(), depth.end());
}

ParamType CircuitSeq::get_parameter_value(Context *ctx, int para_idx) {
  return ctx->get_param_value(para_idx);
}

bool CircuitSeq::qubit_used(int qubit_index) const {
  return outputs[qubit_index] != wires[qubit_index].get();
}

bool CircuitSeq::input_param_used(int param_index) const {
  assert(wires[get_num_qubits() + param_index]->type ==
         CircuitWire::input_param);
  assert(wires[get_num_qubits() + param_index]->index == param_index);
  return !wires[get_num_qubits() + param_index]->output_gates.empty();
}

std::pair<InputParamMaskType, std::vector<InputParamMaskType>>
CircuitSeq::get_input_param_mask() const {
  std::vector<InputParamMaskType> param_mask(get_num_total_parameters());
  for (int i = 0; i < get_num_input_parameters(); i++) {
    param_mask[i] = 1 << i;
  }
  for (int i = get_num_input_parameters(); i < get_num_total_parameters();
       i++) {
    param_mask[i] = 0;
    assert(parameters[i]->input_gates.size() == 1);
    for (auto &input_wire : parameters[i]->input_gates[0]->input_wires) {
      param_mask[i] |= param_mask[input_wire->index];
    }
  }
  InputParamMaskType usage_mask{0};
  for (auto &circuit_gate : gates) {
    // Only consider quantum gate usages of parameters
    if (circuit_gate->gate->is_parametrized_gate()) {
      for (auto &input_wire : circuit_gate->input_wires) {
        if (input_wire->is_parameter()) {
          usage_mask |= param_mask[input_wire->index];
        }
      }
    }
  }
  return std::make_pair(usage_mask, param_mask);
}

void CircuitSeq::generate_hash_values(
    Context *ctx, const ComplexType &hash_value,
    const PhaseShiftIdType &phase_shift_id,
    const std::vector<ParamType> &param_values, CircuitSeqHashType *main_hash,
    std::vector<std::pair<CircuitSeqHashType, PhaseShiftIdType>> *other_hash) {
  if (kFingerprintInvariantUnderPhaseShift) {
#ifdef USE_ARBLIB
    auto val = hash_value.abs();
    auto max_error = hash_value.get_abs_max_error();
    assert(max_error < kCircuitSeqHashMaxError);
#else
    auto val = std::abs(hash_value);
#endif
    *main_hash = (CircuitSeqHashType)std::floor((long double)val /
                                                (2 * kCircuitSeqHashMaxError));
    // Besides rounding the hash value down, we might want to round it
    // up to account for floating point errors.
    other_hash->emplace_back(*main_hash + 1, phase_shift_id);
    return;
  }

  auto val = hash_value.real() * kCircuitSeqHashAlpha +
             hash_value.imag() * (1 - kCircuitSeqHashAlpha);
  *main_hash = (CircuitSeqHashType)std::floor((long double)val /
                                              (2 * kCircuitSeqHashMaxError));
  // Besides rounding the hash value down, we might want to round it up to
  // account for floating point errors.
  other_hash->emplace_back(*main_hash + 1, phase_shift_id);
}

CircuitSeqHashType CircuitSeq::hash(Context *ctx) {
  if (hash_value_valid_) {
    return hash_value_;
  }
  const Vector &input_dis = ctx->get_generated_input_dis(get_num_qubits());
  Vector output_dis;
  auto input_parameters =
      ctx->get_generated_parameters(get_num_input_parameters());
  std::vector<ParamType> all_parameters;
  evaluate(input_dis, input_parameters, output_dis, &all_parameters);
  ComplexType dot_product =
      output_dis.dot(ctx->get_generated_hashing_dis(get_num_qubits()));

  original_fingerprint_ = dot_product;

  other_hash_values_.clear();
  generate_hash_values(ctx, dot_product, kNoPhaseShift, all_parameters,
                       &hash_value_, &other_hash_values_);
  hash_value_valid_ = true;

  // Account for phase shifts.
  // If |kFingerprintInvariantUnderPhaseShift| is true,
  // this was already handled above in |generate_hash_values|.
  if (!kFingerprintInvariantUnderPhaseShift && kCheckPhaseShiftInGenerator) {
    // We try the simplest version first:
    // Apply phase shift for e^(ip) or e^(-ip) for p being a parameter
    // (either input or internal).
    CircuitSeqHashType tmp;
    assert(all_parameters.size() == get_num_total_parameters());
    const int num_total_params = get_num_total_parameters();
    for (int i = 0; i < num_total_params; i++) {
      const auto &param = all_parameters[i];
      ComplexType shifted =
          dot_product * ComplexType{std::cos(param), std::sin(param)};
      generate_hash_values(ctx, shifted, i, all_parameters, &tmp,
                           &other_hash_values_);
      other_hash_values_.emplace_back(tmp, i);
      shifted = dot_product * ComplexType{std::cos(param), -std::sin(param)};
      generate_hash_values(ctx, shifted, i + num_total_params, all_parameters,
                           &tmp, &other_hash_values_);
      other_hash_values_.emplace_back(tmp, i + num_total_params);
    }
    if (kCheckPhaseShiftOfPiOver4) {
      // Check phase shift of pi/4, 2pi/4, ..., 7pi/4.
      for (int i = 1; i < 8; i++) {
        const double pi = std::acos(-1.0);
        ComplexType shifted = dot_product * ComplexType{std::cos(pi / 4 * i),
                                                        std::sin(pi / 4 * i)};
        generate_hash_values(ctx, shifted, i, all_parameters, &tmp,
                             &other_hash_values_);
        other_hash_values_.emplace_back(tmp,
                                        kCheckPhaseShiftOfPiOver4Index + i);
      }
    }
  }
  return hash_value_;
}

std::vector<Vector> CircuitSeq::get_matrix(Context *ctx) const {
  const auto sz = 1 << get_num_qubits();
  Vector input_dis(sz);
  auto input_parameters =
      ctx->get_generated_parameters(get_num_input_parameters());
  std::vector<ParamType> all_parameters;
  std::vector<Vector> result(sz);
  for (int S = 0; S < sz; S++) {
    input_dis[S] = ComplexType(1);
    if (S > 0) {
      input_dis[S - 1] = ComplexType(0);
    }
    evaluate(input_dis, input_parameters, result[S], &all_parameters);
  }
  return result;
}

bool CircuitSeq::hash_value_valid() const { return hash_value_valid_; }

CircuitSeqHashType CircuitSeq::cached_hash_value() const {
  assert(hash_value_valid_);
  return hash_value_;
}

std::vector<CircuitSeqHashType> CircuitSeq::other_hash_values() const {
  assert(hash_value_valid_);
  std::vector<CircuitSeqHashType> result(other_hash_values_.size());
  for (int i = 0; i < (int)other_hash_values_.size(); i++) {
    result[i] = other_hash_values_[i].first;
  }
  return result;
}

std::vector<std::pair<CircuitSeqHashType, PhaseShiftIdType>>
CircuitSeq::other_hash_values_with_phase_shift_id() const {
  assert(hash_value_valid_);
  return other_hash_values_;
}

bool CircuitSeq::remove_unused_qubits(std::vector<int> unused_qubits) {
  if (unused_qubits.empty()) {
    return true;
  }
  std::sort(unused_qubits.begin(), unused_qubits.end(), std::greater<>());
  for (auto &id : unused_qubits) {
    if (id >= get_num_qubits()) {
      return false;
    }
    if (wires[id]->type != CircuitWire::input_qubit) {
      return false;
    }
    if (wires[id]->index != id) {
      return false;
    }
    if (!wires[id]->output_gates.empty()) {
      // used
      return false;
    }
    wires.erase(wires.begin() + id);
    outputs.erase(outputs.begin() + id);
    num_qubits--;
    for (auto &wire : wires) {
      if (wire->is_qubit() && wire->index > id) {
        wire->index--;
      }
    }
  }
  hash_value_valid_ = false;
  return true;
}

bool CircuitSeq::remove_unused_input_params(
    std::vector<int> unused_input_params) {
  if (unused_input_params.empty()) {
    return true;
  }
  std::sort(unused_input_params.begin(), unused_input_params.end(),
            std::greater<>());
  for (auto &id : unused_input_params) {
    if (id >= get_num_input_parameters()) {
      return false;
    }
    if (wires[get_num_qubits() + id]->type != CircuitWire::input_param) {
      return false;
    }
    if (wires[get_num_qubits() + id]->index != id) {
      return false;
    }
    if (!wires[get_num_qubits() + id]->output_gates.empty()) {
      // used
      return false;
    }
    wires.erase(wires.begin() + get_num_qubits() + id);
    parameters.erase(parameters.begin() + id);
    num_input_parameters--;
    for (auto &wire : wires) {
      if (wire->is_parameter() && wire->index > id) {
        wire->index--;
      }
    }
  }
  hash_value_valid_ = false;
  return true;
}

CircuitSeq &CircuitSeq::shrink_unused_input_parameters() {
  // Warning: the hash function should be designed such that this function
  // doesn't change the hash value.
  if (get_num_input_parameters() == 0) {
    return *this;
  }
  int last_unused_input_param_index = get_num_input_parameters();
  while (last_unused_input_param_index > 0 &&
         wires[get_num_qubits() + last_unused_input_param_index - 1]
             ->output_gates.empty()) {
    last_unused_input_param_index--;
  }
  if (last_unused_input_param_index == get_num_input_parameters()) {
    // no need to shrink
    return *this;
  }
  int num_parameters_shrinked =
      get_num_input_parameters() - last_unused_input_param_index;

  // Erase the parameters and the wires
  parameters.erase(parameters.begin() + last_unused_input_param_index,
                   parameters.begin() + get_num_input_parameters());
  wires.erase(wires.begin() + get_num_qubits() + last_unused_input_param_index,
              wires.begin() + get_num_qubits() + get_num_input_parameters());

  // Update the parameter indices
  for (auto &wire : wires) {
    if (wire->is_parameter() && wire->index >= get_num_input_parameters()) {
      // An internal parameter
      wire->index -= num_parameters_shrinked;
    }
  }

  // Update num_input_parameters
  num_input_parameters -= num_parameters_shrinked;
  return *this;
}

std::unique_ptr<CircuitSeq>
CircuitSeq::clone_and_shrink_unused_input_parameters() const {
  auto cloned_seq = std::make_unique<CircuitSeq>(*this);
  cloned_seq->shrink_unused_input_parameters();
  return cloned_seq;
}

bool CircuitSeq::has_unused_parameter() const {
  for (auto &wire : wires) {
    if (wire->is_parameter() && wire->output_gates.empty()) {
      return true;
    }
  }
  return false;
}

int CircuitSeq::remove_unused_internal_parameters() {
  int num_removed = 0;
  int gate_id = (int)gates.size() - 1;
  while (gate_id >= 0) {
    if (gates[gate_id]->gate->is_parameter_gate()) {
      assert(gates[gate_id]->output_wires.size() == 1);
      if (gates[gate_id]->output_wires[0]->output_gates.empty()) {
        num_removed += remove_gate(gates[gate_id].get());
      }
    }
    gate_id--;
  }
  return num_removed;
}

void CircuitSeq::print(Context *ctx) const {
  for (size_t i = 0; i < gates.size(); i++) {
    CircuitGate *circuit_gate = gates[i].get();
    printf("gate[%zu] type(%d)\n", i, circuit_gate->gate->tp);
    for (size_t j = 0; j < circuit_gate->input_wires.size(); j++) {
      CircuitWire *wire = circuit_gate->input_wires[j];
      if (wire->is_qubit()) {
        printf("    inputs[%zu]: qubit(%d)\n", j, wire->index);
      } else {
        printf("    inputs[%zu]: param(%d)\n", j, wire->index);
      }
    }
  }
}

std::string CircuitSeq::to_string(bool line_number) const {
  std::string result;
  result += "CircuitSeq {\n";
  const int num_gates = (int)gates.size();
  for (int i = 0; i < num_gates; i++) {
    if (line_number) {
      char buffer[20]; // enough to store any int
      int max_line_number_width =
          std::max(1, (int)std::ceil(std::log10(num_gates - 0.01)));
      sprintf(buffer, "%*d", max_line_number_width, i);
      result += std::string(buffer) + ": ";
    } else {
      result += "  ";
    }
    result += gates[i]->to_string();
    result += "\n";
  }
  result += "}\n";
  return result;
}

std::string CircuitSeq::to_json() const {
  std::string result;
  result += "[";

  // basic info
  result += "[";
  result += std::to_string(get_num_qubits());
  result += ",";
  result += std::to_string(get_num_input_parameters());
  result += ",";
  result += std::to_string(get_num_total_parameters());
  result += ",";
  result += std::to_string(get_num_gates());
  result += ",";

  result += "[";
  if (hash_value_valid_) {
    bool first_other_hash_value = true;
    for (const auto &val : other_hash_values_with_phase_shift_id()) {
      if (first_other_hash_value) {
        first_other_hash_value = false;
      } else {
        result += ",";
      }
      static char buffer[64];
      auto [ptr, ec] =
          std::to_chars(buffer, buffer + sizeof(buffer), val.first, /*base=*/
                        16);
      assert(ec == std::errc());
      auto hash_value = std::string(buffer, ptr);
      if (kCheckPhaseShiftInGenerator && val.second != kNoPhaseShift) {
        // hash value and phase shift id
        result += "[\"" + hash_value + "\"," + std::to_string(val.second) + "]";
      } else {
        // hash value only
        result += "\"" + hash_value + "\"";
      }
    }
  }
  result += "]";

  result += ",";
  result += "[";
  // std::to_chars for floating-point numbers is not supported by some
  // compilers, including GCC with version < 11.
  result += to_string_with_precision(original_fingerprint_.real(),
                                     /*precision=*/17);
  result += ",";
  result += to_string_with_precision(original_fingerprint_.imag(),
                                     /*precision=*/17);
  result += "]";

  result += "],";

  // gates
  const int num_gates = (int)gates.size();
  result += "[";
  for (int i = 0; i < num_gates; i++) {
    result += "[";
    result += "\"" + gate_type_name(gates[i]->gate->tp) + "\", ";
    if (gates[i]->output_wires.size() == 1) {
      result += "[\"" + gates[i]->output_wires[0]->to_string() + "\"],";
    } else if (gates[i]->output_wires.size() == 2) {
      result += "[\"" + gates[i]->output_wires[0]->to_string();
      result += "\", \"" + gates[i]->output_wires[1]->to_string();
      result += "\"],";
    } else {
      assert(false && "A circuit gate should have 1 or 2 outputs.");
    }
    result += "[";
    for (int j = 0; j < (int)gates[i]->input_wires.size(); j++) {
      result += "\"" + gates[i]->input_wires[j]->to_string() + "\"";
      if (j != (int)gates[i]->input_wires.size() - 1) {
        result += ", ";
      }
    }
    result += "]]";
    if (i + 1 != num_gates)
      result += ",";
  }
  result += "]";

  result += "]\n";
  return result;
}

std::unique_ptr<CircuitSeq> CircuitSeq::read_json(Context *ctx,
                                                  std::istream &fin) {
  fin.ignore(std::numeric_limits<std::streamsize>::max(), '[');

  // basic info
  int num_qubits, num_input_params, num_total_params, num_gates;
  fin.ignore(std::numeric_limits<std::streamsize>::max(), '[');
  fin >> num_qubits;
  fin.ignore(std::numeric_limits<std::streamsize>::max(), ',');
  fin >> num_input_params;
  fin.ignore(std::numeric_limits<std::streamsize>::max(), ',');
  fin >> num_total_params;
  fin.ignore(std::numeric_limits<std::streamsize>::max(), ',');
  fin >> num_gates;

  // TODO: Do not generate the distribution here -- we should generate
  //  earlier to make the result more deterministic.
  ctx->get_and_gen_hashing_dis(num_qubits);
  ctx->get_and_gen_input_dis(num_qubits);
  ctx->get_and_gen_parameters(num_input_params);

  // ignore other hash values
  fin.ignore(std::numeric_limits<std::streamsize>::max(), '[');
  while (true) {
    char ch;
    fin.get(ch);
    while (ch != '[' && ch != ']') {
      fin.get(ch);
    }
    if (ch == '[') {
      // A hash value with a phase shift id.
      fin.ignore(std::numeric_limits<std::streamsize>::max(), ']');
    } else {
      // ch == ']'
      break;
    }
  }

  fin.ignore(std::numeric_limits<std::streamsize>::max(), ']');
  fin.ignore(std::numeric_limits<std::streamsize>::max(), ',');

  auto result = std::make_unique<CircuitSeq>(num_qubits, num_input_params);

  // gates
  fin.ignore(std::numeric_limits<std::streamsize>::max(), '[');
  while (true) {
    char ch;
    fin.get(ch);
    while (ch != '[' && ch != ']') {
      fin.get(ch);
    }
    if (ch == ']') {
      break;
    }

    // New gate
    fin.ignore(std::numeric_limits<std::streamsize>::max(), '\"');
    std::string name;
    std::getline(fin, name, '\"');
    auto gate_type = to_gate_type(name);
    Gate *gate = ctx->get_gate(gate_type);

    std::vector<int> input_qubits, input_params, output_qubits, output_params;
    auto read_indices = [&](std::vector<int> &qubit_indices,
                            std::vector<int> &param_indices) {
      fin.ignore(std::numeric_limits<std::streamsize>::max(), '[');
      while (true) {
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
        fin.ignore(); // '\"'
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

    int output_param_index;
    result->add_gate(input_qubits, input_params, gate, &output_param_index);
    if (gate->is_parameter_gate()) {
      assert(output_param_index == output_params[0]);
    }
  }

  fin.ignore(std::numeric_limits<std::streamsize>::max(), ']');

  return result;
}

std::unique_ptr<CircuitSeq>
CircuitSeq::from_qasm_file(Context *ctx, const std::string &filename) {
  QASMParser parser(ctx);
  CircuitSeq *seq = nullptr;
  parser.load_qasm(filename, seq);
  std::unique_ptr<CircuitSeq> ret;
  ret.reset(seq); // transfer ownership of |seq|
  return ret;
}

bool CircuitSeq::to_qasm_file(Context *ctx, const std::string &filename,
                              int param_precision) const {
  std::ofstream fout(filename);
  if (!fout.is_open()) {
    return false;
  }
  fout << "OPENQASM 2.0;\n"
          "include \"qelib1.inc\";\n"
          "qreg q["
       << get_num_qubits() << "];\n";
  for (auto &gate : gates) {
    fout << gate->to_qasm_style_string(ctx, param_precision);
  }
  fout.close();
  return true;
}

bool CircuitSeq::canonical_representation(
    std::unique_ptr<CircuitSeq> *output_seq, bool output) const {
  if (output) {
    // |output_seq| cannot be nullptr but its content can (and should)
    // be nullptr.
    assert(output_seq);
    // This deletes the content |output_seq| previously stored.
    *output_seq = std::make_unique<CircuitSeq>(get_num_qubits(),
                                               get_num_input_parameters());
  }

  bool this_is_canonical_representation = true;
  // map this CircuitSeq to the canonical representation
  int num_mapped_circuit_gates = 0;

  // Check if all parameter gates are at the beginning.
  bool have_quantum_gates = false;
  for (auto &circuit_gate : gates) {
    if (circuit_gate->gate->is_parameter_gate()) {
      if (have_quantum_gates) {
        this_is_canonical_representation = false;
        if (!output) {
          // no side effects, early return
          return false;
        }
      }
      num_mapped_circuit_gates++;
      if (output) {
        int output_param_index;
        std::vector<int> param_indices;
        for (auto &input_wire : circuit_gate->input_wires) {
          assert(input_wire->is_parameter());
          param_indices.push_back(input_wire->index);
        }
        (*output_seq)
            ->add_gate({}, param_indices, circuit_gate->gate,
                       &output_param_index);
      }
    } else {
      have_quantum_gates = true;
    }
  }

  std::unordered_map<CircuitWire *, int> wire_id;
  std::unordered_map<CircuitGate *, int> gate_id;
  for (int i = 0; i < (int)wires.size(); i++) {
    wire_id[wires[i].get()] = i;
  }
  for (int i = 0; i < (int)gates.size(); i++) {
    gate_id[gates[i].get()] = i;
  }

  class CompareByMinQubitIndex {
  public:
    bool operator()(CircuitGate *gate1, CircuitGate *gate2) const {
      // std::priority_queue maintains the largest element,
      // so we use ">" to get the gate with minimum qubit index.
      return gate1->get_min_qubit_index() > gate2->get_min_qubit_index();
    }
  };

  std::priority_queue<CircuitGate *, std::vector<CircuitGate *>,
                      CompareByMinQubitIndex>
      free_gates;

  std::vector<CircuitWire *> free_wires;
  free_wires.reserve(get_num_qubits() + get_num_input_parameters());

  // Construct the |free_wires| vector with the input qubit wires.
  std::vector<int> wire_in_degree(wires.size(), 0);
  std::vector<int> gate_in_degree(gates.size(), 0);
  for (auto &wire : wires) {
    if (wire->is_parameter()) {
      wire_in_degree[wire_id[wire.get()]] = -1;
      continue;
    }
    wire_in_degree[wire_id[wire.get()]] = (int)wire->input_gates.size();
    if (!wire_in_degree[wire_id[wire.get()]]) {
      free_wires.push_back(wire.get());
    }
  }
  for (auto &circuit_gate : gates) {
    gate_in_degree[gate_id[circuit_gate.get()]] = 0;
    for (auto &input_wire : circuit_gate->input_wires) {
      if (input_wire->is_qubit()) {
        gate_in_degree[gate_id[circuit_gate.get()]]++;
      }
    }
  }

  while (!free_wires.empty() || !free_gates.empty()) {
    // Remove the wires in |free_wires|.
    for (auto &wire : free_wires) {
      for (auto &output_gate : wire->output_gates) {
        if (!--gate_in_degree[gate_id[output_gate]]) {
          free_gates.push(output_gate);
        }
      }
    }
    free_wires.clear();

    if (!free_gates.empty()) {
      // Find the smallest free circuit_gate (gate).
      CircuitGate *smallest_free_gate = free_gates.top();
      free_gates.pop();

      // Map |smallest_free_gate| (a quantum gate).
      if (gates[num_mapped_circuit_gates].get() != smallest_free_gate) {
        this_is_canonical_representation = false;
        if (!output) {
          // no side effects, early return
          return false;
        }
      }
      num_mapped_circuit_gates++;
      if (output) {
        std::vector<int> qubit_indices, param_indices;
        for (auto &input_wire : smallest_free_gate->input_wires) {
          if (input_wire->is_qubit()) {
            qubit_indices.push_back(input_wire->index);
          } else {
            param_indices.push_back(input_wire->index);
          }
        }
        int output_param_index;
        (*output_seq)
            ->add_gate(qubit_indices, param_indices, smallest_free_gate->gate,
                       &output_param_index);
        if (smallest_free_gate->gate->is_parameter_gate()) {
          assert(smallest_free_gate->output_wires[0]->index ==
                 output_param_index);
        }
      }

      // Update the free wires.
      for (auto &output_wire : smallest_free_gate->output_wires) {
        if (!--wire_in_degree[wire_id[output_wire]]) {
          free_wires.push_back(output_wire);
        }
      }
    }
  }

  // The CircuitSeq should have all gates mapped.
  assert(num_mapped_circuit_gates == get_num_gates());

  return this_is_canonical_representation;
}

bool CircuitSeq::is_canonical_representation() const {
  return canonical_representation(nullptr, false);
}

bool CircuitSeq::to_canonical_representation() {
  std::unique_ptr<CircuitSeq> output_seq;
  if (!canonical_representation(&output_seq, true)) {
    clone_from(*output_seq, {}, {});
    return true;
  }
  return false;
}

[[nodiscard]] std::unique_ptr<CircuitSeq>
CircuitSeq::random_gate_permutation(size_t seed,
                                    int *result_permutation) const {
  auto output_seq = std::make_unique<CircuitSeq>(get_num_qubits(),
                                                 get_num_input_parameters());
  std::mt19937 rng(seed);
  // Add all parameter "gates" first.
  for (auto &circuit_gate : gates) {
    if (circuit_gate->gate->is_parameter_gate()) {
      output_seq->add_gate(circuit_gate.get());
    }
  }

  int num_mapped_circuit_gates = output_seq->get_num_gates();

  std::unordered_map<CircuitWire *, int> wire_id;
  std::unordered_map<CircuitGate *, int> gate_id;
  for (int i = 0; i < (int)wires.size(); i++) {
    wire_id[wires[i].get()] = i;
  }
  for (int i = 0; i < (int)gates.size(); i++) {
    gate_id[gates[i].get()] = i;
  }

  std::vector<CircuitGate *> free_gates;
  std::vector<CircuitWire *> free_wires;
  free_wires.reserve(get_num_qubits() + get_num_input_parameters());

  // Construct the |free_wires| vector with the input qubit wires.
  std::vector<int> wire_in_degree(wires.size(), 0);
  std::vector<int> gate_in_degree(gates.size(), 0);
  for (auto &wire : wires) {
    if (wire->is_parameter()) {
      wire_in_degree[wire_id[wire.get()]] = -1;
      continue;
    }
    wire_in_degree[wire_id[wire.get()]] = (int)wire->input_gates.size();
    if (!wire_in_degree[wire_id[wire.get()]]) {
      free_wires.push_back(wire.get());
    }
  }
  for (auto &circuit_gate : gates) {
    gate_in_degree[gate_id[circuit_gate.get()]] = 0;
    for (auto &input_wire : circuit_gate->input_wires) {
      if (input_wire->is_qubit()) {
        gate_in_degree[gate_id[circuit_gate.get()]]++;
      }
    }
  }

  while (!free_wires.empty() || !free_gates.empty()) {
    // Remove the wires in |free_wires|.
    for (auto &wire : free_wires) {
      for (auto &output_gate : wire->output_gates) {
        if (!--gate_in_degree[gate_id[output_gate]]) {
          free_gates.push_back(output_gate);
        }
      }
    }
    free_wires.clear();

    if (!free_gates.empty()) {
      // Find a random free circuit_gate (gate)
      // in [0, (int)free_gates.size() - 1].
      int random_loc = std::uniform_int_distribution<int>(
          0, (int)free_gates.size() - 1)(rng);
      CircuitGate *free_gate = free_gates[random_loc];
      if (result_permutation != nullptr) {
        // Record the permutation.
        result_permutation[gate_id[free_gate]] = num_mapped_circuit_gates;
      }
      num_mapped_circuit_gates++;
      free_gates.erase(free_gates.begin() + random_loc);

      output_seq->add_gate(free_gate);

      // Update the free wires.
      for (auto &output_wire : free_gate->output_wires) {
        if (!--wire_in_degree[wire_id[output_wire]]) {
          free_wires.push_back(output_wire);
        }
      }
    }
  }
  assert(num_mapped_circuit_gates == get_num_gates());
  return output_seq;
}

std::unique_ptr<CircuitSeq>
CircuitSeq::get_permuted_seq(const std::vector<int> &qubit_permutation,
                             const std::vector<int> &param_permutation) const {
  auto result = std::make_unique<CircuitSeq>(0, 0);
  result->clone_from(*this, qubit_permutation, param_permutation);
  return result;
}

void CircuitSeq::clone_from(const CircuitSeq &other,
                            const std::vector<int> &qubit_permutation,
                            const std::vector<int> &param_permutation) {
  num_qubits = other.num_qubits;
  num_input_parameters = other.num_input_parameters;
  original_fingerprint_ = other.original_fingerprint_;
  std::unordered_map<CircuitWire *, CircuitWire *> wires_mapping;
  std::unordered_map<CircuitGate *, CircuitGate *> gates_mapping;
  wires.clear();
  wires.reserve(other.wires.size());
  gates.clear();
  gates.reserve(other.gates.size());
  outputs.clear();
  outputs.reserve(other.outputs.size());
  parameters.clear();
  parameters.reserve(other.parameters.size());
  if (qubit_permutation.empty() && param_permutation.empty()) {
    // A simple clone.
    hash_value_ = other.hash_value_;
    other_hash_values_ = other.other_hash_values_;
    hash_value_valid_ = other.hash_value_valid_;
    for (int i = 0; i < (int)other.wires.size(); i++) {
      wires.emplace_back(std::make_unique<CircuitWire>(*(other.wires[i])));
      assert(wires[i].get() !=
             other.wires[i].get()); // make sure we make a copy
      wires_mapping[other.wires[i].get()] = wires[i].get();
    }
  } else {
    // We need to invalidate the hash value.
    hash_value_valid_ = false;
    assert(qubit_permutation.size() == num_qubits);
    wires.resize(other.wires.size());
    for (int i = 0; i < num_qubits; i++) {
      assert(qubit_permutation[i] >= 0 && qubit_permutation[i] < num_qubits);
      wires[qubit_permutation[i]] =
          std::make_unique<CircuitWire>(*(other.wires[i]));
      wires[qubit_permutation[i]]->index = qubit_permutation[i]; // update index
      assert(wires[qubit_permutation[i]].get() != other.wires[i].get());
      wires_mapping[other.wires[i].get()] = wires[qubit_permutation[i]].get();
    }
    const int num_permuted_parameters =
        std::min(num_input_parameters, (int)param_permutation.size());
    for (int i_inc = 0; i_inc < num_permuted_parameters; i_inc++) {
      assert(param_permutation[i_inc] >= 0 &&
             param_permutation[i_inc] < num_input_parameters);
      const int i = num_qubits + i_inc;
      wires[num_qubits + param_permutation[i_inc]] =
          std::make_unique<CircuitWire>(*(other.wires[i]));
      wires[num_qubits + param_permutation[i_inc]]->index =
          param_permutation[i_inc]; // update index
      assert(wires[num_qubits + param_permutation[i_inc]].get() !=
             other.wires[i].get());
      wires_mapping[other.wires[i].get()] =
          wires[num_qubits + param_permutation[i_inc]].get();
    }
    for (int i = num_qubits + num_permuted_parameters;
         i < (int)other.wires.size(); i++) {
      wires[i] = std::make_unique<CircuitWire>(*(other.wires[i]));
      if (wires[i]->is_qubit()) {
        wires[i]->index = qubit_permutation[wires[i]->index]; // update index
      }
      assert(wires[i].get() != other.wires[i].get());
      wires_mapping[other.wires[i].get()] = wires[i].get();
    }
  }
  for (int i = 0; i < (int)other.gates.size(); i++) {
    gates.emplace_back(std::make_unique<CircuitGate>(*(other.gates[i])));
    assert(gates[i].get() != other.gates[i].get());
    gates_mapping[other.gates[i].get()] = gates[i].get();
  }
  for (auto &wire : wires) {
    for (auto &circuit_gate : wire->input_gates) {
      circuit_gate = gates_mapping[circuit_gate];
    }
    for (auto &circuit_gate : wire->output_gates) {
      circuit_gate = gates_mapping[circuit_gate];
    }
  }
  for (auto &circuit_gate : gates) {
    for (auto &wire : circuit_gate->input_wires) {
      wire = wires_mapping[wire];
    }
    for (auto &wire : circuit_gate->output_wires) {
      wire = wires_mapping[wire];
    }
  }
  for (auto &wire : other.outputs) {
    outputs.emplace_back(wires_mapping[wire]);
  }
  for (auto &wire : other.parameters) {
    parameters.emplace_back(wires_mapping[wire]);
  }
}

std::vector<CircuitGate *> CircuitSeq::first_quantum_gates() const {
  std::vector<CircuitGate *> result;
  std::unordered_set<CircuitGate *> depend_on_other_gates;
  depend_on_other_gates.reserve(gates.size());
  for (const auto &circuit_gate : gates) {
    if (circuit_gate->gate->is_parameter_gate()) {
      continue;
    }
    if (depend_on_other_gates.find(circuit_gate.get()) ==
        depend_on_other_gates.end()) {
      result.push_back(circuit_gate.get());
    }
    for (const auto &output_wire : circuit_gate->output_wires) {
      for (const auto &output_gate : output_wire->output_gates) {
        depend_on_other_gates.insert(output_gate);
      }
    }
  }
  return result;
}

std::vector<CircuitGate *> CircuitSeq::last_quantum_gates() const {
  std::vector<CircuitGate *> result;
  for (const auto &circuit_gate : gates) {
    if (circuit_gate->gate->is_parameter_gate()) {
      continue;
    }
    bool all_output = true;
    for (const auto &output_wire : circuit_gate->output_wires) {
      if (outputs[output_wire->index] != output_wire) {
        all_output = false;
        break;
      }
    }
    if (all_output) {
      result.push_back(circuit_gate.get());
    }
  }
  return result;
}

bool CircuitSeq::same_gate(const CircuitSeq &seq1, int index1,
                           const CircuitSeq &seq2, int index2) {
  assert(seq1.get_num_gates() > index1);
  assert(seq2.get_num_gates() > index2);
  return same_gate(seq1.gates[index1].get(), seq2.gates[index2].get());
}

bool CircuitSeq::same_gate(CircuitGate *gate1, CircuitGate *gate2) {
  if (gate1->gate != gate2->gate) {
    return false;
  }
  if (gate1->input_wires.size() != gate2->input_wires.size()) {
    return false;
  }
  if (gate1->output_wires.size() != gate2->output_wires.size()) {
    return false;
  }
  for (int i = 0; i < (int)gate1->output_wires.size(); i++) {
    if (gate1->output_wires[i]->type != gate2->output_wires[i]->type) {
      return false;
    }
    if (gate1->output_wires[i]->index != gate2->output_wires[i]->index &&
        gate1->output_wires[i]->type != CircuitWire::internal_param) {
      return false;
    }
  }
  for (int i = 0; i < (int)gate1->input_wires.size(); i++) {
    if (gate1->input_wires[i]->type != gate2->input_wires[i]->type) {
      return false;
    }
    if (gate1->input_wires[i]->index != gate2->input_wires[i]->index &&
        gate1->input_wires[i]->type != CircuitWire::internal_param) {
      return false;
    }
    if (gate1->input_wires[i]->type == CircuitWire::internal_param) {
      // Internal parameters are checked recursively.
      assert(gate1->input_wires[i]->input_gates.size() == 1);
      assert(gate2->input_wires[i]->input_gates.size() == 1);
      if (!same_gate(gate1->input_wires[i]->input_gates[0],
                     gate2->input_wires[i]->input_gates[0])) {
        return false;
      }
    }
  }
  return true;
}

} // namespace quartz
