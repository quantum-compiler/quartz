#include "circuitseq.h"

#include "quartz/context/context.h"
#include "quartz/gate/gate.h"
#include "quartz/parser/qasm_parser.h"
#include "quartz/utils/string_utils.h"

#include <algorithm>
#include <cassert>
#include <charconv>
#include <fstream>
#include <optional>
#include <queue>
#include <unordered_set>
#include <utility>

namespace quartz {
CircuitSeq::CircuitSeq(int num_qubits)
    : num_qubits(num_qubits), hash_value_(0), hash_value_valid_(false) {
  wires.reserve(num_qubits);
  outputs.reserve(num_qubits);
  // Initialize num_qubits qubits
  for (int i = 0; i < num_qubits; i++) {
    auto wire = std::make_unique<CircuitWire>();
    wire->type = CircuitWire::input_qubit;
    wire->index = i;
    outputs.push_back(wire.get());
    wires.push_back(std::move(wire));
  }
}

CircuitSeq::CircuitSeq(const CircuitSeq &other) {
  clone_from(other, {}, {}, nullptr);
}

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
  if (num_qubits != other.num_qubits) {
    return false;
  }
  if (wires.size() != other.wires.size() ||
      gates.size() != other.gates.size()) {
    return false;
  }
  std::unordered_map<CircuitWire *, CircuitWire *> wires_mapping;
  for (int i = 0; i < (int)wires.size(); i++) {
    wires_mapping[wires[i].get()] = other.wires[i].get();
  }
  for (int i = 0; i < (int)gates.size(); i++) {
    if (!CircuitGate::equivalent(gates[i].get(), other.gates[i].get(),
                                 wires_mapping,
                                 /*update_mapping=*/false, nullptr)) {
      return false;
    }
  }
  return true;
}

bool CircuitSeq::topologically_equivalent(const CircuitSeq &other) const {
  if (this == &other) {
    return true;
  }
  if (num_qubits != other.num_qubits) {
    return false;
  }
  if (wires.size() != other.wires.size() ||
      gates.size() != other.gates.size()) {
    return false;
  }
  // Mapping from this circuit to the other circuit
  std::unordered_map<CircuitWire *, CircuitWire *> wires_mapping;
  std::queue<CircuitWire *> wires_to_search;
  std::unordered_map<CircuitGate *, int> gate_remaining_in_degree;
  for (int i = 0; i < num_qubits; i++) {
    wires_mapping[wires[i].get()] = other.wires[i].get();
    wires_to_search.push(wires[i].get());
  }
  // Topological sort on this circuit
  while (!wires_to_search.empty()) {
    auto this_wire = wires_to_search.front();
    auto other_wire = wires_mapping[this_wire];
    assert(other_wire);
    wires_to_search.pop();
    if (this_wire->output_gates.size() != other_wire->output_gates.size()) {
      return false;
    }
    for (int i = 0; i < (int)this_wire->output_gates.size(); i++) {
      auto this_gate = this_wire->output_gates[i];
      if (gate_remaining_in_degree.count(this_gate) == 0) {
        // A new gate
        gate_remaining_in_degree[this_gate] = this_gate->gate->get_num_qubits();
      }
      if (!--gate_remaining_in_degree[this_gate]) {
        // Check if this gate is the same as the other gate
        auto other_gate = other_wire->output_gates[i];
        if (!CircuitGate::equivalent(this_gate, other_gate, wires_mapping,
                                     /*update_mapping=*/true,
                                     &wires_to_search)) {
          return false;
        }
      }
    }
  }
  return true;  // equivalent
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
  if (kUseRowRepresentationToCompare) {
    for (int i = 0; i < num_qubits; i++) {
      // Compare all gates on qubit i.
      auto this_ptr = wires[i].get();
      auto other_ptr = other.wires[i].get();
      std::optional<bool> compare_outcome = std::nullopt;
      while (this_ptr != outputs[i]) {
        if (other_ptr == other.outputs[i]) {
          // This circuit sequence has more gates on qubit i,
          // so this circuit is greater.
          return false;
        }
        assert(this_ptr->output_gates.size() == 1);
        assert(other_ptr->output_gates.size() == 1);
        auto this_gate = this_ptr->output_gates[0];
        auto other_gate = other_ptr->output_gates[0];
        if (!compare_outcome.has_value()) {
          if (this_gate->gate->tp != other_gate->gate->tp) {
            compare_outcome = this_gate->gate->tp < other_gate->gate->tp;
          } else {
            assert(this_gate->input_wires.size() ==
                   other_gate->input_wires.size());
            assert(this_gate->output_wires.size() ==
                   other_gate->output_wires.size());
            for (int j = 0; j < (int)this_gate->input_wires.size(); j++) {
              if (this_gate->input_wires[j]->is_qubit() !=
                  other_gate->input_wires[j]->is_qubit()) {
                compare_outcome = this_gate->input_wires[j]->is_qubit();
                break;
              }
              if (this_gate->input_wires[j]->index !=
                  other_gate->input_wires[j]->index) {
                compare_outcome = this_gate->input_wires[j]->index <
                                  other_gate->input_wires[j]->index;
                break;
              }
            }
          }
        }
        // No need to compare output wires for quantum gates.
        bool found_output_wire = false;
        for (auto &output_wire : this_gate->output_wires) {
          if (output_wire->index == i) {
            found_output_wire = true;
            this_ptr = output_wire;
            break;
          }
        }
        assert(found_output_wire);
        found_output_wire = false;
        for (auto &output_wire : other_gate->output_wires) {
          if (output_wire->index == i) {
            found_output_wire = true;
            other_ptr = output_wire;
            break;
          }
        }
        assert(found_output_wire);
      }
      if (other_ptr != other.outputs[i]) {
        // The other circuit sequence has more gates on qubit i,
        // so this circuit is less.
        return true;
      }
      // Two circuit sequences have the same number of gates on qubit i.
      // Compare the contents.
      if (compare_outcome.has_value()) {
        return compare_outcome.value();
      }
    }
  } else {
    for (int i = 0; i < (int)gates.size(); i++) {
      if (gates[i]->gate->tp != other.gates[i]->gate->tp) {
        return gates[i]->gate->tp < other.gates[i]->gate->tp;
      }
      assert(gates[i]->input_wires.size() ==
             other.gates[i]->input_wires.size());
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
  }
  return false;  // fully equivalent
}

bool CircuitSeq::add_gate(const std::vector<int> &qubit_indices,
                          const std::vector<int> &parameter_indices, Gate *gate,
                          const Context *ctx) {
  if (!gate->is_quantum_gate())
    return false;
  if (gate->get_num_qubits() != qubit_indices.size())
    return false;
  if (gate->get_num_parameters() != parameter_indices.size())
    return false;
  // qubit indices must stay in range
  for (auto qubit_idx : qubit_indices)
    if ((qubit_idx < 0) || (qubit_idx >= get_num_qubits()))
      return false;
  auto circuit_gate = std::make_unique<CircuitGate>();
  circuit_gate->gate = gate;
  for (auto qubit_idx : qubit_indices) {
    circuit_gate->input_wires.push_back(outputs[qubit_idx]);
    outputs[qubit_idx]->output_gates.push_back(circuit_gate.get());
  }
  for (auto para_idx : parameter_indices) {
    auto param_wire = ctx->get_param_wire(para_idx);
    // parameter indices must stay in range
    if (param_wire == nullptr) {
      return false;
    }
    circuit_gate->input_wires.push_back(param_wire);
  }
  for (auto qubit_idx : qubit_indices) {
    auto wire = std::make_unique<CircuitWire>();
    wire->type = CircuitWire::internal_qubit;
    wire->index = qubit_idx;
    wire->input_gates.push_back(circuit_gate.get());
    circuit_gate->output_wires.push_back(wire.get());
    outputs[qubit_idx] = wire.get();  // Update outputs
    wires.push_back(std::move(wire));
  }
  gates.push_back(std::move(circuit_gate));
  hash_value_valid_ = false;
  return true;
}

bool CircuitSeq::add_gate(CircuitGate *gate, const Context *ctx) {
  std::vector<int> qubit_indices;
  std::vector<int> parameter_indices;
  for (auto &wire : gate->input_wires) {
    if (wire->is_qubit()) {
      qubit_indices.push_back(wire->index);
    } else {
      parameter_indices.push_back(wire->index);
    }
  }
  return add_gate(qubit_indices, parameter_indices, gate->gate, ctx);
}

bool CircuitSeq::insert_gate(int insert_position,
                             const std::vector<int> &qubit_indices,
                             const std::vector<int> &parameter_indices,
                             Gate *gate, const Context *ctx) {
  if (!gate->is_quantum_gate())
    return false;
  if (insert_position < 0 || insert_position > (int)gates.size())
    return false;
  if (gate->get_num_qubits() != qubit_indices.size())
    return false;
  if (gate->get_num_parameters() != parameter_indices.size())
    return false;
  // qubit indices must stay in range
  for (auto qubit_idx : qubit_indices)
    if ((qubit_idx < 0) || (qubit_idx >= get_num_qubits()))
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
    auto wire = std::make_unique<CircuitWire>();
    wire->type = CircuitWire::internal_qubit;
    wire->index = qubit_idx;
    wire->input_gates.push_back(circuit_gate.get());
    circuit_gate->output_wires.push_back(wire.get());
    if (outputs[qubit_idx] == previous_wires[qubit_idx]) {
      outputs[qubit_idx] = wire.get();  // Update outputs
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
    previous_wires[qubit_idx]->output_gates.push_back(circuit_gate.get());
  }
  for (auto para_idx : parameter_indices) {
    auto param_wire = ctx->get_param_wire(para_idx);
    // parameter indices must stay in range
    if (param_wire == nullptr) {
      return false;
    }
    circuit_gate->input_wires.push_back(param_wire);
  }
  gates.insert(gates.begin() + insert_position, std::move(circuit_gate));
  hash_value_valid_ = false;
  return true;
}

bool CircuitSeq::insert_gate(int insert_position, CircuitGate *gate,
                             const Context *ctx) {
  std::vector<int> qubit_indices;
  std::vector<int> parameter_indices;
  for (auto &wire : gate->input_wires) {
    if (wire->is_qubit()) {
      qubit_indices.push_back(wire->index);
    } else {
      parameter_indices.push_back(wire->index);
    }
  }
  return insert_gate(insert_position, qubit_indices, parameter_indices,
                     gate->gate, ctx);
}

bool CircuitSeq::remove_last_gate() {
  if (gates.empty()) {
    return false;
  }

  auto *circuit_gate = gates.back().get();
  auto *gate = circuit_gate->gate;
  // Remove gates from input wires.
  for (auto *input_wire : circuit_gate->input_wires) {
    if (input_wire->is_qubit()) {
      assert(!input_wire->output_gates.empty());
      assert(input_wire->output_gates.back() == circuit_gate);
      input_wire->output_gates.pop_back();
    }
  }

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

  // Remove the circuit_gate.
  gates.pop_back();

  hash_value_valid_ = false;
  return true;
}

bool CircuitSeq::remove_gate(int gate_position) {
  if (gate_position < 0 || gate_position >= (int)gates.size()) {
    return false;
  }
  CircuitGate *circuit_gate = gates[gate_position].get();
  auto *gate = circuit_gate->gate;
  assert(gate->is_quantum_gate());
  remove_quantum_gate_from_graph(circuit_gate);
  // Remove the gate.
  gates.erase(gates.begin() + gate_position);
  hash_value_valid_ = false;
  return true;
}

bool CircuitSeq::remove_gate(CircuitGate *circuit_gate) {
  auto gate_pos = std::find_if(
      gates.begin(), gates.end(),
      [&](std::unique_ptr<CircuitGate> &p) { return p.get() == circuit_gate; });
  if (gate_pos == gates.end()) {
    return false;
  }
  auto *gate = circuit_gate->gate;
  assert(gate->is_quantum_gate());
  remove_quantum_gate_from_graph(circuit_gate);
  // Remove the gate.
  gates.erase(gate_pos);
  hash_value_valid_ = false;
  return true;
}

bool CircuitSeq::remove_gate_near_end(CircuitGate *circuit_gate) {
  auto gate_pos = std::find_if(
      gates.rbegin(), gates.rend(),
      [&](std::unique_ptr<CircuitGate> &p) { return p.get() == circuit_gate; });
  if (gate_pos == gates.rend()) {
    return false;
  }
  return remove_gate((int)(gates.rend() - gate_pos) - 1);
}

bool CircuitSeq::remove_first_quantum_gate() {
  for (auto &circuit_gate : gates) {
    if (circuit_gate->gate->is_quantum_gate()) {
      return remove_gate(circuit_gate.get());
    }
  }
  return false;  // nothing removed
}

int CircuitSeq::remove_swap_gates() {
  std::vector<int> qubit_permutation(num_qubits);
  for (int i = 0; i < num_qubits; i++) {
    qubit_permutation[i] = i;
  }
  std::vector<CircuitGate *> to_remove;
  auto output_wires_to_be_removed =
      std::make_unique<std::unordered_set<CircuitWire *>>();
  for (auto &circuit_gate : gates) {
    if (circuit_gate->gate->tp == GateType::swap) {
      to_remove.push_back(circuit_gate.get());
      assert((int)circuit_gate->output_wires.size() == 2);
      // apply the logical swap gate
      std::swap(qubit_permutation[circuit_gate->output_wires[0]->index],
                qubit_permutation[circuit_gate->output_wires[1]->index]);
      // maintain the graph topology
      std::swap(circuit_gate->output_wires[0], circuit_gate->output_wires[1]);

      remove_quantum_gate_from_graph(
          circuit_gate.get(),
          /*assert_no_logical_qubit_permutation=*/false,
          output_wires_to_be_removed.get());
      // the output wires are going to be removed,
      // so we do not need to adjust their indices
    } else {
      for (auto &output_wire : circuit_gate->output_wires) {
        if (output_wire->is_qubit()) {
          output_wire->index = qubit_permutation[output_wire->index];
        }
      }
    }
  }
  if (to_remove.empty()) {
    return 0;
  }
  int to_remove_ptr = 0;
  auto original_gates = std::move(gates);
  for (auto &circuit_gate : original_gates) {
    if (to_remove_ptr < (int)to_remove.size() &&
        to_remove[to_remove_ptr] == circuit_gate.get()) {
      // remove this gate
      to_remove_ptr++;
    } else {
      gates.push_back(std::move(circuit_gate));
    }
  }
  auto original_wires = std::move(wires);
  for (auto &wire : original_wires) {
    if (output_wires_to_be_removed->find(wire.get()) ==
        output_wires_to_be_removed->end()) {
      // keep this wire
      wires.push_back(std::move(wire));
    }
  }
  return (int)to_remove.size();
}

bool CircuitSeq::evaluate(const Vector &input_dis,
                          const std::vector<ParamType> &parameter_values,
                          Vector &output_dis) const {
  // We should have 2**n entries for the distribution
  if (input_dis.size() != (1 << get_num_qubits()))
    return false;
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
        params.push_back(parameter_values[input_wire->index]);
      }
    }
    // A quantum gate. Update the distribution.
    assert(gates[i]->gate->is_quantum_gate());
    auto *mat = gates[i]->gate->get_matrix(params);
    output_dis.apply_matrix(mat, qubit_indices);
  }
  return true;
}

int CircuitSeq::get_num_qubits() const { return num_qubits; }

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

InputParamMaskType CircuitSeq::get_input_param_usage_mask(
    const std::vector<InputParamMaskType> &param_masks) const {
  InputParamMaskType usage_mask{0};
  for (auto &circuit_gate : gates) {
    // A quantum gate using parameters.
    if (circuit_gate->gate->is_parametrized_gate()) {
      for (auto &input_wire : circuit_gate->input_wires) {
        if (input_wire->is_parameter()) {
          usage_mask |= param_masks[input_wire->index];
        }
      }
    }
  }
  return usage_mask;
}

std::vector<int> CircuitSeq::get_input_param_indices(Context *ctx) const {
  auto mask = get_input_param_usage_mask(ctx->get_param_masks());
  std::vector<int> ret;
  for (int i = 0; mask; i++) {
    if (mask & (((InputParamMaskType)1) << i)) {
      mask ^= ((InputParamMaskType)1) << i;
      ret.push_back(i);
    }
  }
  return ret;
}

std::vector<int> CircuitSeq::get_directly_used_param_indices() const {
  std::vector<int> ret;
  for (auto &gate : gates) {
    for (auto &input_wire : gate->input_wires) {
      if (input_wire->is_parameter()) {
        ret.push_back(input_wire->index);
      }
    }
  }
  std::sort(ret.begin(), ret.end());
  auto last = std::unique(ret.begin(), ret.end());
  ret.erase(last, ret.end());
  return ret;
}

std::vector<CircuitGate *> CircuitSeq::get_param_expr_ops(Context *ctx) const {
  // Vector used as a stack
  std::vector<int> s = get_directly_used_param_indices();
  std::unordered_set<int> visited_params;
  std::vector<CircuitGate *> result_ops;

  // Make smaller indices appear earlier.
  std::reverse(s.begin(), s.end());

  // Non-recursive DFS
  while (!s.empty()) {
    int current_param = s.back();
    if (!ctx->param_is_expression(current_param)) {
      s.pop_back();
      continue;
    }
    if (visited_params.find(current_param) != visited_params.end()) {
      // visited, add to result
      result_ops.push_back(ctx->get_param_wire(current_param)->input_gates[0]);
      s.pop_back();
      continue;
    }
    // mark visited and search for dependencies
    visited_params.insert(current_param);
    for (auto &input_wire :
         ctx->get_param_wire(current_param)->input_gates[0]->input_wires) {
      int index = input_wire->index;
      if (visited_params.find(index) == visited_params.end() &&
          ctx->param_is_expression(index)) {
        s.push_back(index);
      }
    }
  }
  return result_ops;
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
  const Vector &input_dis = ctx->get_and_gen_input_dis(get_num_qubits());
  Vector output_dis;
  auto input_parameters = ctx->get_all_generated_parameters();
  auto all_parameters = ctx->compute_parameters(input_parameters);
  evaluate(input_dis, all_parameters, output_dis);
  ComplexType dot_product =
      output_dis.dot(ctx->get_and_gen_hashing_dis(get_num_qubits()));

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
    const int num_total_params = (int)all_parameters.size();
    for (int i = 0; i < num_total_params; i++) {
      const auto &param = all_parameters[i];
      ComplexType shifted =
          dot_product * (ComplexType{std::cos(param), std::sin(param)});
      generate_hash_values(ctx, shifted, i, all_parameters, &tmp,
                           &other_hash_values_);
      other_hash_values_.emplace_back(tmp, i);
      shifted = dot_product * (ComplexType{std::cos(param), -std::sin(param)});
      generate_hash_values(ctx, shifted, i + num_total_params, all_parameters,
                           &tmp, &other_hash_values_);
      other_hash_values_.emplace_back(tmp, i + num_total_params);
    }
    if (kCheckPhaseShiftOfPiOver4) {
      // Check phase shift of pi/4, 2pi/4, ..., 7pi/4.
      for (int i = 1; i < 8; i++) {
        const double pi = std::acos(-1.0);
        ComplexType shifted = dot_product * (ComplexType{std::cos(pi / 4 * i),
                                                         std::sin(pi / 4 * i)});
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
  auto input_parameters = ctx->get_all_generated_parameters();
  auto all_parameters = ctx->compute_parameters(input_parameters);
  std::vector<Vector> result(sz);
  for (int S = 0; S < sz; S++) {
    input_dis[S] = ComplexType(1);
    if (S > 0) {
      input_dis[S - 1] = ComplexType(0);
    }
    evaluate(input_dis, all_parameters, result[S]);
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
      char buffer[20];  // enough to store any int
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

std::string CircuitSeq::to_json(bool keep_hash_value) const {
  std::string result;
  result += "[";

  // basic info (meta data)
  result += "[";
  result += std::to_string(get_num_qubits());
  result += ",";
  result += std::to_string(get_num_gates());

  if (keep_hash_value) {
    result += ",[";
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
          result +=
              "[\"" + hash_value + "\"," + std::to_string(val.second) + "]";
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
  }

  // end of meta data
  result += "], ";

  // gates
  const int num_gates = (int)gates.size();
  result += "[";
  for (int i = 0; i < num_gates; i++) {
    result += gates[i]->to_json();
    if (i + 1 != num_gates)
      result += ", ";
  }
  result += "]";

  result += "]\n";
  return result;
}

std::unique_ptr<CircuitSeq> CircuitSeq::read_json(Context *ctx,
                                                  std::istream &fin) {
  fin.ignore(std::numeric_limits<std::streamsize>::max(), '[');

  // basic info
  int num_qubits, num_gates;
  fin.ignore(std::numeric_limits<std::streamsize>::max(), '[');
  fin >> num_qubits;
  fin.ignore(std::numeric_limits<std::streamsize>::max(), ',');
  fin >> num_gates;

  char ch;
  fin.get(ch);
  while (ch != '[' && ch != ']') {
    fin.get(ch);
  }
  if (ch == '[') {
    // ignore other hash values
    while (true) {
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
  }

  fin.ignore(std::numeric_limits<std::streamsize>::max(), ',');

  auto result = std::make_unique<CircuitSeq>(num_qubits);

  // gates
  fin.ignore(std::numeric_limits<std::streamsize>::max(), '[');
  while (true) {
    fin.get(ch);
    while (ch != '[' && ch != ']') {
      fin.get(ch);
    }
    if (ch == ']') {
      break;
    }

    // New gate
    Gate *gate;
    std::vector<int> input_qubits, input_params, output_qubits, output_params;
    CircuitGate::read_json(fin, ctx, input_qubits, input_params, output_qubits,
                           output_params, gate);
    result->add_gate(input_qubits, input_params, gate, ctx);
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
  ret.reset(seq);  // transfer ownership of |seq|
  return ret;
}

std::unique_ptr<CircuitSeq>
CircuitSeq::from_qasm_style_string(Context *ctx, const std::string &str) {
  QASMParser parser(ctx);
  CircuitSeq *seq = nullptr;
  parser.load_qasm_str(str, seq);
  std::unique_ptr<CircuitSeq> ret;
  ret.reset(seq);  // transfer ownership of |seq|
  return ret;
}

std::string CircuitSeq::to_qasm_style_string(Context *ctx,
                                             int param_precision) const {
  std::string result = "OPENQASM 2.0;\n"
                       "include \"qelib1.inc\";\n"
                       "qreg q[";
  result += std::to_string(get_num_qubits());
  result += "];\n";
  for (auto &gate : gates) {
    result += gate->to_qasm_style_string(ctx, param_precision);
  }
  return result;
}

bool CircuitSeq::to_qasm_file(Context *ctx, const std::string &filename,
                              int param_precision) const {
  std::ofstream fout(filename);
  if (!fout.is_open()) {
    return false;
  }
  fout << to_qasm_style_string(ctx, param_precision);
  fout.close();
  return true;
}

bool CircuitSeq::canonical_representation(
    std::unique_ptr<CircuitSeq> *output_seq, const Context *ctx,
    bool output) const {
  if (output) {
    // |output_seq| cannot be nullptr but its content can (and should)
    // be nullptr.
    assert(output_seq);
    // This deletes the content |output_seq| previously stored.
    *output_seq = std::make_unique<CircuitSeq>(get_num_qubits());
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
        std::vector<int> param_indices;
        for (auto &input_wire : circuit_gate->input_wires) {
          assert(input_wire->is_parameter());
          param_indices.push_back(input_wire->index);
        }
        (*output_seq)->add_gate({}, param_indices, circuit_gate->gate, ctx);
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
  free_wires.reserve(get_num_qubits());

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
        (*output_seq)
            ->add_gate(qubit_indices, param_indices, smallest_free_gate->gate,
                       ctx);
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
  return canonical_representation(nullptr, nullptr, false);
}

bool CircuitSeq::to_canonical_representation(const Context *ctx) {
  std::unique_ptr<CircuitSeq> output_seq;
  if (!canonical_representation(&output_seq, ctx, true)) {
    clone_from(*output_seq, {}, {}, ctx);
    return true;
  }
  return false;
}

[[nodiscard]] std::unique_ptr<CircuitSeq> CircuitSeq::get_gate_permutation(
    const Context *ctx,
    const std::function<int(const std::vector<CircuitGate *> &)> &gate_chooser,
    int *result_permutation) const {
  auto output_seq = std::make_unique<CircuitSeq>(get_num_qubits());
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
  free_wires.reserve(get_num_qubits());

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
      int location;
      if (gate_chooser == nullptr) {
        // Find a random free circuit_gate (gate)
        // in [0, (int)free_gates.size() - 1].
        static std::mt19937 rng(0);
        location = std::uniform_int_distribution<int>(
            0, (int)free_gates.size() - 1)(rng);
      } else {
        location = gate_chooser(free_gates);
      }
      assert(location >= 0 && location < (int)free_gates.size());
      CircuitGate *free_gate = free_gates[location];
      if (result_permutation != nullptr) {
        // Record the permutation.
        result_permutation[gate_id[free_gate]] = num_mapped_circuit_gates;
      }
      num_mapped_circuit_gates++;
      free_gates.erase(free_gates.begin() + location);

      output_seq->add_gate(free_gate, ctx);

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
                             const std::vector<int> &input_param_permutation,
                             Context *ctx) const {
  auto result = std::make_unique<CircuitSeq>(0);
  if (input_param_permutation.empty()) {
    result->clone_from(*this, qubit_permutation, input_param_permutation, ctx);
  } else {
    auto all_param_permutation =
        ctx->get_param_permutation(input_param_permutation);
    result->clone_from(*this, qubit_permutation, all_param_permutation, ctx);
  }
  return result;
}

std::unique_ptr<CircuitSeq>
CircuitSeq::get_suffix_seq(const std::unordered_set<CircuitGate *> &start_gates,
                           Context *ctx) const {
  // For topological sort
  std::unordered_map<CircuitGate *, int> gate_remaining_in_degree;
  for (auto &gate : start_gates) {
    gate_remaining_in_degree[gate] = 0;  // ready to include
  }
  auto result = std::make_unique<CircuitSeq>(get_num_qubits());
  // The result should be a subsequence of this circuit
  for (auto &gate : gates) {
    if (gate_remaining_in_degree.count(gate.get()) > 0 &&
        gate_remaining_in_degree[gate.get()] <= 0) {
      result->add_gate(gate.get(), ctx);
      for (auto &output_wire : gate->output_wires) {
        for (auto &output_gate : output_wire->output_gates) {
          // For topological sort
          if (gate_remaining_in_degree.count(output_gate) == 0) {
            gate_remaining_in_degree[output_gate] =
                output_gate->gate->get_num_qubits();
          }
          gate_remaining_in_degree[output_gate]--;
        }
      }
    }
  }
  return result;
}

void CircuitSeq::clone_from(const CircuitSeq &other,
                            const std::vector<int> &qubit_permutation,
                            const std::vector<int> &param_permutation,
                            const Context *ctx) {
  num_qubits = other.num_qubits;
  original_fingerprint_ = other.original_fingerprint_;
  std::unordered_map<CircuitWire *, CircuitWire *> wires_mapping;
  std::unordered_map<CircuitGate *, CircuitGate *> gates_mapping;
  wires.clear();
  wires.reserve(other.wires.size());
  gates.clear();
  gates.reserve(other.gates.size());
  outputs.clear();
  outputs.reserve(other.outputs.size());
  for (int i = 0; i < (int)other.gates.size(); i++) {
    gates.emplace_back(std::make_unique<CircuitGate>(*(other.gates[i])));
    assert(gates[i].get() != other.gates[i].get());
    gates_mapping[other.gates[i].get()] = gates[i].get();
  }
  if (qubit_permutation.empty() && param_permutation.empty()) {
    // A simple clone.
    hash_value_ = other.hash_value_;
    other_hash_values_ = other.other_hash_values_;
    hash_value_valid_ = other.hash_value_valid_;
  } else {
    // We need to invalidate the hash value.
    hash_value_valid_ = false;
  }
  if (qubit_permutation.empty()) {
    for (int i = 0; i < (int)other.wires.size(); i++) {
      wires.emplace_back(std::make_unique<CircuitWire>(*(other.wires[i])));
      assert(wires[i].get() !=
             other.wires[i].get());  // make sure we make a copy
      wires_mapping[other.wires[i].get()] = wires[i].get();
    }
  } else {
    assert(qubit_permutation.size() == num_qubits);
    wires.resize(other.wires.size());
    for (int i = 0; i < num_qubits; i++) {
      assert(qubit_permutation[i] >= 0 && qubit_permutation[i] < num_qubits);
      wires[qubit_permutation[i]] =
          std::make_unique<CircuitWire>(*(other.wires[i]));
      wires[qubit_permutation[i]]->index =
          qubit_permutation[i];  // update index
      assert(wires[qubit_permutation[i]].get() != other.wires[i].get());
      wires_mapping[other.wires[i].get()] = wires[qubit_permutation[i]].get();
    }
    for (int i = num_qubits; i < (int)other.wires.size(); i++) {
      wires[i] = std::make_unique<CircuitWire>(*(other.wires[i]));
      wires[i]->index = qubit_permutation[wires[i]->index];  // update index
      wires_mapping[other.wires[i].get()] = wires[i].get();
    }
  }
  if (!param_permutation.empty()) {
    for (auto &circuit_gate : gates) {
      for (auto &input_wire : circuit_gate->input_wires) {
        if (input_wire->is_parameter() &&
            input_wire->index < (int)param_permutation.size() &&
            input_wire->index != param_permutation[input_wire->index]) {
          // permute the parameter
          input_wire =
              ctx->get_param_wire(param_permutation[input_wire->index]);
        }
      }
    }
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
      if (wire->is_qubit()) {
        wire = wires_mapping[wire];
      }
    }
    for (auto &wire : circuit_gate->output_wires) {
      wire = wires_mapping[wire];
    }
  }
  for (auto &wire : other.outputs) {
    outputs.emplace_back(wires_mapping[wire]);
  }
}

std::unique_ptr<CircuitSeq> CircuitSeq::get_rz_to_t(Context *ctx) const {
  auto result = std::make_unique<CircuitSeq>(num_qubits);
  for (auto &gate : gates) {
    if (gate->gate->tp == GateType::rz) {
      auto val = ctx->get_param_value(gate->input_wires.back()->index);
      auto val_div_pi_4_float = val * 4 / PI;
      int val_div_pi_4 = (int)std::round(val_div_pi_4_float);
      assert(std::abs(val_div_pi_4_float - val_div_pi_4) < 1e-6);
      val_div_pi_4 %= 8;
      if (val_div_pi_4 > 4) {
        val_div_pi_4 -= 8;
      }
      auto qubit_indices = gate->get_qubit_indices();
      if (val_div_pi_4 == -4 || val_div_pi_4 == 4) {
        result->add_gate(qubit_indices, {}, ctx->get_gate(GateType::z),
                         nullptr);
      }
      if (val_div_pi_4 == 2 || val_div_pi_4 == 3) {
        result->add_gate(qubit_indices, {}, ctx->get_gate(GateType::s),
                         nullptr);
      }
      if (val_div_pi_4 == -2 || val_div_pi_4 == -3) {
        result->add_gate(qubit_indices, {}, ctx->get_gate(GateType::sdg),
                         nullptr);
      }
      if (val_div_pi_4 == 1 || val_div_pi_4 == 3) {
        result->add_gate(qubit_indices, {}, ctx->get_gate(GateType::t),
                         nullptr);
      }
      if (val_div_pi_4 == -1 || val_div_pi_4 == -3) {
        result->add_gate(qubit_indices, {}, ctx->get_gate(GateType::tdg),
                         nullptr);
      }
    } else {
      bool ok = result->add_gate(gate.get(), ctx);
      assert(ok);
    }
  }
  return std::move(result);
}

std::unique_ptr<CircuitSeq> CircuitSeq::get_ccz_to_cx_rz(Context *ctx) const {
  auto t = ctx->get_new_param_id(0.25 * PI);
  auto tdg = ctx->get_new_param_id(-0.25 * PI);
  auto result = std::make_unique<CircuitSeq>(num_qubits);
  for (auto &gate : gates) {
    if (gate->gate->tp == GateType::ccz) {
      auto qubit_indices = gate->get_qubit_indices();
      auto q0 = qubit_indices[0];
      auto q1 = qubit_indices[1];
      auto q2 = qubit_indices[2];
      bool ok;
      ok = result->add_gate({q1, q2}, {}, ctx->get_gate(GateType::cx), ctx);
      assert(ok);
      ok = result->add_gate({q2}, {tdg}, ctx->get_gate(GateType::rz), ctx);
      assert(ok);
      ok = result->add_gate({q0, q2}, {}, ctx->get_gate(GateType::cx), ctx);
      assert(ok);
      ok = result->add_gate({q2}, {t}, ctx->get_gate(GateType::rz), ctx);
      assert(ok);
      ok = result->add_gate({q1, q2}, {}, ctx->get_gate(GateType::cx), ctx);
      assert(ok);
      ok = result->add_gate({q2}, {tdg}, ctx->get_gate(GateType::rz), ctx);
      assert(ok);
      ok = result->add_gate({q0, q2}, {}, ctx->get_gate(GateType::cx), ctx);
      assert(ok);
      ok = result->add_gate({q0, q1}, {}, ctx->get_gate(GateType::cx), ctx);
      assert(ok);
      ok = result->add_gate({q1}, {tdg}, ctx->get_gate(GateType::rz), ctx);
      assert(ok);
      ok = result->add_gate({q0, q1}, {}, ctx->get_gate(GateType::cx), ctx);
      assert(ok);
      ok = result->add_gate({q0}, {t}, ctx->get_gate(GateType::rz), ctx);
      assert(ok);
      ok = result->add_gate({q1}, {t}, ctx->get_gate(GateType::rz), ctx);
      assert(ok);
      ok = result->add_gate({q2}, {t}, ctx->get_gate(GateType::rz), ctx);
    } else {
      bool ok = result->add_gate(gate.get(), ctx);
      assert(ok);
    }
  }
  return std::move(result);
}

void CircuitSeq::remove_quantum_gate_from_graph(
    CircuitGate *circuit_gate, bool assert_no_logical_qubit_permutation,
    std::unordered_set<CircuitWire *> *output_wires_to_be_removed) {
  // Remove gates from input wires.
  for (auto *input_wire : circuit_gate->input_wires) {
    if (input_wire->is_qubit()) {
      assert(!input_wire->output_gates.empty());
      auto it = std::find(input_wire->output_gates.begin(),
                          input_wire->output_gates.end(), circuit_gate);
      assert(it != input_wire->output_gates.end());
      input_wire->output_gates.erase(it);
    }
  }
  int num_outputs = (int)circuit_gate->output_wires.size();
  int j = 0;
  for (int i = 0; i < num_outputs; i++, j++) {
    // Match the input qubits and the output qubits.
    while (j < (int)circuit_gate->input_wires.size() &&
           !circuit_gate->input_wires[j]->is_qubit()) {
      j++;
    }
    assert(j < (int)circuit_gate->input_wires.size());
    if (assert_no_logical_qubit_permutation) {
      assert(circuit_gate->input_wires[j]->index ==
             circuit_gate->output_wires[i]->index);
    }
    if (outputs[circuit_gate->output_wires[i]->index] ==
        circuit_gate->output_wires[i]) {
      // Restore the outputs.
      outputs[circuit_gate->output_wires[i]->index] =
          circuit_gate->input_wires[j];
    }
    if (circuit_gate->output_wires[i]->output_gates.empty()) {
      // Remove the qubit wires.
      if (output_wires_to_be_removed) {
        output_wires_to_be_removed->insert(circuit_gate->output_wires[i]);
      } else {
        auto it = std::find_if(
            wires.begin(), wires.end(), [&](std::unique_ptr<CircuitWire> &p) {
              return p.get() == circuit_gate->output_wires[i];
            });
        assert(it != wires.end());
        wires.erase(it);
      }
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
      if (output_wires_to_be_removed) {
        output_wires_to_be_removed->insert(circuit_gate->output_wires[i]);
      } else {
        auto it = std::find_if(
            wires.begin(), wires.end(), [&](std::unique_ptr<CircuitWire> &p) {
              return p.get() == circuit_gate->output_wires[i];
            });
        assert(it != wires.end());
        wires.erase(it);
      }
    }
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

std::vector<int> CircuitSeq::first_quantum_gate_positions() const {
  std::vector<int> result;
  std::unordered_set<CircuitGate *> depend_on_other_gates;
  depend_on_other_gates.reserve(gates.size());
  for (int i = 0; i < (int)gates.size(); i++) {
    CircuitGate *circuit_gate = gates[i].get();
    if (circuit_gate->gate->is_parameter_gate()) {
      continue;
    }
    if (depend_on_other_gates.find(circuit_gate) ==
        depend_on_other_gates.end()) {
      result.push_back(i);
    }
    for (const auto &output_wire : circuit_gate->output_wires) {
      for (const auto &output_gate : output_wire->output_gates) {
        depend_on_other_gates.insert(output_gate);
      }
    }
  }
  return result;
}

bool CircuitSeq::is_one_of_last_gates(CircuitGate *circuit_gate) const {
  for (const auto &output_wire : circuit_gate->output_wires) {
    if (outputs[output_wire->index] != output_wire) {
      return false;
    }
  }
  return true;
}

std::vector<CircuitGate *> CircuitSeq::last_quantum_gates() const {
  std::vector<CircuitGate *> result;
  for (const auto &circuit_gate : gates) {
    if (circuit_gate->gate->is_parameter_gate()) {
      continue;
    }
    if (is_one_of_last_gates(circuit_gate.get())) {
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

}  // namespace quartz
