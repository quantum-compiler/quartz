#include "context.h"

#include "../gate/all_gates.h"
#include "quartz/circuitseq/circuitseq.h"

#include <cassert>
#include <cmath>
#include <iostream>
#include <mutex>
#include <random>

namespace quartz {
Context::Context(const std::vector<GateType> &supported_gates)
    : global_unique_id(16384), supported_gates_(supported_gates) {
  gates_.reserve(supported_gates.size());
  for (const auto &gate : supported_gates) {
    insert_gate(gate);
    if (gates_[gate]->is_parameter_gate()) {
      supported_parameter_ops_.emplace_back(gate);
    } else {
      supported_quantum_gates_.emplace_back(gate);
    }
  }
}

Context::Context(const std::vector<GateType> &supported_gates, int num_qubits,
                 int num_input_symbolic_params)
    : Context(supported_gates) {
  gen_input_and_hashing_dis(num_qubits);
  get_and_gen_parameters(num_input_symbolic_params);
  for (int i = 0; i < num_input_symbolic_params; i++) {
    get_new_param_id();
  }
}

size_t Context::next_global_unique_id(void) {
  static std::mutex lock;
  std::lock_guard<std::mutex> lg(lock);
  return global_unique_id++;
}

void Context::set_generated_parameter(int id, ParamType param) {
  random_parameters_[id] = param;
}

Gate *Context::get_gate(GateType tp) {
  const auto it = gates_.find(tp);
  if (it != gates_.end())
    return it->second.get();
  else {
    std::cerr << "Gate " << gate_type_name(tp)
              << " not in current context (gate set)." << std::endl;
    assert(false);
    return nullptr;
  }
}

Gate *Context::get_general_controlled_gate(GateType tp,
                                           const std::vector<bool> &state) {
  auto it1 = general_controlled_gates_.find(tp);
  if (it1 == general_controlled_gates_.end()) {
    it1 = general_controlled_gates_.insert(
        general_controlled_gates_.end(),
        std::make_pair(
            tp,
            std::unordered_map<std::vector<bool>, std::unique_ptr<Gate>>()));
  }
  auto it2 = it1->second.find(state);
  if (it2 == it1->second.end()) {
    // Create a new general controlled gate if not found.
    Gate *controlled_gate = get_gate(tp);
    if (!controlled_gate) {
      return nullptr;
    }
    it2 = it1->second.insert(
        it1->second.end(),
        std::make_pair(state, std::make_unique<GeneralControlledGate>(
                                  controlled_gate, state)));
  }
  return it2->second.get();
}

bool Context::insert_gate(GateType tp) {
  if (gates_.count(tp) > 0) {
    return false;
  }
  std::unique_ptr<Gate> new_gate;

#define PER_GATE(x, XGate)                                                     \
  case GateType::x:                                                            \
    new_gate = std::make_unique<XGate>();                                      \
    break;

  switch (tp) {
#include "../gate/gates.inc.h"
  }

#undef PER_GATE

  gates_[tp] = std::move(new_gate);
  return true;
}

const std::vector<GateType> &Context::get_supported_gates() const {
  return supported_gates_;
}

const std::vector<GateType> &Context::get_supported_parameter_ops() const {
  return supported_parameter_ops_;
}

const std::vector<GateType> &Context::get_supported_quantum_gates() const {
  return supported_quantum_gates_;
}

void Context::gen_input_and_hashing_dis(const int num_qubits) {
  assert(num_qubits >= 0);
  assert(random_input_distribution_.size() ==
         random_hashing_distribution_.size());
  while (random_input_distribution_.size() <= num_qubits) {
    random_input_distribution_.emplace_back(
        Vector::random_generate((int)random_input_distribution_.size(), &gen));
    random_hashing_distribution_.emplace_back(Vector::random_generate(
        (int)random_hashing_distribution_.size(), &gen));
  }
}

std::vector<ParamType> Context::get_and_gen_parameters(const int num_params) {
  assert(num_params >= 0);
  if (random_parameters_.size() < num_params) {
    static ParamType pi = std::acos((ParamType)-1.0);
    static std::uniform_real_distribution<ParamType> dis_real(-pi, pi);
    while (random_parameters_.size() < num_params) {
      random_parameters_.emplace_back(dis_real(gen));
    }
  }
  return std::vector<ParamType>(random_parameters_.begin(),
                                random_parameters_.begin() + num_params);
}

const Vector &Context::get_generated_input_dis(int num_qubits) const {
  if (0 <= num_qubits && num_qubits < random_input_distribution_.size())
    return random_input_distribution_[num_qubits];
  else {
    std::cerr << "Currently random_input_distribution_.size() = "
              << random_input_distribution_.size()
              << " , but the queried num_qubits = " << num_qubits << std::endl
              << "Please generate enough random_input_distribution_ in advance"
                 " or use Context::gen_input_and_hashing_dis ."
              << std::endl;
    assert(false);
    return {};
  }
}

const Vector &Context::get_generated_hashing_dis(int num_qubits) const {
  if (0 <= num_qubits && num_qubits < random_hashing_distribution_.size())
    return random_hashing_distribution_[num_qubits];
  else {
    std::cerr
        << "Currently random_hashing_distribution_.size() = "
        << random_hashing_distribution_.size()
        << " , but the queried num_qubits = " << num_qubits << std::endl
        << "Please generate enough random_hashing_distribution_ in advance"
           " or use Context::gen_input_and_hashing_dis ."
        << std::endl;
    assert(false);
    return {};
  }
}

std::vector<ParamType> Context::get_generated_parameters(int num_params) const {
  if (0 <= num_params && num_params <= random_parameters_.size())
    return std::vector<ParamType>(random_parameters_.begin(),
                                  random_parameters_.begin() + num_params);
  else {
    std::cerr << "Currently random_parameters_.size() = "
              << random_parameters_.size()
              << " , but the queried num_params = " << num_params << std::endl
              << "Please generate enough random_parameters_ in advance"
                 " or use Context::get_and_gen_parameters ."
              << std::endl;
    assert(false);
    return std::vector<ParamType>(num_params);
  }
}

std::vector<ParamType> Context::get_all_generated_parameters() const {
  return random_parameters_;
}

void Context::set_representative(std::unique_ptr<CircuitSeq> seq) {
  representatives_[seq->hash(this)] = seq.get();
  representative_seqs_.emplace_back(std::move(seq));
}

void Context::clear_representatives() {
  representatives_.clear();
  representative_seqs_.clear();
}

bool Context::get_possible_representative(const CircuitSeqHashType &hash_value,
                                          CircuitSeq *&representative) const {
  auto it = representatives_.find(hash_value);
  if (it == representatives_.end()) {
    return false;
  }
  representative = it->second;
  return true;
}

ParamType Context::get_param_value(int id) const {
  assert(id >= 0 && id < (int)parameter_values_.size());
  assert(!is_parameter_symbolic_[id]);
  return parameter_values_[id];
}

void Context::set_param_value(int id, const ParamType &param) {
  assert(id >= 0 && id < (int)is_parameter_symbolic_.size());
  assert(!is_parameter_symbolic_[id]);
  while (id >= (int)parameter_values_.size()) {
    parameter_values_.emplace_back();
  }
  parameter_values_[id] = param;
}

std::vector<ParamType> Context::get_all_param_values() const {
  return parameter_values_;
}

int Context::get_new_param_id(const ParamType &param) {
  int id = (int)is_parameter_symbolic_.size();
  is_parameter_symbolic_.push_back(false);
  auto wire = std::make_unique<CircuitWire>();
  wire->type = CircuitWire::input_param;
  wire->index = id;
  parameter_wires_.push_back(std::move(wire));
  set_param_value(id, param);
  return id;
}

int Context::get_new_param_id() {
  int id = (int)is_parameter_symbolic_.size();
  is_parameter_symbolic_.push_back(true);
  auto wire = std::make_unique<CircuitWire>();
  wire->type = CircuitWire::input_param;
  wire->index = id;
  parameter_wires_.push_back(std::move(wire));
  return id;
}

int Context::get_new_param_expression_id(
    const std::vector<int> &parameter_indices, Gate *op) {
  bool is_symbolic = false;
  for (auto &input_id : parameter_indices) {
    assert(input_id >= 0 && input_id < (int)is_parameter_symbolic_.size());
    if (param_is_symbolic(input_id)) {
      is_symbolic = true;
    }
  }
  if (!is_symbolic) {
    // A concrete parameter, no need to create an expression.
    // Compute the value directly.
    std::vector<ParamType> input_params;
    input_params.reserve(parameter_indices.size());
    for (auto &input_id : parameter_indices) {
      input_params.push_back(get_param_value(input_id));
    }
    return get_new_param_id(op->compute(input_params));
  }
  int id = (int)is_parameter_symbolic_.size();
  is_parameter_symbolic_.push_back(true);
  auto circuit_gate = std::make_unique<CircuitGate>();
  circuit_gate->gate = op;
  for (auto &input_id : parameter_indices) {
    circuit_gate->input_wires.push_back(parameter_wires_[input_id].get());
    parameter_wires_[input_id]->output_gates.push_back(circuit_gate.get());
  }
  auto wire = std::make_unique<CircuitWire>();
  wire->type = CircuitWire::internal_param;
  wire->index = id;
  wire->input_gates.push_back(circuit_gate.get());
  circuit_gate->output_wires.push_back(wire.get());
  parameter_wires_.push_back(std::move(wire));
  parameter_expressions_.push_back(std::move(circuit_gate));
  return id;
}

int Context::get_num_parameters() const {
  return (int)is_parameter_symbolic_.size();
}

bool Context::param_is_symbolic(int id) const {
  return id >= 0 && id < (int)is_parameter_symbolic_.size() &&
         is_parameter_symbolic_[id];
}

bool Context::param_has_value(int id) const {
  return id >= 0 && id < (int)is_parameter_symbolic_.size() &&
         !is_parameter_symbolic_[id];
}

bool Context::param_is_expression(int id) const {
  return id >= 0 && id < (int)parameter_wires_.size() &&
         !parameter_wires_[id]->input_gates.empty();
}

CircuitWire *Context::get_param_wire(int id) const {
  if (id >= 0 && id < (int)parameter_wires_.size()) {
    return parameter_wires_[id].get();
  } else {
    return nullptr;  // out of range
  }
}

std::vector<int> Context::get_param_permutation(
    const std::vector<int> &input_param_permutation) {
  int num_parameters = (int)is_parameter_symbolic_.size();
  std::vector<int> result = input_param_permutation;
  result.resize(num_parameters, -1);  // fill with -1
  for (int i = (int)input_param_permutation.size(); i < num_parameters; i++) {
    if (param_is_expression(i)) {
      auto gate = get_param_wire(i)->input_gates[0];
      std::vector<int> input_indices;
      input_indices.reserve(gate->input_wires.size());
      for (auto &wire : gate->input_wires) {
        assert(wire->index < i);
        input_indices.push_back(result[wire->index]);  // get permuted input
      }
      auto input_0 = get_param_wire(input_indices[0]);
      for (auto &potential_gate : input_0->output_gates) {
        // same gate type (pointer comparison)
        if (potential_gate->gate == gate->gate &&
            input_indices.size() == potential_gate->input_wires.size()) {
          bool same_indices = true;
          for (int j = 0; j < (int)input_indices.size(); j++) {
            if (input_indices[j] != potential_gate->input_wires[j]->index) {
              same_indices = false;
              break;
            }
          }
          if (same_indices) {
            // found permuted expression
            result[i] = potential_gate->output_wires[0]->index;
            break;
          }
        }
      }
      if (result[i] == -1) {
        // still not found, create new expression
        result[i] = get_new_param_expression_id(input_indices, gate->gate);
      }
    } else {
      // not an expression, map to itself
      result[i] = i;
    }
  }
  return result;
}

void Context::generate_parameter_expressions(
    int max_num_operators_per_expression) {
  assert(max_num_operators_per_expression == 1);
  int num_input_parameters = (int)is_parameter_symbolic_.size();
  for (const auto &idx : get_supported_parameter_ops()) {
    Gate *op = get_gate(idx);
    if (op->get_num_parameters() == 1) {
      std::vector<int> param_indices(1);
      for (param_indices[0] = 0; param_indices[0] < num_input_parameters;
           param_indices[0]++) {
        get_new_param_expression_id(param_indices, op);
      }
    } else if (op->get_num_parameters() == 2) {
      // Case: 0-qubit operators with 2 parameters
      std::vector<int> param_indices(2);
      for (param_indices[0] = 0; param_indices[0] < num_input_parameters;
           param_indices[0]++) {
        for (param_indices[1] = 0; param_indices[1] < num_input_parameters;
             param_indices[1]++) {
          if (op->is_commutative() && param_indices[0] > param_indices[1]) {
            // For commutative operators, enforce param_indices[0]
            // <= param_indices[1]
            continue;
          }
          get_new_param_expression_id(param_indices, op);
        }
      }
    } else {
      assert(false && "Unsupported gate type");
    }
  }
}

double Context::random_number() {
  static std::uniform_real_distribution<double> dis_real(0, 1);
  return dis_real(gen);
}

bool Context::has_parameterized_gate() const {
  for (auto it = gates_.begin(); it != gates_.end(); ++it) {
    if (it->second->is_parametrized_gate())
      return true;
  }
  return false;
}

Context union_contexts(Context *ctx_0, Context *ctx_1) {
  std::vector<GateType> union_vector;
  std::set<GateType> gate_set_0(ctx_0->get_supported_gates().begin(),
                                ctx_0->get_supported_gates().end());
  std::set<GateType> gate_set_1(ctx_1->get_supported_gates().begin(),
                                ctx_1->get_supported_gates().end());
  for (auto tp : gate_set_0)
    union_vector.push_back(tp);
  for (auto tp : gate_set_1) {
    if (gate_set_0.find(tp) == gate_set_0.end()) {
      union_vector.push_back(tp);
    }
  }
  return Context(union_vector);
}

}  // namespace quartz
