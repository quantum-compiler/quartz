#include "context.h"

#include "quartz/circuitseq/circuitseq.h"
#include "quartz/gate/all_gates.h"
#include "quartz/utils/string_utils.h"

#include <cassert>
#include <iostream>
#include <mutex>
#include <random>
#include <set>

namespace quartz {
Context::Context(const std::vector<GateType> &supported_gates,
                 ParamInfo *param_info)
    : global_unique_id(16384), supported_gates_(supported_gates),
      param_info_(param_info) {
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
                 ParamInfo *param_info)
    : Context(supported_gates, param_info) {
  gen_input_and_hashing_dis(num_qubits);
  generate_parameter_expressions();
}

size_t Context::next_global_unique_id(void) {
  static std::mutex lock;
  std::lock_guard<std::mutex> lg(lock);
  return global_unique_id++;
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

bool Context::has_gate(GateType tp) const {
  return gates_.find(tp) != gates_.end();
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

const Vector &Context::get_and_gen_input_dis(int num_qubits) {
  gen_input_and_hashing_dis(num_qubits);
  return get_generated_input_dis(num_qubits);
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

const Vector &Context::get_and_gen_hashing_dis(int num_qubits) {
  gen_input_and_hashing_dis(num_qubits);
  return get_generated_hashing_dis(num_qubits);
}

std::vector<ParamType> Context::get_all_generated_parameters() const {
  return param_info_->get_all_generated_parameters();
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
  return param_info_->get_param_value(id);
}

void Context::set_param_value(int id, const ParamType &param) {
  return param_info_->set_param_value(id, param);
}

std::vector<ParamType> Context::get_all_input_param_values() const {
  return param_info_->get_all_input_param_values();
}

int Context::get_new_param_id(const ParamType &param) {
  return param_info_->get_new_param_id(param);
}

int Context::get_new_param_id() { return param_info_->get_new_param_id(); }

int Context::get_new_param_expression_id(
    const std::vector<int> &parameter_indices, Gate *op) {
  return param_info_->get_new_param_expression_id(parameter_indices, op);
}

int Context::get_num_parameters() const {
  return param_info_->get_num_parameters();
}

int Context::get_num_input_symbolic_parameters() const {
  return param_info_->get_num_input_symbolic_parameters();
}

bool Context::param_is_symbolic(int id) const {
  return param_info_->param_is_symbolic(id);
}

bool Context::param_has_value(int id) const {
  return param_info_->param_has_value(id);
}

bool Context::param_is_expression(int id) const {
  return param_info_->param_is_expression(id);
}

CircuitWire *Context::get_param_wire(int id) const {
  return param_info_->get_param_wire(id);
}

std::vector<ParamType>
Context::compute_parameters(const std::vector<ParamType> &input_parameters) {
  return param_info_->compute_parameters(input_parameters);
}

std::vector<int> Context::get_param_permutation(
    const std::vector<int> &input_param_permutation) {
  int num_parameters = (int)param_info_->is_parameter_symbolic_.size();
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
  int num_input_parameters = (int)param_info_->is_parameter_symbolic_.size();
  assert(num_input_parameters > 0);
  if (!param_info_->parameter_expressions_.empty()) {
    std::cerr << "Context::generate_parameter_expressions() called twice for a "
                 "single ParamInfo object. Please use different ParamInfo "
                 "objects for different Context objects."
              << std::endl;
    assert(false);
  }
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

std::vector<InputParamMaskType> Context::get_param_masks() const {
  return param_info_->get_param_masks();
}

std::string Context::param_info_to_json() const {
  std::string result = "[";
  result += "[";
  result += std::to_string(param_info_->is_parameter_symbolic_.size());
  for (int i = 0; i < (int)param_info_->is_parameter_symbolic_.size(); i++) {
    result += ", ";
    if (param_is_expression(i)) {
      result += param_info_->parameter_wires_[i]->input_gates[0]->to_json();
    } else if (param_info_->is_parameter_symbolic_[i]) {
      result += "\"\"";
    } else {
      result += to_string_with_precision(param_info_->parameter_values_[i],
                                         /*precision=*/17);
    }
  }
  result += "], ";
  result += to_json_style_string_with_precision(param_info_->random_parameters_,
                                                /*precision=*/17);
  result += "]";
  return result;
}

void Context::load_param_info_from_json(std::istream &fin) {
  fin.ignore(std::numeric_limits<std::streamsize>::max(), '[');
  fin.ignore(std::numeric_limits<std::streamsize>::max(), '[');
  int num_params;
  fin >> num_params;
  param_info_->is_parameter_symbolic_.clear();
  param_info_->is_parameter_symbolic_.reserve(num_params);
  param_info_->parameter_wires_.clear();
  param_info_->parameter_wires_.reserve(num_params);
  param_info_->parameter_values_.clear();
  param_info_->parameter_values_.reserve(num_params);
  param_info_->parameter_expressions_.clear();
  for (int i = 0; i < num_params; i++) {
    char ch;
    fin >> ch;
    while (ch != '[' && ch != '\"' && ch != '-' && !std::isdigit(ch) &&
           ch != ']') {
      fin >> ch;
    }
    assert(ch != ']');
    if (ch == '[') {
      // parameter expression
      Gate *gate;
      std::vector<int> input_qubits, input_params, output_qubits, output_params;
      CircuitGate::read_json(fin, this, input_qubits, input_params,
                             output_qubits, output_params, gate);
      int id = get_new_param_expression_id(input_params, gate);
      assert(id == i);
    } else if (ch == '\"') {
      // symbolic parameter
      fin >> ch;
      assert(ch == '\"');  // ""
      int id = get_new_param_id();
      assert(id == i);
    } else {
      // concrete parameter
      fin.unget();
      ParamType val;
      fin >> val;
      int id = get_new_param_id(val);
      assert(id == i);
    }
  }
  fin.ignore(std::numeric_limits<std::streamsize>::max(), ',');
  bool ret = read_json_style_vector(fin, param_info_->random_parameters_);
  assert(ret);
}

ParamInfo *Context::get_param_info() const { return param_info_; }

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
  assert(ctx_0->get_param_info() == ctx_1->get_param_info());
  std::vector<GateType> union_vector;
  const auto &ctx_0_gates = ctx_0->get_supported_gates();
  const auto &ctx_1_gates = ctx_1->get_supported_gates();
  std::set<GateType> gate_set_0(ctx_0_gates.begin(), ctx_0_gates.end());
  std::set<GateType> gate_set_1(ctx_1_gates.begin(), ctx_1_gates.end());
  for (auto tp : gate_set_0)
    union_vector.push_back(tp);
  for (auto tp : gate_set_1) {
    if (gate_set_0.find(tp) == gate_set_0.end()) {
      union_vector.push_back(tp);
    }
  }
  return Context(union_vector, ctx_0->get_param_info());
}

}  // namespace quartz
