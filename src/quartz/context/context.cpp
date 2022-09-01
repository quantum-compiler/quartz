#include "context.h"
#include "../dag/dag.h"
#include "../gate/all_gates.h"

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
      supported_parameter_gates_.emplace_back(gate);
    } else {
      supported_quantum_gates_.emplace_back(gate);
    }
  }
}

Context::Context(const std::vector<GateType> &supported_gates,
                 const int num_qubits, const int num_params)
    : Context(supported_gates) {
  get_and_gen_input_dis(num_qubits);
  get_and_gen_hashing_dis(num_qubits);
  get_and_gen_parameters(num_params);
}

size_t Context::next_global_unique_id(void) {
  static std::mutex lock;
  std::lock_guard<std::mutex> lg(lock);
  return global_unique_id++;
}

void Context::set_generated_parameter(int id, ParamType param) {
  get_generated_parameters(id);
  random_parameters_[id] = param;
}

Gate *Context::get_gate(GateType tp) {
  const auto it = gates_.find(tp);
  if (it != gates_.end())
    return it->second.get();
  else
    return nullptr;
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

const std::vector<GateType> &Context::get_supported_parameter_gates() const {
  return supported_parameter_gates_;
}

const std::vector<GateType> &Context::get_supported_quantum_gates() const {
  return supported_quantum_gates_;
}

const Vector &Context::get_and_gen_input_dis(const int num_qubits) {
  assert(num_qubits >= 0);
  while (random_input_distribution_.size() <= num_qubits) {
    random_input_distribution_.emplace_back(
        Vector::random_generate(random_input_distribution_.size(), &gen));
  }
  return random_input_distribution_[num_qubits];
}

const Vector &Context::get_and_gen_hashing_dis(const int num_qubits) {
  assert(num_qubits >= 0);
  while (random_hashing_distribution_.size() <= num_qubits) {
    random_hashing_distribution_.emplace_back(
        Vector::random_generate(random_hashing_distribution_.size(), &gen));
  }
  return random_hashing_distribution_[num_qubits];
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
                 " or use Context::get_and_gen_input_dis ."
              << std::endl;
    assert(false);
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
           " or use Context::get_and_gen_hashing_dis ."
        << std::endl;
    assert(false);
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
  }
}

std::vector<ParamType> Context::get_all_generated_parameters() const {
  return random_parameters_;
}

DAG *Context::get_possible_representative(DAG *dag) {
  return representatives_[dag->hash(this)];
}

void Context::set_representative(std::unique_ptr<DAG> dag) {
  representatives_[dag->hash(this)] = dag.get();
  representative_dags_.emplace_back(std::move(dag));
}

void Context::clear_representatives() {
  representatives_.clear();
  representative_dags_.clear();
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

} // namespace quartz
