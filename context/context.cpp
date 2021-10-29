#include "context.h"
#include "../gate/all_gates.h"
#include "../dag/dag.h"

#include <cassert>
#include <cmath>
#include <random>

Context::Context(const std::vector<GateType> &supported_gates)
    : supported_gates_(supported_gates), global_unique_id(100) {
  gates_.reserve(supported_gates.size());
  for (const auto &gate : supported_gates) {
	insert_gate(gate);
	if (gates_[gate]->is_parameter_gate()) {
	  supported_parameter_gates_.emplace_back(gate);
	}
	else {
	  supported_quantum_gates_.emplace_back(gate);
	}
  }
}

size_t Context::next_global_unique_id(void) { return global_unique_id++; }

Gate *Context::get_gate(GateType tp) { return gates_[tp].get(); }

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

const Vector &Context::get_generated_input_dis(int num_qubits) {
  assert(num_qubits >= 0);
  if (random_input_distribution_.size() <= num_qubits) {
	random_input_distribution_.resize(num_qubits + 1);
  }
  if (random_input_distribution_[num_qubits].size() == 0) {
	random_input_distribution_[num_qubits] =
	    Vector::random_generate(num_qubits);
  }
  return random_input_distribution_[num_qubits];
}

const Vector &Context::get_generated_hashing_dis(int num_qubits) {
  assert(num_qubits >= 0);
  if (random_hashing_distribution_.size() <= num_qubits) {
	random_hashing_distribution_.resize(num_qubits + 1);
  }
  if (random_hashing_distribution_[num_qubits].size() == 0) {
	random_hashing_distribution_[num_qubits] =
	    Vector::random_generate(num_qubits);
  }
  return random_hashing_distribution_[num_qubits];
}

std::vector<ParamType> Context::get_generated_parameters(int num_params) {
  assert(num_params >= 0);
  if (random_parameters_.size() < num_params) {
	// Standard mersenne_twister_engine seeded with 0
	static std::mt19937 gen(0);
	static ParamType pi = std::acos((ParamType)-1.0);
	static std::uniform_real_distribution<ParamType> dis_real(-pi, pi);
	while (random_parameters_.size() < num_params) {
	  random_parameters_.emplace_back(dis_real(gen));
	}
  }
  return std::vector<ParamType>(random_parameters_.begin(),
                                random_parameters_.begin() + num_params);
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
