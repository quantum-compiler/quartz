#include "param_info.h"

#include <cassert>

namespace quartz {

ParamInfo::ParamInfo(int num_input_symbolic_params) {
  gen_random_parameters(num_input_symbolic_params);
  for (int i = 0; i < num_input_symbolic_params; i++) {
    get_new_param_id();
  }
}

void ParamInfo::gen_random_parameters(int num_params) {
  assert(num_params >= 0);
  if (random_parameters_.size() < num_params) {
    random_parameters_.reserve(num_params);
    static const ParamType pi = std::acos((ParamType)-1.0);
    static std::uniform_real_distribution<ParamType> dis_real(-pi, pi);
    while (random_parameters_.size() < num_params) {
      random_parameters_.emplace_back(dis_real(gen));
    }
  }
}

std::vector<ParamType> ParamInfo::get_all_generated_parameters() const {
  return random_parameters_;
}

ParamType ParamInfo::get_param_value(int id) const {
  assert(id >= 0 && id < (int)parameter_values_.size());
  assert(!is_parameter_symbolic_[id]);
  return parameter_values_[id];
}

void ParamInfo::set_param_value(int id, const ParamType &param) {
  assert(id >= 0 && id < (int)is_parameter_symbolic_.size());
  assert(!is_parameter_symbolic_[id]);
  while (id >= (int)parameter_values_.size()) {
    parameter_values_.emplace_back();
  }
  parameter_values_[id] = param;
}

std::vector<ParamType> ParamInfo::get_all_input_param_values() const {
  return parameter_values_;
}

int ParamInfo::get_new_param_id(const ParamType &param) {
  int id = (int)is_parameter_symbolic_.size();
  is_parameter_symbolic_.push_back(false);
  auto wire = std::make_unique<CircuitWire>();
  wire->type = CircuitWire::input_param;
  wire->index = id;
  parameter_wires_.push_back(std::move(wire));
  set_param_value(id, param);
  return id;
}

int ParamInfo::get_new_param_id() {
  int id = (int)is_parameter_symbolic_.size();
  is_parameter_symbolic_.push_back(true);
  auto wire = std::make_unique<CircuitWire>();
  wire->type = CircuitWire::input_param;
  wire->index = id;
  parameter_wires_.push_back(std::move(wire));
  return id;
}

int ParamInfo::get_new_param_expression_id(
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

int ParamInfo::get_num_parameters() const {
  return (int)is_parameter_symbolic_.size();
}

int ParamInfo::get_num_input_symbolic_parameters() const {
  return (int)random_parameters_.size();
}

bool ParamInfo::param_is_symbolic(int id) const {
  return id >= 0 && id < (int)is_parameter_symbolic_.size() &&
         is_parameter_symbolic_[id];
}

bool ParamInfo::param_has_value(int id) const {
  return id >= 0 && id < (int)is_parameter_symbolic_.size() &&
         !is_parameter_symbolic_[id];
}

bool ParamInfo::param_is_expression(int id) const {
  return id >= 0 && id < (int)parameter_wires_.size() &&
         !parameter_wires_[id]->input_gates.empty();
}

CircuitWire *ParamInfo::get_param_wire(int id) const {
  if (id >= 0 && id < (int)parameter_wires_.size()) {
    return parameter_wires_[id].get();
  } else {
    return nullptr;  // out of range
  }
}

std::vector<ParamType>
ParamInfo::compute_parameters(const std::vector<ParamType> &input_parameters) {
  auto result = input_parameters;
  result.resize(is_parameter_symbolic_.size());
  for (auto &expr : parameter_expressions_) {
    std::vector<ParamType> params;
    for (const auto &input_wire : expr->input_wires) {
      params.push_back(result[input_wire->index]);
    }
    assert(expr->output_wires.size() == 1);
    const auto &output_wire = expr->output_wires[0];
    result[output_wire->index] = expr->gate->compute(params);
  }
  return result;
}

std::vector<InputParamMaskType> ParamInfo::get_param_masks() const {
  std::vector<InputParamMaskType> param_mask(is_parameter_symbolic_.size());
  for (int i = 0; i < (int)param_mask.size(); i++) {
    if (!param_is_expression(i)) {
      param_mask[i] = ((InputParamMaskType)1) << i;
    }
  }
  for (auto &expr : parameter_expressions_) {
    const auto &output_wire = expr->output_wires[0];
    param_mask[output_wire->index] = 0;
    for (const auto &input_wire : expr->input_wires) {
      param_mask[output_wire->index] |= param_mask[input_wire->index];
    }
  }
  return param_mask;
}
}  // namespace quartz
