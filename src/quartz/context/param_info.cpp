#include "param_info.h"

#include "quartz/utils/string_utils.h"

#include <cassert>

namespace quartz {

bool is_symbolic_constant(Gate *op) {
  bool rv = false;
  if (op->tp == GateType::pi) {
    rv = true;
  }
  return rv;
}

ParamInfo::ParamInfo(int num_input_symbolic_params, bool is_halved) {
  gen_random_parameters(num_input_symbolic_params);
  for (int i = 0; i < num_input_symbolic_params; i++) {
    get_new_param_id(is_halved);
  }
}

void ParamInfo::gen_random_parameters(int num_params) {
  assert(num_params >= 0);
  if (random_parameters_.size() < num_params) {
    random_parameters_.reserve(num_params);
#ifdef USE_RATIONAL
    static const long long kDenominator = 1ll << 62;
    static std::uniform_int_distribution<long long> dis_int(-kDenominator,
                                                            kDenominator);
    while (random_parameters_.size() < num_params) {
      random_parameters_.emplace_back(dis_int(gen), kDenominator);
    }
#else
    static const ParamType pi = std::acos((ParamType)-1.0);
    static std::uniform_real_distribution<ParamType> dis_real(-pi, pi);
    while (random_parameters_.size() < num_params) {
      random_parameters_.emplace_back(dis_real(gen));
    }
#endif
  }
}

std::vector<ParamType> ParamInfo::get_all_generated_parameters() const {
  return random_parameters_;
}

ParamType ParamInfo::get_param_value(int id) const {
  assert(id >= 0 && id < (int)parameter_class_.size());
  assert(parameter_class_[id].is_const());
  return parameter_values_[id];
}

std::string ParamInfo::get_param_symbolic_string(int id, int precision,
                                                 int operator_precedence,
                                                 bool is_param_halved) const {
  assert(id >= 0 && id < (int)parameter_class_.size());
  assert(parameter_class_[id].is_const());
  if (parameter_class_[id] == ParamClass::concrete_const) {
    std::ostringstream out;
    out.precision(precision);
    out << parameter_values_[id] * (ParamType)(is_param_halved ? 2 : 1);
    return out.str();
  } else if (parameter_class_[id] == ParamClass::arithmetic_int) {
    return std::to_string((long long)parameter_values_[id] *
                          (is_param_halved ? 2 : 1));
  }
  assert(parameter_class_[id] == ParamClass::symbolic_constexpr);
  auto op = parameter_wires_[id]->input_gates[0]->gate->tp;
  int current_precedence = -1;
  if (op == GateType::add) {
    current_precedence = 1;
  } else if (op == GateType::mult || op == GateType::pi ||
             op == GateType::neg) {
    current_precedence = 2;
  } else {
    assert(false);
  }
  std::vector<std::string> child_str;
  child_str.reserve(parameter_wires_[id]->input_gates[0]->input_wires.size());
  for (auto &input_wire : parameter_wires_[id]->input_gates[0]->input_wires) {
    // Only pass the flag |is_param_halved| if the operator is +, -, or * only
    // for RHS
    // 2*(a+b) ==> 2*a + 2*b
    // 2*(-a) ==> -2*a
    // 2*(a*b) ==> a * (2*b)
    child_str.emplace_back(ParamInfo::get_param_symbolic_string(
        input_wire->index, precision, current_precedence,
        is_param_halved && (op == GateType::add || op == GateType::neg ||
                            (op == GateType::mult && child_str.size() == 1))));
  }
  std::string result;
  if (op == GateType::add) {
    result = child_str[0] + "+" + child_str[1];
  } else if (op == GateType::mult) {
    result = child_str[0] + "*" + child_str[1];
  } else if (op == GateType::pi) {
    if (is_param_halved) {
      if (parameter_class_[parameter_wires_[id]
                               ->input_gates[0]
                               ->input_wires[0]
                               ->index] == ParamClass::arithmetic_int &&
          (long long)parameter_values_[parameter_wires_[id]
                                           ->input_gates[0]
                                           ->input_wires[0]
                                           ->index] %
                  2 ==
              0) {
        // 2*pi/6 ==> pi/3
        return "pi/" +
               std::to_string((long long)parameter_values_[parameter_wires_[id]
                                                               ->input_gates[0]
                                                               ->input_wires[0]
                                                               ->index] /
                              2);
      } else {
        result = "2*pi/" + child_str[0];
      }
    } else {
      result = "pi/" + child_str[0];
    }
  } else if (op == GateType::neg) {
    result = "-" + child_str[0];
  }
  if (current_precedence < operator_precedence) {
    result = "(" + result + ")";
  }
  return result;
}

void ParamInfo::set_param_value(int id, const ParamType &param) {
  assert(id >= 0 && id < (int)parameter_class_.size());
  assert(parameter_class_[id] == ParamClass::concrete_const ||
         parameter_class_[id] == ParamClass::arithmetic_int);
  while (id >= (int)parameter_values_.size()) {
    parameter_values_.emplace_back();
  }
  parameter_values_[id] = param;
}

std::vector<ParamType> ParamInfo::get_all_input_param_values() const {
  return parameter_values_;
}

int ParamInfo::get_new_param_id(const ParamType &param) {
  int id = (int)parameter_class_.size();
  assert(id == (int)parameter_class_.size());
  parameter_class_.emplace_back(ParamClass::concrete_const);
  auto wire = std::make_unique<CircuitWire>();
  wire->type = CircuitWire::input_param;
  wire->index = id;
  parameter_wires_.push_back(std::move(wire));
  set_param_value(id, param);
  return id;
}

int ParamInfo::get_new_arithmetic_param_id(const ParamType &param) {
  int id = (int)parameter_class_.size();
  assert(id == (int)parameter_class_.size());
#ifdef USE_RATIONAL
  assert(param.denominator() == Int(1) &&
         "Did you set use_symbolic_pi to true but used floating-point "
         "expressions like pi*0.5?");
#else
  assert((int)param == param &&
         "Did you set use_symbolic_pi to true but used floating-point "
         "expressions like pi*0.5?");
#endif
  parameter_class_.emplace_back(ParamClass::arithmetic_int);
  auto wire = std::make_unique<CircuitWire>();
  wire->type = CircuitWire::input_param;
  wire->index = id;
  parameter_wires_.push_back(std::move(wire));
  set_param_value(id, param);
  return id;
}

int ParamInfo::get_new_param_id(bool is_halved) {
  int id = (int)parameter_class_.size();
  parameter_class_.emplace_back(is_halved ? ParamClass::symbolic_halved
                                          : ParamClass::symbolic);
  // Make sure to generate a random parameter for each symbolic parameter.
  gen_random_parameters(id + 1);
  auto wire = std::make_unique<CircuitWire>();
  wire->type = CircuitWire::input_param;
  wire->index = id;
  parameter_wires_.push_back(std::move(wire));
  return id;
}

int ParamInfo::get_new_param_expression_id(
    const std::vector<int> &parameter_indices, Gate *op) {
  bool is_symbolic = is_symbolic_constant(op);
  bool is_const = true;
  for (auto &input_id : parameter_indices) {
    assert(input_id >= 0 && input_id < (int)parameter_class_.size());
    if (parameter_class_[input_id].is_symbolic()) {
      is_symbolic = true;
    }
    if (!parameter_class_[input_id].is_const()) {
      is_const = false;
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
  int id = (int)parameter_class_.size();
  parameter_class_.emplace_back(is_const ? ParamClass::symbolic_constexpr
                                         : ParamClass::expression);
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
  return (int)parameter_class_.size();
}

int ParamInfo::get_num_input_symbolic_parameters() const {
  return (int)random_parameters_.size();
}

bool ParamInfo::param_is_symbolic(int id) const {
  return id >= 0 && id < (int)parameter_class_.size() &&
         parameter_class_[id].is_symbolic();
}

bool ParamInfo::param_is_const(int id) const {
  return id >= 0 && id < (int)parameter_class_.size() &&
         parameter_class_[id].is_const();
}

bool ParamInfo::param_is_expression(int id) const {
  return id >= 0 && id < (int)parameter_wires_.size() &&
         !parameter_wires_[id]->input_gates.empty();
}

bool ParamInfo::param_is_halved(int id) const {
  return id >= 0 && id < (int)parameter_class_.size() &&
         parameter_class_[id] == ParamClass::symbolic_halved;
  // TODO: halved parameter expressions
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
  // Creates a param list, assuming that all symbolic params are defined first.
  auto result = input_parameters;
  result.resize(parameter_class_.size());
  // Populates constant parameters.
  for (int i = 0; i < result.size(); ++i) {
    if (parameter_class_[i].is_const() &&
        parameter_class_[i] != ParamClass::symbolic_constexpr) {
      result[i] = parameter_values_[i];
    }
  }
  // Populates expression parameters.
  for (auto &expr : parameter_expressions_) {
    std::vector<ParamType> params;
    for (const auto &input_wire : expr->input_wires) {
      params.push_back(result[input_wire->index]);
    }
    assert(expr->output_wires.size() == 1);
    const auto &output_wire = expr->output_wires[0];
    result[output_wire->index] = expr->gate->compute(params);
  }
  // Returns populated list.
  return result;
}

std::vector<InputParamMaskType> ParamInfo::get_param_masks() const {
  std::vector<InputParamMaskType> param_mask(parameter_class_.size());
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

std::string ParamInfo::to_json() const {
  std::string result = "[";
  result += "[";
  result += std::to_string(parameter_class_.size());
  for (int i = 0; i < (int)parameter_class_.size(); i++) {
    result += ", ";
    if (parameter_class_[i] == ParamClass::arithmetic_int) {
      // arithmetic int
      result += std::to_string((long long)parameter_values_[i]);
    } else if (parameter_class_[i].is_input()) {
      if (parameter_class_[i].is_symbolic()) {
        // input symbolic
        result += "\"\"";
        // TODO: halved parameter
      } else {
        // input concrete
        result += to_string_with_precision(parameter_values_[i],
                                           /*precision=*/17);
      }
    } else {
      // expression
      result += parameter_wires_[i]->input_gates[0]->to_json();
    }
  }
  result += "], ";
  result += to_json_style_string_with_precision(random_parameters_,
                                                /*precision=*/17);
  result += "]";
  return result;
}
}  // namespace quartz
