#pragma once

#include "quartz/circuitseq/circuitgate.h"
#include "quartz/circuitseq/circuitwire.h"
#include "quartz/gate/gate_utils.h"
#include "quartz/utils/utils.h"

#include <random>
#include <vector>

namespace quartz {

/**
 * Determines if a constant expression should be treated as a symbolic value.
 */
bool is_symbolic_constant(Gate *op);

/**
 * The class a parameter belongs. See ParamClass::Value for each case.
 */
class ParamClass {
 public:
  enum Value : uint8_t {
    symbolic,         // an input symbolic parameter like theta
    symbolic_halved,  // an input symbolic parameter like theta/2, when loading
                      // circuits the parameter is halved, i.e., if we input 1.2
                      // we will store 0.6; when printing circuits the parameter
                      // is doubled, i.e., if we store 0.6 we will print 1.2.
                      // This is for gates with matrices including theta/2. See
                      // also Gate::is_param_halved(int).
    concrete_const,   // a concrete, floating-point parameter (with a value)
                      // like 1.2, from the input OpenQASM file written by
                      // QASMParser
    symbolic_constexpr,  // an at-least-partially symbolic constant expression
                         // like pi/2+1.2. Fully concrete constant expressions
                         // like 1.2+0.1 are expanded directly to 1.3.
    expression,          // an expression involving at least one input symbolic
                         // parameter like theta*2+1.2
    arithmetic_int,      // only used for arithmetic expressions but not quantum
                         // gates, must be an integer like 2
  };
  ParamClass() = default;
  constexpr ParamClass(Value v) : value_(v) {}
  explicit operator bool() const = delete;
  constexpr bool operator==(ParamClass a) const { return value_ == a.value_; }
  constexpr bool operator!=(ParamClass a) const { return value_ != a.value_; }
  /**
   * @return True iff the parameter has a known constant value.
   */
  [[nodiscard]] constexpr bool is_const() const {
    return value_ == concrete_const || value_ == symbolic_constexpr ||
           value_ == arithmetic_int;
  }
  /**
   * @return True iff the parameter is or contains anything symbolic
   * (including pi).
   */
  [[nodiscard]] constexpr bool is_symbolic() const {
    return value_ == symbolic || value_ == symbolic_halved ||
           value_ == expression || value_ == symbolic_constexpr;
  }
  /**
   * @return True iff the parameter is an input symbolic or concrete parameter.
   */
  [[nodiscard]] constexpr bool is_input() const {
    return value_ == symbolic || value_ == symbolic_halved ||
           value_ == concrete_const;
  }

 private:
  Value value_;
};

class ParamInfo {
 public:
  /**
   * Default constructor: initialize 0 parameters.
   */
  ParamInfo() : ParamInfo(0, false) {}
  /**
   * Initialize |num_input_symbolic_params| input symbolic parameters.
   */
  explicit ParamInfo(int num_input_symbolic_params, bool is_halved);

  /**
   * Generate random values for random testing for input symbolic parameters.
   * The results are stored in |random_parameters_|.
   * The size of |random_parameters_| should be equal to the number of input
   * symbolic parameters.
   * @param num_params The number of input symbolic parameters.
   */
  void gen_random_parameters(int num_params);

  [[nodiscard]] std::vector<ParamType> get_all_generated_parameters() const;

  /**
   * Get the value of a concrete parameter.
   */
  [[nodiscard]] ParamType get_param_value(int id) const;
  /**
   * Get the value of a symbolic constant expression.
   * @param id The expression id.
   * @param precision The precision of concrete constant parameters in the
   * returned string.
   * @param operator_precedence The operator precedence from outside.
   * If the inner precedence is strictly lower (in number) than the outer one,
   * then we must add parentheses.
   * 0: no need to add parentheses outside
   * 1: add
   * 2: mult, pi, neg  // expressions like PI/4*-3 are allowed
   * @param is_param_halved
   * If true, the result is the stored value multiplied by 2.
   * @return The string for the symbolic constant expression.
   */
  [[nodiscard]] std::string
  get_param_symbolic_string(int id, int precision,
                            int operator_precedence, bool is_param_halved) const;
  /**
   * Set the value of a concrete parameter.
   */
  void set_param_value(int id, const ParamType &param);
  /**
   * A convenient method to return all concrete parameter values.
   * @return A vector such that the |i|-th index stores the value of the
   * parameter with index |i|.
   */
  [[nodiscard]] std::vector<ParamType> get_all_input_param_values() const;
  /**
   * Create a new concrete parameter.
   * @return The index of the new concrete parameter.
   */
  int get_new_param_id(const ParamType &param);
  /**
   * Create a new concrete (integer) parameter that can only be used in
   * arithmetic expressions.
   * @return The index of the new concrete parameter.
   */
  int get_new_arithmetic_param_id(const ParamType &param);
  /**
   * Create a new symbolic parameter.
   * @param is_halved If true, then used by a gate with period 4*pi.
   * @return The index of the new symbolic parameter.
   */
  int get_new_param_id(bool is_halved);
  /**
   * Create a new parameter expression. If all input parameters are concrete,
   * compute the result directly instead of creating the expression.
   * @param parameter_indices The input parameter indices.
   * @param op The operator of the parameter expression.
   * @return The index of the new parameter expression.
   */
  int get_new_param_expression_id(const std::vector<int> &parameter_indices,
                                  Gate *op);
  [[nodiscard]] int get_num_parameters() const;
  [[nodiscard]] int get_num_input_symbolic_parameters() const;
  [[nodiscard]] bool param_is_symbolic(int id) const;
  [[nodiscard]] bool param_is_const(int id) const;
  [[nodiscard]] bool param_is_expression(int id) const;
  [[nodiscard]] bool param_is_halved(int id) const;

  [[nodiscard]] CircuitWire *get_param_wire(int id) const;

  /**
   * Compute all parameters given the input symbolic parameter values.
   * @param input_parameters The input symbolic parameter values.
   * @return A vector containing all parameters.
   */
  [[nodiscard]] std::vector<ParamType>
  compute_parameters(const std::vector<ParamType> &input_parameters);

  /**
   * Compute and return which input parameters are used in each of the
   * parameter expressions.
   * @return The mask for each parameter expression.
   */
  [[nodiscard]] std::vector<InputParamMaskType> get_param_masks() const;

  [[nodiscard]] std::string to_json() const;

  // These three vectors should always have the same size. Each entry represents
  // a parameter.
  std::vector<ParamType> parameter_values_;
  std::vector<std::unique_ptr<CircuitWire>> parameter_wires_;
  std::vector<ParamClass> parameter_class_;

  std::vector<bool> is_parameter_symbolic_;
  std::vector<bool> is_parameter_halved_;
  // A holder for parameter expressions.
  std::vector<std::unique_ptr<CircuitGate>> parameter_expressions_;

  // For random testing.
  std::vector<ParamType> random_parameters_;
  // Standard mersenne_twister_engine seeded with 0 for random testing.
  std::mt19937 gen{0};
};
}  // namespace quartz
