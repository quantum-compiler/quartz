#pragma once

#include "quartz/circuitseq/circuitgate.h"
#include "quartz/circuitseq/circuitwire.h"
#include "quartz/gate/gate_utils.h"
#include "quartz/utils/utils.h"

#include <random>
#include <vector>

namespace quartz {

class ParamInfo {
 public:
  /**
   * Default constructor: initialize 0 parameters.
   */
  ParamInfo() : ParamInfo(0) {}
  /**
   * Initialize |num_input_symbolic_params| input symbolic parameters.
   */
  explicit ParamInfo(int num_input_symbolic_params);

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
   * Create a new symbolic parameter.
   * @return The index of the new symbolic parameter.
   */
  int get_new_param_id();
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
  [[nodiscard]] bool param_has_value(int id) const;
  [[nodiscard]] bool param_is_expression(int id) const;

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

  // Each parameter can be either a concrete parameter (with a value),
  // an input symbolic parameter, or a symbolic parameter expression.
  // The concrete parameters are from the input QASM file,
  // written by QASMParser.
  // These three vectors should always have the same size.
  std::vector<ParamType> parameter_values_;
  std::vector<std::unique_ptr<CircuitWire>> parameter_wires_;
  std::vector<bool> is_parameter_symbolic_;
  // A holder for parameter expressions.
  std::vector<std::unique_ptr<CircuitGate>> parameter_expressions_;

  // For random testing.
  std::vector<ParamType> random_parameters_;
  // Standard mersenne_twister_engine seeded with 0 for random testing.
  std::mt19937 gen{0};
};
}  // namespace quartz
