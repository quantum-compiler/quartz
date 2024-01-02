#pragma once

#include "quartz/circuitseq/circuitwire.h"
#include "quartz/gate/gate_utils.h"
#include "quartz/math/vector.h"
#include "quartz/utils/utils.h"

#include <algorithm>
#include <memory>
#include <random>
#include <set>
#include <unordered_map>
#include <vector>

namespace quartz {
class CircuitSeq;
class CircuitGate;

class Context {
 public:
  explicit Context(const std::vector<GateType> &supported_gates);
  /**
   * Generate the random testing values for 2^|num_qubits| coefficients
   * and |num_input_symbolic_params| parameters.
   */
  Context(const std::vector<GateType> &supported_gates, int num_qubits,
          int num_input_symbolic_params);

  Gate *get_gate(GateType tp);
  Gate *get_general_controlled_gate(GateType tp,
                                    const std::vector<bool> &state);
  [[nodiscard]] const std::vector<GateType> &get_supported_gates() const;
  [[nodiscard]] const std::vector<GateType> &
  get_supported_parameter_ops() const;
  [[nodiscard]] const std::vector<GateType> &
  get_supported_quantum_gates() const;
  /**
   * Two deterministic (random) distributions for each number of qubits.
   */
  void gen_input_and_hashing_dis(int num_qubits);
  [[nodiscard]] const Vector &get_generated_input_dis(int num_qubits) const;
  [[nodiscard]] const Vector &get_generated_hashing_dis(int num_qubits) const;
  [[nodiscard]] std::vector<ParamType> get_all_generated_parameters() const;
  size_t next_global_unique_id();

  [[nodiscard]] bool has_parameterized_gate() const;

  // A hacky function: set a generated parameter.
  void set_generated_parameter(int id, ParamType param);

  // These three functions are used in |Verifier::redundant()| for a version
  // of RepGen algorithm that does not invoke Python verifier.
  void set_representative(std::unique_ptr<CircuitSeq> seq);
  void clear_representatives();
  bool get_possible_representative(const CircuitSeqHashType &hash_value,
                                   CircuitSeq *&representative) const;

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
   * Derive the permutation of all parameter expressions from an input
   * parameter permutation.
   * @param input_param_permutation The input parameter permutation.
   * @return The permutation of all parameters (expressions).
   */
  [[nodiscard]] std::vector<int>
  get_param_permutation(const std::vector<int> &input_param_permutation);

  /**
   * Generate all parameter expressions using existing parameters and
   * operators. After calling this function, no new parameters or parameter
   * expressions should be created in order for the generator to work properly.
   * @param max_num_operators_per_expression Restrict each parameter expression
   * to use at most |max_num_operators_per_expression| operators.
   * Currently we only support |max_num_operators_per_expression| = 1.
   */
  void generate_parameter_expressions(int max_num_operators_per_expression = 1);

  /**
   * Compute and return which input parameters are used in each of the
   * parameter expressions.
   * @return The mask for each parameter expression.
   */
  [[nodiscard]] std::vector<InputParamMaskType> get_param_masks() const;

  // This function generates a deterministic series of random numbers
  // ranging [0, 1].
  double random_number();

 private:
  bool insert_gate(GateType tp);

  /**
   * Generate random values for random testing for input symbolic parameters.
   * The results are stored in |random_parameters_|.
   * The size of |random_parameters_| should be equal to the number of input
   * symbolic parameters.
   * @param num_params The number of input symbolic parameters.
   */
  void gen_random_parameters(int num_params);

  size_t global_unique_id;
  std::unordered_map<GateType, std::unique_ptr<Gate>> gates_;
  std::unordered_map<
      GateType, std::unordered_map<std::vector<bool>, std::unique_ptr<Gate>>>
      general_controlled_gates_;
  std::vector<GateType> supported_gates_;
  std::vector<GateType> supported_parameter_ops_;
  std::vector<GateType> supported_quantum_gates_;
  std::vector<Vector> random_input_distribution_;
  std::vector<Vector> random_hashing_distribution_;
  std::vector<ParamType> random_parameters_;

  // A vector to store the representative circuit sequences.
  std::vector<std::unique_ptr<CircuitSeq>> representative_seqs_;
  std::unordered_map<CircuitSeqHashType, CircuitSeq *> representatives_;
  // Standard mersenne_twister_engine seeded with 0
  std::mt19937 gen{0};

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
};

// TODO: This function does not consider the parameters
Context union_contexts(Context *ctx_0, Context *ctx_1);

}  // namespace quartz
