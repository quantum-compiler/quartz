#pragma once

#include "../gate/gate.h"
#include "../math/vector.h"
#include "../utils/utils.h"
#include "circuitgate.h"
#include "circuitwire.h"

#include <functional>
#include <istream>
#include <string>
#include <unordered_set>

namespace quartz {

class Context;

class CircuitSeq {
 public:
  // TODO: Input parameters should be handled in Context instead of here
  explicit CircuitSeq(int num_qubits);
  CircuitSeq(const CircuitSeq &other);  // clone a CircuitSeq
  [[nodiscard]] std::unique_ptr<CircuitSeq> clone() const;
  /**
   * Compare if two circuit sequences are fully equivalent except for the hash
   * value.
   * @param other The other circuit sequence to be compared.
   * @return True iff two circuit sequences are fully equivalent.
   */
  [[nodiscard]] bool fully_equivalent(const CircuitSeq &other) const;
  /**
   * Compare if two circuits are topologically equivalent.
   * X(Q0) X(Q1) and X(Q1) X(Q0) are topologically equivalent but not
   * fully equivalent.
   * @param other The other circuit sequence to be compared.
   * @return True iff two circuit sequences are topologically equivalent.
   */
  [[nodiscard]] bool topologically_equivalent(const CircuitSeq &other) const;
  /**
   * Compute the hash value and compare if two circuit sequences are fully
   * equivalent including the hash value.
   * @param ctx The context to compute the hash value.
   * @param other The other circuit sequence to be compared.
   * @return True iff two circuit sequences are fully equivalent.
   */
  [[nodiscard]] bool fully_equivalent(Context *ctx, CircuitSeq &other);
  /**
   * Compare two circuit sequences first by the qubit count (fewer is less),
   * then by the gate count (fewer is less), then by the gate sequence.
   * If |kUseRowRepresentationToCompare| is true, compare the gates on qubit 0
   * first (fewer is less, then compare by the content), then qubit 1, ...
   * @param other The other circuit sequence to compare with.
   * @return True iff this circuit sequence is strictly less than the other
   * circuit sequence.
   */
  [[nodiscard]] bool less_than(const CircuitSeq &other) const;

  /**
   * Add a gate at the end of the circuit sequence.
   * @param qubit_indices The qubit indices of the gate.
   * @param parameter_indices The parameter indices of the gate.
   * @param gate The gate type.
   * @param ctx The context for parameters.
   * @return True iff the insertion is successful.
   */
  bool add_gate(const std::vector<int> &qubit_indices,
                const std::vector<int> &parameter_indices, Gate *gate,
                const Context *ctx);
  /**
   * Add a gate at the end of the circuit sequence.
   * @param gate The gate object. A gate with the same type, the same qubit
   * indices, and the same parameter indices will be inserted.
   * @param ctx The context for parameters.
   * @return True iff the insertion is successful.
   */
  bool add_gate(CircuitGate *gate, const Context *ctx);
  /**
   * Insert a gate to any position of the circuit sequence.
   * Warning: remove_last_gate() cannot be called anymore after calling
   * insert_gate().
   * @param insert_position The gate position to insert.
   * @param qubit_indices The qubit indices of the gate.
   * @param parameter_indices The parameter indices of the gate.
   * @param gate The gate type.
   * @param ctx The context for parameters.
   * @return True iff the insertion is successful.
   */
  bool insert_gate(int insert_position, const std::vector<int> &qubit_indices,
                   const std::vector<int> &parameter_indices, Gate *gate,
                   const Context *ctx);
  /**
   * Insert a gate to any position of the circuit sequence.
   * Warning: remove_last_gate() cannot be called anymore after calling
   * insert_gate().
   * @param insert_position The gate position to insert.
   * @param gate The gate object. A gate with the same type, the same qubit
   * indices, and the same parameter indices will be inserted.
   * @param ctx The context for parameters.
   * @return True iff the insertion is successful.
   */
  bool insert_gate(int insert_position, CircuitGate *gate, const Context *ctx);
  /**
   * Remove the last gate, assuming add_gate() was just called.
   * @return True iff the removal is successful.
   */
  bool remove_last_gate();

  /**
   * Remove a quantum gate in O(|get_num_gates()| - |gate_position|).
   * @param gate_position The position of the gate to be removed (0-indexed).
   * @return True iff the removal is successful.
   */
  bool remove_gate(int gate_position);
  /**
   * Remove a quantum gate in O(|get_num_gates()|).
   * @param circuit_gate The gate to be removed.
   * @return True iff the removal is successful.
   */
  bool remove_gate(CircuitGate *circuit_gate);
  /**
   * Remove a quantum gate in O(|get_num_gates()| - |gate_position|).
   * @param circuit_gate The gate to be removed.
   * @return True iff the removal is successful.
   */
  bool remove_gate_near_end(CircuitGate *circuit_gate);
  /**
   * Remove the first quantum gate (if there is one).
   * @return True iff the removal is successful.
   */
  bool remove_first_quantum_gate();
  /**
   * Remove all swap gates, adjusting logical qubit indices correspondingly.
   * The time complexity is
   * O((total number of gates) + (total number of wires)) (linear).
   * @return The number of gates removed.
   */
  int remove_swap_gates();
  /**
   * Evaluate the output distribution given input distribution and parameters.
   * @param input_dis The input distribution.
   * @param parameter_values All parameter values, computed by
   * Context::compute_parameters().
   * @param output_dis The output distribution to write to.
   * @return True iff the evaluation is successful.
   */
  bool evaluate(const Vector &input_dis,
                const std::vector<ParamType> &parameter_values,
                Vector &output_dis) const;
  [[nodiscard]] int get_num_qubits() const;
  [[nodiscard]] int get_num_gates() const;
  [[nodiscard]] int get_circuit_depth() const;
  [[nodiscard]] static ParamType get_parameter_value(Context *ctx,
                                                     int para_idx);
  [[nodiscard]] bool qubit_used(int qubit_index) const;
  /**
   * Returns the input parameters used in this CircuitSeq as a mask.
   * @param param_masks The result of |Context::get_param_masks()|.
   */
  [[nodiscard]] InputParamMaskType get_input_param_usage_mask(
      const std::vector<InputParamMaskType> &param_masks) const;
  /**
   * Returns the input parameters used in this CircuitSeq (sorted).
   */
  [[nodiscard]] std::vector<int> get_input_param_indices(Context *ctx) const;
  /**
   * Returns all parameter (expression) indices directly used used in this
   * CircuitSeq (sorted).
   */
  [[nodiscard]] std::vector<int> get_directly_used_param_indices() const;
  /**
   * Returns all operations needed for all parameter expressions used in this
   * CircuitSeq (in a topological order).
   */
  [[nodiscard]] std::vector<CircuitGate *>
  get_param_expr_ops(Context *ctx) const;
  CircuitSeqHashType hash(Context *ctx);
  // Evaluate the output distribution 2^|num_qubits| times, with the i-th
  // time the input distribution being a vector with only the i-th entry
  // equals to 1 and all other entries equal to 0.
  [[nodiscard]] std::vector<Vector> get_matrix(Context *ctx) const;
  [[nodiscard]] bool hash_value_valid() const;
  [[nodiscard]] CircuitSeqHashType cached_hash_value() const;
  [[nodiscard]] std::vector<CircuitSeqHashType> other_hash_values() const;
  [[nodiscard]] std::vector<std::pair<CircuitSeqHashType, PhaseShiftIdType>>
  other_hash_values_with_phase_shift_id() const;

  // Remove the qubit set of |unused_qubits|, given that they are unused.
  // Returns false iff an error occurs.
  bool remove_unused_qubits(std::vector<int> unused_qubits);

  void print(Context *ctx) const;
  [[nodiscard]] std::string to_string(bool line_number = false) const;
  [[nodiscard]] std::string to_json(bool keep_hash_value = true) const;
  static std::unique_ptr<CircuitSeq> read_json(Context *ctx, std::istream &fin);
  static std::unique_ptr<CircuitSeq>
  from_qasm_file(Context *ctx, const std::string &filename);
  static std::unique_ptr<CircuitSeq>
  from_qasm_style_string(Context *ctx, const std::string &str);
  std::string
  to_qasm_style_string(Context *ctx,
                       int param_precision = kDefaultQASMParamPrecision) const;
  bool to_qasm_file(Context *ctx, const std::string &filename,
                    int param_precision = kDefaultQASMParamPrecision) const;

  /**
   * Canonical representation is a sequence representation of a
   * circuit such that the sequence is the lexicographically smallest
   * topological order of the circuit, where the gates are compared by:
   * 1. The qubit indices in lexicographical order.
   *    If the smallest qubit indices two gates operate on
   *    are different, the gate with the smaller one is considered
   *    smaller. For example,
   *      (q1, q4) < (q2, q3);
   *      (q1, q3) < (q1, q4);
   *      (q1, q3) < (q2);
   *      (q1) < (q1, q3).
   *    Note that two gates operating on the same qubit is impossible
   *    for the topological order of one circuit, but we still define it
   *    for the completeness of comparing different circuits.
   * 2. If the qubit indices are all the same, compare the gate type.
   *
   * The parameter "gates" are placed at the beginning.
   *
   * This functions guarantees that if and only if two sequence
   * representations share the same canonical representation, they have
   * the same circuit representation.
   *
   * @param output_seq If |output| is true, store the canonical representation
   * into |output_seq|.
   * The parameter |output_seq| should be a pointer containing nullptr
   * (otherwise its content will be deleted).
   * @param ctx The context to construct the canonical representation.
   * Only when |output| is false, it is OK to pass in a nullptr here.
   * @param output Whether to output the canonical representation. Default is
   * true.
   * @return True iff the CircuitSeq is already under the canonical
   * representation.
   */
  bool canonical_representation(std::unique_ptr<CircuitSeq> *output_seq,
                                const Context *ctx, bool output = true) const;
  [[nodiscard]] bool is_canonical_representation() const;
  /**
   * Convert this CircuitSeq to canonical representation.
   * @param ctx The context to construct the canonical representation.
   * @return True iff this is NOT canonical representation
   * (so the function modifies this CircuitSeq).
   */
  bool to_canonical_representation(const Context *ctx);

  /**
   * Permute the quantum gates. This function topologically sorts
   * the sequence and picks one quantum gate among all choices
   * each time.
   * @param ctx The context to construct the new circuit.
   * @param gate_chooser The function used to pick the quantum gate to be
   * placed the first each time, invoked the same number of times as the number
   * of quantum gates. This function takes as input an std::vector of
   * potential quantum gates, and returns the index of the quantum gate to be
   * placed first.
   * Default (nullptr) is uniformly random.
   * @param result_permutation Store the result permutation in this array
   * if it is not nullptr. Requires the size to be at least the number of
   * quantum gates if not nullptr. The behavior is undefined if there is
   * a non-quantum gate and this parameter is not nullptr. Default is nullptr.
   * @return The permuted circuit sequence.
   */
  [[nodiscard]] std::unique_ptr<CircuitSeq> get_gate_permutation(
      const Context *ctx,
      const std::function<int(const std::vector<CircuitGate *> &)>
          &gate_chooser = nullptr,
      int *result_permutation = nullptr) const;
  /**
   * Permute the qubits and input parameters.
   * @param qubit_permutation The qubit permutation. The size must be the
   * same as the number of qubits.
   * @param input_param_permutation The input parameter permutation. If the size
   * is smaller than the total number of input parameters, this function only
   * permutes a prefix of input parameters corresponding to |param_permutation|.
   * @param ctx The context, only needed when |input_param_permutation| is not
   * empty. When |input_param_permutation| is empty, it is safe to pass in a
   * nullptr.
   * @return The permuted circuit sequence.
   */
  [[nodiscard]] std::unique_ptr<CircuitSeq>
  get_permuted_seq(const std::vector<int> &qubit_permutation,
                   const std::vector<int> &input_param_permutation,
                   Context *ctx) const;
  /**
   * Get a circuit with |start_gates| and all gates topologically after them.
   * @param start_gates The first gates at each qubit to include in the
   * circuit to return.
   */
  [[nodiscard]] std::unique_ptr<CircuitSeq>
  get_suffix_seq(const std::unordered_set<CircuitGate *> &start_gates,
                 Context *ctx) const;

  /**
   * Get a circuit which replaces RZ gates with T, Tdg, S, Sdg, and Z gates.
   * Requires all Rz gates to have parameters that are multiples of Pi/4.
   */
  std::unique_ptr<CircuitSeq> get_rz_to_t(Context *ctx) const;

  /**
   * Get a circuit which replaces CCZ gates with CX and RZ gates.
   * Use only one decomposition:
   * ccz q0 q1 q2 = cx q1 q2; rz q2 -0.25pi; cx q0 q2; rz q2 0.25pi; cx q1 q2;
   * rz q2 -0.25pi; cx q0 q2; cx q0 q1; rz q1 -0.25pi; cx q0 q1; rz q0 0.25pi;
   * rz q1 0.25pi; rz q2 0.25pi;
   */
  std::unique_ptr<CircuitSeq> get_ccz_to_cx_rz(Context *ctx) const;

  /**
   * Returns quantum gates which do not topologically depend on any other
   * quantum gates.
   * @return The pointers to the first quantum gates.
   */
  [[nodiscard]] std::vector<CircuitGate *> first_quantum_gates() const;
  /**
   * Returns quantum gates which do not topologically depend on any other
   * quantum gates.
   * @return The positions (0-indexed) of the first quantum gates.
   */
  [[nodiscard]] std::vector<int> first_quantum_gate_positions() const;
  /**
   * Check if a quantum gate can appear at last in some topological
   * order of the CircuitSeq.
   * @param circuit_gate The pointer to a quantum gate in the circuit.
   * @return True iff the gate can appear at last in some topological
   * order of the CircuitSeq.
   */
  [[nodiscard]] bool is_one_of_last_gates(CircuitGate *circuit_gate) const;
  /**
   * Returns quantum gates which can appear at last in some topological
   * order of the CircuitSeq.
   * @return The pointers to the last quantum gates.
   */
  [[nodiscard]] std::vector<CircuitGate *> last_quantum_gates() const;

  static bool same_gate(const CircuitSeq &seq1, int index1,
                        const CircuitSeq &seq2, int index2);

  static bool same_gate(CircuitGate *gate1, CircuitGate *gate2);

 private:
  /**
   * Clone the circuit from another circuit sequence.
   * @param other The source circuit sequence.
   * @param qubit_permutation The qubit permutation (optional).
   * If not empty, the size must be the same as the number of qubits.
   * @param param_permutation The parameter permutation (optional).
   * If empty, |ctx| is not used and all parameters will be from the same
   * context as |other|.
   * @param ctx The context, only needed when |param_permutation| is not empty.
   * When |param_permutation| is empty, it is safe to pass in a nullptr.
   * @return The permuted circuit sequence.
   */
  void clone_from(const CircuitSeq &other,
                  const std::vector<int> &qubit_permutation,
                  const std::vector<int> &param_permutation,
                  const Context *ctx);

  /**
   * Remove a quantum gate from the graph, remove its output wires by default,
   * adjust the pointers, but do not remove the gate from |gates|.
   * The time complexity is
   * O((number of wires removed) * (total number of wires)) by default,
   * or O(number of wires of the gate) if the wires are not removed.
   * @param circuit_gate The gate to be removed.
   * @param assert_no_logical_qubit_permutation If this variable is true (by
   * default), assert that removing this gate will not cause future qubit
   * indices being swapped.
   * @param output_wires_to_be_removed If this variable is nullptr, remove
   * the output wires; otherwise, store the output wires to be removed into
   * this set.
   */
  void remove_quantum_gate_from_graph(
      CircuitGate *circuit_gate,
      bool assert_no_logical_qubit_permutation = true,
      std::unordered_set<CircuitWire *> *output_wires_to_be_removed = nullptr);

  // A helper function used by |CircuitSeqHashType hash(Context *ctx)|.
  static void generate_hash_values(
      Context *ctx, const ComplexType &hash_value,
      const PhaseShiftIdType &phase_shift_id,
      const std::vector<ParamType> &param_values, CircuitSeqHashType *main_hash,
      std::vector<std::pair<CircuitSeqHashType, PhaseShiftIdType>> *other_hash);

 public:
  std::vector<std::unique_ptr<CircuitWire>> wires;
  std::vector<std::unique_ptr<CircuitGate>> gates;
  std::vector<CircuitWire *> outputs;

 private:
  int num_qubits;
  CircuitSeqHashType hash_value_;
  // For both floating-point error tolerance
  // and equivalences under a phase shift.
  // The first component of the pair is the hash value,
  // and the second component is the ID of the phase shifted.
  // For now, the ID is hard-coded as follows:
  //   - |kNoPhaseShift|: no shift
  //   - p \in [0, ctx->get_num_parameters()):
  //       shifted by e^(i * (p-th parameter))
  //   - p \in [ctx->get_num_parameters(), 2 *
  //   ctx->get_num_parameters()):
  //       shifted by e^(-i * ((p - ctx->get_num_parameters())-th
  //       parameter))
  std::vector<std::pair<CircuitSeqHashType, PhaseShiftIdType>>
      other_hash_values_;
  ComplexType original_fingerprint_;
  bool hash_value_valid_;
};

class UniquePtrCircuitSeqComparator {
 public:
  bool operator()(const std::unique_ptr<CircuitSeq> &seq1,
                  const std::unique_ptr<CircuitSeq> &seq2) const {
    if (!seq1 || !seq2) {
      // nullptr
      return !seq2;
    }
    return seq1->less_than(*seq2);
  }
};
}  // namespace quartz
