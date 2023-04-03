#pragma once

#include "../gate/gate.h"
#include "../math/vector.h"
#include "../utils/utils.h"
#include "circuitgate.h"
#include "circuitwire.h"

#include <istream>
#include <string>

namespace quartz {

class Context;

class CircuitSeq {
public:
  CircuitSeq(int num_qubits, int num_input_parameters);
  CircuitSeq(const CircuitSeq &other); // clone a CircuitSeq
  [[nodiscard]] std::unique_ptr<CircuitSeq> clone() const;
  [[nodiscard]] bool fully_equivalent(const CircuitSeq &other) const;
  [[nodiscard]] bool fully_equivalent(Context *ctx, CircuitSeq &other);
  [[nodiscard]] bool less_than(const CircuitSeq &other) const;

  bool add_gate(const std::vector<int> &qubit_indices,
                const std::vector<int> &parameter_indices, Gate *gate,
                int *output_para_index);
  bool add_gate(CircuitGate *gate);
  // Insert a gate to any position of the circuit sequence.
  // Warning: remove_last_gate() cannot be called anymore after calling
  // insert_gate().
  bool insert_gate(int insert_position, const std::vector<int> &qubit_indices,
                   const std::vector<int> &parameter_indices, Gate *gate,
                   int *output_para_index);
  bool insert_gate(int insert_position, CircuitGate *gate);
  void add_input_parameter();
  bool remove_last_gate();

  // Generate all possible parameter gates at the beginning.
  // TODO: Currently we only support |max_recursion_depth == 1|.
  void generate_parameter_gates(Context *ctx, int max_recursion_depth = 1);

  // Return the total number of gates removed.
  // The time complexity is O((number of gates removed) *
  // ((total number of wires) + (total number of gates))).
  int remove_gate(CircuitGate *circuit_gate);
  int remove_first_quantum_gate();
  // Evaluate the output distribution given input distribution and
  // input parameters. Also output all parameter values (including input
  // and internal parameters) when |parameter_values| is not nullptr.
  bool evaluate(const Vector &input_dis,
                const std::vector<ParamType> &input_parameters,
                Vector &output_dis,
                std::vector<ParamType> *parameter_values = nullptr) const;
  [[nodiscard]] int get_num_qubits() const;
  [[nodiscard]] int get_num_input_parameters() const;
  [[nodiscard]] int get_num_total_parameters() const;
  [[nodiscard]] int get_num_internal_parameters() const;
  [[nodiscard]] int get_num_gates() const;
  [[nodiscard]] int get_circuit_depth() const;
  [[nodiscard]] ParamType get_parameter_value(Context *ctx, int para_idx) const;
  [[nodiscard]] bool qubit_used(int qubit_index) const;
  // Used by a parameter gate is considered as used here.
  [[nodiscard]] bool input_param_used(int param_index) const;
  // Returns a pair. The first component denotes the input parameters
  // already used in this CircuitSeq. The second component denotes the input
  // parameters used in each of the parameters in this CircuitSeq.
  [[nodiscard]] std::pair<InputParamMaskType, std::vector<InputParamMaskType>>
  get_input_param_mask() const;
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

  // Remove the parameter set of |unused_input_params|, given that they
  // are unused input parameters Returns false iff an error occurs.
  bool remove_unused_input_params(std::vector<int> unused_input_params);

  // Remove a suffix of unused input parameters.
  CircuitSeq &shrink_unused_input_parameters();
  [[nodiscard]] std::unique_ptr<CircuitSeq>
  clone_and_shrink_unused_input_parameters() const;
  [[nodiscard]] bool has_unused_parameter() const;
  // Returns the number of internal parameters removed.
  int remove_unused_internal_parameters();
  void print(Context *ctx) const;
  [[nodiscard]] std::string to_string(bool line_number = false) const;
  [[nodiscard]] std::string to_json() const;
  static std::unique_ptr<CircuitSeq> read_json(Context *ctx, std::istream &fin);
  static std::unique_ptr<CircuitSeq>
  from_qasm_file(Context *ctx, const std::string &filename);

  // Returns true iff the CircuitSeq is already under the canonical
  // representation.
  // Canonical representation is a sequence representation of a
  // circuit such that the sequence is the lexicographically smallest
  // topological order of the circuit, where the gates are compared by:
  // 1. The qubit indices in lexicographical order.
  //    If the smallest qubit indices two gates operate on
  //    are different, the gate with the smaller one is considered
  //    smaller. For example,
  //      (q1, q4) < (q2, q3);
  //      (q1, q3) < (q1, q4);
  //      (q1, q3) < (q2);
  //      (q1) < (q1, q3).
  //    Note that two gates operating on the same qubit is impossible
  //    for the topological order of one circuit, but we still define it
  //    for the completeness of comparing different circuits.
  // 2. If the qubit indices are all the same, compare the gate type.
  //
  // The parameter "gates" are placed at the beginning.
  //
  // If |output| is true, store the canonical representation into
  // |output_seq|.
  // The parameter |output_seq| should be a pointer containing nullptr
  // (otherwise its content will be deleted).
  // This functions guarantees that if and only if two sequence
  // representations share the same canonical representation, they have
  // the same circuit representation.
  bool canonical_representation(std::unique_ptr<CircuitSeq> *output_seq,
                                bool output = true) const;
  [[nodiscard]] bool is_canonical_representation() const;
  // Returns true iff this is NOT canonical representation
  // (so the function changes this CircuitSeq to canonical representation).
  bool to_canonical_representation();
  [[nodiscard]] std::unique_ptr<CircuitSeq>
  get_permuted_seq(const std::vector<int> &qubit_permutation,
                   const std::vector<int> &param_permutation) const;

  // Returns quantum gates which do not topologically depend on any other
  // quantum gates.
  std::vector<CircuitGate *> first_quantum_gates() const;
  // Returns quantum gates which can appear at last in some topological
  // order of the CircuitSeq.
  std::vector<CircuitGate *> last_quantum_gates() const;

  static bool same_gate(const CircuitSeq &seq1, int index1,
                        const CircuitSeq &seq2, int index2);

  static bool same_gate(CircuitGate *gate1, CircuitGate *gate2);

private:
  void clone_from(const CircuitSeq &other,
                  const std::vector<int> &qubit_permutation,
                  const std::vector<int> &param_permutation);

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
  std::vector<CircuitWire *> parameters;

private:
  int num_qubits, num_input_parameters;
  CircuitSeqHashType hash_value_;
  // For both floating-point error tolerance
  // and equivalences under a phase shift.
  // The first component of the pair is the hash value,
  // and the second component is the id of the phase shifted.
  // For now, the id is hard-coded as follows:
  //   - |kNoPhaseShift|: no shift
  //   - p \in [0, get_num_total_parameters()):
  //       shifted by e^(i * (p-th parameter))
  //   - p \in [get_num_total_parameters(), 2 *
  //   get_num_total_parameters()):
  //       shifted by e^(-i * ((p - get_num_total_parameters())-th
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
} // namespace quartz
