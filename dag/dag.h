#pragma once

#include "dagnode.h"
#include "daghyperedge.h"
#include "../utils/utils.h"
#include "../gate/gate.h"
#include "../math/vector.h"

#include <istream>
#include <string>

class Context;

class DAG {
 public:
  DAG(int num_qubits, int num_input_parameters);
  DAG(const DAG &other);  // clone a DAG
  [[nodiscard]] std::unique_ptr<DAG> clone() const;
  [[nodiscard]] bool fully_equivalent(const DAG &other) const;
  [[nodiscard]] bool fully_equivalent(Context *ctx, DAG &other);
  [[nodiscard]] bool less_than(const DAG &other) const;

  bool add_gate(const std::vector<int> &qubit_indices,
                const std::vector<int> &parameter_indices,
                Gate *gate,
                int *output_para_index);
  void add_input_parameter();
  bool remove_last_gate();

  // Generate all possible parameter gates at the beginning.
  // TODO: Currently we only support |max_recursion_depth == 1|.
  void generate_parameter_gates(Context *ctx, int max_recursion_depth = 1);

  // Return the total number of gates removed.
  // The time complexity is O((number of gates removed) *
  // ((total number of nodes) + (total number of edges))).
  int remove_gate(DAGHyperEdge *edge);
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
  [[nodiscard]] bool qubit_used(int qubit_index) const;
  [[nodiscard]] bool input_param_used(int param_index) const;
  DAGHashType hash(Context *ctx);
  [[nodiscard]] bool hash_value_valid() const;
  [[nodiscard]] DAGHashType cached_hash_value() const;
  [[nodiscard]] std::vector<DAGHashType> other_hash_values() const;
  [[nodiscard]] std::vector<std::pair<DAGHashType,
                                      PhaseShiftIdType>> other_hash_values_with_phase_shift_id() const;

  // Remove the qubit set of |unused_qubits|, given that they are unused.
  // Returns false iff an error occurs.
  bool remove_unused_qubits(std::vector<int> unused_qubits);

  // Remove the parameter set of |unused_input_params|, given that they are
  // unused input parameters
  // Returns false iff an error occurs.
  bool remove_unused_input_params(std::vector<int> unused_input_params);

  // Remove a suffix of unused input parameters.
  DAG &shrink_unused_input_parameters();
  [[nodiscard]] std::unique_ptr<DAG> clone_and_shrink_unused_input_parameters() const;
  [[nodiscard]] bool has_unused_parameter() const;
  // Returns the number of internal parameters removed.
  int remove_unused_internal_parameters();
  void print(Context *ctx) const;
  [[nodiscard]] std::string to_string() const;
  [[nodiscard]] std::string to_json() const;
  static std::unique_ptr<DAG> read_json(Context *ctx, std::istream &fin);

  // Returns true iff the DAG is already under the minimal circuit
  // representation.
  // Minimal circuit representation is a sequence representation of a circuit
  // such that:
  // 1. The gates are ordered column by column. If we see each circuit as
  //    a grid of gates such that each row represents a qubit, and put each
  //    gate at the leftmost possible position, gate A is placed before gate B
  //    iff A is in a column before B or they are in the same column and
  //    the smallest qubit index of A is smaller than the smallest qubit index
  //    of B.
  // 2. The parameter "gates" are placed at the beginning.
  // If |output| is true, store the minimal circuit representation into
  // |output_dag|.
  // The parameter |output_dag| should be a pointer containing nullptr
  // (otherwise its content will be deleted).
  // This functions guarantees that if two sequence representations
  // share the same "minimal_circuit_representation", they have the same
  // circuit representation.
  bool minimal_circuit_representation(std::unique_ptr<DAG> *output_dag,
                                      bool output = true) const;
  [[nodiscard]] bool is_minimal_circuit_representation() const;
  [[nodiscard]] std::unique_ptr<DAG> get_permuted_dag(const std::vector<int> &qubit_permutation,
                                                      const std::vector<int> &param_permutation) const;

  // Returns quantum gates which do not topologically depend on any other
  // quantum gates.
  std::vector<DAGHyperEdge *> first_quantum_gates() const;
  // Returns quantum gates which can appear at last in some topological order
  // of the DAG.
  std::vector<DAGHyperEdge *> last_quantum_gates() const;

  static bool same_gate(const DAG &dag1,
                        int index1,
                        const DAG &dag2,
                        int index2);

  static bool same_gate(DAGHyperEdge *edge1, DAGHyperEdge *edge2);

 private:
  void clone_from(const DAG &other,
                  const std::vector<int> &qubit_permutation,
                  const std::vector<int> &param_permutation);

  // A helper function used by |DAGHashType hash(Context *ctx)|.
  static void generate_hash_values(Context *ctx,
                                   const ComplexType &hash_value,
                                   const PhaseShiftIdType &phase_shift_id,
                                   const std::vector<ParamType> &param_values,
                                   DAGHashType *main_hash,
                                   std::vector<std::pair<DAGHashType,
                                                         PhaseShiftIdType>> *other_hash);

 public:
  std::vector<std::unique_ptr<DAGNode>> nodes;
  std::vector<std::unique_ptr<DAGHyperEdge>> edges;
  // The gates' information is owned by edges.
  std::vector<DAGNode *> outputs;
  std::vector<DAGNode *> parameters;

 private:
  int num_qubits, num_input_parameters;
  DAGHashType hash_value_;
  // For both floating-point error tolerance
  // and equivalences under a phase shift.
  // The first component of the pair is the hash value,
  // and the second component is the id of the phase shifted.
  // For now, the id is hard-coded as follows:
  //   - |kNoPhaseShift|: no shift
  //   - p \in [0, get_num_total_parameters()):
  //       shifted by e^(i * (p-th parameter))
  //   - p \in [get_num_total_parameters(), 2 * get_num_total_parameters()):
  //       shifted by e^(-i * ((p - get_num_total_parameters())-th parameter))
  std::vector<std::pair<DAGHashType, PhaseShiftIdType>> other_hash_values_;
  ComplexType original_fingerprint_;
  bool hash_value_valid_;
};

class UniquePtrDAGComparator {
 public:
  bool operator()(const std::unique_ptr<DAG> &dag1,
                  const std::unique_ptr<DAG> &dag2) const {
    if (!dag1 || !dag2) {
      // nullptr
      return !dag2;
    }
    return dag1->less_than(*dag2);
  }
};
