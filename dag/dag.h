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

  std::vector<std::unique_ptr<DAGNode>> nodes;
  std::vector<std::unique_ptr<DAGHyperEdge>> edges;
  // The gates' information is owned by edges.
  std::vector<DAGNode *> outputs;
  std::vector<DAGNode *> parameters;

  bool add_gate(const std::vector<int> &qubit_indices,
                const std::vector<int> &parameter_indices,
                Gate *gate,
                int *output_para_index);
  void add_input_parameter();
  bool remove_last_gate();

  // Return the total number of gates removed.
  // The time complexity is O((number of gates removed) *
  // ((total number of nodes) + (total number of edges))).
  int remove_gate(DAGHyperEdge *edge);
  bool evaluate(const Vector &input_dis,
                const std::vector<ParamType> &input_parameters,
                Vector &output_dis) const;
  [[nodiscard]] int get_num_qubits() const;
  [[nodiscard]] int get_num_input_parameters() const;
  [[nodiscard]] int get_num_total_parameters() const;
  [[nodiscard]] int get_num_gates() const;
  [[nodiscard]] bool qubit_used(int qubit_index) const;
  DAGHashType hash(Context *ctx);
  DAG &shrink_unused_input_parameters();
  [[nodiscard]] std::unique_ptr<DAG> clone_and_shrink_unused_input_parameters() const;
  [[nodiscard]] bool has_unused_parameter() const;
  void print(Context *ctx) const;
  [[nodiscard]] std::string to_string() const;
  [[nodiscard]] std::string to_json() const;
  static std::unique_ptr<DAG> read_json(Context *ctx, std::istream &fin);

  // Returns true iff the DAG is already under the minimal representation.
  // If |output| is true, store the minimal representation into |output_dag|,
  // and store the permutation of qubits and parameters into |qubit_permutation|
  // and |param_permutation|.
  // The parameter |output_dag| should be a pointer containing nullptr
  // (otherwise its content will be deleted),
  // and the parameters |qubit_permutation| and |param_permutation| being
  // nullptr means that the permutation is not required to store.
  // |qubit_permutation[0] == 1| means that the qubit Q0 in this DAG maps to
  // the qubit Q1 in the minimal representation.
  bool minimal_representation(std::unique_ptr<DAG> *output_dag,
                              std::vector<int> *qubit_permutation = nullptr,
                              std::vector<int> *param_permutation = nullptr,
                              bool output = true) const;
  [[nodiscard]] bool is_minimal_representation() const;
  [[nodiscard]] std::unique_ptr<DAG> get_permuted_dag(const std::vector<int> &qubit_permutation,
                                                      const std::vector<int> &param_permutation) const;

 private:
  void clone_from(const DAG &other,
                  const std::vector<int> &qubit_permutation,
                  const std::vector<int> &param_permutation);

  int num_qubits, num_input_parameters;
  DAGHashType hash_value_;
  bool hash_value_valid_;
};

class UniquePtrDAGComparator {
 public:
  bool operator()(const std::unique_ptr<DAG> &dag1,
                  const std::unique_ptr<DAG> &dag2) const {
    return dag1->less_than(*dag2);
  }
};
