#pragma once

#include "../gate/gate_utils.h"
#include "../math/vector.h"
#include "../utils/utils.h"

#include <unordered_map>
#include <vector>
#include <memory>

class DAG;

class Context {
 public:
  explicit Context(const std::vector<GateType> &supported_gates);
  Gate *get_gate(GateType tp);
  [[nodiscard]] const std::vector<GateType> &get_supported_gates() const;
  [[nodiscard]] const std::vector<GateType> &get_supported_parameter_gates() const;
  [[nodiscard]] const std::vector<GateType> &get_supported_quantum_gates() const;
  // Two deterministic (random) distributions for each number of qubits.
  const Vector &get_generated_input_dis(int num_qubits);
  const Vector &get_generated_hashing_dis(int num_qubits);
  std::vector<ParamType> get_generated_parameters(int num_params);
  size_t next_global_unique_id();
  bool has_representative(DAGHashType hash_value, int equiv_id) const;
  std::vector<DAG *> get_possible_representatives(DAG *dag);

  // Warning: this presumes that DAGs with the same hash value are equivalent.
  void set_representative(std::unique_ptr<DAG> dag);

  // This function is recommended.
  void set_representative(std::unique_ptr<DAG> dag, int equiv_id);

  void clear_representatives();

 private:
  bool insert_gate(GateType tp);
  size_t global_unique_id;
  std::unordered_map<GateType, std::unique_ptr<Gate>> gates_;
  std::vector<GateType> supported_gates_;
  std::vector<GateType> supported_parameter_gates_;
  std::vector<GateType> supported_quantum_gates_;
  std::vector<Vector> random_input_distribution_;
  std::vector<Vector> random_hashing_distribution_;
  std::vector<ParamType> random_parameters_;

  // A vector to store the representative DAGs.
  std::vector<std::unique_ptr<DAG>> representative_dags_;

  std::unordered_map<DAGHashType, std::vector<DAG *>> representatives_;
};
