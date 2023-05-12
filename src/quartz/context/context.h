#pragma once

#include "../gate/gate_utils.h"
#include "../math/vector.h"
#include "../utils/utils.h"

#include <algorithm>
#include <memory>
#include <random>
#include <set>
#include <unordered_map>
#include <vector>

namespace quartz {
class CircuitSeq;

class Context {
public:
  explicit Context(const std::vector<GateType> &supported_gates);
  Context(const std::vector<GateType> &supported_gates, const int num_qubits,
          const int num_params);
  Gate *get_gate(GateType tp);
  [[nodiscard]] const std::vector<GateType> &get_supported_gates() const;
  [[nodiscard]] const std::vector<GateType> &
  get_supported_parameter_gates() const;
  [[nodiscard]] const std::vector<GateType> &
  get_supported_quantum_gates() const;
  // Two deterministic (random) distributions for each number of qubits.
  const Vector &get_and_gen_input_dis(int num_qubits);
  const Vector &get_and_gen_hashing_dis(int num_qubits);
  std::vector<ParamType> get_and_gen_parameters(int num_params);
  const Vector &get_generated_input_dis(int num_qubits) const;
  const Vector &get_generated_hashing_dis(int num_qubits) const;
  std::vector<ParamType> get_generated_parameters(int num_params) const;
  std::vector<ParamType> get_all_generated_parameters() const;
  size_t next_global_unique_id();

  bool has_parameterized_gate() const;

  // A hacky function: set a generated parameter.
  void set_generated_parameter(int id, ParamType param);

  // These three functions are used in |Verifier::redundant()| for a version
  // of RepGen algorithm that does not invoke Python verifier.
  void set_representative(std::unique_ptr<CircuitSeq> seq);
  void clear_representatives();
  bool get_possible_representative(const CircuitSeqHashType &hash_value,
                                   CircuitSeq *&representative) const;

  ParamType get_param_value(int id) const;
  void set_param_value(int id, const ParamType &param);
  // TODO: Use this function when generating symbolic parameters
  int get_new_param_id(bool is_symbolic);
  // TODO: This function should not be needed
  int get_num_parameters() const;
  bool param_is_symbolic(int id) const;
  bool param_has_value(int id) const;

  // This function generates a deterministic series of random numbers
  // ranging [0, 1].
  double random_number();

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

  // A vector to store the representative circuit sequences.
  std::vector<std::unique_ptr<CircuitSeq>> representative_seqs_;
  std::unordered_map<CircuitSeqHashType, CircuitSeq *> representatives_;
  // Standard mersenne_twister_engine seeded with 0
  std::mt19937 gen{0};

  // The concrete parameters are from the input QASM file,
  // written by QASMParser.
  std::vector<ParamType> parameters_;
  std::vector<bool> is_parameter_symbolic_;
  int num_parameters_{0};
};

// TODO: This function does not consider the parameters
Context union_contexts(Context *ctx_0, Context *ctx_1);

} // namespace quartz
