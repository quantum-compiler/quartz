#pragma once

#include "../gate/gate_utils.h"
#include "../math/vector.h"

#include <unordered_map>
#include <vector>
#include <memory>

class Context {
 public:
  explicit Context(const std::vector<GateType> &supported_gates);
  Gate *get_gate(GateType tp);
  [[nodiscard]] const std::vector<GateType> &get_supported_gates() const;
  // Two deterministic (random) distributions for each number of qubits.
  const Vector &get_generated_input_dis(int num_qubits);
  const Vector &get_generated_hashing_dis(int num_qubits);
  std::vector<ParamType> get_generated_parameters(int num_params);

 private:
  bool insert_gate(GateType tp);

  std::unordered_map<GateType, std::unique_ptr<Gate>> gates_;
  std::vector<GateType> supported_gates_;
  std::vector<Vector> random_input_distribution_;
  std::vector<Vector> random_hashing_distribution_;
  std::vector<ParamType> random_parameters_;
};
