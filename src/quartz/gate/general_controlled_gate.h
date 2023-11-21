#pragma once

#include "../math/matrix.h"
#include "gate.h"

#include <assert.h>

namespace quartz {
class GeneralControlledGate : public Gate {
 public:
  GeneralControlledGate(Gate *controlled_gate, const std::vector<bool> &state)
      : controlled_gate_(controlled_gate), state_(state),
        Gate(controlled_gate->tp, controlled_gate->num_qubits,
             controlled_gate->num_parameters) {}
  void permute_matrix(MatrixBase *mat) {
    assert(num_qubits <= 30);
    for (int i = 0; i < (1 << num_qubits); i++) {
      int new_i = i;
      for (int k = 0; k < (int)state_.size(); k++) {
        if (!state_[k]) {
          new_i ^= (1 << k);
        }
      }
      if (new_i < i) {
        // already swapped
        continue;
      }
      for (int j = 0; j < (1 << num_qubits); j++) {
        int new_j = j;
        for (int k = 0; k < (int)state_.size(); k++) {
          if (!state_[k]) {
            new_j ^= (1 << k);
          }
        }
        std::swap((*mat)[i][j], (*mat)[new_i][new_j]);
      }
    }
  }
  // TODO: Cache the results, avoid memory leakage
  MatrixBase *get_matrix() override {
    auto result = new MatrixBase(*controlled_gate_->get_matrix());
    // permute_matrix(result);
    return result;
  }
  // TODO: Cache the results, avoid memory leakage
  MatrixBase *get_matrix(const std::vector<ParamType> &params) override {
    // auto result = new MatrixBase(*controlled_gate_->get_matrix(params));
    // printf("here3\n");
    // permute_matrix(result);
    // printf("here4\n");
    // return result;
    return controlled_gate_->get_matrix(params);
  }
  [[nodiscard]] bool is_symmetric() const override {
    // Same as the original gate
    return controlled_gate_->is_symmetric();
  }
  [[nodiscard]] bool is_sparse() const override {
    // Same as the original gate
    return controlled_gate_->is_sparse();
  }
  [[nodiscard]] bool is_diagonal() const override {
    // Same as the original gate
    return controlled_gate_->is_diagonal();
  }
  [[nodiscard]] int get_num_control_qubits() const override {
    // Same as the original gate
    return controlled_gate_->get_num_control_qubits();
  }
  [[nodiscard]] std::vector<bool> get_control_state() const override {
    return state_;
  }
  Gate *controlled_gate_;
  std::vector<bool> state_;
};
}  // namespace quartz
