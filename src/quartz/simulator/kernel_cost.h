#pragma once

#include "quartz/gate/gate_utils.h"

#include <functional>
#include <vector>

namespace quartz {

using KernelCostType = double;

/**
 * A class for the cost function of kernels.
 */
class KernelCost {
 public:
  /**
   * The cost function of kernels.
   * @param fusion_kernel_costs An array of costs for fusion kernels.
   * The value of the 0-th index in the array should be 0 because it stands
   * for a 0-qubit kernel.
   * @param shared_memory_init_cost The cost of an empty shared-memory kernel.
   * @param shared_memory_gate_cost The cost function of each gate in a
   * shared-memory kernel.
   * @param shared_memory_total_qubits The maximum size of a shared-memory
   * kernel.
   * @param shared_memory_cacheline_qubits Any shared-memory kernel must
   * contain this number of the least significant qubits.
   */
  KernelCost(
      const std::vector<KernelCostType> &fusion_kernel_costs,
      const KernelCostType &shared_memory_init_cost,
      const std::function<KernelCostType(GateType)> &shared_memory_gate_cost,
      int shared_memory_total_qubits, int shared_memory_cacheline_qubits)
      : fusion_kernel_costs_(fusion_kernel_costs),
        shared_memory_init_cost_(shared_memory_init_cost),
        shared_memory_gate_cost_(shared_memory_gate_cost),
        shared_memory_total_qubits_(shared_memory_total_qubits),
        shared_memory_cacheline_qubits_(shared_memory_cacheline_qubits) {
    if (fusion_kernel_costs.size() <= 1) {
      optimal_fusion_kernel_size_ = 0;
    } else {
      optimal_fusion_kernel_size_ = 1;
      KernelCostType optimal_kernel_size_cost = fusion_kernel_costs[1];
      for (int i = 2; i < (int)fusion_kernel_costs.size(); i++) {
        KernelCostType tmp = fusion_kernel_costs[i] / i;
        if (tmp < optimal_kernel_size_cost) {
          optimal_fusion_kernel_size_ = i;
          optimal_kernel_size_cost = tmp;
        }
      }
    }
  }

  [[nodiscard]] const std::vector<KernelCostType> &
  get_fusion_kernel_costs() const;
  [[nodiscard]] const KernelCostType &get_shared_memory_init_cost() const;
  [[nodiscard]] KernelCostType get_shared_memory_gate_cost(GateType tp) const;
  [[nodiscard]] int get_shared_memory_num_free_qubits() const;
  [[nodiscard]] int get_shared_memory_num_cacheline_qubits() const;
  [[nodiscard]] int get_optimal_fusion_kernel_size() const;

  std::vector<KernelCostType> fusion_kernel_costs_;
  KernelCostType shared_memory_init_cost_;
  std::function<KernelCostType(GateType)> shared_memory_gate_cost_;
  int shared_memory_total_qubits_;
  int shared_memory_cacheline_qubits_;
  int optimal_fusion_kernel_size_;
};

}  // namespace quartz
