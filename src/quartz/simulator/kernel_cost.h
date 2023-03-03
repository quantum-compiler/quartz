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
  KernelCost(
      const std::vector<KernelCostType> &fusion_kernel_costs,
      const KernelCostType &shared_memory_init_cost,
      const std::function<KernelCostType(GateType)> &shared_memory_gate_cost,
      int shared_memory_total_qubits, int shared_memory_cacheline_qubits)
      : fusion_kernel_costs_(fusion_kernel_costs),
        shared_memory_init_cost_(shared_memory_init_cost),
        shared_memory_gate_cost_(shared_memory_gate_cost),
        shared_memory_total_qubits_(shared_memory_total_qubits),
        shared_memory_cacheline_qubits_(shared_memory_cacheline_qubits) {}

  [[nodiscard]] const std::vector<KernelCostType> &
  get_fusion_kernel_costs() const;
  [[nodiscard]] const KernelCostType &get_shared_memory_init_cost() const;
  [[nodiscard]] KernelCostType get_shared_memory_gate_cost(GateType tp) const;
  [[nodiscard]] int get_shared_memory_num_free_qubits() const;

  std::vector<KernelCostType> fusion_kernel_costs_;
  KernelCostType shared_memory_init_cost_;
  std::function<KernelCostType(GateType)> shared_memory_gate_cost_;
  int shared_memory_total_qubits_;
  int shared_memory_cacheline_qubits_;
};

} // namespace quartz
