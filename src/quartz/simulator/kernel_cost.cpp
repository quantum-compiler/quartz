#include "kernel_cost.h"

namespace quartz {

const std::vector<KernelCostType> &
quartz::KernelCost::get_fusion_kernel_costs() const {
  return fusion_kernel_costs_;
}

const KernelCostType &KernelCost::get_shared_memory_init_cost() const {
  return shared_memory_init_cost_;
}

KernelCostType KernelCost::get_shared_memory_gate_cost(GateType tp) const {
  return shared_memory_gate_cost_(tp);
}

int KernelCost::get_shared_memory_num_free_qubits() const {
  return shared_memory_total_qubits_ - shared_memory_cacheline_qubits_;
}

int KernelCost::get_shared_memory_num_cacheline_qubits() const {
  return shared_memory_cacheline_qubits_;
}

int KernelCost::get_optimal_fusion_kernel_size() const {
  return optimal_fusion_kernel_size_;
}

}  // namespace quartz
