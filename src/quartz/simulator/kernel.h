#pragma once

#include "quartz/circuitseq/circuitseq.h"
#include "quartz/simulator/kernel_cost.h"
#include "quartz/utils/utils.h"

#include <string>
#include <vector>

namespace quartz {

enum KernelType { fusion, shared_memory };

std::string kernel_type_name(KernelType tp);
KernelType string_to_kernel_type(const std::string &s);

/**
 * A kernel used in the output of schedule.
 */
class Kernel {
 public:
  Kernel(std::unique_ptr<CircuitSeq> &&gates, const std::vector<int> &qubits,
         KernelType type)
      : gates(std::move(gates)), qubits(qubits), type(type) {}

  [[nodiscard]] std::string to_string() const;

  static Kernel from_qasm_style_string(Context *ctx, const std::string &str);
  [[nodiscard]] std::string
  to_qasm_style_string(Context *ctx,
                       int param_precision = kDefaultQASMParamPrecision) const;
  /**
   * Compute the cost of this kernel, return infinity if the kernel is not
   * feasible.
   * If the kernel type is fusion, the complexity is O(1).
   * If the kernel type is shared-memory, the complexity is
   * O(||qubits|| * cost_function.get_shared_memory_num_cacheline_qubits()
   *   + ||gates|| * 1(customized_shared_memory_gate_cost == nullptr)).
   * @param cost_function The cost function of kernels.
   * @param qubit_layout The qubit layout to determine if a shared-memory
   * kernel is feasible. |qubit_layout[0]| stores the least significant
   * qubit.
   * @param customized_shared_memory_gate_cost Sometimes we may want to
   * customize the sum of the cost of gates in a shared-memory kernel.
   * If this variable is not |nullptr|, we take the value directly instead of
   * adding up the gates in |gates|.
   * @return The cost of this kernel, or infinity if it's not feasible.
   */
  [[nodiscard]] KernelCostType cost(
      const KernelCost &cost_function,
      const std::vector<int> &qubit_layout = {},
      const KernelCostType *customized_shared_memory_gate_cost = nullptr) const;
  /**
   * Add a gate and update the |qubits| set.
   * @param gate The gate to be added, calling |CircuitGate::add_gate(gate)|.
   * @param ctx The context for the circuit sequence.
   * @param is_local_qubit An oracle to return if a qubit is local, assumed to
   * run in constant time.
   * @param customized_non_insular_qubits Sometimes we may want to customize
   * the set of non-insular qubits in a shared-memory kernel.
   * If this variable is not empty and if the kernel type is shared-memory,
   * we use this qubit set to update |qubits|.
   * @return True iff the gate is successfully added.
   */
  bool add_gate(CircuitGate *gate, Context *ctx,
                const std::function<bool(int)> &is_local_qubit,
                const std::vector<int> &customized_non_insular_qubits = {});

  /**
   * Verify if the kernel is executable.
   * @param is_local_qubit An oracle to return if a qubit is local, assumed to
   * run in constant time.
   * @return True iff all non-insular qubits are local and:
   * the qubits of each gate are in the |qubits| set if this is a fusion kernel,
   * or the non-insular qubits of each gate are in the |qubits| set if this is a
   * shared-memory kernel.
   */
  [[nodiscard]] bool
  verify(const std::function<bool(int)> &is_local_qubit) const;

  std::unique_ptr<CircuitSeq> gates;
  std::vector<int> qubits;
  KernelType type;
};

/**
 * A kernel used in the state of dynamic programming when computing the
 * schedule.
 */
struct KernelInDP {
 public:
  // It is unspecified what the kernel type is if the default constructor
  // is called.
  KernelInDP() {}
  KernelInDP(const std::vector<int> &active_qubits,
             const std::vector<int> &touching_qubits, KernelType tp)
      : active_qubits(active_qubits), touching_qubits(touching_qubits), tp(tp) {
  }
  size_t get_hash() const;
  bool operator==(const KernelInDP &b) const;
  // A partial order: only compare the first element of |active_qubits|.
  // Undefined behavior if any of the two |active_qubits| is empty.
  bool operator<(const KernelInDP &b) const;
  [[nodiscard]] std::string to_string() const;
  std::vector<int> active_qubits;
  std::vector<int> touching_qubits;
  KernelType tp;
};

}  // namespace quartz
