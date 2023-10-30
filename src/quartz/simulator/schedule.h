#pragma once

#include "quartz/circuitseq/circuitseq.h"
#include "quartz/context/context.h"
#include "quartz/simulator/kernel.h"
#include "quartz/simulator/kernel_cost.h"

#include <cassert>
#include <cmath>
#include <fstream>
#include <map>
#include <vector>

namespace quartz {
/**
 * A simulation schedule of a circuit sequence on one device.
 */
class Schedule {
 public:
  Schedule(std::unique_ptr<CircuitSeq> &&sequence,
           const std::vector<int> &local_qubit,
           const std::vector<int> &global_qubit,
           int num_shared_memory_cacheline_qubits, Context *ctx);

  // Compute the number of down sets for the circuit sequence.
  [[nodiscard]] size_t num_down_sets();

  [[nodiscard]] bool is_local_qubit(int index) const;

  [[nodiscard]] bool is_shared_memory_cacheline_qubit(int index) const;

  /**
   * Compute which open kernels to merge together at the end using a greedy
   * algorithm, assuming there are no other kernels after the given ones.
   * The greedy algorithm is approximate -- it might not give the optimal
   * result.
   * @param kernel_cost The cost function of kernels.
   * @param open_kernels The non-intersecting open kernels to be merged.
   * @param result_cost The sum of the cost of the resulting merged kernels.
   * @param result_kernels The resulting merged kernels, or nullptr if it is
   * not necessary to record them.
   * @return True iff the computation succeeds.
   */
  bool compute_end_schedule(const KernelCost &kernel_cost,
                            const std::vector<KernelInDP> &open_kernels,
                            KernelCostType &result_cost,
                            std::vector<KernelInDP> *result_kernels) const;

  /**
   * Compute the schedule using dynamic programming.
   * Note that it is possible for this DP to give wrong results only when
   * there are both X gate and controlled gates: it assumes we can swap
   * single-qubit sparse gates with adjacent controlled gates if the
   * single-qubit gate operates on the same qubit as the control qubit of
   * the controlled gate. For example, it may reorder
   * X(Q0) CZ(Q0, Q1)
   * to
   * CZ(Q0, Q1) X(Q0)
   * where the correct schedule should be
   * CZ[0](Q0, Q1) X(Q0).
   * @param kernel_cost The cost function of kernels.
   * @param non_insular_qubit_indices The set of non-insular qubit indices
   * for each gate, if any of them should be considered differently from
   * what we would have computed from the gate itself.
   * @param shared_memory_gate_costs The cost for each gate if put into a
   * shared-memory kernel, if any of them should be considered differently from
   * what we would have computed from the gate itself.
   * @return True iff the computation succeeds. The results are stored in
   * the member variables |kernels| and |cost_|.
   */
  bool compute_kernel_schedule(
      const KernelCost &kernel_cost,
      const std::vector<std::vector<int>> &non_insular_qubit_indices = {},
      const std::vector<KernelCostType> &shared_memory_gate_costs = {});

  /**
   * Compute the schedule using a simple quadratic dynamic programming
   * algorithm.
   * @param kernel_cost The cost function of kernels.
   * @param non_insular_qubit_indices The set of non-insular qubit indices
   * for each gate, if any of them should be considered differently from
   * what we would have computed from the gate itself.
   * @param shared_memory_gate_costs The cost for each gate if put into a
   * shared-memory kernel, if any of them should be considered differently from
   * what we would have computed from the gate itself.
   * @return True iff the computation succeeds. The results are stored in
   * the member variables |kernels| and |cost_|.
   */
  bool compute_kernel_schedule_simple(
      const KernelCost &kernel_cost,
      const std::vector<std::vector<int>> &non_insular_qubit_indices = {},
      const std::vector<KernelCostType> &shared_memory_gate_costs = {});

  bool compute_kernel_schedule_simple_repeat(
      int repeat, const KernelCost &kernel_cost,
      const std::vector<std::vector<int>> &non_insular_qubit_indices = {},
      const std::vector<KernelCostType> &shared_memory_gate_costs = {});

  [[nodiscard]] int get_num_kernels() const;
  void print_kernel_info() const;
  void print_kernel_schedule() const;

  /**
   * Remove kernels with no gates and update the cost.
   * @return The number of empty kernels removed.
   */
  int remove_empty_kernels(const KernelCost &kernel_cost);

  // The result simulation schedule. We will execute the kernels one by one,
  // and each kernel contains a sequence of gates.
  std::vector<Kernel> kernels;
  KernelCostType cost_;

 private:
  // The original circuit sequence.
  std::unique_ptr<CircuitSeq> sequence_;

  // The set of local qubits.
  // |local_qubit_[0]| is the least significant bit.
  std::vector<int> local_qubit_;

  // The set of non-local qubits.
  // |global_qubit_[0]| is the least significant bit.
  std::vector<int> global_qubit_;

  // The mask for local qubits.
  std::vector<bool> local_qubit_mask_;

  // The mask for shared-memory cacheline qubits.
  std::vector<bool> shared_memory_cacheline_qubit_mask_;

  Context *ctx_;
};

/**
 * Compute the schedule using dynamic programming.
 * See more information in Schedule::compute_kernel_schedule().
 * @param sequence The entire circuit sequence.
 * @param local_qubits The local qubits in each stage.
 * @param kernel_cost The cost function of kernels.
 * @param ctx The Context object.
 * @param attach_single_qubit_gates An optimization to reduce the running
 * time of this function. Requires the input circuit to be fully entangled.
 * @return The kernel schedule for each stage.
 */
std::vector<Schedule>
get_schedules(const CircuitSeq &sequence,
              const std::vector<std::vector<int>> &local_qubits,
              const KernelCost &kernel_cost, Context *ctx,
              bool attach_single_qubit_gates, int use_simple_dp_times);

class PythonInterpreter;
std::vector<std::vector<int>>
compute_local_qubits_with_ilp(const CircuitSeq &sequence, int num_local_qubits,
                              Context *ctx, PythonInterpreter *interpreter,
                              int answer_start_with = 1);
}  // namespace quartz
