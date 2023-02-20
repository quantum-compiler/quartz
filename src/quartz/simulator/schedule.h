#pragma once

#include "../context/context.h"
#include "quartz/circuitseq/circuitseq.h"

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
  using KernelCostType = double;

  Schedule(const CircuitSeq &sequence, const std::vector<bool> &local_qubit,
           Context *ctx)
      : sequence_(sequence), local_qubit_(local_qubit), ctx_(ctx) {}

  // Compute the number of down sets for the circuit sequence.
  [[nodiscard]] size_t num_down_sets();

  [[nodiscard]] bool is_local_qubit(int index) const;

  /**
   * Compute which kernels to merge together at the end using a greedy
   * algorithm, assuming there are no other kernels after the given ones.
   * The greedy algorithm is approximate -- it might not give the optimal
   * result.
   * @param kernel_costs kernel_costs[i] represents the cost of an i-qubit
   * kernel.
   * @param kernels The non-intersecting kernels to be merged.
   * @param result_cost The sum of the cost of the resulting merged kernels.
   * @param result_kernels The resulting merged kernels, or nullptr if it is
   * not necessary to record them.
   * @return True iff the computation succeeds.
   */
  static bool
  compute_end_schedule(const std::vector<KernelCostType> &kernel_costs,
                       const std::vector<std::vector<int>> &kernels,
                       KernelCostType &result_cost,
                       std::vector<std::vector<int>> *result_kernels);

  /**
   * Compute the schedule using dynamic programming.
   * A kernel with a single controlled gate (e.g., CX) is assumed to have
   * half of the cost of a kernel with a single corresponding non-controlled
   * gate (e.g., X). A kernel with one CCX is assumed to have 1/4 of the
   * cost of a kernel with X.
   * @param kernel_costs kernel_costs[i] represents the cost of an i-qubit
   * kernel.
   * @return True iff the computation succeeds. The results are stored in
   * the member variables |kernels|, |kernel_qubits|, and |cost_|.
   */
  bool compute_kernel_schedule(const std::vector<KernelCostType> &kernel_costs);

  [[nodiscard]] int get_num_kernels() const;
  void print_kernel_schedule() const;

  // The result simulation schedule. We will execute the kernels one by one,
  // and each kernel contains a sequence of gates.
  std::vector<CircuitSeq> kernels;
  std::vector<std::vector<int>> kernel_qubits;
  KernelCostType cost_;

private:
  // The original circuit sequence.
  CircuitSeq sequence_;

  // The mask for local qubits.
  std::vector<bool> local_qubit_;

  Context *ctx_;
};

/**
 * Compute the schedule using dynamic programming.
 * See more information in Schedule::compute_kernel_schedule().
 * @param sequence The entire circuit sequence.
 * @param local_qubits The local qubits in each stage.
 * @param kernel_costs kernel_costs[i] represents the cost of an i-qubit
 * kernel.
 * @param ctx The Context object.
 * @param absorb_single_qubit_gates An optimization to reduce the running
 * time of this function. Requires the input circuit to be fully entangled.
 * TODO: check if it's entangled or not
 * @return The kernel schedule for each stage.
 */
std::vector<Schedule>
get_schedules(const CircuitSeq &sequence,
              const std::vector<std::vector<bool>> &local_qubits,
              const std::vector<Schedule::KernelCostType> &kernel_costs,
              Context *ctx, bool absorb_single_qubit_gates = true);

class PythonInterpreter;
std::vector<std::vector<bool>>
compute_local_qubits_with_ilp(const CircuitSeq &sequence, int num_local_qubits,
                              Context *ctx, PythonInterpreter *interpreter);
} // namespace quartz
