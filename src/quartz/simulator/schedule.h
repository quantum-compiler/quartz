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
  Schedule(std::unique_ptr<CircuitSeq> &&sequence, int num_local_qubits,
           const std::vector<int> &qubit_layout,
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
   * @param max_num_dp_states The maximum number of DP states per iteration.
   * @param shrink_to_num_dp_states If the number of DP states exceeds
   * |max_num_dp_states|, shrink the number of DP states to this number.
   * If this number is 0, use ceil(|max_num_dp_states| / 2).
   * @return True iff the computation succeeds. The results are stored in
   * the member variables |kernels| and |cost_|.
   */
  bool compute_kernel_schedule(
      const KernelCost &kernel_cost,
      const std::vector<std::vector<int>> &non_insular_qubit_indices = {},
      const std::vector<KernelCostType> &shared_memory_gate_costs = {},
      int max_num_dp_states = 500, int shrink_to_num_dp_states = 0);

  /**
   * Compute the schedule using a simple quadratic dynamic programming
   * algorithm. The memory complexity is also quadratic.
   * @param kernel_cost The cost function of kernels.
   * @param is_local_qubit An oracle to return if a qubit is local, assumed to
   * run in constant time.
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
      const std::function<bool(int)> &is_local_qubit,
      const std::vector<std::vector<int>> &non_insular_qubit_indices = {},
      const std::vector<KernelCostType> &shared_memory_gate_costs = {});

  /**
   * Similar to above, but compute in a reversed order.
   * The space complexity becomes linear.
   */
  bool compute_kernel_schedule_simple_reversed(
      const KernelCost &kernel_cost,
      const std::function<bool(int)> &is_local_qubit,
      const std::vector<std::vector<int>> &non_insular_qubit_indices = {},
      const std::vector<KernelCostType> &shared_memory_gate_costs = {});

  bool compute_kernel_schedule_simple_repeat(
      int repeat, const KernelCost &kernel_cost,
      const std::function<bool(int)> &is_local_qubit,
      const std::vector<std::vector<int>> &non_insular_qubit_indices = {},
      const std::vector<KernelCostType> &shared_memory_gate_costs = {});

  bool compute_kernel_schedule_greedy_pack_fusion(
      const KernelCost &kernel_cost,
      const std::function<bool(int)> &is_local_qubit, int num_qubits_to_pack);

  [[nodiscard]] int get_num_kernels() const;
  void print_kernel_info() const;
  void print_kernel_schedule() const;

  [[nodiscard]] const std::vector<int> &get_qubit_layout() const;
  void print_qubit_layout(int num_global_qubits) const;
  [[nodiscard]] std::vector<std::pair<int, int>>
  get_local_swaps_from_previous_stage(const Schedule &prev_schedule) const;

  /**
   * Remove kernels with no gates and update the cost.
   * @return The number of empty kernels removed.
   */
  int remove_empty_kernels(const KernelCost &kernel_cost);
  static Schedule from_file(Context *ctx, const std::string &filename);
  bool save_to_file(Context *ctx, const std::string &filename,
                    int param_precision = 15) const;

  // The result simulation schedule. We will execute the kernels one by one,
  // and each kernel contains a sequence of gates.
  std::vector<Kernel> kernels;
  KernelCostType cost_;

 private:
  // The original circuit sequence.
  std::unique_ptr<CircuitSeq> sequence_;

  int num_local_qubits_;

  // |qubit_layout_[0]| is the least significant bit.
  // The first |num_local_qubits_| qubits are local.
  std::vector<int> qubit_layout_;

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
 * @param num_local_qubits The number of local qubits per device.
 * @param qubit_layout The qubit layout in each stage.
 * @param kernel_cost The cost function of kernels.
 * @param ctx The Context object.
 * @param attach_single_qubit_gates An optimization to reduce the running
 * time of this function. Requires the input circuit to be fully entangled.
 * @param max_num_dp_states
 * If this variable is negative,
 * use the greedy algorithm to pack towards this number of qubits (absolute
 * value).
 * If this variable is 0,
 * use the simple DP algorithm.
 * If this variable is positive, set the |max_num_dp_states| to this number
 * in the DP algorithm.
 * @param cache_file_name_prefix If this variable is not empty,
 * use the cached files if possible (in this case, no other variables are used),
 * and compute the schedules and cache them into files otherwise.
 * @return The kernel schedule for each stage.
 */
std::vector<Schedule>
get_schedules(const CircuitSeq &sequence, int num_local_qubits,
              const std::vector<std::vector<int>> &qubit_layout,
              const KernelCost &kernel_cost, Context *ctx,
              bool attach_single_qubit_gates, int max_num_dp_states = 500,
              const std::string &cache_file_name_prefix = "");

class PythonInterpreter;

/**
 * Compute the local qubits using single-level ILP.
 * @param sequence The entire circuit sequence.
 * @param num_local_qubits The number of local qubits per device.
 * @param ctx The Context object.
 * @param interpreter The Python interpreter.
 * @param answer_start_with We know that the number of stages is at least this
 * number (default is 1). A larger number, if guaranteed to be correct,
 * may accelerate this function.
 * @return The local qubits for each stage.
 */
std::vector<std::vector<int>>
compute_local_qubits_with_ilp(const CircuitSeq &sequence, int num_local_qubits,
                              Context *ctx, PythonInterpreter *interpreter,
                              int answer_start_with = 1);

/**
 * Compute the qubit layout using two-level ILP.
 * @param sequence The entire circuit sequence.
 * @param num_local_qubits The number of local qubits per device.
 * @param num_regional_qubits The number of regional qubits per node.
 * @param ctx The Context object.
 * @param interpreter The Python interpreter.
 * @param answer_start_with We know that the number of stages is at least this
 * number (default is 1). A larger number, if guaranteed to be correct,
 * may accelerate this function.
 * @return The qubit layout for each stage.
 */
std::vector<std::vector<int>> compute_qubit_layout_with_ilp(
    const CircuitSeq &sequence, int num_local_qubits, int num_regional_qubits,
    Context *ctx, PythonInterpreter *interpreter, int answer_start_with = 1);

/**
 * Call the above two functions sequentially, use the cached files if possible.
 */
std::vector<Schedule> get_schedules_with_ilp(
    const CircuitSeq &sequence, int num_local_qubits, int num_regional_qubits,
    const KernelCost &kernel_cost, Context *ctx, PythonInterpreter *interpreter,
    bool attach_single_qubit_gates, int max_num_dp_states = 500,
    const std::string &cache_file_name_prefix = "", int answer_start_with = 1);

/**
 * Verify the schedule by checking the well-formedness of each kernel and then
 * random testing an input state and running the simulation.
 * If |random_test_times| > 0:
 * Requires the sequence to have no more than 30 qubits. Requires an exponential
 * amount of memory to the number of qubits (~48 GiB for 30 qubits). The time
 * complexity is O(2^(number of qubits) * (number of gates)).
 * @return True iff simulating the sequence itself and running the simulation
 * on the schedules yield the same result for each input state tested.
 */
bool verify_schedule(Context *ctx, const CircuitSeq &sequence,
                     const std::vector<Schedule> &schedules,
                     int random_test_times = 1);
}  // namespace quartz
