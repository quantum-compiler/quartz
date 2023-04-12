#include "schedule.h"
#include "quartz/pybind/pybind.h"

#include <deque>
#include <queue>
#include <stack>
#include <unordered_set>

namespace quartz {

Schedule::Schedule(const CircuitSeq &sequence,
                   const std::vector<int> &local_qubit,
                   const std::vector<int> &global_qubit,
                   int num_shared_memory_cacheline_qubits, Context *ctx)
    : cost_(), sequence_(sequence), local_qubit_(local_qubit),
      global_qubit_(global_qubit), ctx_(ctx) {
  assert(local_qubit.size() + global_qubit.size() == sequence.get_num_qubits());
  assert(local_qubit.size() >= num_shared_memory_cacheline_qubits);
  local_qubit_mask_.assign(sequence.get_num_qubits(), false);
  shared_memory_cacheline_qubit_mask_.assign(sequence.get_num_qubits(), false);
  for (int i = 0; i < (int)local_qubit.size(); i++) {
    local_qubit_mask_[local_qubit[i]] = true;
    if (i < num_shared_memory_cacheline_qubits) {
      shared_memory_cacheline_qubit_mask_[i] = true;
    }
  }
}

size_t Schedule::num_down_sets() {
  const int num_qubits = sequence_.get_num_qubits();
  const int num_gates = sequence_.get_num_gates();
  // Each std::vector<bool> object stores a mask of which gates are in the
  // down set.
  std::unordered_set<std::vector<bool>> down_sets;
  std::queue<std::vector<bool>> to_search;
  std::vector<bool> empty_set(num_gates, false);
  down_sets.insert(empty_set);
  to_search.push(empty_set);

  long long tmp = 1; // debug
  while (!to_search.empty()) {
    auto current_set = to_search.front();
    to_search.pop();
    /*std::cout << "searching ";
    for (auto bit : current_set) {
      std::cout << (int)bit;
    }
    std::cout << std::endl;*/
    if (down_sets.size() >= tmp) {
      std::cout << "down sets: " << down_sets.size() << std::endl;
      tmp *= 2;
    }
    std::vector<bool> qubit_used(num_qubits, false);
    for (int i = 0; i < num_gates; i++) {
      if (!current_set[i]) {
        bool can_add = true;
        for (auto wire : sequence_.gates[i]->input_wires) {
          if (wire->is_qubit() && qubit_used[wire->index]) {
            can_add = false;
            break;
          }
        }
        if (can_add) {
          auto new_set = current_set;
          new_set[i] = true;
          if (down_sets.count(new_set) == 0) {
            down_sets.insert(new_set);
            to_search.push(new_set);
          }
        }
        for (auto wire : sequence_.gates[i]->input_wires) {
          if (wire->is_qubit()) {
            // If we decide to not add this gate, we cannot add any gates using
            // the qubits it operates on later.
            qubit_used[wire->index] = true;
          }
        }
      }
    }
  }
  return down_sets.size();
}

bool Schedule::is_local_qubit(int index) const {
  return local_qubit_mask_[index];
}

bool Schedule::is_shared_memory_cacheline_qubit(int index) const {
  return shared_memory_cacheline_qubit_mask_[index];
}

bool Schedule::compute_end_schedule(
    const KernelCost &kernel_cost, const std::vector<KernelInDP> &open_kernels,
    KernelCostType &result_cost,
    std::vector<KernelInDP> *result_kernels) const {
  const auto &kernel_costs = kernel_cost.get_fusion_kernel_costs();
  const int num_kernels = (int)open_kernels.size();
  const int optimal_fusion_kernel_size =
      kernel_cost.get_optimal_fusion_kernel_size();
  bool at_least_one_shared_memory_kernel = false;
  std::vector<std::vector<int>> fusion_kernels_of_size(
      optimal_fusion_kernel_size);
  const int max_shared_memory_kernel_size =
      kernel_cost.get_shared_memory_num_free_qubits();
  std::vector<std::vector<int>> shared_memory_kernels_of_size(
      max_shared_memory_kernel_size);
  if (result_kernels != nullptr) {
    result_kernels->clear();
  }
  result_cost = 0;
  for (int i = 0; i < num_kernels; i++) {
    if (open_kernels[i].tp == KernelType::fusion) {
      // Assume all fusion kernels are not empty.
      assert(!open_kernels[i].active_qubits.empty());
      if (open_kernels[i].active_qubits.size() >= optimal_fusion_kernel_size) {
        // Greedily try to use the optimal kernel size for fusion kernels.
        // Copy the large fusion kernels to the result.
        if (result_kernels != nullptr) {
          result_kernels->emplace_back(open_kernels[i]);
        }
        result_cost += kernel_costs[open_kernels[i].active_qubits.size()];
      } else {
        fusion_kernels_of_size[open_kernels[i].active_qubits.size()].push_back(
            i);
      }
    } else if (open_kernels[i].tp == KernelType::shared_memory) {
      int kernel_size = (int)open_kernels[i].active_qubits.size();
      for (auto &qubit : open_kernels[i].active_qubits) {
        if (is_shared_memory_cacheline_qubit(qubit)) {
          // Exclude shared-memory cacheline qubits.
          kernel_size--;
        }
      }
      if (kernel_size >= max_shared_memory_kernel_size) {
        assert(kernel_size == max_shared_memory_kernel_size);
        // Copy the large shared-memory open_kernels to the result.
        if (result_kernels != nullptr) {
          result_kernels->emplace_back(open_kernels[i]);
        }
        result_cost += kernel_cost.get_shared_memory_init_cost();
      } else {
        shared_memory_kernels_of_size[kernel_size].push_back(i);
        // In case this shared-memory kernel consists of only cacheline qubits,
        // we need to record that this shared-memory kernel exists.
        at_least_one_shared_memory_kernel = true;
      }
    } else {
      assert(false);
    }
  }

  // Greedily merge the small fusion kernels.
  while (true) {
    bool has_any_remaining_kernel = false;
    KernelInDP current_kernel({}, {}, KernelType::fusion);
    int current_kernel_size = 0;
    for (int i = optimal_fusion_kernel_size - 1; i >= 1; i--) {
      while (!fusion_kernels_of_size[i].empty()) {
        if (current_kernel_size + i <= optimal_fusion_kernel_size) {
          // Add a kernel of size i to the current kernel.
          if (result_kernels != nullptr) {
            current_kernel.active_qubits.insert(
                current_kernel.active_qubits.end(),
                open_kernels[fusion_kernels_of_size[i].back()]
                    .active_qubits.begin(),
                open_kernels[fusion_kernels_of_size[i].back()]
                    .active_qubits.end());
            current_kernel.touching_qubits.insert(
                current_kernel.touching_qubits.end(),
                open_kernels[fusion_kernels_of_size[i].back()]
                    .touching_qubits.begin(),
                open_kernels[fusion_kernels_of_size[i].back()]
                    .touching_qubits.end());
          }
          current_kernel_size += i;
          fusion_kernels_of_size[i].pop_back();
        } else {
          has_any_remaining_kernel = true;
          break;
        }
      }
    }
    if (current_kernel_size != 0) {
      result_cost += kernel_costs[current_kernel_size];
      if (result_kernels != nullptr) {
        // Sort active qubits.
        std::sort(current_kernel.active_qubits.begin(),
                  current_kernel.active_qubits.end());
        // Fusion kernels should not have |touching_qubits|.
        assert(current_kernel.touching_qubits.empty());
        result_kernels->emplace_back(std::move(current_kernel));
      }
    }
    if (!has_any_remaining_kernel) {
      break;
    }
  }

  // Greedily merge the small shared-memory kernels.
  while (true) {
    bool has_any_remaining_kernel = false;
    KernelInDP current_kernel({}, {}, KernelType::shared_memory);
    int current_kernel_size = 0;
    for (int i = max_shared_memory_kernel_size - 1; i >= 0; i--) {
      while (!shared_memory_kernels_of_size[i].empty()) {
        if (current_kernel_size + i <= max_shared_memory_kernel_size) {
          // Add a kernel of size i to the current kernel.
          if (result_kernels != nullptr) {
            current_kernel.active_qubits.insert(
                current_kernel.active_qubits.end(),
                open_kernels[shared_memory_kernels_of_size[i].back()]
                    .active_qubits.begin(),
                open_kernels[shared_memory_kernels_of_size[i].back()]
                    .active_qubits.end());
            current_kernel.touching_qubits.insert(
                current_kernel.touching_qubits.end(),
                open_kernels[shared_memory_kernels_of_size[i].back()]
                    .touching_qubits.begin(),
                open_kernels[shared_memory_kernels_of_size[i].back()]
                    .touching_qubits.end());
          }
          current_kernel_size += i;
          shared_memory_kernels_of_size[i].pop_back();
        } else {
          has_any_remaining_kernel = true;
          break;
        }
      }
    }
    if (current_kernel_size != 0 || at_least_one_shared_memory_kernel) {
      // Even if |current_kernel_size| is 0, we can have one shared-memory
      // kernel with only cacheline qubits (but exactly one shared-memory
      // kernel in this case).
      at_least_one_shared_memory_kernel = false;
      result_cost += kernel_cost.get_shared_memory_init_cost();
      if (result_kernels != nullptr) {
        // Sort and eliminate duplicate qubits.
        std::sort(current_kernel.active_qubits.begin(),
                  current_kernel.active_qubits.end());
        int num_qubits = 0;
        for (auto &index : current_kernel.active_qubits) {
          num_qubits = std::max(num_qubits, index + 1);
        }
        for (auto &index : current_kernel.touching_qubits) {
          num_qubits = std::max(num_qubits, index + 1);
        }
        std::vector<bool> in_current_kernel(num_qubits, false);
        for (auto &index : current_kernel.active_qubits) {
          in_current_kernel[index] = true;
        }
        // Recompute the touching qubits.
        std::vector<int> new_touching_qubits;
        for (auto &index : current_kernel.touching_qubits) {
          if (!in_current_kernel[index]) {
            new_touching_qubits.push_back(index);
            in_current_kernel[index] = true;
          }
        }
        current_kernel.touching_qubits = new_touching_qubits;
        std::sort(current_kernel.touching_qubits.begin(),
                  current_kernel.touching_qubits.end());
        result_kernels->emplace_back(std::move(current_kernel));
      }
    }
    if (!has_any_remaining_kernel) {
      break;
    }
  }
  return true;
}

bool Schedule::compute_kernel_schedule(
    const KernelCost &kernel_cost,
    const std::vector<std::vector<int>> &non_insular_qubit_indices,
    const std::vector<KernelCostType> &shared_memory_gate_costs) {
  const int num_qubits = sequence_.get_num_qubits();
  const int num_gates = sequence_.get_num_gates();
  auto kernel_costs = kernel_cost.get_fusion_kernel_costs();
  const int max_fusion_kernel_size = (int)kernel_costs.size() - 1;
  const int shared_memory_kernel_size =
      kernel_cost.get_shared_memory_num_free_qubits();
  const int shared_memory_cacheline_size =
      kernel_cost.get_shared_memory_num_cacheline_qubits();
  const int max_kernel_size =
      std::max(max_fusion_kernel_size, shared_memory_kernel_size);
  // We need to be able to execute at least 1-qubit gates.
  assert(max_kernel_size >= 2);
  // We have not computed the schedule before.
  assert(kernels.empty());
  // Either to not customize |non_insular_qubit_indices|, or to customize the
  // non-insular qubit indices for each gate.
  assert(non_insular_qubit_indices.empty() ||
         (int)non_insular_qubit_indices.size() == num_gates);
  // Either to not customize |shared_memory_gate_costs|, or to customize the
  // cost for each gate.
  assert(shared_memory_gate_costs.empty() ||
         (int)shared_memory_gate_costs.size() == num_gates);

  // A state for dynamic programming.
  struct Status {
  public:
    static size_t get_hash(const std::vector<int> &s) {
      size_t result = 5381;
      for (const auto &i : s) {
        result = result * 33 + i;
      }
      return result;
    }
    void compute_hash() {
      // XXX: We are not using |absorbing_kernels| to compute the hash.
      // If we decide to use it in the future, we need to search for every
      // place we modify |absorbing_kernels| and update all places where we
      // update the hash manually.
      hash = 0;
      for (const auto &s : open_kernels) {
        hash ^= s.get_hash();
      }
    }
    void insert_set(const KernelInDP &kernel) {
      int insert_position = 0;
      while (insert_position < (int)open_kernels.size() &&
             open_kernels[insert_position] < kernel) {
        insert_position++;
      }
      open_kernels.insert(open_kernels.begin() + insert_position, kernel);
    }
    [[nodiscard]] bool check_valid() const {
      std::vector<bool> has_qubit;
      for (int i = 0; i < (int)open_kernels.size(); i++) {
        for (int j = 0; j < (int)open_kernels[i].active_qubits.size(); j++) {
          while (open_kernels[i].active_qubits[j] >= has_qubit.size()) {
            has_qubit.push_back(false);
          }
          if (has_qubit[open_kernels[i].active_qubits[j]]) {
            std::cerr << "Invalid status: qubit "
                      << open_kernels[i].active_qubits[j]
                      << " is active in two open kernels." << std::endl;
            std::cerr << to_string() << std::endl;
            return false;
          }
          has_qubit[open_kernels[i].active_qubits[j]] = true;
        }
      }
      for (int i = 0; i < (int)absorbing_kernels.size(); i++) {
        for (int j = 0; j < (int)absorbing_kernels[i].active_qubits.size();
             j++) {
          while (absorbing_kernels[i].active_qubits[j] >= has_qubit.size()) {
            has_qubit.push_back(false);
          }
          if (has_qubit[absorbing_kernels[i].active_qubits[j]]) {
            std::cerr << "Invalid status: qubit "
                      << absorbing_kernels[i].active_qubits[j]
                      << " is active in two kernels." << std::endl;
            std::cerr << to_string() << std::endl;
            return false;
          }
          has_qubit[absorbing_kernels[i].active_qubits[j]] = true;
        }
      }
      for (int i = 0; i < (int)open_kernels.size(); i++) {
        for (int j = 0; j < (int)open_kernels[i].touching_qubits.size(); j++) {
          while (open_kernels[i].touching_qubits[j] >= has_qubit.size()) {
            has_qubit.push_back(false);
          }
          if (has_qubit[open_kernels[i].touching_qubits[j]]) {
            std::cerr << "Invalid status: qubit "
                      << open_kernels[i].touching_qubits[j]
                      << " is active in one kernel and touching in another "
                         "open kernel."
                      << std::endl;
            std::cerr << to_string() << std::endl;
            return false;
          }
        }
      }
      for (int i = 0; i < (int)absorbing_kernels.size(); i++) {
        for (int j = 0; j < (int)absorbing_kernels[i].touching_qubits.size();
             j++) {
          while (absorbing_kernels[i].touching_qubits[j] >= has_qubit.size()) {
            has_qubit.push_back(false);
          }
          if (has_qubit[absorbing_kernels[i].touching_qubits[j]]) {
            std::cerr << "Invalid status: qubit "
                      << absorbing_kernels[i].touching_qubits[j]
                      << " is active in one kernel and touching in another "
                         "absorbing kernel."
                      << std::endl;
            std::cerr << to_string() << std::endl;
            return false;
          }
        }
      }
      return true;
    }
    bool operator==(const Status &b) const {
      if (open_kernels.size() != b.open_kernels.size()) {
        return false;
      }
      for (int i = 0; i < (int)open_kernels.size(); i++) {
        if (!(open_kernels[i] == b.open_kernels[i])) {
          return false;
        }
      }
      if (absorbing_kernels.size() != b.absorbing_kernels.size()) {
        return false;
      }
      for (int i = 0; i < (int)absorbing_kernels.size(); i++) {
        if (!(absorbing_kernels[i] == b.absorbing_kernels[i])) {
          return false;
        }
      }
      return true;
    }
    [[nodiscard]] std::string to_string() const {
      std::string result;
      result += "{";
      for (int i = 0; i < (int)open_kernels.size(); i++) {
        result += open_kernels[i].to_string();
        if (i != (int)open_kernels.size() - 1) {
          result += ", ";
        }
      }
      if (!absorbing_kernels.empty()) {
        if (!open_kernels.empty()) {
          result += ", ";
        }
        result += "absorbing {";
        for (int i = 0; i < (int)absorbing_kernels.size(); i++) {
          result += absorbing_kernels[i].to_string();
          if (i != (int)absorbing_kernels.size() - 1) {
            result += ", ";
          }
        }
        result += "}";
      }
      result += "}";
      return result;
    }
    std::vector<KernelInDP> open_kernels;
    // The collection of closed kernels such that, as long as there is a kernel
    // in this collection and a gate in the sequence that is not depending on
    // any qubits not in the kernel, we can execute the gate at no cost if
    // this is a fusion kernel or the gate execution cost if this is a
    // shared-memory kernel.
    std::vector<KernelInDP> absorbing_kernels;
    size_t hash;
  };
  class StatusHash {
  public:
    size_t operator()(const Status &s) const { return s.hash; }
  };
  struct LocalSchedule {
  public:
    [[nodiscard]] std::string to_string() const {
      std::string result;
      result += "{";
      for (int i = 0; i < (int)kernels.size(); i++) {
        result += kernels[i].to_string();
        if (i != (int)kernels.size() - 1) {
          result += ", ";
        }
      }
      result += "}";
      return result;
    }
    std::vector<KernelInDP> kernels;
  };
  // f[i][S].first: first i (1-indexed) gates, "status of open kernels" S,
  // min cost of the closed kernels.
  // f[i][S].second: the schedule to achieve min cost.
  std::unordered_map<Status, std::pair<KernelCostType, LocalSchedule>,
                     StatusHash>
      f[2];
  Status initial_status;
  initial_status.compute_hash();
  f[0][initial_status] = std::make_pair(0, LocalSchedule());
  auto update_f =
      [](std::unordered_map<Status, std::pair<KernelCostType, LocalSchedule>,
                            StatusHash> &f,
         const Status &s, const KernelCostType &new_cost,
         const LocalSchedule &local_schedule) {
        auto it = f.find(s);
        if (it == f.end()) {
          f.insert(std::make_pair(s, std::make_pair(new_cost, local_schedule)));
        } else {
          if (new_cost < it->second.first) {
            it->second = std::make_pair(new_cost, local_schedule);
          }
        }
      };
  constexpr bool debug = false;
  if (debug) {
    std::cout << "Start DP:" << std::endl;
    std::cout << "Local qubits: ";
    for (auto &i : local_qubit_) {
      std::cout << i << ", ";
    }
    std::cout << std::endl;
    std::cout << "Global qubits: ";
    for (auto &i : global_qubit_) {
      std::cout << i << ", ";
    }
    std::cout << std::endl;
    std::cout << sequence_.to_string(true) << std::endl;
  }
  // The main DP loop.
  for (int i = 0; i < num_gates; i++) {
    // Update from f[i & 1] to f[~i & 1].
    auto &f_prev = f[i & 1];
    auto &f_next = f[~i & 1];
    if (debug) {
      std::cout << "DP: i=" << i << " size=" << f_prev.size() << std::endl;
    }
    f_next.clear();
    // Get the qubit indices of the current gate.
    auto &current_gate = *sequence_.gates[i];
    KernelCostType current_gate_cost;
    if (shared_memory_gate_costs.empty()) {
      current_gate_cost =
          kernel_cost.get_shared_memory_gate_cost(current_gate.gate->tp);
    } else {
      current_gate_cost = shared_memory_gate_costs[i];
    }
    std::vector<bool> current_index(num_qubits, false);
    std::vector<int> current_indices;
    std::vector<bool> current_non_insular_index(num_qubits, false);
    std::vector<int> current_non_insular_indices;
    if (!non_insular_qubit_indices.empty()) {
      current_non_insular_indices = non_insular_qubit_indices[i];
    } else {
      current_non_insular_indices =
          current_gate.get_non_insular_qubit_indices();
    }
    for (auto &qubit : current_non_insular_indices) {
      assert(is_local_qubit(qubit));
      current_non_insular_index[qubit] = true;
    }
    current_indices.reserve(current_gate.input_wires.size());
    for (auto &input_wire : current_gate.input_wires) {
      if (input_wire->is_qubit()) {
        // We do not care about global qubits here.
        // XXX: this also means that we do not check dependencies on global
        // gates.
        // If the circuit sequence is
        // CCX(0,1,2) CCX(0,3,4) X(0) CCX(0,1,2) CCX(0,3,4)
        // and 0 is a global qubit,
        // we may put two CCX(0,1,2)s into one kernel
        // and two CCX(0,3,4)s into another kernel.
        if (is_local_qubit(input_wire->index)) {
          current_index[input_wire->index] = true;
          current_indices.push_back(input_wire->index);
          if (debug) {
            if (current_non_insular_index[input_wire->index]) {
              std::cout << "non-insular ";
            }
            std::cout << "current index " << input_wire->index << std::endl;
          }
        }
      }
    }
    std::sort(current_indices.begin(), current_indices.end());
    std::sort(current_non_insular_indices.begin(),
              current_non_insular_indices.end());

    // Precompute the single-gate kernels because they may be used many times
    // during the DP.
    // Note that the |touching_qubits| is always empty in fusion kernels.
    KernelInDP current_single_gate_fusion_kernel(current_indices, {},
                                                 KernelType::fusion);
    auto current_single_gate_fusion_kernel_hash =
        current_single_gate_fusion_kernel.get_hash();
    KernelInDP current_single_gate_shared_memory_kernel(
        current_non_insular_indices, {}, KernelType::shared_memory);
    current_single_gate_shared_memory_kernel.touching_qubits.reserve(
        current_indices.size() - current_non_insular_indices.size());
    for (auto &qubit : current_indices) {
      if (!current_non_insular_index[qubit]) {
        current_single_gate_shared_memory_kernel.touching_qubits.push_back(
            qubit);
      }
    }
    auto current_single_gate_shared_memory_kernel_hash =
        current_single_gate_shared_memory_kernel.get_hash();

    // TODO: make these numbers configurable
    constexpr int kMaxNumOfStatus = 2000;
    constexpr int kShrinkToNumOfStatus = 1000;
    if (f_prev.size() > kMaxNumOfStatus) {
      // Pruning.
      std::vector<std::pair<
          KernelCostType,
          std::unordered_map<Status, std::pair<KernelCostType, LocalSchedule>,
                             StatusHash>::iterator>>
          costs;
      if (debug) {
        std::cout << "Shrink f[" << i << "] from " << f_prev.size()
                  << " elements to " << kShrinkToNumOfStatus << " elements."
                  << std::endl;
      }
      costs.reserve(f_prev.size());
      KernelCostType lowest_cost, highest_cost; // for debugging
      for (auto it = f_prev.begin(); it != f_prev.end(); it++) {
        // Use the current "end" cost as a heuristic.
        // TODO: profile the running time of |compute_end_schedule| and see
        //  if an approximation optimization is necessary.
        KernelCostType result_cost;
        compute_end_schedule(kernel_cost, it->first.open_kernels, result_cost,
                             /*result_kernels=*/nullptr);
        costs.emplace_back(std::make_pair(result_cost + it->second.first, it));
        if (debug) {
          if (it == f_prev.begin() ||
              result_cost + it->second.first > highest_cost) {
            highest_cost = result_cost + it->second.first;
          }
          if (it == f_prev.begin() ||
              result_cost + it->second.first < lowest_cost) {
            lowest_cost = result_cost + it->second.first;
          }
        }
      }
      // Retrieve the first |kShrinkToNumOfStatus| lowest cost.
      std::nth_element(
          costs.begin(), costs.begin() + kShrinkToNumOfStatus, costs.end(),
          [](const auto &p1, const auto &p2) { return p1.first < p2.first; });
      std::unordered_map<Status, std::pair<KernelCostType, LocalSchedule>,
                         StatusHash>
          new_f;
      for (int j = 0; j < kShrinkToNumOfStatus; j++) {
        // Extract the node and insert to the new unordered_map.
        new_f.insert(f_prev.extract(costs[j].second));
      }
      f_prev = new_f;
      if (debug) {
        std::cout << "Costs shrank from [" << lowest_cost << ", "
                  << highest_cost << "] to [" << lowest_cost << ", "
                  << costs[kShrinkToNumOfStatus - 1].first << "]." << std::endl;
      }
    }

    if (debug) {
      KernelCostType tmp_best_cost = 1e100;
      Status tmp_status;
      LocalSchedule tmp_schedule;
      for (auto &it : f_prev) {
        auto &current_status = it.first;
        auto &current_cost = it.second.first;
        auto &current_local_schedule = it.second.second;
        KernelCostType tmp;
        compute_end_schedule(kernel_cost, current_status.open_kernels, tmp,
                             nullptr);
        tmp += current_cost;
        if (tmp < tmp_best_cost) {
          tmp_best_cost = tmp;
          tmp_status = current_status;
          tmp_schedule = current_local_schedule;
        }
      }
      std::cout << "best cost before i=" << i << ": " << tmp_best_cost
                << std::endl;
      std::cout << "open kernels: " << tmp_status.to_string() << std::endl;
      std::cout << "closed kernels: " << tmp_schedule.to_string() << std::endl;
    }

    for (auto &it : f_prev) {
      auto &current_status = it.first;
      auto &current_cost = it.second.first;
      auto &current_local_schedule = it.second.second;
      assert(current_status.check_valid());

      if (current_indices.empty()) {
        // A global gate. Directly update.
        update_f(f_next, current_status, current_cost, current_local_schedule);
        continue;
      }

      // Count the number of all qubits absorbed into active qubits first.
      int absorbing_set_index = -1;
      int absorb_count = 0;
      for (int j = 0; j < (int)current_status.absorbing_kernels.size(); j++) {
        for (auto &qubit : current_status.absorbing_kernels[j].active_qubits) {
          if (current_index[qubit]) {
            absorb_count++;
            if (absorbing_set_index == -1) {
              absorbing_set_index = j;
            } else if (absorbing_set_index != j) {
              // Touching 2 absorbing kernels' active qubits,
              // so not absorb-able.
              absorbing_set_index = -2;
              break;
            }
          }
        }
        if (absorbing_set_index == -2) {
          break;
        }
      }
      // Count the number of all qubits absorbed into touching qubits.
      if (absorbing_set_index >= 0) {
        for (auto &qubit : current_status.absorbing_kernels[absorbing_set_index]
                               .touching_qubits) {
          if (current_index[qubit]) {
            absorb_count++;
          }
        }
      } else if (absorbing_set_index == -1 &&
                 current_non_insular_indices.empty()) {
        assert(absorb_count == 0);
        // If no absorbing kernels' active qubits touch the current gate's
        // qubits, we can still probably absorb the gate if this gate does
        // not have any non-insular qubits.
        for (int j = 0; j < (int)current_status.absorbing_kernels.size(); j++) {
          absorb_count = 0;
          for (auto &qubit :
               current_status.absorbing_kernels[j].touching_qubits) {
            if (current_index[qubit]) {
              absorb_count++;
            }
          }
          if (absorb_count == current_indices.size()) {
            // This kernel absorbs the gate.
            absorbing_set_index = j;
            break;
          }
        }
      }
      if (absorbing_set_index >= 0 && absorb_count == current_indices.size()) {
        // We then check if all non-insular qubits are active.
        absorb_count = 0;
        for (auto &qubit : current_status.absorbing_kernels[absorbing_set_index]
                               .active_qubits) {
          if (current_non_insular_index[qubit]) {
            absorb_count++;
          }
        }
        if (absorb_count == current_non_insular_indices.size()) {
          // Optimization:
          // The current gate is absorbed by a previous kernel.
          // Directly update.
          assert(current_status.absorbing_kernels[absorbing_set_index].tp ==
                     KernelType::fusion ||
                 current_status.absorbing_kernels[absorbing_set_index].tp ==
                     KernelType::shared_memory);
          if (current_status.absorbing_kernels[absorbing_set_index].tp ==
              KernelType::shared_memory) {
            current_cost += current_gate_cost;
          }
          update_f(f_next, current_status, current_cost,
                   current_local_schedule);
          continue;
        }
      }

      // Count the number of qubits intersecting with each open kernel.
      const int num_kernels = current_status.open_kernels.size();
      std::vector<int> intersect_kernel_indices, active_size, touching_size;
      for (int j = 0; j < num_kernels; j++) {
        bool touching_set_has_j = false;
        for (auto &index : current_status.open_kernels[j].active_qubits) {
          if (current_index[index]) {
            if (!touching_set_has_j) {
              touching_set_has_j = true;
              intersect_kernel_indices.push_back(j);
              active_size.push_back(1);
              touching_size.push_back(0);
            } else {
              active_size.back()++;
            }
          }
        }
        for (auto &index : current_status.open_kernels[j].touching_qubits) {
          if (current_index[index]) {
            if (!touching_set_has_j) {
              touching_set_has_j = true;
              intersect_kernel_indices.push_back(j);
              active_size.push_back(0);
              touching_size.push_back(1);
            } else {
              touching_size.back()++;
            }
          }
        }
      }
      if (intersect_kernel_indices.size() == 1 &&
          active_size[0] + touching_size[0] == current_indices.size()) {
        int target_active_count = 0;
        for (auto &qubit :
             current_status.open_kernels[intersect_kernel_indices[0]]
                 .active_qubits) {
          if (current_non_insular_index[qubit]) {
            target_active_count++;
          }
        }
        if (target_active_count == current_non_insular_indices.size()) {
          // Optimization:
          // The current gate is touching exactly one open kernel
          // and is subsumed by that kernel.
          // Directly update.
          assert(current_status.open_kernels[intersect_kernel_indices[0]].tp ==
                     KernelType::fusion ||
                 current_status.open_kernels[intersect_kernel_indices[0]].tp ==
                     KernelType::shared_memory);
          if (current_status.open_kernels[intersect_kernel_indices[0]].tp ==
              KernelType::shared_memory) {
            current_cost += current_gate_cost;
          }
          update_f(f_next, current_status, current_cost,
                   current_local_schedule);
          continue;
        }
      }

      // For shared-memory kernels, we only need to have the target qubits
      // in the absorbing kernel; but we need to have all control qubits
      // not in any open kernels (nor other absorbing kernels, which is
      // already checked before).
      auto new_absorbing_kernels = current_status.absorbing_kernels;

      // Update |absorbing_kernels|.
      // Loop in reverse order so that we do not need to worry about
      // the index change during removal.
      for (int k = (int)new_absorbing_kernels.size() - 1; k >= 0; k--) {
        // Remove all qubits touching the current gate from the
        // active qubits in |absorbing_kernels|.
        // Loop in reverse order so that we do not need to worry about
        // the index change during removal.
        for (int j = (int)new_absorbing_kernels[k].active_qubits.size() - 1;
             j >= 0; j--) {
          if (current_index[new_absorbing_kernels[k].active_qubits[j]]) {
            new_absorbing_kernels[k].active_qubits.erase(
                new_absorbing_kernels[k].active_qubits.begin() + j);
          }
        }
        // Remove all target qubits of the current gate from the
        // touching qubits in |absorbing_kernels|.
        // Loop in reverse order so that we do not need to worry about
        // the index change during removal.
        for (int j = (int)new_absorbing_kernels[k].touching_qubits.size() - 1;
             j >= 0; j--) {
          if (current_non_insular_index[new_absorbing_kernels[k]
                                            .touching_qubits[j]]) {
            new_absorbing_kernels[k].touching_qubits.erase(
                new_absorbing_kernels[k].touching_qubits.begin() + j);
          }
        }
        // To make sorting easier, we remove the absorbing kernel whenever
        // the active qubits set becomes empty (even if the touching qubits
        // set is not empty).
        if (new_absorbing_kernels[k].active_qubits.empty()) {
          new_absorbing_kernels.erase(new_absorbing_kernels.begin() + k);
        }
      }
      // Sort the absorbing kernels in ascending order.
      std::sort(new_absorbing_kernels.begin(), new_absorbing_kernels.end());
      auto update_absorbing_kernels_for_new_fusion_kernel = [&]() {
        // For fusion kernels, we need to update the |absorbing_kernels|'s
        // |touching_qubits| again.
        // Loop in reverse order so that we do not need to worry about
        // the index change during removal.
        for (int k = (int)new_absorbing_kernels.size() - 1; k >= 0; k--) {
          // Remove all qubits of the current gate from the
          // touching qubits in |absorbing_kernels|.
          // Loop in reverse order so that we do not need to worry about
          // the index change during removal.
          for (int j = (int)new_absorbing_kernels[k].touching_qubits.size() - 1;
               j >= 0; j--) {
            if (current_index[new_absorbing_kernels[k].touching_qubits[j]]) {
              new_absorbing_kernels[k].touching_qubits.erase(
                  new_absorbing_kernels[k].touching_qubits.begin() + j);
            }
          }
        }
      };

      if (intersect_kernel_indices.empty()) {
        // Optimization:
        // The current gate is not touching any kernels on the frontier.
        // Directly add the gate to the frontier.
        auto new_status = current_status;
        new_status.insert_set(current_single_gate_shared_memory_kernel);
        new_status.absorbing_kernels = new_absorbing_kernels;
        new_status.hash ^= current_single_gate_shared_memory_kernel_hash;
        update_f(f_next, new_status, current_cost + current_gate_cost,
                 current_local_schedule);

        update_absorbing_kernels_for_new_fusion_kernel();
        new_status = current_status;
        new_status.insert_set(current_single_gate_fusion_kernel);
        new_status.absorbing_kernels = new_absorbing_kernels;
        new_status.hash ^= current_single_gate_fusion_kernel_hash;
        update_f(f_next, new_status, current_cost, current_local_schedule);
        continue;
      }

      // Main search in the DP.
      // Keep track of the schedule during the search.
      LocalSchedule local_schedule = current_local_schedule;
      std::vector<KernelInDP> absorbing_kernels_stack;
      // Keep track of which kernels are merged during the search.
      std::vector<bool> kernel_merged(num_kernels, false);
      for (auto &index : intersect_kernel_indices) {
        // These kernels are already considered -- treat them as merged.
        kernel_merged[index] = true;
      }
      auto search_merging_kernels =
          [&](auto &this_ref, const KernelInDP &current_gate_kernel,
              const KernelInDP &current_merging_kernel,
              const KernelCostType &cost, int touching_set_index,
              int kernel_index) -> void {
        if (kernel_index == num_kernels) {
          // We have searched all kernels to merge or not with this
          // "touching set".
          touching_set_index++;
          auto new_cost = cost;
          if (!current_merging_kernel.active_qubits.empty() ||
              !current_merging_kernel.touching_qubits.empty()) {
            // Because we are not merging this kernel with the current
            // gate, we need to record the merged kernel.
            local_schedule.kernels.push_back(current_merging_kernel);
            assert(current_merging_kernel.tp == KernelType::fusion ||
                   current_merging_kernel.tp == KernelType::shared_memory);
            // We record the cost here when closing the kernel.
            if (current_merging_kernel.tp == KernelType::fusion) {
              assert(current_merging_kernel.touching_qubits.empty());
              new_cost +=
                  kernel_costs[current_merging_kernel.active_qubits.size()];
            } else {
              new_cost += kernel_cost.get_shared_memory_init_cost();
              // We compute the |touching_qubits| for |current_merging_kernel|
              // here. We assume every qubit that is not active can be
              // touching here.
              std::vector<bool> touching(num_qubits, true);
              for (auto &index : current_merging_kernel.active_qubits) {
                touching[index] = false;
              }
              for (auto &kernel : current_status.open_kernels) {
                for (auto &index : kernel.active_qubits) {
                  touching[index] = false;
                }
              }
              // TODO: this is a bit conservative: instead of not including
              //  them in |touching_qubits| here, we can also remove the
              //  previous absorbing kernel and include the qubits in
              //  |touching_qubits| here. (This is for the following
              //  absorbing kernel, not for the |local_schedule| here.)
              for (auto &kernel : current_status.absorbing_kernels) {
                for (auto &index : kernel.active_qubits) {
                  touching[index] = false;
                }
              }
              local_schedule.kernels.back().touching_qubits.clear();
              for (int index = 0; index < num_qubits; index++) {
                if (touching[index]) {
                  local_schedule.kernels.back().touching_qubits.push_back(
                      index);
                }
              }
            }
            std::sort(local_schedule.kernels.back().active_qubits.begin(),
                      local_schedule.kernels.back().active_qubits.end());
            // |touching_qubits| is guaranteed to be sorted here.

            // Compute the absorbing kernel.
            KernelInDP absorbing_kernel({}, {}, current_merging_kernel.tp);
            for (auto &index : local_schedule.kernels.back().active_qubits) {
              if (!current_index[index]) {
                // As long as the current gate does not block the qubit |index|,
                // we can execute a gate at the qubit |index| later in the
                // kernel |current_merging_kernel|.
                absorbing_kernel.active_qubits.push_back(index);
              }
            }
            for (auto &index : local_schedule.kernels.back().touching_qubits) {
              if (!current_non_insular_index[index] &&
                  (!current_index[index] ||
                   current_gate_kernel.tp == KernelType::shared_memory)) {
                // Similarly, we block the target qubits in the
                // |touching_qubits| set.
                // If the current gate is in a fusion kernel, we also need
                // to block the insular qubits in the current gate in the
                // |touching_qubits| set.
                absorbing_kernel.touching_qubits.push_back(index);
              }
            }
            absorbing_kernels_stack.push_back(absorbing_kernel);
          }
          if (touching_set_index == (int)intersect_kernel_indices.size()) {
            // We have searched everything.
            // Create the new Status object.
            Status new_status;
            for (int j = 0; j < num_kernels; j++) {
              if (!kernel_merged[j]) {
                new_status.open_kernels.push_back(
                    current_status.open_kernels[j]);
              }
            }
            // Insert the new open kernel.
            new_status.insert_set(current_gate_kernel);
            // Insert the absorbing kernels.
            new_status.absorbing_kernels = new_absorbing_kernels;
            new_status.absorbing_kernels.reserve(
                new_absorbing_kernels.size() + absorbing_kernels_stack.size());
            // Push the absorbing kernels if |active_qubits| is not empty.
            for (auto &absorbing_kernel : absorbing_kernels_stack) {
              if (!absorbing_kernel.active_qubits.empty()) {
                new_status.absorbing_kernels.push_back(absorbing_kernel);
              }
            }
            // Sort the absorbing sets in ascending order.
            std::sort(new_status.absorbing_kernels.begin(),
                      new_status.absorbing_kernels.end());
            new_status.compute_hash();
            update_f(f_next, new_status, new_cost, local_schedule);
          } else {
            // Start a new iteration of searching.
            // Try to merge the "touching set" with the current gate first.
            bool merging_is_always_better = false;
            if (current_status
                    .open_kernels[intersect_kernel_indices[touching_set_index]]
                    .tp == current_gate_kernel.tp) {
              // We only merge kernels of the same type.
              auto new_gate_kernel = current_gate_kernel;
              // We update the |touching_qubits| for |current_gate_kernel| here.
              assert(current_gate_kernel.tp == KernelType::fusion ||
                     current_gate_kernel.tp == KernelType::shared_memory);
              if (current_gate_kernel.tp == KernelType::fusion) {
                // Put everything into |active_qubits| if it is a fusion kernel.
                for (auto &index :
                     current_status
                         .open_kernels
                             [intersect_kernel_indices[touching_set_index]]
                         .active_qubits) {
                  if (!current_index[index]) {
                    new_gate_kernel.active_qubits.push_back(index);
                  }
                }
              } else {
                // For shared-memory kernels, handle |active_qubits| and
                // |touching_qubits| separately.
                for (auto &index :
                     current_status
                         .open_kernels
                             [intersect_kernel_indices[touching_set_index]]
                         .active_qubits) {
                  if (!current_non_insular_index[index]) {
                    new_gate_kernel.active_qubits.push_back(index);
                    auto index_pos =
                        std::find(new_gate_kernel.touching_qubits.begin(),
                                  new_gate_kernel.touching_qubits.end(), index);
                    if (index_pos != new_gate_kernel.touching_qubits.end()) {
                      // Remove this active qubit from |touching_qubits|.
                      new_gate_kernel.touching_qubits.erase(index_pos);
                    }
                  }
                }
                for (auto &index :
                     current_status
                         .open_kernels
                             [intersect_kernel_indices[touching_set_index]]
                         .touching_qubits) {
                  if (!current_index[index]) {
                    if (std::find(new_gate_kernel.active_qubits.begin(),
                                  new_gate_kernel.active_qubits.end(), index) ==
                        new_gate_kernel.active_qubits.end()) {
                      if (std::find(new_gate_kernel.touching_qubits.begin(),
                                    new_gate_kernel.touching_qubits.end(),
                                    index) ==
                          new_gate_kernel.touching_qubits.end()) {
                        // Only add to |touching_qubits| if it doesn't exist in
                        // |active_qubits| or |touching_qubits|.
                        new_gate_kernel.touching_qubits.push_back(index);
                      }
                    }
                  }
                }
              }
              if (new_gate_kernel.active_qubits.size() ==
                  current_gate_kernel.active_qubits.size()) {
                // An optimization: if we merge a kernel with the current gate
                // and the size of active qubits remain unchanged, we always
                // want to merge the kernel with the current gate.
                merging_is_always_better = true;
              }
              bool kernel_size_ok = false;
              if (current_gate_kernel.tp == KernelType::fusion) {
                assert(new_gate_kernel.touching_qubits.empty());
                kernel_size_ok = new_gate_kernel.active_qubits.size() <=
                                 max_fusion_kernel_size;
              } else {
                if (new_gate_kernel.active_qubits.size() <=
                    shared_memory_kernel_size) {
                  kernel_size_ok = true;
                } else {
                  // Check cacheline qubits.
                  int num_cacheline_qubits = 0;
                  for (const auto &qubit : new_gate_kernel.active_qubits) {
                    if (is_shared_memory_cacheline_qubit(qubit)) {
                      num_cacheline_qubits++;
                    }
                  }
                  kernel_size_ok = new_gate_kernel.active_qubits.size() -
                                       num_cacheline_qubits <=
                                   shared_memory_kernel_size;
                }
              }
              if (kernel_size_ok) {
                std::sort(new_gate_kernel.active_qubits.begin(),
                          new_gate_kernel.active_qubits.end());
                std::sort(new_gate_kernel.touching_qubits.begin(),
                          new_gate_kernel.touching_qubits.end());
                // If we merge a kernel with the current gate, we do not need
                // to search for other kernels to merge together.
                // So the |kernel_index| is |num_kernels| instead of 0.
                this_ref(this_ref, new_gate_kernel,
                         /*current_merging_kernel=*/KernelInDP(), new_cost,
                         touching_set_index,
                         /*kernel_index=*/num_kernels);
              }
            }
            if (!merging_is_always_better) {
              // Try to not merge the "touching set" with the current gate.
              this_ref(this_ref, current_gate_kernel,
                       /*current_merging_kernel=*/
                       current_status.open_kernels
                           [intersect_kernel_indices[touching_set_index]],
                       new_cost, touching_set_index, /*kernel_index=*/0);
            }
          }
          if (!current_merging_kernel.active_qubits.empty() ||
              !current_merging_kernel.touching_qubits.empty()) {
            // Restore the merged kernel stack.
            local_schedule.kernels.pop_back();
            absorbing_kernels_stack.pop_back();
          }
          return;
        }
        // We can always try not to merge with this kernel.
        this_ref(this_ref, current_gate_kernel, current_merging_kernel, cost,
                 touching_set_index, kernel_index + 1);
        if (kernel_merged[kernel_index]) {
          // This kernel is already considered. Continue to the next one.
          return;
        }
        if (current_status.open_kernels[kernel_index].tp !=
            current_merging_kernel.tp) {
          // We don't merge kernels with different types.
          return;
        }
        assert(current_merging_kernel.tp == KernelType::fusion ||
               current_merging_kernel.tp == KernelType::shared_memory);
        bool kernel_size_ok = false;
        if (current_merging_kernel.tp == KernelType::fusion) {
          assert(current_merging_kernel.touching_qubits.empty());
          assert(current_status.open_kernels[kernel_index]
                     .touching_qubits.empty());
          kernel_size_ok = current_merging_kernel.active_qubits.size() +
                               current_status.open_kernels[kernel_index]
                                   .active_qubits.size() <=
                           max_fusion_kernel_size;
        } else {
          if (current_merging_kernel.active_qubits.size() +
                  current_status.open_kernels[kernel_index]
                      .active_qubits.size() <=
              shared_memory_kernel_size) {
            kernel_size_ok = true;
          } else {
            // Check cacheline qubits.
            int num_cacheline_qubits = 0;
            for (const auto &qubit : current_merging_kernel.active_qubits) {
              if (is_shared_memory_cacheline_qubit(qubit)) {
                num_cacheline_qubits++;
              }
            }
            for (const auto &qubit :
                 current_status.open_kernels[kernel_index].active_qubits) {
              if (is_shared_memory_cacheline_qubit(qubit)) {
                num_cacheline_qubits++;
              }
            }
            kernel_size_ok = current_merging_kernel.active_qubits.size() +
                                 current_status.open_kernels[kernel_index]
                                     .active_qubits.size() -
                                 num_cacheline_qubits <=
                             shared_memory_kernel_size;
          }
        }
        if (!kernel_size_ok) {
          // The kernel would be too large if we merge this one.
          // Continue to the next one.
          return;
        }
        // Merge this kernel.
        auto new_merging_kernel = current_merging_kernel;
        new_merging_kernel.active_qubits.insert(
            new_merging_kernel.active_qubits.end(),
            current_status.open_kernels[kernel_index].active_qubits.begin(),
            current_status.open_kernels[kernel_index].active_qubits.end());
        // We do not record the set of |touching_qubits| here;
        // we will compute them right before doing the DP transition.
        kernel_merged[kernel_index] = true;
        // Continue the search.
        this_ref(this_ref, current_gate_kernel, new_merging_kernel, cost,
                 touching_set_index, kernel_index + 1);
        // Restore the |kernel_merged| status.
        kernel_merged[kernel_index] = false;
      };
      // Start the search with touching_set_index=-1 and
      // kernel_index=num_kernels, so that we can decide whether to merge the
      // first "touching set" with the current gate or not inside the search.
      // For shared-memory kernels, we account for the gate cost now.
      search_merging_kernels(
          search_merging_kernels,
          /*current_gate_kernel=*/current_single_gate_shared_memory_kernel,
          /*current_merging_kernel=*/KernelInDP(),
          /*cost=*/current_cost + current_gate_cost,
          /*touching_set_index=*/-1, /*kernel_index=*/num_kernels);

      update_absorbing_kernels_for_new_fusion_kernel();
      search_merging_kernels(
          search_merging_kernels,
          /*current_gate_kernel=*/current_single_gate_fusion_kernel,
          /*current_merging_kernel=*/KernelInDP(), /*cost=*/current_cost,
          /*touching_set_index=*/-1, /*kernel_index=*/num_kernels);
    }
  } // end of for (int i = 0; i < num_gates; i++)
  if (f[num_gates & 1].empty()) {
    return false;
  }
  KernelCostType min_cost;
  LocalSchedule result_schedule;
  for (auto &it : f[num_gates & 1]) {
    // Compute the end schedule and get the one with minimal total cost.
    KernelCostType cost;
    std::vector<KernelInDP> end_schedule;
    compute_end_schedule(kernel_cost, it.first.open_kernels, cost,
                         &end_schedule);
    if (result_schedule.kernels.empty() || cost + it.second.first < min_cost) {
      min_cost = cost + it.second.first;
      result_schedule = it.second.second;
      result_schedule.kernels.insert(result_schedule.kernels.end(),
                                     end_schedule.begin(), end_schedule.end());
    }
  }
  if (result_schedule.kernels.empty()) {
    // All gates in this schedule are global.
    // We need to create an empty kernel for them.
    result_schedule.kernels.emplace_back(std::vector<int>(), std::vector<int>(),
                                         KernelType::fusion);
  }
  // Translate the |result_schedule| into |kernels|.
  kernels.reserve(result_schedule.kernels.size());
  std::vector<bool> executed(num_gates, false);
  cost_ = min_cost;
  int start_gate_index = 0; // an optimization
  for (auto &s : result_schedule.kernels) {
    // Greedily execute a kernel.
    CircuitSeq current_seq(num_qubits, sequence_.get_num_input_parameters());
    std::vector<bool> active_in_kernel(num_qubits, false);
    for (auto &index : s.active_qubits) {
      active_in_kernel[index] = true;
    }
    std::vector<bool> touched_in_kernel(num_qubits, false);
    for (auto &index : s.touching_qubits) {
      touched_in_kernel[index] = true;
    }
    // A non-insular qubit of a gate blocks this qubit.
    std::vector<bool> qubit_blocked(num_qubits, false);
    // An insular qubit of a gate blocks this qubit.
    std::vector<bool> qubit_insularly_blocked(num_qubits, false);
    for (int i = start_gate_index; i < num_gates; i++) {
      if (executed[i]) {
        continue;
      }
      bool executable = true;
      for (auto &wire : sequence_.gates[i]->input_wires) {
        if (wire->is_qubit() && qubit_blocked[wire->index]) {
          executable = false;
          break;
        }
      }
      std::vector<int> current_non_insular_indices;
      if (!non_insular_qubit_indices.empty()) {
        current_non_insular_indices = non_insular_qubit_indices[i];
      } else {
        current_non_insular_indices =
            sequence_.gates[i]->get_non_insular_qubit_indices();
      }
      for (auto &qubit : current_non_insular_indices) {
        if (qubit_insularly_blocked[qubit]) {
          executable = false;
          break;
        }
      }
      if (s.tp == KernelType::fusion) {
        // For fusion kernels, we require all local qubits to be active.
        for (auto &qubit : sequence_.gates[i]->get_qubit_indices()) {
          if (is_local_qubit(qubit) && !active_in_kernel[qubit]) {
            executable = false;
            break;
          }
        }
      } else {
        // For shared-memory kernels, we only require all non-insular qubits
        // to be active.
        for (auto &qubit : current_non_insular_indices) {
          if (!active_in_kernel[qubit]) {
            executable = false;
            break;
          }
        }
      }
      if (executable) {
        // Execute the gate.
        executed[i] = true;
        current_seq.add_gate(sequence_.gates[i].get());
      } else {
        // Block the non-insular qubits.
        for (auto &qubit : current_non_insular_indices) {
          qubit_blocked[qubit] = true;
        }
        // "Insularly" block the insular qubits so that we cannot execute any
        // gate with the same non-insular qubit in this kernel.
        // TODO: if we execute any gate with the same insular qubit later
        //  in this kernel, we need to adjust the states for X gates
        //  accordingly.
        for (auto &qubit : sequence_.gates[i]->get_insular_qubit_indices()) {
          qubit_insularly_blocked[qubit] = true;
        }
      }
    }
    kernels.emplace_back(current_seq, s.active_qubits, s.tp);
    while (start_gate_index < num_gates && executed[start_gate_index]) {
      start_gate_index++;
    }
  }
  if (start_gate_index != num_gates) {
    std::cerr << "Gate number " << start_gate_index
              << " is not executed yet in the kernel schedule." << std::endl;
    assert(false);
  }
  return true;
}

int Schedule::get_num_kernels() const { return (int)kernels.size(); }

void Schedule::print_kernel_schedule() const {
  const int num_kernels = get_num_kernels();
  std::cout << "Kernel schedule with " << num_kernels
            << " kernels: cost = " << cost_ << ", local qubits";
  for (auto &qubit : local_qubit_) {
    std::cout << " " << qubit;
  }
  std::cout << std::endl;
  for (int i = 0; i < num_kernels; i++) {
    std::cout << "Kernel " << i << ": ";
    std::cout << kernels[i].to_string() << std::endl;
  }
}

std::vector<Schedule>
get_schedules(const CircuitSeq &sequence,
              const std::vector<std::vector<int>> &local_qubits,
              const KernelCost &kernel_cost, Context *ctx,
              bool attach_single_qubit_gates) {
  constexpr bool debug = false;
  std::vector<Schedule> result;
  result.reserve(local_qubits.size());
  const int num_qubits = sequence.get_num_qubits();
  const int num_gates = sequence.get_num_gates();
  std::vector<int> current_local_qubit_layout;
  std::vector<int> current_global_qubit_layout;
  std::vector<bool> local_qubit_mask(num_qubits, false);
  std::vector<bool> executed(num_gates, false);
  int start_gate_index = 0; // an optimization

  // Variables for |attach_single_qubit_gates|=true.
  // |single_qubit_gate_indices[i]|: the indices of single-qubit gates at the
  // |i|-th qubit after the last multi-qubit gate at the |i|-th qubit,
  // ignoring all global qubits;
  // only computed when |attach_single_qubit_gates| is true
  std::vector<std::vector<int>> single_qubit_gate_indices(num_qubits);
  // |has_dense_single_qubit_gate[i]|: if |single_qubit_gate_indices[i]|
  // includes a dense single-qubit gate or not;
  // only computed when |attach_single_qubit_gates| is true
  std::vector<bool> has_dense_single_qubit_gate(num_qubits, false);
  // |last_gate_index[i]|: the last gate touching the |i|-th qubit;
  // only computed when |attach_single_qubit_gates| is true
  std::vector<int> last_gate_index(num_qubits, -1);
  // |gate_indices[gate_string]|: the indices of gates with to_string() being
  // |gate_string|, used to restore the single-qubit gates;
  // only computed when |attach_single_qubit_gates| is true
  std::unordered_map<std::string, std::queue<int>> gate_indices;
  // |attach_front[j]|: the indices of single-qubit gates to be attached
  // to the |j|-th gate (which should be a multi-qubit gate);
  // only computed when |attach_single_qubit_gates| is true
  std::vector<std::vector<int>> attach_front(num_gates);
  std::vector<std::vector<int>> attach_back(num_gates);
  // |dp_sequence_position[j]|: the stage and the position of the |j|-th gate in
  // the original sequence in |current_seq|;
  // only computed when |attach_single_qubit_gates| is true
  // |non_insular_qubit_indices[i][j]|: the set of non-insular qubits of the
  // |j|-th gate in |current_seq| in the |i|-th stage;
  // only computed when |attach_single_qubit_gates| is true
  // |shared_memory_gate_costs[i][j]|: the total cost of gates in shared-memory
  // kernels after attaching the gates;
  // only computed when |attach_single_qubit_gates| is true
  std::vector<std::pair<int, int>> dp_sequence_position(num_gates);
  std::vector<std::vector<std::vector<int>>> non_insular_qubit_indices(
      local_qubits.size());
  std::vector<std::vector<KernelCostType>> shared_memory_gate_costs(
      local_qubits.size());

  // |should_flip_control_qubit[i][j]|: if the |j|-th qubit in the |i|-th gate
  // should be flipped if we remove all single-qubit sparse non-diagonal gates
  // (e.g., X gate)
  // |flipping[i]|: if the |i|-th qubit is flipped now if we remove them
  std::vector<std::vector<bool>> should_flip_control_qubit(num_gates);
  std::vector<bool> flipping(num_qubits, false);

  if (debug) {
    std::cout << "get_schedules for " << sequence.to_string(true) << std::endl;
  }

  for (int i = 0; i < num_gates; i++) {
    if (sequence.gates[i]->gate->get_num_qubits() == 1 &&
        sequence.gates[i]->gate->is_sparse() &&
        !sequence.gates[i]->gate->is_diagonal()) {
      flipping[sequence.gates[i]->get_min_qubit_index()].flip();
    } else if (sequence.gates[i]->gate->get_num_control_qubits() > 0) {
      for (auto &qubit : sequence.gates[i]->get_control_qubit_indices()) {
        should_flip_control_qubit[i].push_back(flipping[qubit]);
      }
    }
  }

  int num_stage = -1;
  for (auto &local_qubit : local_qubits) {
    num_stage++;
    // Convert vector<int> to vector<bool>.
    local_qubit_mask.assign(num_qubits, false);
    if (debug) {
      std::cout << "local qubits: ";
    }
    for (auto &i : local_qubit) {
      local_qubit_mask[i] = true;
      if (debug) {
        std::cout << i << " ";
      }
    }
    if (debug) {
      std::cout << std::endl;
    }
    if (current_local_qubit_layout.empty()) {
      // First iteration. Take the initial layout from |local_qubits[0]|.
      current_local_qubit_layout = local_qubit;
      current_global_qubit_layout.reserve(num_qubits - local_qubit.size());
      // The global qubits are sorted in ascending order.
      for (int i = 0; i < num_qubits; i++) {
        if (!local_qubit_mask[i]) {
          current_global_qubit_layout.push_back(i);
        }
      }
    } else {
      // Update the layout.
      // We should have the same number of local qubits.
      assert(local_qubit.size() == current_local_qubit_layout.size());
      int num_global_swaps = 0;
      for (auto &i : current_global_qubit_layout) {
        if (local_qubit_mask[i]) {
          // A global-to-local swap.
          num_global_swaps++;
        }
      }
      int should_swap_ptr = 0;
      for (int i = (int)current_local_qubit_layout.size() - num_global_swaps;
           i < (int)current_local_qubit_layout.size(); i++) {
        if (!local_qubit_mask[current_local_qubit_layout[i]]) {
          // This qubit should be swapped with a global qubit,
          // and it is already at the correct position.
          continue;
        }
        while (local_qubit_mask[current_local_qubit_layout[should_swap_ptr]]) {
          should_swap_ptr++;
        }
        // Find the first local qubit that should be swapped with a global
        // qubit, and perform a local swap.
        std::swap(current_local_qubit_layout[i],
                  current_local_qubit_layout[should_swap_ptr]);
        should_swap_ptr++;
      }
      for (auto &i : current_global_qubit_layout) {
        if (local_qubit_mask[i]) {
          // Swap the new local qubit with the qubit at its corresponding
          // position.
          std::swap(
              i,
              current_local_qubit_layout
                  [(int)current_local_qubit_layout.size() - num_global_swaps]);
          num_global_swaps--;
        }
      }
    }
    if (debug) {
      std::cout << "current layout: local ";
      for (auto &i : current_local_qubit_layout) {
        std::cout << i << " ";
      }
      std::cout << ", global ";
      for (auto &i : current_global_qubit_layout) {
        std::cout << i << " ";
      }
      std::cout << std::endl;
    }

    // Returns the total shared-memory gate costs.
    auto do_attach_single_qubit_gates =
        [&single_qubit_gate_indices, &has_dense_single_qubit_gate, &kernel_cost,
         &sequence, &debug](std::vector<std::vector<int>> &attach_to,
                            int gate_id, int qubit) {
          if (debug) {
            if (!single_qubit_gate_indices[qubit].empty()) {
              std::cout << "Attach single qubit gates";
              for (auto &index : single_qubit_gate_indices[qubit]) {
                std::cout << " " << index;
              }
              std::cout << " to " << gate_id << std::endl;
            }
          }
          KernelCostType result = 0;
          for (auto &index : single_qubit_gate_indices[qubit]) {
            result += kernel_cost.get_shared_memory_gate_cost(
                sequence.gates[index]->gate->tp);
          }
          attach_to[gate_id].insert(attach_to[gate_id].end(),
                                    single_qubit_gate_indices[qubit].begin(),
                                    single_qubit_gate_indices[qubit].end());
          single_qubit_gate_indices[qubit].clear();
          has_dense_single_qubit_gate[qubit] = false;
          return result;
        };

    CircuitSeq current_seq(num_qubits, sequence.get_num_input_parameters());
    std::vector<bool> qubit_blocked(num_qubits, false);
    for (int i = start_gate_index; i < num_gates; i++) {
      if (executed[i]) {
        continue;
      }
      // XXX: Assume that there are no parameter gates.
      assert(sequence.gates[i]->gate->is_quantum_gate());
      // Greedily try to execute the gates.
      bool executable = true;
      for (auto &wire : sequence.gates[i]->input_wires) {
        if (wire->is_qubit() && qubit_blocked[wire->index]) {
          executable = false;
          break;
        }
      }
      std::vector<int> non_insular_qubits =
          sequence.gates[i]->get_non_insular_qubit_indices();
      for (auto &qubit : non_insular_qubits) {
        if (!local_qubit_mask[qubit]) {
          executable = false;
          break;
        }
      }
      if (executable) {
        // Execute the gate.
        executed[i] = true;
        if (attach_single_qubit_gates) {
          // Get local qubits.
          std::vector<int> current_local_qubits;
          for (auto &wire : sequence.gates[i]->input_wires) {
            if (wire->is_qubit() && local_qubit_mask[wire->index]) {
              current_local_qubits.push_back(wire->index);
            }
          }
          if (current_local_qubits.size() == 1) {
            // Do not put single-qubit gates into |current_seq|.
            // Update |single_qubit_gate_indices| and
            // |has_dense_single_qubit_gate| instead.
            single_qubit_gate_indices[current_local_qubits[0]].push_back(i);
            // Note that global qubits, if any, must be control qubits;
            // so we can directly use Gate::is_sparse() here to check if
            // the matrix on the local qubit is sparse.
            if (!sequence.gates[i]->gate->is_sparse()) {
              has_dense_single_qubit_gate[current_local_qubits[0]] = true;
            }

            if (sequence.gates[i]->gate->tp == GateType::cx ||
                sequence.gates[i]->gate->tp == GateType::ccx) {
              std::cerr << "Warning: CX or CCX gate attached to another gate. "
                        << "The schedule may be different for each device, "
                        << "but we only output the schedule for the device "
                        << "where CX or CCX is not executed here." << std::endl;
            }
          } else {
            // Either a global gate or a multi-qubit gate.
            current_seq.add_gate(sequence.gates[i].get());
            // Attach single-qubit gates to this gate.
            auto gate_cost = kernel_cost.get_shared_memory_gate_cost(
                sequence.gates[i]->gate->tp);
            for (auto &qubit : non_insular_qubits) {
              gate_cost += do_attach_single_qubit_gates(attach_front, i, qubit);
            }
            for (auto &qubit : sequence.gates[i]->get_insular_qubit_indices()) {
              if (has_dense_single_qubit_gate[qubit]) {
                // We need to attach single-qubit gates to an insular qubit
                // if there is any dense single-qubit gate.
                if (last_gate_index[qubit] != -1 &&
                    std::find(
                        non_insular_qubit_indices
                            [dp_sequence_position[last_gate_index[qubit]].first]
                            [dp_sequence_position[last_gate_index[qubit]]
                                 .second]
                                .begin(),
                        non_insular_qubit_indices
                            [dp_sequence_position[last_gate_index[qubit]].first]
                            [dp_sequence_position[last_gate_index[qubit]]
                                 .second]
                                .end(),
                        qubit) !=
                        non_insular_qubit_indices
                            [dp_sequence_position[last_gate_index[qubit]].first]
                            [dp_sequence_position[last_gate_index[qubit]]
                                 .second]
                                .end()) {
                  // It's better to attach them to the last gate because
                  // it's already non-insular.
                  gate_cost += do_attach_single_qubit_gates(
                      attach_back, last_gate_index[qubit], qubit);
                } else {
                  gate_cost +=
                      do_attach_single_qubit_gates(attach_front, i, qubit);
                  // This qubit is no longer insular.
                  non_insular_qubits.push_back(qubit);
                }
              }
            }

            // Only update them for gates that are passed into the DP.
            // Update |dp_sequence_position|, |non_insular_qubit_indices|,
            // and |shared_memory_gate_costs|.
            dp_sequence_position[i] = std::make_pair(
                num_stage, (int)non_insular_qubit_indices[num_stage].size());
            non_insular_qubit_indices[num_stage].push_back(
                std::move(non_insular_qubits));
            shared_memory_gate_costs[num_stage].push_back(gate_cost);
            // Update |last_gate_index|.
            for (auto &wire : sequence.gates[i]->input_wires) {
              if (wire->is_qubit()) {
                last_gate_index[wire->index] = i;
              }
            }
            // Update |gate_indices|.
            gate_indices[sequence.gates[i]->to_string()].push(i);
          }
        } else {
          current_seq.add_gate(sequence.gates[i].get());
        }
      } else {
        // Block the qubits.
        for (auto &wire : sequence.gates[i]->input_wires) {
          if (wire->is_qubit()) {
            qubit_blocked[wire->index] = true;
          }
        }
      }
    }

    auto attach_back_and_update_insular_and_cost =
        [&has_dense_single_qubit_gate, &non_insular_qubit_indices,
         &shared_memory_gate_costs, &dp_sequence_position, &last_gate_index,
         &do_attach_single_qubit_gates, &attach_back](int qubit) {
          if (has_dense_single_qubit_gate[qubit]) {
            if (std::find(
                    non_insular_qubit_indices
                        [dp_sequence_position[last_gate_index[qubit]].first]
                        [dp_sequence_position[last_gate_index[qubit]].second]
                            .begin(),
                    non_insular_qubit_indices
                        [dp_sequence_position[last_gate_index[qubit]].first]
                        [dp_sequence_position[last_gate_index[qubit]].second]
                            .end(),
                    qubit) ==
                non_insular_qubit_indices
                    [dp_sequence_position[last_gate_index[qubit]].first]
                    [dp_sequence_position[last_gate_index[qubit]].second]
                        .end()) {
              // This qubit is no longer insular.
              non_insular_qubit_indices
                  [dp_sequence_position[last_gate_index[qubit]].first]
                  [dp_sequence_position[last_gate_index[qubit]].second]
                      .push_back(qubit);
            }
          }
          shared_memory_gate_costs
              [dp_sequence_position[last_gate_index[qubit]].first]
              [dp_sequence_position[last_gate_index[qubit]].second] +=
              do_attach_single_qubit_gates(attach_back, last_gate_index[qubit],
                                           qubit);
        };

    if (num_stage == (int)local_qubits.size() - 1) {
      // The last stage. We need to execute all the remaining single-qubit
      // gates.
      for (int qubit = 0; qubit < num_qubits; qubit++) {
        if (!single_qubit_gate_indices[qubit].empty()) {
          if (last_gate_index[qubit] == -1) {
            std::cerr << "Qubit " << qubit << " is not entangled." << std::endl;
            assert(false);
          }
          attach_back_and_update_insular_and_cost(qubit);
        }
      }
    } else {
      // Not the last stage, but we need to execute any single-qubit gate
      // that operates on a qubit that will become global in the next stage.
      for (int qubit = 0; qubit < num_qubits; qubit++) {
        if (!single_qubit_gate_indices[qubit].empty() &&
            std::find(local_qubits[num_stage + 1].begin(),
                      local_qubits[num_stage + 1].end(),
                      qubit) == local_qubits[num_stage + 1].end()) {
          if (last_gate_index[qubit] == -1) {
            if (debug) {
              std::cout << "Single-qubit gate "
                        << single_qubit_gate_indices[qubit][0] << ": "
                        << sequence.gates[single_qubit_gate_indices[qubit][0]]
                               ->to_string()
                        << "not removed." << std::endl;
            }
            int gate_id = single_qubit_gate_indices[qubit][0];
            current_seq.add_gate(sequence.gates[gate_id].get());
            // Only update them for gates that are passed into the DP.
            // Update |dp_sequence_position| and |non_insular_qubit_indices|.
            dp_sequence_position[gate_id] = std::make_pair(
                num_stage, (int)non_insular_qubit_indices[num_stage].size());
            non_insular_qubit_indices[num_stage].push_back(
                sequence.gates[gate_id]->get_non_insular_qubit_indices());
            shared_memory_gate_costs[num_stage].push_back(
                kernel_cost.get_shared_memory_gate_cost(
                    sequence.gates[gate_id]->gate->tp));
            // Update |last_gate_index|.
            for (auto &wire : sequence.gates[gate_id]->input_wires) {
              if (wire->is_qubit()) {
                last_gate_index[wire->index] = gate_id;
              }
            }
            // Update |gate_indices|.
            gate_indices[sequence.gates[gate_id]->to_string()].push(gate_id);
            // Remove the first gate because it is already added to the DP.
            single_qubit_gate_indices[qubit].erase(
                single_qubit_gate_indices[qubit].begin());
          }
          attach_back_and_update_insular_and_cost(qubit);
        }
      }
    }

    result.emplace_back(
        current_seq, current_local_qubit_layout, current_global_qubit_layout,
        kernel_cost.get_shared_memory_num_cacheline_qubits(), ctx);
    while (start_gate_index < num_gates && executed[start_gate_index]) {
      start_gate_index++;
    }
  }
  if (start_gate_index != num_gates) {
    std::cerr << "Gate number " << start_gate_index
              << " is not executed yet in the schedule." << std::endl;
    assert(false);
  }
  num_stage = -1;
  for (auto &schedule : result) {
    num_stage++;
    schedule.compute_kernel_schedule(kernel_cost,
                                     non_insular_qubit_indices[num_stage],
                                     shared_memory_gate_costs[num_stage]);
  }
  if (attach_single_qubit_gates) {
    // Restore the single-qubit gates.
    // Adjust controlled gates.
    flipping.assign(num_qubits, false);
    for (auto &schedule : result) {
      for (int i = 0; i < schedule.get_num_kernels(); i++) {
        // Returns the number of gates inserted.
        auto insert_single_qubit_gates =
            [&schedule, &i, &sequence, &flipping](
                const std::vector<int> &gate_indices, int insert_location) {
              for (auto &gate_index : gate_indices) {
                if (sequence.gates[gate_index]->gate->get_num_qubits() == 1 &&
                    sequence.gates[gate_index]->gate->is_sparse() &&
                    !sequence.gates[gate_index]->gate->is_diagonal()) {
                  flipping[sequence.gates[gate_index]->get_min_qubit_index()]
                      .flip();
                }
                schedule.kernels[i].gates.insert_gate(
                    insert_location, sequence.gates[gate_index].get());
                insert_location++;
              }
              return gate_indices.size();
            };

        // Purposely using |schedule.kernels[i].gates.gates.size()| because we
        // may modify |schedule.kernels[i].gates.gates| in this loop.
        for (int j = 0; j < (int)schedule.kernels[i].gates.gates.size(); j++) {
          auto &gate = schedule.kernels[i].gates.gates[j];
          auto &gate_indices_queue = gate_indices[gate->to_string()];
          assert(!gate_indices_queue.empty());

          const int original_index = gate_indices_queue.front();
          gate_indices_queue.pop();
          j += insert_single_qubit_gates(attach_front[original_index], j);

          // We execute the gate now.
          std::vector<bool> control_state;
          if (schedule.kernels[i]
                  .gates.gates[j]
                  ->gate->get_num_control_qubits() > 0) {
            assert(schedule.kernels[i]
                       .gates.gates[j]
                       ->gate->get_num_control_qubits() ==
                   (int)should_flip_control_qubit[original_index].size());
            int k = 0;
            for (auto &qubit : schedule.kernels[i]
                                   .gates.gates[j]
                                   ->get_control_qubit_indices()) {
              control_state.push_back(
                  should_flip_control_qubit[original_index][k] ^ flipping[k] ^
                  1);
              k++;
            }
            if (!std::all_of(control_state.begin(), control_state.end(),
                             [](bool v) { return v; })) {
              // Not a simple controlled gate
              schedule.kernels[i].gates.gates[j]->gate =
                  ctx->get_general_controlled_gate(
                      schedule.kernels[i].gates.gates[j]->gate->tp,
                      control_state);
            }
          }
          if (schedule.kernels[i].gates.gates[j]->gate->get_num_qubits() == 1 &&
              schedule.kernels[i].gates.gates[j]->gate->is_sparse() &&
              !schedule.kernels[i].gates.gates[j]->gate->is_diagonal()) {
            flipping[schedule.kernels[i].gates.gates[j]->get_min_qubit_index()]
                .flip();
          }

          j += insert_single_qubit_gates(attach_back[original_index], j + 1);
        }
      }
    }
  }
  return result;
}

std::vector<std::vector<int>>
compute_local_qubits_with_ilp(const CircuitSeq &sequence, int num_local_qubits,
                              Context *ctx, PythonInterpreter *interpreter,
                              int answer_start_with) {
  const int num_qubits = sequence.get_num_qubits();
  const int num_gates = sequence.get_num_gates();
  std::vector<std::vector<int>> circuit_gate_qubits;
  std::vector<int> circuit_gate_executable_type;
  std::unordered_map<CircuitGate *, int> gate_index;
  std::vector<std::vector<int>> out_gate(num_gates);
  circuit_gate_qubits.reserve(num_gates);
  circuit_gate_executable_type.reserve(num_gates);
  gate_index.reserve(num_gates);
  for (int i = 0; i < num_gates; i++) {
    circuit_gate_qubits.push_back(sequence.gates[i]->get_qubit_indices());
    int executable_type;
    // 0 is always executable
    // 1 is the target qubits must be local-only
    // 2 is local-only
    if (sequence.gates[i]->gate->get_num_qubits() == 1) {
      if (sequence.gates[i]->gate->is_sparse()) {
        // A single-qubit gate is always executable if it is "sparse".
        executable_type = 0;
      } else {
        // Otherwise, we require the qubit to be local-only.
        executable_type = 2;
      }
    } else if (sequence.gates[i]->gate->get_num_control_qubits() > 0) {
      if (sequence.gates[i]->gate->is_symmetric()) {
        // A controlled gate is always executable if every qubit can be a
        // control qubit.
        executable_type = 0;
      } else {
        // The target qubits must be local-only.
        // We assume there is only 1 target qubit in the Python code.
        assert(sequence.gates[i]->gate->get_num_control_qubits() ==
               sequence.gates[i]->gate->get_num_qubits() - 1);
        executable_type = 1;
      }
    } else {
      // For all non-controlled multi-qubit gates,
      // we require all qubits to be local-only.
      // Note: although the SWAP gate can be executed globally,
      // it cannot be executed when it's partial global and partial local,
      // so we restrict it to be local-only here.
      executable_type = 2;
    }
    circuit_gate_executable_type.push_back(executable_type);
    gate_index[sequence.gates[i].get()] = i;
  }
  for (int i = 0; i < num_gates; i++) {
    for (const auto &output_wire : sequence.gates[i]->output_wires) {
      for (const auto &output_gate : output_wire->output_gates) {
        out_gate[i].push_back(gate_index[output_gate]);
      }
    }
  }
  for (int num_iterations = answer_start_with; true; num_iterations++) {
    auto result = interpreter->solve_ilp(
        circuit_gate_qubits, circuit_gate_executable_type, out_gate, num_qubits,
        num_local_qubits, num_iterations);
    if (!result.empty()) {
      return result;
    }
  }
}
} // namespace quartz
