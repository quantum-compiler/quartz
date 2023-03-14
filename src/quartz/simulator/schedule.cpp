#include "schedule.h"
#include "quartz/pybind/pybind.h"

#include <queue>
#include <stack>
#include <unordered_set>

namespace quartz {

Schedule::Schedule(const CircuitSeq &sequence,
                   const std::vector<int> &local_qubit,
                   const std::vector<int> &global_qubit, Context *ctx)
    : cost_(), sequence_(sequence), local_qubit_(local_qubit),
      global_qubit_(global_qubit), ctx_(ctx) {
  assert(local_qubit.size() + global_qubit.size() == sequence.get_num_qubits());
  local_qubit_mask_.assign(sequence.get_num_qubits(), false);
  for (auto &i : local_qubit) {
    local_qubit_mask_[i] = true;
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

bool Schedule::compute_end_schedule(
    const KernelCost &kernel_cost,
    const std::vector<std::pair<std::vector<int>, KernelType>> &kernels,
    const std::vector<bool> &is_shared_memory_cacheline_qubit,
    KernelCostType &result_cost,
    std::vector<std::pair<std::vector<int>, KernelType>> *result_kernels) {
  const auto &kernel_costs = kernel_cost.get_fusion_kernel_costs();
  const int num_kernels = (int)kernels.size();
  const int optimal_fusion_kernel_size =
      kernel_cost.get_optimal_fusion_kernel_size();
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
    assert(kernels[i].first.size() > 0);
    if (kernels[i].second == KernelType::fusion) {
      if (kernels[i].first.size() >= optimal_fusion_kernel_size) {
        // Greedily try to use the optimal kernel size for fusion kernels.
        // Copy the large fusion kernels to the result.
        if (result_kernels != nullptr) {
          result_kernels->emplace_back(kernels[i]);
        }
        result_cost += kernel_costs[kernels[i].first.size()];
      } else {
        fusion_kernels_of_size[kernels[i].first.size()].push_back(i);
      }
    } else if (kernels[i].second == KernelType::shared_memory) {
      int kernel_size = kernels[i].first.size();
      for (auto &qubit : kernels[i].first) {
        if (is_shared_memory_cacheline_qubit[qubit]) {
          // Exclude shared-memory cacheline qubits.
          kernel_size--;
        }
      }
      if (kernel_size >= max_shared_memory_kernel_size) {
        assert(kernel_size == max_shared_memory_kernel_size);
        // Copy the large shared-memory kernels to the result.
        if (result_kernels != nullptr) {
          result_kernels->emplace_back(kernels[i]);
        }
        result_cost += kernel_cost.get_shared_memory_init_cost();
      } else {
        shared_memory_kernels_of_size[kernel_size].push_back(i);
      }
    } else {
      assert(false);
    }
  }

  // Greedily merge the small fusion kernels.
  while (true) {
    bool has_any_remaining_kernel = false;
    std::vector<int> current_kernel;
    int current_kernel_size = 0;
    for (int i = optimal_fusion_kernel_size - 1; i >= 1; i--) {
      while (!fusion_kernels_of_size[i].empty()) {
        if (current_kernel_size + i <= optimal_fusion_kernel_size) {
          // Add a kernel of size i to the current kernel.
          if (result_kernels != nullptr) {
            current_kernel.insert(
                current_kernel.end(),
                kernels[fusion_kernels_of_size[i].back()].first.begin(),
                kernels[fusion_kernels_of_size[i].back()].first.end());
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
        std::sort(current_kernel.begin(), current_kernel.end());
        result_kernels->emplace_back(std::move(current_kernel),
                                     KernelType::fusion);
      }
    }
    if (!has_any_remaining_kernel) {
      break;
    }
  }

  // Greedily merge the small shared-memory kernels.
  while (true) {
    bool has_any_remaining_kernel = false;
    std::vector<int> current_kernel;
    int current_kernel_size = 0;
    for (int i = max_shared_memory_kernel_size - 1; i >= 1; i--) {
      while (!shared_memory_kernels_of_size[i].empty()) {
        if (current_kernel_size + i <= max_shared_memory_kernel_size) {
          // Add a kernel of size i to the current kernel.
          if (result_kernels != nullptr) {
            current_kernel.insert(
                current_kernel.end(),
                kernels[shared_memory_kernels_of_size[i].back()].first.begin(),
                kernels[shared_memory_kernels_of_size[i].back()].first.end());
          }
          current_kernel_size += i;
          shared_memory_kernels_of_size[i].pop_back();
        } else {
          has_any_remaining_kernel = true;
          break;
        }
      }
    }
    if (current_kernel_size != 0) {
      result_cost += kernel_cost.get_shared_memory_init_cost();
      if (result_kernels != nullptr) {
        std::sort(current_kernel.begin(), current_kernel.end());
        result_kernels->emplace_back(std::move(current_kernel),
                                     KernelType::shared_memory);
      }
    }
    if (!has_any_remaining_kernel) {
      break;
    }
  }
  return true;
}

bool Schedule::compute_kernel_schedule(const KernelCost &kernel_cost) {
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
  std::vector<bool> is_shared_memory_cacheline_qubit(num_qubits, false);
  assert(local_qubit_.size() >= shared_memory_cacheline_size);
  for (int i = 0; i < shared_memory_cacheline_size; i++) {
    // These are the shared-memory cacheline qubits.
    is_shared_memory_cacheline_qubit[local_qubit_[i]] = true;
  }
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
      // XXX: We are not using |absorbing_qubits| to compute the hash.
      // If we decide to use it in the future, we need to search for every
      // place we modify |absorbing_qubits| and update all places where we
      // update the hash manually.
      hash = 0;
      for (const auto &s : sets) {
        hash ^= get_hash(s.first) + (size_t)s.second;
      }
    }
    void insert_set(const std::vector<int> &s, KernelType tp) {
      int insert_position = 0;
      while (insert_position < (int)sets.size() &&
             sets[insert_position].first[0] < s[0]) {
        insert_position++;
      }
      sets.insert(sets.begin() + insert_position, std::make_pair(s, tp));
    }
    [[nodiscard]] bool check_valid() const {
      std::vector<bool> has_qubit;
      for (int i = 0; i < (int)sets.size(); i++) {
        for (int j = 0; j < (int)sets[i].first.size(); j++) {
          while (sets[i].first[j] >= has_qubit.size()) {
            has_qubit.push_back(false);
          }
          if (has_qubit[sets[i].first[j]]) {
            std::cerr << "Invalid status: qubit " << sets[i].first[j]
                      << " appears twice." << std::endl;
            std::cerr << to_string() << std::endl;
            return false;
          }
          has_qubit[sets[i].first[j]] = true;
        }
      }
      for (int i = 0; i < (int)absorbing_qubits.size(); i++) {
        for (int j = 0; j < (int)absorbing_qubits[i].first.size(); j++) {
          while (absorbing_qubits[i].first[j] >= has_qubit.size()) {
            has_qubit.push_back(false);
          }
          if (has_qubit[absorbing_qubits[i].first[j]]) {
            std::cerr << "Invalid status: qubit "
                      << absorbing_qubits[i].first[j] << " appears twice."
                      << std::endl;
            std::cerr << to_string() << std::endl;
            return false;
          }
          has_qubit[absorbing_qubits[i].first[j]] = true;
        }
      }
      return true;
    }
    bool operator==(const Status &b) const {
      if (sets.size() != b.sets.size()) {
        return false;
      }
      for (int i = 0; i < (int)sets.size(); i++) {
        if (sets[i].first.size() != b.sets[i].first.size()) {
          return false;
        }
        if (sets[i].second != b.sets[i].second) {
          return false;
        }
        for (int j = 0; j < (int)sets[i].first.size(); j++) {
          if (sets[i].first[j] != b.sets[i].first[j]) {
            return false;
          }
        }
      }
      if (absorbing_qubits.size() != b.absorbing_qubits.size()) {
        return false;
      }
      for (int i = 0; i < (int)absorbing_qubits.size(); i++) {
        if (absorbing_qubits[i].first.size() !=
            b.absorbing_qubits[i].first.size()) {
          return false;
        }
        for (int j = 0; j < (int)absorbing_qubits[i].first.size(); j++) {
          if (absorbing_qubits[i].first[j] != b.absorbing_qubits[i].first[j]) {
            return false;
          }
        }
      }
      return true;
    }
    [[nodiscard]] std::string to_string() const {
      std::string result;
      result += "{";
      for (int i = 0; i < (int)sets.size(); i++) {
        result += kernel_type_name(sets[i].second);
        result += "{";
        for (int j = 0; j < (int)sets[i].first.size(); j++) {
          result += std::to_string(sets[i].first[j]);
          if (j != (int)sets[i].first.size() - 1) {
            result += ", ";
          }
        }
        result += "}";
        if (i != (int)sets.size() - 1) {
          result += ", ";
        }
      }
      if (!absorbing_qubits.empty()) {
        if (!sets.empty()) {
          result += ", ";
        }
        result += "absorbing {";
        for (int i = 0; i < (int)absorbing_qubits.size(); i++) {
          result += kernel_type_name(absorbing_qubits[i].second);
          result += "{";
          for (int j = 0; j < (int)absorbing_qubits[i].first.size(); j++) {
            result += std::to_string(absorbing_qubits[i].first[j]);
            if (j != (int)absorbing_qubits[i].first.size() - 1) {
              result += ", ";
            }
          }
          result += "}";
          if (i != (int)absorbing_qubits.size() - 1) {
            result += ", ";
          }
        }
        result += "}";
      }
      result += "}";
      return result;
    }
    std::vector<std::pair<std::vector<int>, KernelType>> sets;
    // The collection of sets of qubits such that although we have terminated
    // some kernels operating on them, as long as there is a set in this
    // collection and a gate in the sequence that is not depending on any
    // qubits not in the set, we can execute the gate at no cost.
    std::vector<std::pair<std::vector<int>, KernelType>> absorbing_qubits;
    size_t hash;
  };
  class StatusHash {
  public:
    size_t operator()(const Status &s) const { return s.hash; }
  };
  struct LocalSchedule {
  public:
    [[nodiscard]] bool check_valid() const {
      std::vector<bool> has_qubit;
      for (int i = 0; i < (int)sets.size(); i++) {
        for (int j = 0; j < (int)sets[i].first.size(); j++) {
          while (sets[i].first[j] >= has_qubit.size()) {
            has_qubit.push_back(false);
          }
          if (has_qubit[sets[i].first[j]]) {
            std::cerr << "Invalid local schedule: qubit " << sets[i].first[j]
                      << " appears twice." << std::endl;
            return false;
          }
          has_qubit[sets[i].first[j]] = true;
        }
      }
      return true;
    }
    [[nodiscard]] std::string to_string() const {
      std::string result;
      result += "{";
      for (int i = 0; i < (int)sets.size(); i++) {
        result += kernel_type_name(sets[i].second);
        result += "{";
        for (int j = 0; j < (int)sets[i].first.size(); j++) {
          result += std::to_string(sets[i].first[j]);
          if (j != (int)sets[i].first.size() - 1) {
            result += ", ";
          }
        }
        result += "}";
        if (i != (int)sets.size() - 1) {
          result += ", ";
        }
      }
      result += "}";
      return result;
    }
    std::vector<std::pair<std::vector<int>, KernelType>> sets;
  };
  // f[i][S].first: first i (1-indexed) gates, "status of kernels on the
  // frontier" S, min cost of the kernels not on the frontier.
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
    std::cout << sequence_.to_string() << std::endl;
  }
  for (int i = 0; i < num_gates; i++) {
    if (debug) {
      std::cout << "DP: i=" << i << " size=" << f[i & 1].size() << std::endl;
    }
    // Update from f[i & 1] to f[~i & 1].
    f[~i & 1].clear();
    // Get the qubit indices of the current gate.
    auto &current_gate = *sequence_.gates[i];
    auto current_gate_cost =
        kernel_cost.get_shared_memory_gate_cost(current_gate.gate->tp);
    std::vector<bool> current_index(num_qubits, false);
    std::vector<int> current_indices;
    current_indices.reserve(current_gate.input_wires.size());
    for (auto &input_wire : current_gate.input_wires) {
      // We do not care about global qubits here.
      if (input_wire->is_qubit() && is_local_qubit(input_wire->index)) {
        current_index[input_wire->index] = true;
        current_indices.push_back(input_wire->index);
        if (debug) {
          std::cout << "current index " << input_wire->index << std::endl;
        }
      }
    }
    std::sort(current_indices.begin(), current_indices.end());
    auto current_indices_hash = Status::get_hash(current_indices);

    // TODO: make these numbers configurable
    constexpr int kMaxNumOfStatus = 4000;
    constexpr int kShrinkToNumOfStatus = 2000;
    if (f[i & 1].size() > kMaxNumOfStatus) {
      // Pruning.
      std::vector<std::pair<
          KernelCostType,
          std::unordered_map<Status, std::pair<KernelCostType, LocalSchedule>,
                             StatusHash>::iterator>>
          costs;
      if (debug) {
        std::cout << "Shrink f[" << i << "] from " << f[i & 1].size()
                  << " elements to " << kShrinkToNumOfStatus << " elements."
                  << std::endl;
      }
      costs.reserve(f[i & 1].size());
      KernelCostType lowest_cost, highest_cost; // for debugging
      for (auto it = f[i & 1].begin(); it != f[i & 1].end(); it++) {
        // Use the current "end" cost as a heuristic.
        // TODO: profile the running time of |compute_end_schedule| and see
        //  if an approximation optimization is necessary.
        KernelCostType result_cost;
        compute_end_schedule(kernel_cost, it->first.sets,
                             is_shared_memory_cacheline_qubit, result_cost,
                             /*result_kernels=*/nullptr);
        costs.emplace_back(std::make_pair(result_cost + it->second.first, it));
        if (debug) {
          if (it == f[i & 1].begin() ||
              result_cost + it->second.first > highest_cost) {
            highest_cost = result_cost + it->second.first;
          }
          if (it == f[i & 1].begin() ||
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
        new_f.insert(f[i & 1].extract(costs[j].second));
      }
      f[i & 1] = new_f;
      if (debug) {
        std::cout << "Costs shrank from [" << lowest_cost << ", "
                  << highest_cost << "] to [" << lowest_cost << ", "
                  << costs[kShrinkToNumOfStatus - 1].first << "]." << std::endl;
      }
    }

    // for debugging
    KernelCostType tmp_best_cost = 1e100;
    Status tmp_status;
    LocalSchedule tmp_schedule;

    for (auto &it : f[i & 1]) {
      auto &current_status = it.first;
      auto &current_cost = it.second.first;
      auto &current_local_schedule = it.second.second;
      assert(current_status.check_valid());
      if (debug) {
        KernelCostType tmp;
        compute_end_schedule(kernel_cost, current_status.sets,
                             is_shared_memory_cacheline_qubit, tmp, nullptr);
        tmp += current_cost;
        if (tmp < tmp_best_cost) {
          tmp_best_cost = tmp;
          tmp_status = current_status;
          tmp_schedule = current_local_schedule;
        }
      }

      int absorbing_set_index = -1;
      int absorb_count = 0;
      for (int j = 0; j < (int)current_status.absorbing_qubits.size(); j++) {
        for (auto &qubit : current_status.absorbing_qubits[j].first) {
          if (current_index[qubit]) {
            absorb_count++;
            if (absorbing_set_index == -1) {
              absorbing_set_index = j;
            } else if (absorbing_set_index != j) {
              // not absorb-able
              absorbing_set_index = -2;
              break;
            }
          }
        }
      }
      if (absorbing_set_index >= 0 && absorb_count == current_indices.size()) {
        // Optimization:
        // The current gate is absorbed by a previous kernel.
        // Directly update.
        assert(current_status.absorbing_qubits[absorbing_set_index].second ==
                   KernelType::fusion ||
               current_status.absorbing_qubits[absorbing_set_index].second ==
                   KernelType::shared_memory);
        if (current_status.absorbing_qubits[absorbing_set_index].second ==
            KernelType::shared_memory) {
          current_cost += current_gate_cost;
        }
        update_f(f[~i & 1], current_status, current_cost,
                 current_local_schedule);
        continue;
      }

      const int num_kernels = current_status.sets.size();
      std::vector<int> touching_set_indices, touching_size;
      for (int j = 0; j < num_kernels; j++) {
        bool touching_set_has_j = false;
        for (auto &index : current_status.sets[j].first) {
          if (current_index[index]) {
            if (!touching_set_has_j) {
              touching_set_has_j = true;
              touching_set_indices.push_back(j);
              touching_size.push_back(1);
            } else {
              touching_size.back()++;
            }
          }
        }
      }
      if (touching_set_indices.size() == 1 &&
          touching_size[0] == current_indices.size()) {
        assert(absorb_count == 0);
        // Optimization:
        // The current gate is touching exactly one kernel on the frontier,
        // and is subsumed by that kernel.
        // Directly update.
        assert(current_status.sets[touching_set_indices[0]].second ==
                   KernelType::fusion ||
               current_status.sets[touching_set_indices[0]].second ==
                   KernelType::shared_memory);
        if (current_status.sets[touching_set_indices[0]].second ==
            KernelType::shared_memory) {
          current_cost += current_gate_cost;
        }
        update_f(f[~i & 1], current_status, current_cost,
                 current_local_schedule);
        continue;
      }
      auto new_absorbing_qubits = current_status.absorbing_qubits;
      if (absorb_count != 0) {
        // Remove all qubits touching the current gate from the
        // |absorbing_qubits| set collection.
        // Loop in reverse order so that we do not need to worry about
        // the index change during removal.
        for (int k = (int)new_absorbing_qubits.size() - 1; k >= 0; k--) {
          // Loop in reverse order so that we do not need to worry about
          // the index change during removal.
          for (int j = (int)new_absorbing_qubits[k].first.size() - 1; j >= 0;
               j--) {
            if (current_index[new_absorbing_qubits[k].first[j]]) {
              new_absorbing_qubits[k].first.erase(
                  new_absorbing_qubits[k].first.begin() + j);
            }
          }
          if (new_absorbing_qubits[k].first.empty()) {
            new_absorbing_qubits.erase(new_absorbing_qubits.begin() + k);
          }
        }
        // Sort the absorbing sets in ascending order.
        std::sort(new_absorbing_qubits.begin(), new_absorbing_qubits.end(),
                  [](auto &s1, auto &s2) { return s1.first[0] < s2.first[0]; });
      }
      if (touching_set_indices.empty()) {
        // Optimization:
        // The current gate is not touching any kernels on the frontier.
        // Directly add the gate to the frontier.
        auto new_status = current_status;
        new_status.insert_set(current_indices, KernelType::fusion);
        new_status.absorbing_qubits = new_absorbing_qubits;
        new_status.hash ^= current_indices_hash;
        update_f(f[~i & 1], new_status, current_cost, current_local_schedule);

        new_status = current_status;
        new_status.insert_set(current_indices, KernelType::shared_memory);
        new_status.absorbing_qubits = new_absorbing_qubits;
        new_status.hash ^= current_indices_hash;
        current_cost += current_gate_cost;
        update_f(f[~i & 1], new_status, current_cost, current_local_schedule);
        continue;
      }
      // Keep track of the schedule during the search.
      LocalSchedule local_schedule = current_local_schedule;
      std::vector<std::pair<std::vector<int>, KernelType>> absorbing_sets_stack;
      // Keep track of which kernels are merged during the search.
      std::vector<bool> kernel_merged(num_kernels, false);
      for (auto &index : touching_set_indices) {
        // These kernels are already considered -- treat them as merged.
        kernel_merged[index] = true;
      }
      auto search_merging_kernels =
          [&](auto &this_ref, const std::vector<int> &current_gate_kernel,
              KernelType current_gate_kernel_type,
              const std::vector<int> &current_merging_kernel,
              KernelType current_merging_kernel_type,
              const KernelCostType &cost, int touching_set_index,
              int kernel_index) -> void {
        if (kernel_index == num_kernels) {
          // We have searched all kernels to merge or not with this
          // "touching set".
          touching_set_index++;
          auto new_cost = cost;
          if (!current_merging_kernel.empty()) {
            // Because we are not merging this kernel with the current
            // gate, we need to record the merged kernel.
            local_schedule.sets.emplace_back(current_merging_kernel,
                                             current_merging_kernel_type);
            std::sort(local_schedule.sets.back().first.begin(),
                      local_schedule.sets.back().first.end());
            new_cost += kernel_costs[current_merging_kernel.size()];
            std::vector<int> absorbing_set;
            for (auto &index : current_merging_kernel) {
              if (!current_index[index]) {
                // As long as the current gate does not block the qubit |index|,
                // we can execute a gate at the qubit |index| later in the
                // kernel |current_merging_kernel|.
                absorbing_set.push_back(index);
              }
            }
            std::sort(absorbing_set.begin(), absorbing_set.end());
            absorbing_sets_stack.emplace_back(absorbing_set,
                                              current_merging_kernel_type);
          }
          if (touching_set_index == (int)touching_set_indices.size()) {
            // We have searched everything.
            // Create the new Status object.
            Status new_status;
            for (int j = 0; j < num_kernels; j++) {
              if (!kernel_merged[j]) {
                new_status.sets.push_back(current_status.sets[j]);
              }
            }
            // Insert the new open kernel.
            new_status.insert_set(current_gate_kernel,
                                  current_gate_kernel_type);
            // Insert the absorbing kernels.
            new_status.absorbing_qubits = new_absorbing_qubits;
            new_status.absorbing_qubits.insert(
                new_status.absorbing_qubits.end(), absorbing_sets_stack.begin(),
                absorbing_sets_stack.end());
            // Sort the absorbing sets in ascending order.
            std::sort(
                new_status.absorbing_qubits.begin(),
                new_status.absorbing_qubits.end(),
                [](auto &s1, auto &s2) { return s1.first[0] < s2.first[0]; });
            new_status.compute_hash();
            update_f(f[~i & 1], new_status, new_cost, local_schedule);
          } else {
            // Start a new iteration of searching.
            // Try to merge the "touching set" with the current gate first.
            bool merging_is_always_better = false;
            if (current_status.sets[touching_set_indices[touching_set_index]]
                    .second == current_gate_kernel_type) {
              // We only merge kernels of the same type.
              auto new_gate_kernel = current_gate_kernel;
              for (auto &index :
                   current_status.sets[touching_set_indices[touching_set_index]]
                       .first) {
                if (!current_index[index]) {
                  new_gate_kernel.push_back(index);
                }
              }
              if (new_gate_kernel.size() == current_gate_kernel.size()) {
                // An optimization: if we merge a kernel with the current gate
                // and the size remain unchanged, we always want to merge the
                // kernel with the current gate.
                merging_is_always_better = true;
              }
              assert(current_gate_kernel_type == KernelType::fusion ||
                     current_gate_kernel_type == KernelType::shared_memory);
              bool kernel_size_ok = false;
              if (current_gate_kernel_type == KernelType::fusion) {
                kernel_size_ok =
                    new_gate_kernel.size() <= max_fusion_kernel_size;
              } else {
                if (new_gate_kernel.size() <= shared_memory_kernel_size) {
                  kernel_size_ok = true;
                } else {
                  // Check cacheline qubits.
                  int num_cacheline_qubits = 0;
                  for (const auto &qubit : new_gate_kernel) {
                    if (is_shared_memory_cacheline_qubit[qubit]) {
                      num_cacheline_qubits++;
                    }
                  }
                  kernel_size_ok =
                      new_gate_kernel.size() - num_cacheline_qubits <=
                      shared_memory_kernel_size;
                }
              }
              if (kernel_size_ok) {
                std::sort(new_gate_kernel.begin(), new_gate_kernel.end());
                // If we merge a kernel with the current gate, we do not need
                // to search for other kernels to merge together.
                this_ref(this_ref, new_gate_kernel, current_gate_kernel_type,
                         /*current_merging_kernel=*/std::vector<int>(),
                         /*current_merging_kernel_type=*/KernelType(), new_cost,
                         touching_set_index,
                         /*kernel_index=*/num_kernels);
              }
            }
            if (!merging_is_always_better) {
              // Try to not merge the "touching set" with the current gate.
              this_ref(
                  this_ref, current_gate_kernel, current_gate_kernel_type,
                  /*current_merging_kernel=*/
                  current_status.sets[touching_set_indices[touching_set_index]]
                      .first,
                  /*current_merging_kernel_type=*/
                  current_status.sets[touching_set_indices[touching_set_index]]
                      .second,
                  new_cost, touching_set_index, /*kernel_index=*/0);
            }
          }
          if (!current_merging_kernel.empty()) {
            // Restore the merged kernel stack.
            local_schedule.sets.pop_back();
            absorbing_sets_stack.pop_back();
          }
          return;
        }
        // We can always try not to merge with this kernel.
        this_ref(this_ref, current_gate_kernel, current_gate_kernel_type,
                 current_merging_kernel, current_merging_kernel_type, cost,
                 touching_set_index, kernel_index + 1);
        if (kernel_merged[kernel_index]) {
          // This kernel is already considered. Continue to the next one.
          return;
        }
        if (current_status.sets[kernel_index].second !=
            current_merging_kernel_type) {
          // We don't merge kernels with different types.
          return;
        }
        assert(current_merging_kernel_type == KernelType::fusion ||
               current_merging_kernel_type == KernelType::shared_memory);
        bool kernel_size_ok = false;
        if (current_merging_kernel_type == KernelType::fusion) {
          kernel_size_ok = current_merging_kernel.size() +
                               current_status.sets[kernel_index].first.size() <=
                           max_fusion_kernel_size;
        } else {
          if (current_merging_kernel.size() +
                  current_status.sets[kernel_index].first.size() <=
              shared_memory_kernel_size) {
            kernel_size_ok = true;
          } else {
            // Check cacheline qubits.
            int num_cacheline_qubits = 0;
            for (const auto &qubit : current_merging_kernel) {
              if (is_shared_memory_cacheline_qubit[qubit]) {
                num_cacheline_qubits++;
              }
            }
            for (const auto &qubit : current_status.sets[kernel_index].first) {
              if (is_shared_memory_cacheline_qubit[qubit]) {
                num_cacheline_qubits++;
              }
            }
            kernel_size_ok =
                current_merging_kernel.size() +
                    current_status.sets[kernel_index].first.size() -
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
        new_merging_kernel.insert(
            new_merging_kernel.end(),
            current_status.sets[kernel_index].first.begin(),
            current_status.sets[kernel_index].first.end());
        kernel_merged[kernel_index] = true;
        // Continue the search.
        this_ref(this_ref, current_gate_kernel, current_gate_kernel_type,
                 new_merging_kernel, current_merging_kernel_type, cost,
                 touching_set_index, kernel_index + 1);
        // Restore the |kernel_merged| status.
        kernel_merged[kernel_index] = false;
      };
      // Start the search with touching_set_index=-1 and
      // kernel_index=num_kernels, so that we can decide whether to merge the
      // first "touching set" with the current gate or not inside the search.
      search_merging_kernels(
          search_merging_kernels, /*current_gate_kernel=*/current_indices,
          /*current_gate_kernel_type=*/KernelType::fusion,
          /*current_merging_kernel=*/std::vector<int>(),
          /*current_merging_kernel_type=*/KernelType(), /*cost=*/current_cost,
          /*touching_set_index=*/-1, /*kernel_index=*/num_kernels);
      // For shared-memory kernels, we account for the gate cost now.
      search_merging_kernels(
          search_merging_kernels, /*current_gate_kernel=*/current_indices,
          /*current_gate_kernel_type=*/KernelType::shared_memory,
          /*current_merging_kernel=*/std::vector<int>(),
          /*current_merging_kernel_type=*/KernelType(),
          /*cost=*/current_cost + current_gate_cost,
          /*touching_set_index=*/-1, /*kernel_index=*/num_kernels);
    }
    if (debug) {
      std::cout << "best cost before i=" << i << ": " << tmp_best_cost
                << std::endl;
      std::cout << "open kernels: " << tmp_status.to_string() << std::endl;
      std::cout << "closed kernels: " << tmp_schedule.to_string() << std::endl;
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
    std::vector<std::pair<std::vector<int>, KernelType>> end_schedule;
    compute_end_schedule(kernel_cost, it.first.sets,
                         is_shared_memory_cacheline_qubit, cost, &end_schedule);
    if (result_schedule.sets.empty() || cost + it.second.first < min_cost) {
      min_cost = cost + it.second.first;
      result_schedule = it.second.second;
      result_schedule.sets.insert(result_schedule.sets.end(),
                                  end_schedule.begin(), end_schedule.end());
    }
  }
  // Translate the |result_schedule| into |kernels|.
  kernels.reserve(result_schedule.sets.size());
  std::vector<bool> executed(num_gates, false);
  cost_ = min_cost;
  int start_gate_index = 0; // an optimization
  for (auto &s : result_schedule.sets) {
    // Greedily execute a kernel.
    CircuitSeq current_seq(num_qubits, sequence_.get_num_input_parameters());
    std::vector<bool> local_in_kernel(num_qubits, false);
    for (auto &index : s.first) {
      local_in_kernel[index] = true;
    }
    std::vector<bool> qubit_blocked(num_qubits, false);
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
      for (auto &wire : sequence_.gates[i]->input_wires) {
        // We do not care about global qubits, but we need the local qubit to
        // be local in this kernel.
        if (wire->is_qubit() && is_local_qubit(wire->index) &&
            !local_in_kernel[wire->index]) {
          executable = false;
          break;
        }
      }
      if (executable) {
        // Execute the gate.
        executed[i] = true;
        current_seq.add_gate(sequence_.gates[i].get());
      } else {
        // Block the qubits.
        for (auto &wire : sequence_.gates[i]->input_wires) {
          if (wire->is_qubit()) {
            qubit_blocked[wire->index] = true;
          }
        }
      }
    }
    kernels.emplace_back(current_seq, s.first, s.second);
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
            << " kernels: cost = " << cost_ << std::endl;
  for (int i = 0; i < num_kernels; i++) {
    std::cout << "Kernel " << i << ": ";
    std::cout << kernels[i].to_string() << std::endl;
  }
}

std::vector<Schedule>
get_schedules(const CircuitSeq &sequence,
              const std::vector<std::vector<int>> &local_qubits,
              const KernelCost &kernel_cost, Context *ctx,
              bool absorb_single_qubit_gates) {
  std::vector<Schedule> result;
  result.reserve(local_qubits.size());
  const int num_qubits = sequence.get_num_qubits();
  const int num_gates = sequence.get_num_gates();
  std::vector<int> current_local_qubit_layout;
  std::vector<int> current_global_qubit_layout;
  std::vector<bool> local_qubit_mask(num_qubits, false);
  std::vector<bool> executed(num_gates, false);
  int start_gate_index = 0; // an optimization

  // Variables for |absorb_single_qubit_gates|=true.
  std::vector<std::pair<int, std::unordered_set<int>>>
      single_qubit_gate_indices;
  // |last_gate_index[i]|: the last gate touching the |i|-th qubit;
  // only computed when |absorb_single_qubit_gates| is true
  std::vector<int> last_gate_index(num_qubits, -1);
  // |gate_indices[gate_string]|: the indices of gates with to_string() being
  // |gate_string|;
  // only computed when |absorb_single_qubit_gates| is true
  std::unordered_map<std::string, std::queue<int>> gate_indices;

  for (auto &local_qubit : local_qubits) {
    // Convert vector<int> to vector<bool>.
    local_qubit_mask.assign(num_qubits, false);
    for (auto &i : local_qubit) {
      local_qubit_mask[i] = true;
    }
    std::cout << std::endl;
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
      if (sequence.gates[i]->gate->get_num_control_qubits() > 0) {
        if (!sequence.gates[i]->gate->is_symmetric()) {
          // For controlled gates, we only need to have all target qubits local
          // when it's not symmetric.
          int control_qubit_count =
              sequence.gates[i]->gate->get_num_control_qubits();
          for (auto &wire : sequence.gates[i]->input_wires) {
            if (wire->is_qubit() && control_qubit_count--) {
              // Skip the control qubits.
              continue;
            }
            if (wire->is_qubit() && !local_qubit_mask[wire->index]) {
              executable = false;
              break;
            }
          }
        }
      } else if (sequence.gates[i]->gate->get_num_qubits() > 1 ||
                 !sequence.gates[i]->gate->is_sparse()) {
        // For non-controlled multi-qubit gates and non-sparse single-qubit
        // gates, we need to have all qubits local.
        for (auto &wire : sequence.gates[i]->input_wires) {
          if (wire->is_qubit() && !local_qubit_mask[wire->index]) {
            executable = false;
            break;
          }
        }
      }
      if (executable) {
        // Execute the gate.
        executed[i] = true;
        if (absorb_single_qubit_gates) {
          // Count the number of local qubits.
          int num_local_qubit = 0;
          for (auto &wire : sequence.gates[i]->input_wires) {
            if (wire->is_qubit() && local_qubit_mask[wire->index]) {
              num_local_qubit++;
            }
          }
          assert(num_local_qubit > 0);
          if (num_local_qubit == 1) {
            // Do not put single-qubit gates into |current_seq|.
            // Compute the dependency here.
            std::unordered_set<int> depending_gates;
            for (auto &wire : sequence.gates[i]->input_wires) {
              if (wire->is_qubit()) {
                if (last_gate_index[wire->index] != -1) {
                  // Use std::unordered_set to remove duplicates automatically.
                  depending_gates.insert(last_gate_index[wire->index]);
                }
              }
            }
            single_qubit_gate_indices.emplace_back(i, depending_gates);
          } else {
            current_seq.add_gate(sequence.gates[i].get());
          }

          // Update |last_gate_index|.
          for (auto &wire : sequence.gates[i]->input_wires) {
            if (wire->is_qubit()) {
              last_gate_index[wire->index] = i;
            }
          }
          // Update |gate_indices|.
          gate_indices[sequence.gates[i]->to_string()].push(i);
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
    result.emplace_back(current_seq, current_local_qubit_layout,
                        current_global_qubit_layout, ctx);
    while (start_gate_index < num_gates && executed[start_gate_index]) {
      start_gate_index++;
    }
  }
  if (start_gate_index != num_gates) {
    std::cerr << "Gate number " << start_gate_index
              << " is not executed yet in the schedule." << std::endl;
    assert(false);
  }
  for (auto &schedule : result) {
    schedule.compute_kernel_schedule(kernel_cost);
  }
  if (!single_qubit_gate_indices.empty()) {
    std::vector<int> single_qubit_gate_to_execute;
    // Restore the single-qubit gates.
    std::vector<std::vector<int>> next_single_qubit_gates(num_gates);
    for (int i = 0; i < (int)single_qubit_gate_indices.size(); i++) {
      auto &single_qubit_gate = single_qubit_gate_indices[i];
      // Record the dependencies.
      for (auto &index : single_qubit_gate.second) {
        next_single_qubit_gates[index].push_back(i);
      }
      // If there is no dependency, we can execute it.
      if (single_qubit_gate.second.empty()) {
        single_qubit_gate_to_execute.push_back(single_qubit_gate.first);
      }
    }
    int current_kernel_qubits_last_updated_i = -1;
    std::vector<bool> current_kernel_qubits(num_qubits, false);
    for (auto &schedule : result) {
      // avoid cache conflict with other |schedule|s
      current_kernel_qubits_last_updated_i = -1;
      for (int i = 0; i < schedule.get_num_kernels(); i++) {
        auto try_to_execute_single_qubit_gates =
            [&](int insert_location) -> bool {
          if (single_qubit_gate_to_execute.empty()) {
            return false;
          }
          if (current_kernel_qubits_last_updated_i != i) {
            current_kernel_qubits_last_updated_i = i;
            current_kernel_qubits.assign(num_qubits, false);
            for (auto &qubit : schedule.kernels[i].qubits) {
              current_kernel_qubits[qubit] = true;
            }
          }
          bool executed_any_gate = false;
          for (int k = 0; k < (int)single_qubit_gate_to_execute.size(); k++) {
            int index = single_qubit_gate_to_execute[k];
            bool executable = true;
            for (auto &wire : sequence.gates[index]->input_wires) {
              if (wire->is_qubit()) {
                if (schedule.is_local_qubit(wire->index)) {
                  if (!current_kernel_qubits[wire->index]) {
                    executable = false; // not local in this kernel
                    break;
                  }
                } else {
                  if (!sequence.gates[index]->gate->is_sparse()) {
                    executable = false; // not executable globally
                    break;
                  }
                }
              }
            }
            if (executable) {
              executed_any_gate = true;
              // Insert the gate to
              // |schedule.kernels[i].gates[insert_location]|.
              // Note that we don't update |next_single_qubit_gates| here.
              schedule.kernels[i].gates.insert_gate(
                  insert_location, sequence.gates[index].get());
              assert(schedule.kernels[i].type == KernelType::fusion ||
                     schedule.kernels[i].type == KernelType::shared_memory);
              if (schedule.kernels[i].type == KernelType::shared_memory) {
                schedule.cost_ += kernel_cost.get_shared_memory_gate_cost(
                    sequence.gates[index]->gate->tp);
              }
              // Erase the gate from |single_qubit_gate_to_execute|.
              single_qubit_gate_to_execute.erase(
                  single_qubit_gate_to_execute.begin() + k);
              // Because we have |k++| at the end of this iteration of the
              // for loop, we need to cancel the effect here.
              k--;
            }
          }
          return executed_any_gate;
        };

        try_to_execute_single_qubit_gates(0);

        // Purposely using |schedule.kernels[i].gates.gates.size()| because we
        // may modify |schedule.kernels[i].gates.gates| in this loop.
        for (int j = 0; j < (int)schedule.kernels[i].gates.gates.size(); j++) {
          auto &gate = schedule.kernels[i].gates.gates[j];
          auto &gate_indices_queue = gate_indices[gate->to_string()];
          assert(!gate_indices_queue.empty());

          // We execute the gate now.
          const int original_index = gate_indices_queue.front();
          for (auto &next_single_qubit_gate :
               next_single_qubit_gates[original_index]) {
            single_qubit_gate_indices[next_single_qubit_gate].second.erase(
                original_index);
            if (single_qubit_gate_indices[next_single_qubit_gate]
                    .second.empty()) {
              single_qubit_gate_to_execute.push_back(
                  single_qubit_gate_indices[next_single_qubit_gate].first);
              // Insert the gate after this gate.
              try_to_execute_single_qubit_gates(j + 1);
            }
          }
          gate_indices_queue.pop();
        }
      }
    }
    for (auto &single_qubit_gate : single_qubit_gate_to_execute) {
      std::cerr << "Single-qubit gate " << single_qubit_gate
                << " is not executed yet in the schedule because there are no "
                   "local kernels."
                << std::endl;
      assert(false);
    }
    for (auto &single_qubit_gate : single_qubit_gate_indices) {
      if (!single_qubit_gate.second.empty()) {
        std::cerr << "Single-qubit gate " << single_qubit_gate.first
                  << " is not executed yet in the schedule because of "
                     "unresolved dependencies."
                  << std::endl;
        assert(false);
      }
    }
  }
  return result;
}

std::vector<std::vector<int>>
compute_local_qubits_with_ilp(const CircuitSeq &sequence, int num_local_qubits,
                              Context *ctx, PythonInterpreter *interpreter) {
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
  for (int num_iterations = 1; true; num_iterations++) {
    auto result = interpreter->solve_ilp(
        circuit_gate_qubits, circuit_gate_executable_type, out_gate, num_qubits,
        num_local_qubits, num_iterations);
    if (!result.empty()) {
      return result;
    }
  }
}
} // namespace quartz
