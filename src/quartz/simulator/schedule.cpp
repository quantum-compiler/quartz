#include "schedule.h"
#include "quartz/pybind/pybind.h"

#include <queue>
#include <stack>
#include <unordered_set>

namespace quartz {

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

bool Schedule::is_local_qubit(int index) const { return local_qubit_[index]; }

bool Schedule::compute_end_schedule(
    const std::vector<KernelCostType> &kernel_costs,
    const std::vector<std::vector<int>> &kernels,
    Schedule::KernelCostType &result_cost,
    std::vector<std::vector<int>> *result_kernels) {
  const int num_kernels = kernels.size();
  const int max_kernel_size = (int)kernel_costs.size() - 1;
  std::vector<std::vector<int>> kernels_of_size(max_kernel_size + 1);
  for (int i = 0; i < num_kernels; i++) {
    assert(kernels[i].size() <= max_kernel_size);
    assert(kernels[i].size() > 0);
    kernels_of_size[kernels[i].size()].push_back(i);
  }
  // Greedily try to use the optimal kernel size.
  int optimal_kernel_size = 1;
  KernelCostType optimal_kernel_size_cost = kernel_costs[1];
  for (int i = 2; i < max_kernel_size; i++) {
    KernelCostType tmp = kernel_costs[i] / i;
    if (tmp < optimal_kernel_size_cost) {
      optimal_kernel_size = i;
      optimal_kernel_size_cost = tmp;
    }
  }
  if (result_kernels != nullptr) {
    result_kernels->clear();
  }
  result_cost = 0;
  // Copy the large kernels to the result.
  for (int i = max_kernel_size; i >= optimal_kernel_size; i--) {
    for (auto &index : kernels_of_size[i]) {
      if (result_kernels != nullptr) {
        result_kernels->emplace_back(kernels[index]);
      }
      result_cost += kernel_costs[i];
    }
  }
  // Greedily select the small kernels.
  while (true) {
    bool has_any_remaining_kernel = false;
    std::vector<int> current_kernel;
    int current_kernel_size = 0;
    for (int i = optimal_kernel_size - 1; i >= 1; i--) {
      while (!kernels_of_size[i].empty()) {
        if (current_kernel.size() + i <= optimal_kernel_size) {
          // Add a kernel of size i to the current kernel.
          if (result_kernels != nullptr) {
            current_kernel.insert(current_kernel.end(),
                                  kernels[kernels_of_size[i].back()].begin(),
                                  kernels[kernels_of_size[i].back()].end());
          }
          current_kernel_size += i;
          kernels_of_size[i].pop_back();
        } else {
          has_any_remaining_kernel = true;
          break;
        }
      }
    }
    if (!current_kernel.empty()) {
      result_cost += kernel_costs[current_kernel_size];
      if (result_kernels != nullptr) {
        std::sort(current_kernel.begin(), current_kernel.end());
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
    const std::vector<KernelCostType> &kernel_costs) {
  // We need to be able to execute at least 1-qubit gates.
  assert(kernel_costs.size() >= 2);
  // We have not computed the schedule before.
  assert(kernels.empty());
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
        hash ^= get_hash(s);
      }
    }
    void insert_set(const std::vector<int> &s) {
      int insert_position = 0;
      while (insert_position < (int)sets.size() &&
             sets[insert_position][0] < s[0]) {
        insert_position++;
      }
      sets.insert(sets.begin() + insert_position, s);
    }
    bool check_valid() const {
      std::vector<bool> has_qubit;
      for (int i = 0; i < (int)sets.size(); i++) {
        for (int j = 0; j < (int)sets[i].size(); j++) {
          while (sets[i][j] >= has_qubit.size()) {
            has_qubit.push_back(false);
          }
          if (has_qubit[sets[i][j]]) {
            std::cerr << "Invalid status: qubit " << sets[i][j]
                      << " appears twice." << std::endl;
            std::cerr << to_string() << std::endl;
            return false;
          }
          has_qubit[sets[i][j]] = true;
        }
      }
      for (int i = 0; i < (int)absorbing_qubits.size(); i++) {
        for (int j = 0; j < (int)absorbing_qubits[i].size(); j++) {
          while (absorbing_qubits[i][j] >= has_qubit.size()) {
            has_qubit.push_back(false);
          }
          if (has_qubit[absorbing_qubits[i][j]]) {
            std::cerr << "Invalid status: qubit " << absorbing_qubits[i][j]
                      << " appears twice." << std::endl;
            std::cerr << to_string() << std::endl;
            return false;
          }
          has_qubit[absorbing_qubits[i][j]] = true;
        }
      }
      return true;
    }
    bool operator==(const Status &b) const {
      if (sets.size() != b.sets.size()) {
        return false;
      }
      for (int i = 0; i < (int)sets.size(); i++) {
        if (sets[i].size() != b.sets[i].size()) {
          return false;
        }
        for (int j = 0; j < (int)sets[i].size(); j++) {
          if (sets[i][j] != b.sets[i][j]) {
            return false;
          }
        }
      }
      if (absorbing_qubits.size() != b.absorbing_qubits.size()) {
        return false;
      }
      for (int i = 0; i < (int)absorbing_qubits.size(); i++) {
        if (absorbing_qubits[i].size() != b.absorbing_qubits[i].size()) {
          return false;
        }
        for (int j = 0; j < (int)absorbing_qubits[i].size(); j++) {
          if (absorbing_qubits[i][j] != b.absorbing_qubits[i][j]) {
            return false;
          }
        }
      }
      return true;
    }
    std::string to_string() const {
      std::string result;
      result += "{";
      for (int i = 0; i < (int)sets.size(); i++) {
        result += "{";
        for (int j = 0; j < (int)sets[i].size(); j++) {
          result += std::to_string(sets[i][j]);
          if (j != (int)sets[i].size() - 1) {
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
          result += "{";
          for (int j = 0; j < (int)absorbing_qubits[i].size(); j++) {
            result += std::to_string(absorbing_qubits[i][j]);
            if (j != (int)absorbing_qubits[i].size() - 1) {
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
    std::vector<std::vector<int>> sets;
    // The collection of sets of qubits such that although we have terminated
    // some kernels operating on them, as long as there is a set in this
    // collection and a gate in the sequence that is not depending on any
    // qubits not in the set, we can execute the gate at no cost.
    std::vector<std::vector<int>> absorbing_qubits;
    size_t hash;
  };
  class StatusHash {
  public:
    size_t operator()(const Status &s) const { return s.hash; }
  };
  struct LocalSchedule {
  public:
    bool check_valid() const {
      std::vector<bool> has_qubit;
      for (int i = 0; i < (int)sets.size(); i++) {
        for (int j = 0; j < (int)sets[i].size(); j++) {
          while (sets[i][j] >= has_qubit.size()) {
            has_qubit.push_back(false);
          }
          if (has_qubit[sets[i][j]]) {
            std::cerr << "Invalid local schedule: qubit " << sets[i][j]
                      << " appears twice." << std::endl;
            return false;
          }
          has_qubit[sets[i][j]] = true;
        }
      }
      return true;
    }
    std::string to_string() const {
      std::string result;
      result += "{";
      for (int i = 0; i < (int)sets.size(); i++) {
        result += "{";
        for (int j = 0; j < (int)sets[i].size(); j++) {
          result += std::to_string(sets[i][j]);
          if (j != (int)sets[i].size() - 1) {
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
    std::vector<std::vector<int>> sets;
  };
  const int num_qubits = sequence_.get_num_qubits();
  const int num_gates = sequence_.get_num_gates();
  const int max_kernel_size = (int)kernel_costs.size() - 1;
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
  // debug
  std::cout << sequence_.to_string() << std::endl;
  for (int i = 0; i < num_gates; i++) {
    // debug
    std::cout << "DP: " << i << " " << f[i & 1].size() << std::endl;
    // Update from f[i & 1] to f[~i & 1].
    f[~i & 1].clear();
    // Get the qubit indices of the current gate.
    auto &current_gate = *sequence_.gates[i];
    std::vector<bool> current_index(num_qubits, false);
    std::vector<int> current_indices;
    current_indices.reserve(current_gate.input_wires.size());
    for (auto &input_wire : current_gate.input_wires) {
      // We do not care about global qubits here.
      if (input_wire->is_qubit() && is_local_qubit(input_wire->index)) {
        current_index[input_wire->index] = true;
        current_indices.push_back(input_wire->index);
        // debug
        std::cout << "current index " << input_wire->index << std::endl;
      }
    }
    std::sort(current_indices.begin(), current_indices.end());
    auto current_indices_hash = Status::get_hash(current_indices);
    // TODO: add an option to directly execute the gate if it's a controlled
    //  gate -- don't forget to remove the corresponding qubits from the
    //  |absorbed_qubits| set if they are partially in the set.

    // TODO: make these numbers configurable
    constexpr int kMaxNumOfStatus = 10000;
    constexpr int kShrinkToNumOfStatus = 5000;
    if (f[i & 1].size() > kMaxNumOfStatus) {
      // Pruning.
      std::vector<std::pair<
          KernelCostType,
          std::unordered_map<Status, std::pair<KernelCostType, LocalSchedule>,
                             StatusHash>::iterator>>
          costs;
      // debug
      std::cout << "Shrink f[" << i << "] from " << f[i & 1].size()
                << " elements to " << kShrinkToNumOfStatus << " elements."
                << std::endl;
      costs.reserve(f[i & 1].size());
      KernelCostType lowest_cost, highest_cost; // for debugging
      for (auto it = f[i & 1].begin(); it != f[i & 1].end(); it++) {
        // Use the current "end" cost as a heuristic.
        // TODO: profile the running time of |compute_end_schedule| and see
        //  if an approximation optimization is necessary.
        KernelCostType result_cost;
        compute_end_schedule(kernel_costs, it->first.sets, result_cost,
                             /*result_kernels=*/nullptr);
        costs.emplace_back(std::make_pair(result_cost + it->second.first, it));
        if (it == f[i & 1].begin() ||
            result_cost + it->second.first > highest_cost) {
          highest_cost = result_cost + it->second.first;
        }
        if (it == f[i & 1].begin() ||
            result_cost + it->second.first < lowest_cost) {
          lowest_cost = result_cost + it->second.first;
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
      // debug
      std::cout << "Costs shrank from [" << lowest_cost << ", " << highest_cost
                << "] to [" << lowest_cost << ", "
                << costs[kShrinkToNumOfStatus - 1].first << "]." << std::endl;
    }

    for (auto &it : f[i & 1]) {
      auto &current_status = it.first;
      auto &current_cost = it.second.first;
      auto &current_local_schedule = it.second.second;
      assert(current_status.check_valid());

      int absorbing_set_index = -1;
      int absorb_count = 0;
      for (int j = 0; j < (int)current_status.absorbing_qubits.size(); j++) {
        for (auto &qubit : current_status.absorbing_qubits[j]) {
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
        // The current gate absorbed by a previous kernel.
        // Directly update.
        update_f(f[~i & 1], current_status, current_cost,
                 current_local_schedule);
        continue;
      }

      const int num_kernels = current_status.sets.size();
      std::vector<int> touching_set_indices, touching_size;
      for (int j = 0; j < num_kernels; j++) {
        bool touching_set_has_j = false;
        for (auto &index : current_status.sets[j]) {
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
          for (int j = (int)new_absorbing_qubits[k].size() - 1; j >= 0; j--) {
            if (current_index[new_absorbing_qubits[k][j]]) {
              new_absorbing_qubits[k].erase(new_absorbing_qubits[k].begin() +
                                            j);
            }
          }
          if (new_absorbing_qubits[k].empty()) {
            new_absorbing_qubits.erase(new_absorbing_qubits.begin() + k);
          }
        }
        // Sort the absorbing sets in ascending order.
        std::sort(new_absorbing_qubits.begin(), new_absorbing_qubits.end(),
                  [](auto &s1, auto &s2) { return s1[0] < s2[0]; });
      }
      if (touching_set_indices.empty()) {
        // Optimization:
        // The current gate is not touching any kernels on the frontier.
        // Directly add the gate to the frontier.
        auto new_status = current_status;
        new_status.insert_set(current_indices);
        new_status.absorbing_qubits = new_absorbing_qubits;
        new_status.hash ^= current_indices_hash;
        update_f(f[~i & 1], new_status, current_cost, current_local_schedule);
        continue;
      }
      // Keep track of the schedule during the search.
      LocalSchedule local_schedule = current_local_schedule;
      std::vector<std::vector<int>> absorbing_sets_stack;
      // Keep track of which kernels are merged during the search.
      std::vector<bool> kernel_merged(num_kernels, false);
      for (auto &index : touching_set_indices) {
        // These kernels are already considered -- treat them as merged.
        kernel_merged[index] = true;
      }
      auto search_merging_kernels =
          [&](auto &this_ref, const std::vector<int> &current_gate_kernel,
              const std::vector<int> &current_merging_kernel,
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
            local_schedule.sets.push_back(current_merging_kernel);
            std::sort(local_schedule.sets.back().begin(),
                      local_schedule.sets.back().end());
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
            absorbing_sets_stack.push_back(absorbing_set);
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
            // Insert the new kernel on the frontier.
            new_status.insert_set(current_gate_kernel);
            new_status.absorbing_qubits = new_absorbing_qubits;
            new_status.absorbing_qubits.insert(
                new_status.absorbing_qubits.end(), absorbing_sets_stack.begin(),
                absorbing_sets_stack.end());
            // Sort the absorbing sets in ascending order.
            std::sort(new_status.absorbing_qubits.begin(),
                      new_status.absorbing_qubits.end(),
                      [](auto &s1, auto &s2) { return s1[0] < s2[0]; });
            new_status.compute_hash();
            /*if (!new_status.check_valid()) {
              for (int j = 0; j < (int)current_gate_kernel.size(); j++) {
                std::cout << current_gate_kernel[j] << std::endl;
              }
              std::cout << current_status.to_string() << std::endl;
              exit(1);
            }*/
            update_f(f[~i & 1], new_status, new_cost, local_schedule);
          } else {
            // Start a new iteration of searching.
            // Try to merge the "touching set" with the current gate first.
            auto new_gate_kernel = current_gate_kernel;
            for (auto &index :
                 current_status
                     .sets[touching_set_indices[touching_set_index]]) {
              if (!current_index[index]) {
                new_gate_kernel.push_back(index);
              }
            }
            if (new_gate_kernel.size() <= max_kernel_size) {
              std::sort(new_gate_kernel.begin(), new_gate_kernel.end());
              // If we merge a kernel with the current gate, we do not need
              // to search for other kernels to merge together.
              this_ref(this_ref, new_gate_kernel,
                       /*current_merging_kernel=*/std::vector<int>(), new_cost,
                       touching_set_index,
                       /*kernel_index=*/num_kernels);
            }
            // An optimization: if we merge a kernel with the current gate
            // and the size remain unchanged, we always want to merge the
            // kernel with the current gate.
            if (new_gate_kernel.size() > current_gate_kernel.size()) {
              // Try to not merge the "touching set" with the current gate.
              this_ref(
                  this_ref, current_gate_kernel,
                  /*current_merging_kernel=*/
                  current_status.sets[touching_set_indices[touching_set_index]],
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
        this_ref(this_ref, current_gate_kernel, current_merging_kernel, cost,
                 touching_set_index, kernel_index + 1);
        if (kernel_merged[kernel_index]) {
          // This kernel is already considered. Continue to the next one.
          return;
        }
        if (current_merging_kernel.size() +
                current_status.sets[kernel_index].size() >
            max_kernel_size) {
          // The kernel would be too large if we merge this one.
          // Continue to the next one.
          return;
        }
        // Merge this kernel.
        auto new_merging_kernel = current_merging_kernel;
        new_merging_kernel.insert(new_merging_kernel.end(),
                                  current_status.sets[kernel_index].begin(),
                                  current_status.sets[kernel_index].end());
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
      search_merging_kernels(
          search_merging_kernels, /*current_gate_kernel=*/current_indices,
          /*current_merging_kernel=*/std::vector<int>(), /*cost=*/current_cost,
          /*touching_set_index=*/-1, /*kernel_index=*/num_kernels);
    }
  }
  if (f[num_gates & 1].empty()) {
    return false;
  }
  KernelCostType min_cost;
  LocalSchedule result_schedule;
  for (auto &it : f[num_gates & 1]) {
    // Compute the end schedule and get the one with minimal total cost.
    KernelCostType cost;
    std::vector<std::vector<int>> end_schedule;
    compute_end_schedule(kernel_costs, it.first.sets, cost, &end_schedule);
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
  kernel_qubits = result_schedule.sets;
  cost_ = min_cost;
  int start_gate_index = 0; // an optimization
  for (auto &s : result_schedule.sets) {
    // Greedily execute a kernel.
    CircuitSeq current_seq(num_qubits, sequence_.get_num_input_parameters());
    std::vector<bool> local_in_kernel(num_qubits, false);
    for (auto &index : s) {
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
    kernels.emplace_back(current_seq);
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

void Schedule::print_kernel_schedule() const {
  assert(kernels.size() == kernel_qubits.size());
  const int num_kernels = kernels.size();
  std::cout << "Kernel schedule with " << num_kernels
            << " kernels: cost = " << cost_ << std::endl;
  for (int i = 0; i < num_kernels; i++) {
    std::cout << "Kernel " << i << ": qubits [";
    for (int j = 0; j < (int)kernel_qubits[i].size(); j++) {
      std::cout << kernel_qubits[i][j];
      if (j != (int)kernel_qubits[i].size() - 1) {
        std::cout << ", ";
      }
    }
    std::cout << "], gates ";
    std::cout << kernels[i].to_string() << std::endl;
  }
}

std::vector<Schedule>
get_schedules(const CircuitSeq &sequence,
              const std::vector<std::vector<bool>> &local_qubits,
              Context *ctx) {
  std::vector<Schedule> result;
  result.reserve(local_qubits.size());
  const int num_qubits = sequence.get_num_qubits();
  const int num_gates = sequence.get_num_gates();
  std::vector<bool> executed(num_gates, false);
  int start_gate_index = 0; // an optimization
  for (auto &local_qubit : local_qubits) {
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
      if (!sequence.gates[i]->gate->is_sparse()) {
        for (auto &wire : sequence.gates[i]->input_wires) {
          if (wire->is_qubit() && !local_qubit[wire->index]) {
            executable = false;
            break;
          }
        }
      }
      if (executable) {
        // Execute the gate.
        executed[i] = true;
        current_seq.add_gate(sequence.gates[i].get());
      } else {
        // Block the qubits.
        for (auto &wire : sequence.gates[i]->input_wires) {
          if (wire->is_qubit()) {
            qubit_blocked[wire->index] = true;
          }
        }
      }
    }
    // debug
    std::cout << current_seq.to_string() << std::endl;
    result.emplace_back(current_seq, local_qubit, ctx);
    while (start_gate_index < num_gates && executed[start_gate_index]) {
      start_gate_index++;
    }
  }
  if (start_gate_index != num_gates) {
    std::cerr << "Gate number " << start_gate_index
              << " is not executed yet in the schedule." << std::endl;
    assert(false);
  }
  return result;
}

std::vector<std::vector<bool>>
compute_local_qubits_with_ilp(const CircuitSeq &sequence, int num_local_qubits,
                              Context *ctx, PythonInterpreter *interpreter) {
  const int num_qubits = sequence.get_num_qubits();
  const int num_gates = sequence.get_num_gates();
  std::vector<std::vector<int>> circuit_gate_qubits;
  std::vector<bool> circuit_gate_is_sparse;
  std::unordered_map<CircuitGate *, int> gate_index;
  std::vector<std::vector<int>> out_gate(num_gates);
  circuit_gate_qubits.reserve(num_gates);
  circuit_gate_is_sparse.reserve(num_gates);
  gate_index.reserve(num_gates);
  for (int i = 0; i < num_gates; i++) {
    circuit_gate_qubits.push_back(sequence.gates[i]->get_qubit_indices());
    circuit_gate_is_sparse.push_back(sequence.gates[i]->gate->is_sparse());
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
        circuit_gate_qubits, circuit_gate_is_sparse, out_gate, num_qubits,
        num_local_qubits, num_iterations);
    if (!result.empty()) {
      // convert vector<int> to vector<bool>
      std::vector<std::vector<bool>> ret(result.size());
      for (int i = 0; i < (int)result.size(); i++) {
        ret[i] = std::vector<bool>(num_qubits, false);
        for (auto j : result[i]) {
          ret[i][j] = true;
        }
      }
      return ret;
    }
  }
}
} // namespace quartz
