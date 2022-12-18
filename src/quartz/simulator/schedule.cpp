#include "schedule.h"

#include <queue>
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

  while (!to_search.empty()) {
    auto current_set = to_search.front();
    to_search.pop();
    /*std::cout << "searching ";
    for (auto bit : current_set) {
      std::cout << (int)bit;
    }
    std::cout << std::endl;*/
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
    for (int i = start_gate_index; i < num_gates; i++) {
      if (executed[i]) {
        continue;
      }
      // XXX: Assume that there are no parameter gates.
      assert(sequence.gates[i]->gate->is_quantum_gate());
      // Greedily try to execute the gates.
      bool executable = true;
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
      }
    }
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
} // namespace quartz
