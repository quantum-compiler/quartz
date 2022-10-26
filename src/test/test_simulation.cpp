#include "quartz/context/context.h"
#include "quartz/gate/gate_utils.h"
#include "quartz/parser/qasm_parser.h"
#include "quartz/tasograph/tasograph.h"

#include <algorithm>
#include <iostream>
#include <unordered_map>

using namespace quartz;

int main() {
  Context ctx({GateType::input_qubit, GateType::input_param, GateType::h,
               GateType::u2, GateType::cx, GateType::cp, GateType::swap});
  auto graph = Graph::from_qasm_file(
      &ctx, "circuit/MQTBench_40q/qft_indep_qiskit_40.qasm");
  auto dag = graph->to_dag();
  std::cout << dag->to_string() << std::endl;
  int num_local_qubits = 30;
  int num_qubits = dag->get_num_qubits();
  std::unordered_map<DAGHyperEdge *, bool> executed;
  std::vector<bool> local_qubit(num_qubits, false);
  for (int i = 0; i < num_local_qubits; i++) {
    local_qubit[i] = true;
  }
  int num_shuffles = 0;
  while (true) {
    bool all_done = true;
    std::vector<bool> executable(num_qubits, true);
    for (auto &gate : dag->edges) {
      if (gate->gate->is_quantum_gate() && !executed[gate.get()]) {
        bool ok = true;
        for (auto &output : gate->output_nodes) {
          if (!executable[output->index]) {
            ok = false;
          }
        }
        if (!gate->gate->is_sparse()) {
          for (auto &output : gate->output_nodes) {
            if (!local_qubit[output->index]) {
              ok = false;
            }
          }
        }
        if (ok) {
          // execute
          for (auto &output : gate->output_nodes) {
            std::cout << output->index << " ";
          }
          std::cout << "execute\n";
          executed[gate.get()] = true;
        } else {
          // not executable, block the qubits
          all_done = false;
          for (auto &output : gate->output_nodes) {
            executable[output->index] = false;
          }
        }
      }
    }
    if (all_done) {
      break;
    }
    num_shuffles++;
    // count global and local gates
    std::vector<bool> first_unexecuted_gate(num_qubits, false);
    std::vector<int> local_gates(num_qubits, 0);
    std::vector<int> global_gates(num_qubits, 0);
    bool first = true;
    for (auto &gate : dag->edges) {
      if (gate->gate->is_quantum_gate() && !executed[gate.get()]) {
        bool local = true;
        if (!gate->gate->is_sparse()) {
          for (auto &output : gate->output_nodes) {
            if (!local_qubit[output->index]) {
              local = false;
            }
          }
        }
        for (auto &output : gate->output_nodes) {
          if (local) {
            local_gates[output->index]++;
          } else {
            global_gates[output->index]++;
          }
          if (first) {
            first_unexecuted_gate[output->index] = true;
          }
        }
        first = false;
      }
    }
    auto cmp = [&](int a, int b) {
      if (first_unexecuted_gate[b])
        return false;
      if (first_unexecuted_gate[a])
        return true;
      if (global_gates[a] != global_gates[b]) {
        return global_gates[a] > global_gates[b];
      }
      return local_gates[a] > local_gates[b];
    };
    std::vector<int> candidate_indices(num_qubits, 0);
    for (int i = 0; i < num_qubits; i++) {
      candidate_indices[i] = i;
      local_qubit[i] = false;
    }
    std::sort(candidate_indices.begin(), candidate_indices.end(), cmp);
    std::cout << "Shuffle " << num_shuffles << ": {";
    for (int i = 0; i < num_local_qubits; i++) {
      local_qubit[candidate_indices[i]] = true;
      std::cout << candidate_indices[i];
      if (i < num_local_qubits - 1) {
        std::cout << ", ";
      }
    }
    std::cout << "}" << std::endl;
  }
  std::cout << num_shuffles << " shuffles." << std::endl;
  return 0;
}
