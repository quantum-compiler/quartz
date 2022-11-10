#include "quartz/context/context.h"
#include "quartz/gate/gate_utils.h"
#include "quartz/parser/qasm_parser.h"
#include "quartz/tasograph/tasograph.h"

#include <algorithm>
#include <iostream>
#include <unordered_map>

using namespace quartz;

int num_shuffles_by_heuristics(CircuitSeq *seq, int num_local_qubits) {
  int num_qubits = seq->get_num_qubits();
  std::unordered_map<CircuitGate *, bool> executed;
  std::vector<bool> local_qubit(num_qubits, false);
  for (int i = 0; i < num_local_qubits; i++) {
    local_qubit[i] = true;
  }
  int num_shuffles = 0;
  while (true) {
    bool all_done = true;
    std::vector<bool> executable(num_qubits, true);
    for (auto &gate : seq->gates) {
      if (gate->gate->is_quantum_gate() && !executed[gate.get()]) {
        bool ok = true;
        for (auto &output : gate->output_wires) {
          if (!executable[output->index]) {
            ok = false;
          }
        }
        if (!gate->gate->is_sparse()) {
          for (auto &output : gate->output_wires) {
            if (!local_qubit[output->index]) {
              ok = false;
            }
          }
        }
        if (ok) {
          // execute
          /*for (auto &output : gate->output_nodes) {
            std::cout << output->index << " ";
          }
          std::cout << "execute\n";*/
          executed[gate.get()] = true;
        } else {
          // not executable, block the qubits
          all_done = false;
          for (auto &output : gate->output_wires) {
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
    for (auto &gate : seq->gates) {
      if (gate->gate->is_quantum_gate() && !executed[gate.get()]) {
        bool local = true;
        if (!gate->gate->is_sparse()) {
          for (auto &output : gate->output_wires) {
            if (!local_qubit[output->index]) {
              local = false;
            }
          }
        }
        for (auto &output : gate->output_wires) {
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
  return num_shuffles;
}

int main() {
  Context ctx({GateType::input_qubit, GateType::input_param, GateType::h,
               GateType::x, GateType::ry, GateType::u2, GateType::u3,
               GateType::cx, GateType::cz, GateType::cp, GateType::swap});
  std::vector<std::string> circuit_names = {
      "dj",           "ghz",           "graphstate", "qft",
      "qftentangled", "realamprandom", "su2random",  "twolocalrandom",
      "wstate"};
  std::vector<int> num_qubits = {40, 41, 42};
  std::vector<int> num_local_qubits;
  for (int i = 12; i <= 33; i++) {
    num_local_qubits.push_back(i);
  }
  FILE *fout = fopen("result.txt", "w");
  for (auto circuit : circuit_names) {
    fprintf(fout, "%s\n", circuit.c_str());
    for (int num_q : num_qubits) {
      auto graph = Graph::from_qasm_file(
          &ctx, std::string("circuit/MQTBench_") + std::to_string(num_q) +
                    "q/" + circuit + "_indep_qiskit_" + std::to_string(num_q) +
                    ".qasm");
      auto seq = graph->to_circuit_sequence();
      fprintf(fout, "%d", num_q);
      for (int local_q : num_local_qubits) {
        int result = num_shuffles_by_heuristics(seq.get(), local_q);
        fprintf(fout, " %d", result);
      }
      fprintf(fout, "\n");
    }
  }
  fclose(fout);
  return 0;
}
