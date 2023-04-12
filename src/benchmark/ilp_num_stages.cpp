#include "quartz/context/context.h"
#include "quartz/gate/gate_utils.h"
#include "quartz/parser/qasm_parser.h"
#include "quartz/pybind/pybind.h"
#include "quartz/simulator/schedule.h"
#include "quartz/tasograph/tasograph.h"

#include <algorithm>
#include <chrono>
#include <iostream>
#include <unordered_map>
#include <vector>

using namespace quartz;

int num_stages_by_heuristics(CircuitSeq *seq, int num_local_qubits,
                             std::vector<std::vector<bool>> &local_qubits) {
  int num_qubits = seq->get_num_qubits();
  std::unordered_map<CircuitGate *, bool> executed;
  // No initial configuration -- all qubits are global.
  std::vector<bool> local_qubit(num_qubits, false);
  int num_stages = 0;
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
        if (!gate->gate->is_diagonal()) {
          int num_remaining_control_qubits =
              gate->gate->get_num_control_qubits();
          for (auto &output : gate->output_wires) {
            if (output->is_qubit()) {
              num_remaining_control_qubits--;
              if (num_remaining_control_qubits < 0 &&
                  !local_qubit[output->index]) {
                ok = false;
              }
            }
          }
        }
        if (ok) {
          // execute
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
    num_stages++;
    // count global and local gates
    std::vector<bool> first_unexecuted_gate(num_qubits, false);
    std::vector<int> local_gates(num_qubits, 0);
    std::vector<int> global_gates(num_qubits, 0);
    bool first = true;
    for (auto &gate : seq->gates) {
      if (gate->gate->is_quantum_gate() && !executed[gate.get()]) {
        bool local = true;
        if (!gate->gate->is_diagonal()) {
          int num_remaining_control_qubits =
              gate->gate->get_num_control_qubits();
          for (auto &output : gate->output_wires) {
            if (output->is_qubit()) {
              num_remaining_control_qubits--;
              if (num_remaining_control_qubits < 0 &&
                  !local_qubit[output->index]) {
                local = false;
              }
            }
          }
        }
        int num_remaining_control_qubits = gate->gate->get_num_control_qubits();
        for (auto &output : gate->output_wires) {
          if (output->is_qubit()) {
            num_remaining_control_qubits--;
            if (local) {
              local_gates[output->index]++;
            } else {
              global_gates[output->index]++;
            }
            if (first && num_remaining_control_qubits < 0) {
              first_unexecuted_gate[output->index] = true;
            }
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
      if (local_gates[a] != local_gates[b]) {
        return local_gates[a] > local_gates[b];
      }
      // Use the qubit index as a final tiebreaker.
      return a < b;
    };
    std::vector<int> candidate_indices(num_qubits, 0);
    for (int i = 0; i < num_qubits; i++) {
      candidate_indices[i] = i;
      local_qubit[i] = false;
    }
    std::sort(candidate_indices.begin(), candidate_indices.end(), cmp);
    std::cout << "Stage " << num_stages << ": {";
    for (int i = 0; i < num_local_qubits; i++) {
      local_qubit[candidate_indices[i]] = true;
      std::cout << candidate_indices[i];
      if (i < num_local_qubits - 1) {
        std::cout << ", ";
      }
    }
    std::cout << "}" << std::endl;
    local_qubits.push_back(local_qubit);
  }
  std::cout << num_stages << " stages." << std::endl;
  return num_stages;
}

int main() {
  auto start = std::chrono::steady_clock::now();
  init_python_interpreter();
  PythonInterpreter interpreter;
  Context ctx({GateType::input_qubit, GateType::input_param, GateType::h,
               GateType::x, GateType::ry, GateType::u2, GateType::u3,
               GateType::cx, GateType::cz, GateType::cp, GateType::swap});
  std::vector<std::string> circuit_names = {"ae",
                                            "dj",
                                            "ghz",
                                            "graphstate",
                                            "qft",
                                            "qftentangled",
                                            "qpeexact",
                                            "qpeinexact",
                                            "realamprandom",
                                            "su2random",
                                            "twolocalrandom",
                                            "wstate"};
  // 31 or 42 total qubits, 23-33 local qubits
  std::vector<int> num_qubits = {31, 42};
  std::vector<int> num_local_qubits;
  for (int i = 33; i >= 23; i--) {
    num_local_qubits.push_back(i);
  }
  FILE *fout = fopen("result.csv", "w");
  for (auto circuit : circuit_names) {
    fprintf(fout, "%s\n", circuit.c_str());
    for (int num_q : num_qubits) {
      auto seq = CircuitSeq::from_qasm_file(
          &ctx, std::string("circuit/MQTBench_") + std::to_string(num_q) +
                    "q/" + circuit + "_indep_qiskit_" + std::to_string(num_q) +
                    ".qasm");

      fprintf(fout, "%d, ", num_q);
      for (int local_q : num_local_qubits) {
        if (local_q > num_q) {
          continue;
        }
        std::vector<std::vector<bool>> local_qubits_by_heuristics;
        int heuristics_result = num_stages_by_heuristics(
            seq.get(), local_q, local_qubits_by_heuristics);
        fprintf(fout, "%d, ", heuristics_result);
        fflush(fout);
      }
      fprintf(fout, "\n");
    }
    for (int num_q : num_qubits) {
      auto seq = CircuitSeq::from_qasm_file(
          &ctx, std::string("circuit/MQTBench_") + std::to_string(num_q) +
                    "q/" + circuit + "_indep_qiskit_" + std::to_string(num_q) +
                    ".qasm");

      fprintf(fout, "%d, ", num_q);
      int answer_start_with = 1;
      for (int local_q : num_local_qubits) {
        if (local_q > num_q) {
          continue;
        }
        std::vector<std::vector<int>> local_qubits;
        local_qubits = compute_local_qubits_with_ilp(
            *seq, local_q, &ctx, &interpreter, answer_start_with);
        int ilp_result = (int)local_qubits.size();
        for (int j = 0; j < ilp_result; j++) {
          std::cout << "Stage " << j << ": ";
          for (int k = 0; k < local_q; k++) {
            std::cout << local_qubits[j][k] << " ";
          }
          std::cout << std::endl;
        }
        fprintf(fout, "%d, ", ilp_result);
        fflush(fout);
        answer_start_with = ilp_result;
      }
      fprintf(fout, "\n");
    }
  }
  fclose(fout);
  auto end = std::chrono::steady_clock::now();
  std::cout
      << std::chrono::duration_cast<std::chrono::seconds>(end - start).count()
      << " seconds." << std::endl;
  return 0;
}
