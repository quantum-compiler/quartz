#include "quartz/context/context.h"
#include "quartz/gate/gate_utils.h"
#include "quartz/parser/qasm_parser.h"
#include "quartz/pybind/pybind.h"
#include "quartz/simulator/schedule.h"

#include <algorithm>
#include <chrono>
#include <iostream>
#include <vector>

using namespace quartz;

int main() {
  // assume the work directory is in build/
  auto start = std::chrono::steady_clock::now();
  init_python_interpreter();
  PythonInterpreter interpreter;
  ParamInfo param_info;
  Context ctx({GateType::input_qubit, GateType::input_param, GateType::h,
               GateType::x, GateType::ry, GateType::u2, GateType::u3,
               GateType::cx, GateType::cz, GateType::cp, GateType::swap,
               GateType::rz, GateType::p, GateType::ccx, GateType::rx},
              &param_info);
  std::vector<std::string> circuit_names = {
      "ae",           "dj",       "ghz",       "graphstate", "qft",
      "qftentangled", "qpeexact", "su2random", "wstate"};

  std::vector<std::string> circuit_names_nwq = {"bv", "ising", "qsvm", "vqc",
                                                "hhl"};
  // 28-34 total qubits, 28 local qubits
  std::vector<int> num_qubits;
  for (int i = 28; i <= 34; i++) {
    num_qubits.push_back(i);
  }
  std::vector<int> num_local_qubits = {28};
  KernelCost kernel_cost(
      /*fusion_kernel_costs=*/
      {0, 6.4, 6.2, 6.5, 6.4, 6.4, 25.8, 32.4},
      /*shared_memory_init_cost=*/6,
      /*shared_memory_gate_cost=*/
      [](quartz::GateType type) {
        if (type == quartz::GateType::swap)
          return 1000.0;
        else
          return 0.5;
      },
      /*shared_memory_total_qubits=*/10, /*shared_memory_cacheline_qubits=*/3);
  std::vector<int> dp_t = {-5,  0,   2,    4,    10,   16,   20,  32,
                           50,  70,  100,  150,  200,  300,  400, 500,
                           650, 800, 1000, 1500, 2000, 3000, 4000};
  FILE *fout = fopen("../dp_result.csv", "w");
  for (int run_nwq = 0; run_nwq <= 1; run_nwq++) {
    for (const auto &circuit : (run_nwq ? circuit_names_nwq : circuit_names)) {
      fprintf(fout, "%s\n", circuit.c_str());
      std::cout << circuit << std::endl;
      for (int num_q : num_qubits) {
        if (circuit == std::string("hhl")) {
          if (num_q == 28) {
            num_q = 4;
          } else if (num_q == 29) {
            num_q = 7;
          } else if (num_q == 30) {
            num_q = 9;
          } else if (num_q == 31) {
            num_q = 10;
          } else {
            break;
          }
        }
        // requires running test_remove_swap first
        auto seq = CircuitSeq::from_qasm_file(
            &ctx, (run_nwq ? (std::string("../circuit/NWQBench/") + circuit +
                              "_" + (circuit == std::string("hhl") ? "" : "n") +
                              std::to_string(num_q) + ".qasm")
                           : (std::string("../circuit/MQTBench_") +
                              std::to_string(num_q) + "q/" + circuit +
                              "_indep_qiskit_" + std::to_string(num_q) +
                              "_no_swap.qasm")));
        fprintf(fout, "%d, ", num_q);
        int answer_start_with = 1;
        bool has_local_q_at_least_num_q = false;
        for (int local_q : num_local_qubits) {
          if (local_q >= num_q) {
            if (has_local_q_at_least_num_q) {
              break;
            }
            local_q = num_q;
            has_local_q_at_least_num_q = true;
          }
          int regional_q = std::min(2, num_q - local_q);
          int global_q = num_q - local_q - regional_q;
          auto t0 = std::chrono::steady_clock::now();
          std::vector<std::vector<int>> qubit_layout;
          qubit_layout = compute_qubit_layout_with_ilp(
              *seq, local_q, regional_q, &ctx, &interpreter, answer_start_with);
          auto t0end = std::chrono::steady_clock::now();
          int ilp_result = (int)qubit_layout.size();
          std::cout << qubit_layout.size() << " stages." << std::endl;
          auto t1 = std::chrono::steady_clock::now();
          auto schedules = get_schedules(
              *seq, local_q, qubit_layout, kernel_cost, &ctx,
              /*attach_single_qubit_gates=*/true,
              /*max_num_dp_states=*/500,
              /*cache_file_name_prefix=*/circuit + std::to_string(num_q) + "_" +
                  std::to_string(local_q));
          auto t1end = std::chrono::steady_clock::now();
          KernelCostType total_cost = 0;
          for (auto &schedule : schedules) {
            // schedule.print_kernel_info();
            total_cost += schedule.cost_;
          }
          std::cout << "Schedule for " << circuit << " with " << num_q
                    << " qubits and " << local_q
                    << " local qubits:" << std::endl;
          std::cout << "Stage 0: layout ";
          schedules[0].print_qubit_layout(global_q);
          for (int i = 1; i < (int)schedules.size(); i++) {
            auto local_swaps = schedules[i].get_local_swaps_from_previous_stage(
                schedules[i - 1]);
            if (!local_swaps.empty()) {
              std::cout << "  swap";
              for (auto &s : local_swaps) {
                std::cout << " (" << s.first << ", " << s.second << ")";
              }
              std::cout << std::endl;
            }
            std::cout << "Stage " << i << ": layout ";
            schedules[i].print_qubit_layout(global_q);
          }
          verify_schedule(&ctx, *seq, schedules, /*random_test_times=*/0);
          std::vector<double> ts;
          for (int t : dp_t) {
            auto t2 = std::chrono::steady_clock::now();
            schedules =
                get_schedules(*seq, local_q, qubit_layout, kernel_cost, &ctx,
                              /*attach_single_qubit_gates=*/true,
                              /*max_num_dp_states=*/t,
                              /*cache_file_name_prefix=*/"");
            auto t2end = std::chrono::steady_clock::now();
            total_cost = 0;
            for (auto &schedule : schedules) {
              // schedule.print_kernel_info();
              total_cost += schedule.cost_;
            }
            fprintf(fout, "%.1f, ", total_cost);
            fflush(fout);
            ts.push_back(
                (double)std::chrono::duration_cast<std::chrono::microseconds>(
                    t2end - t2)
                    .count() /
                1e6);
          }
          fprintf(fout, "%.6f",
                  (double)std::chrono::duration_cast<std::chrono::milliseconds>(
                      t1end - t1)
                          .count() /
                      1e6);
          for (auto t : ts) {
            fprintf(fout, ", %.6f", t);
          }
          fprintf(fout, ", %.6f",
                  (double)std::chrono::duration_cast<std::chrono::microseconds>(
                      t0end - t0)
                          .count() /
                      1e6);
          answer_start_with = ilp_result;
        }
        fprintf(fout, "\n");
        fflush(fout);
      }
    }
  }
  fclose(fout);
  auto end = std::chrono::steady_clock::now();
  std::cout
      << std::chrono::duration_cast<std::chrono::seconds>(end - start).count()
      << " seconds." << std::endl;
  return 0;
}
