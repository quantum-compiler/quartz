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
  Context ctx({GateType::input_qubit, GateType::input_param, GateType::h,
               GateType::x, GateType::ry, GateType::u2, GateType::u3,
               GateType::cx, GateType::cz, GateType::cp, GateType::swap,
               GateType::rz});
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

  std::vector<std::string> circuit_names_nwq = {"bv", "ising"};
  // 28-34 total qubits, 28 local qubits
  std::vector<int> num_qubits;
  for (int i = 28; i <= 34; i++) {
    num_qubits.push_back(i);
  }
  std::vector<int> num_local_qubits = {28};
  KernelCost kernel_cost(
      /*fusion_kernel_costs=*/{0, 6.4, 6.2, 6.5, 6.4, 6.4, 25.8, 32.4},
      /*shared_memory_init_cost=*/6,
      /*shared_memory_gate_cost=*/
      [](quartz::GateType type) {
        if (type == quartz::GateType::swap)
          return 1000.0;
        else
          return 0.5;
      },
      /*shared_memory_total_qubits=*/10, /*shared_memory_cacheline_qubits=*/3);
  FILE *fout = fopen("../dp_result.csv", "w");
  for (auto circuit : circuit_names) {
    // nwq:
    //  for (auto circuit : circuit_names_nwq) {
    fprintf(fout, "%s\n", circuit.c_str());
    std::cout << circuit << std::endl;
    for (int num_q : num_qubits) {
      auto seq = CircuitSeq::from_qasm_file(
          &ctx, std::string("../circuit/MQTBench_") + std::to_string(num_q) +
                    "q/" + circuit + "_indep_qiskit_" + std::to_string(num_q) +
                    ".qasm");
      // nwq:
      //      auto seq = CircuitSeq::from_qasm_file(
      //          &ctx,
      //          std::string("../../../PycharmProjects/nwqbench/NWQ_Bench/") +
      //                    circuit + "/qasm/" + circuit + "_n" +
      //                    std::to_string(num_q) + ".qasm");
      // TODO: remove swap gates

      fprintf(fout, "%d, ", num_q);
      int answer_start_with = 1;
      for (int local_q : num_local_qubits) {
        if (local_q > num_q) {
          continue;
        }
        auto t0 = std::chrono::steady_clock::now();
        std::vector<std::vector<int>> local_qubits;
        local_qubits = compute_local_qubits_with_ilp(
            *seq, local_q, &ctx, &interpreter, answer_start_with);
        int ilp_result = (int)local_qubits.size();
        std::cout << local_qubits.size() << " stages." << std::endl;
        auto t1 = std::chrono::steady_clock::now();
        auto schedules = get_schedules(*seq, local_qubits, kernel_cost, &ctx,
                                       /*attach_single_qubit_gates=*/true,
                                       /*use_simple_dp_times=*/1);
        KernelCostType total_cost = 0;
        for (auto &schedule : schedules) {
          // schedule.print_kernel_info();
          total_cost += schedule.cost_;
        }
        fprintf(fout, "%.1f, ", total_cost);
        fflush(fout);
        auto t2 = std::chrono::steady_clock::now();
        schedules = get_schedules(*seq, local_qubits, kernel_cost, &ctx,
                                  /*attach_single_qubit_gates=*/true,
                                  /*use_simple_dp_times=*/-1);
        total_cost = 0;
        for (auto &schedule : schedules) {
          // schedule.print_kernel_info();
          total_cost += schedule.cost_;
        }
        fprintf(fout, "%.1f, ", total_cost);
        fflush(fout);
        auto t3 = std::chrono::steady_clock::now();
        schedules = get_schedules(*seq, local_qubits, kernel_cost, &ctx,
                                  /*attach_single_qubit_gates=*/true,
                                  /*use_simple_dp_times=*/10);
        total_cost = 0;
        for (auto &schedule : schedules) {
          // schedule.print_kernel_info();
          total_cost += schedule.cost_;
        }
        fprintf(fout, "%.1f, ", total_cost);
        fflush(fout);
        auto t4 = std::chrono::steady_clock::now();
        schedules = get_schedules(*seq, local_qubits, kernel_cost, &ctx,
                                  /*attach_single_qubit_gates=*/true,
                                  /*use_simple_dp_times=*/100);
        total_cost = 0;
        for (auto &schedule : schedules) {
          // schedule.print_kernel_info();
          total_cost += schedule.cost_;
        }
        fprintf(fout, "%.1f, ", total_cost);
        fflush(fout);
        auto t5 = std::chrono::steady_clock::now();
        schedules = get_schedules(*seq, local_qubits, kernel_cost, &ctx,
                                  /*attach_single_qubit_gates=*/true,
                                  /*use_simple_dp_times=*/0);
        total_cost = 0;
        for (auto &schedule : schedules) {
          // schedule.print_kernel_info();
          total_cost += schedule.cost_;
        }
        fprintf(fout, "%.1f, ", total_cost);
        fflush(fout);
        auto t6 = std::chrono::steady_clock::now();
        fprintf(fout, "%.3f, %.3f, %.3f, %.3f, %.3f",
                (double)std::chrono::duration_cast<std::chrono::milliseconds>(
                    t2 - t1)
                        .count() /
                    1000.0,
                (double)std::chrono::duration_cast<std::chrono::milliseconds>(
                    t3 - t2)
                        .count() /
                    1000.0,
                (double)std::chrono::duration_cast<std::chrono::milliseconds>(
                    t4 - t3)
                        .count() /
                    1000.0,
                (double)std::chrono::duration_cast<std::chrono::milliseconds>(
                    t5 - t4)
                        .count() /
                    1000.0,
                (double)std::chrono::duration_cast<std::chrono::milliseconds>(
                    t6 - t5)
                        .count() /
                    1000.0);
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
