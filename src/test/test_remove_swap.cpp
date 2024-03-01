#include "quartz/gate/gate_utils.h"
#include "quartz/parser/qasm_parser.h"

#include <filesystem>

using namespace quartz;
void remove_swap_for_nwq(Context *ctx) {
  std::vector<std::string> circuit_names = {"bv", "hhl", "ising", "qsvm",
                                            "vqc"};

  std::filesystem::path this_file_path(__FILE__);
  auto circuit_folder =
      this_file_path.parent_path().parent_path().parent_path().append(
          "circuit");
  std::vector<int> num_qubits;
  for (int i = 28; i <= 34; i++) {
    num_qubits.push_back(i);
  }
  num_qubits.push_back(42);
  for (int num_q1 : num_qubits) {
    std::cout << num_q1 << " qubits:" << std::endl;
    for (const auto &circuit : circuit_names) {
      int num_q = num_q1;
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
          continue;
        }
      }
      auto seq = CircuitSeq::from_qasm_file(
          ctx, circuit_folder.string() +
                   (std::string("/NWQBench/") + circuit + "_" +
                    (circuit == std::string("hhl") ? "" : "n") +
                    std::to_string(num_q) + ".qasm"));
      int num_swap = seq->remove_swap_gates();
      std::cout << circuit << ": " << num_swap << " swap gates removed."
                << std::endl;
      seq->to_qasm_file(ctx, circuit_folder.string() +
                                 (std::string("/NWQBench/") + circuit + "_" +
                                  (circuit == std::string("hhl") ? "" : "n") +
                                  std::to_string(num_q) + "_no_swap.qasm"));
    }
  }
}
int main() {
  ParamInfo param_info;
  Context ctx({GateType::input_qubit, GateType::input_param, GateType::h,
               GateType::x, GateType::ry, GateType::u2, GateType::u3,
               GateType::cx, GateType::cz, GateType::cp, GateType::swap,
               GateType::rz, GateType::rx, GateType::p},
              &param_info);
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

  std::filesystem::path this_file_path(__FILE__);
  auto circuit_folder =
      this_file_path.parent_path().parent_path().parent_path().append(
          "circuit");
  std::vector<int> num_qubits;
  for (int i = 28; i <= 34; i++) {
    num_qubits.push_back(i);
  }
  num_qubits.push_back(42);
  for (int num_q : num_qubits) {
    std::cout << num_q << " qubits:" << std::endl;
    for (const auto &circuit : circuit_names) {
      auto seq = CircuitSeq::from_qasm_file(
          &ctx, circuit_folder.string() + "/MQTBench_" + std::to_string(num_q) +
                    "q/" + circuit + "_indep_qiskit_" + std::to_string(num_q) +
                    ".qasm");
      int num_swap = seq->remove_swap_gates();
      std::cout << circuit << ": " << num_swap << " swap gates removed."
                << std::endl;
      seq->to_qasm_file(&ctx, circuit_folder.string() + "/MQTBench_" +
                                  std::to_string(num_q) + "q/" + circuit +
                                  "_indep_qiskit_" + std::to_string(num_q) +
                                  "_no_swap.qasm");
    }
  }
  // remove_swap_for_nwq(&ctx);
  return 0;
}
