#include "quartz/gate/gate_utils.h"
#include "quartz/parser/qasm_parser.h"
#include "quartz/pybind/pybind.h"

#include <filesystem>

using namespace quartz;

int main() {
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

  std::filesystem::path this_file_path(__FILE__);
  auto circuit_folder =
      this_file_path.parent_path().parent_path().parent_path().append(
          "circuit");
  std::vector<int> num_qubits;
  for (int i = 28; i <= 34; i++) {
    num_qubits.push_back(i);
  }
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
  return 0;
}
