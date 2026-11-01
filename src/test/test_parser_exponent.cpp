#include "quartz/circuitseq/circuitseq.h"
#include "quartz/context/context.h"
#include "quartz/parser/qasm_parser.h"

#include <cmath>
#include <iostream>
#include <string>
#include <utility>
#include <vector>

using namespace quartz;

// Rotation angles written in scientific notation (e.g. 9.5e-05, as printed by
// many tools) must not be split at the sign of their exponent: each angle must
// parse to the same circuit as the angle written as a plain decimal.
int main() {
  ParamInfo param_info;
  Context ctx({GateType::input_qubit, GateType::input_param, GateType::add,
               GateType::rz},
              /*num_qubits=*/1, &param_info);
  auto parse = [&](const std::string &angle) {
    QASMParser parser(&ctx);
    CircuitSeq *seq = nullptr;
    const bool ok = parser.load_qasm_str(
        "OPENQASM 2.0;\ninclude \"qelib1.inc\";\nqreg q[1];\nrz(" + angle +
            ") q[0];\n",
        seq);
    return ok ? seq : nullptr;
  };
  const std::vector<std::pair<std::string, std::string>> cases = {
      {"9.58737992428526e-05", "0.0000958737992428526"},
      {"-4.79e-05", "-0.0000479"},
      {"3.451511e-09", "0.000000003451511"},
      {"1.5E+2", "150"},
      {"2e-3", "0.002"},
      {"0.5-1e-3", "0.5-0.001"},
      {"pi*-0.25", "-pi/4"},
  };
  bool all_ok = true;
  for (const auto &[angle, plain] : cases) {
    CircuitSeq *seq1 = parse(angle);
    CircuitSeq *seq2 = parse(plain);
    bool ok = seq1 != nullptr && seq2 != nullptr;
    if (ok) {
      const auto mat1 = seq1->get_matrix(&ctx);
      const auto mat2 = seq2->get_matrix(&ctx);
      for (int i = 0; i < (int)mat1.size(); i++) {
        for (int j = 0; j < (int)mat1[i].size(); j++) {
          ok = ok && std::abs(mat1[i][j] - mat2[i][j]) < 1e-12;
        }
      }
    }
    if (!ok) {
      std::cout << "rz(" << angle << ") is not parsed as rz(" << plain << ")"
                << std::endl;
      all_ok = false;
    }
    delete seq1;
    delete seq2;
  }
  std::cout << (all_ok ? "All angles parsed correctly." : "FAILED")
            << std::endl;
  return all_ok ? 0 : 1;
}
