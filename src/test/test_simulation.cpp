#include "quartz/context/context.h"
#include "quartz/gate/gate_utils.h"
#include "quartz/parser/qasm_parser.h"
#include "quartz/tasograph/tasograph.h"

#include <iostream>

using namespace quartz;

int main() {
  Context ctx({GateType::input_qubit, GateType::input_param, GateType::h,
               GateType::u2, GateType::cx});
  auto graph = Graph::from_qasm_file(
      &ctx, "circuit/MQTBench_40q/dj_indep_qiskit_40.qasm");
  std::cout << "TODO" << std::endl;
  return 0;
}
