#include "quartz/context/context.h"
#include "quartz/gate/gate_utils.h"
#include "quartz/parser/qasm_parser.h"
#include "quartz/tasograph/tasograph.h"

#include <iostream>

using namespace quartz;

int main() {
  Context ctx(
      {GateType::h, GateType::u2, GateType::u3, GateType::cx, GateType::cp});
  auto graph = Graph::from_qasm_file(
      &ctx, "circuit/MQTBench_40q/ae_indep_qiskit_40.qasm");
  std::cout << "TODO" << std::endl;
  return 0;
}
