#include "quartz/parser/qasm_parser.h"
#include "quartz/tasograph/tasograph.h"

using namespace quartz;

#include <iostream>

int main() {
  ParamInfo param_info;
  Context ctx({GateType::input_qubit, GateType::input_param, GateType::t,
               GateType::tdg, GateType::cx},
              &param_info);
  QASMParser qasm_parser(&ctx);
  CircuitSeq *dag = nullptr;
  if (!qasm_parser.load_qasm("circuit/example-circuits/t_cx_tdg.qasm", dag)) {
    std::cout << "Parser failed" << std::endl;
  }

  Graph graph(&ctx, dag);
  graph.to_qasm("temp.qasm", /*print_result=*/true, true);
  graph.draw_circuit("temp.qasm", "temp.png");
}
