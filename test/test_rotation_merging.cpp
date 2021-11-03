#include "test_optimization.h"
#include "../gate/gate_utils.h"

#include <iostream>

int main() {

  Context src_ctx({GateType::input_param, GateType::input_qubit, GateType::t,
                   GateType::tdg, GateType::h, GateType::cx});
  Context dst_ctx({GateType::input_param, GateType::input_qubit, GateType::rz,
                   GateType::h, GateType::cx});
  Context union_ctx({GateType::input_param, GateType::input_qubit, GateType::t,
                     GateType::tdg, GateType::h, GateType::cx, GateType::rz});
  QASMParser qasm_parser(&union_ctx);
  DAG *dag = nullptr;
  if (!qasm_parser.load_qasm("circuit/example-circuits/barenco_tof_3.qasm",
                             dag)) {
	std::cout << "Parser failed" << std::endl;
  }

  TASOGraph::Graph graph(&union_ctx, *dag);
  RuleParser rule_parser({"t q0 = rz q0 0.25pi", // t -> u1
                          "tdg q0 = rz q0 -0.25pi"});
  TASOGraph::Graph *newGraph =
      graph.context_shift(&src_ctx, &dst_ctx, &union_ctx, &rule_parser);
  newGraph->rotation_merging(GateType::cx);
  std::cout << newGraph->total_cost() << " gates in circuit before optimizing."
            << std::endl;
}