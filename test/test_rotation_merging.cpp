#include "../gate/gate_utils.h"
#include "../context/context.h"
#include "../tasograph/tasograph.h"
#include "../parser/qasm_parser.h"

#include <iostream>

int main() {

  Context src_ctx({GateType::input_param, GateType::input_qubit, GateType::t,
                   GateType::tdg, GateType::cx});
  Context dst_ctx({GateType::input_param, GateType::input_qubit, GateType::rz,
                   GateType::cx});
  Context union_ctx({GateType::input_param, GateType::input_qubit, GateType::t,
                     GateType::tdg, GateType::cx, GateType::rz});
  QASMParser qasm_parser(&union_ctx);
  DAG *dag = nullptr;
  if (!qasm_parser.load_qasm("circuit/example-circuits/10_gates.qasm", dag)) {
	std::cout << "Parser failed" << std::endl;
  }

  TASOGraph::Graph graph(&union_ctx, *dag);
  RuleParser rule_parser({"t q0 = rz q0 0.25pi", // t -> u1
                          "tdg q0 = rz q0 -0.25pi"});
  TASOGraph::Graph *newGraph =
      graph.context_shift(&src_ctx, &dst_ctx, &union_ctx, &rule_parser);
  for (auto it = newGraph->inEdges.begin(); it != newGraph->inEdges.end();
       ++it) {
	std::cout << gate_type_name(it->first.ptr->tp) << std::endl;
  }
  newGraph->rotation_merging(GateType::rz);
  std::cout << newGraph->total_cost()
            << " gates in circuit after rotation merging." << std::endl;
}