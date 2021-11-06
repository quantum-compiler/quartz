#include "../gate/gate_utils.h"
#include "../context/context.h"
#include "../tasograph/tasograph.h"
#include "../parser/qasm_parser.h"

#include <iostream>

int main() {

  Context src_ctx({GateType::input_param, GateType::input_qubit, GateType::t,
                   GateType::tdg, GateType::cx, GateType::h});
  Context dst_ctx({GateType::input_param, GateType::input_qubit, GateType::h,
                   GateType::rz, GateType::cx});
  //   Context union_ctx({GateType::input_param, GateType::input_qubit,
  //   GateType::t,
  //                      GateType::tdg, GateType::cx, GateType::rz});
  Context union_ctx = union_contexts(&src_ctx, &dst_ctx);
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
  newGraph->rotation_merging(GateType::rz);
  std::cout << newGraph->total_cost()
            << " gates in circuit after rotation merging." << std::endl;
  newGraph->to_qasm("temp.qasm", /*print_result=*/true);
  //   newGraph->draw_circuit("temp.qasm", "temp.png");
}