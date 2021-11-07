#include "../tasograph/tasograph.h"
#include "../tasograph/substitution.h"
#include "../parser/qasm_parser.h"

int main() {
  std::string benchmark_filename =
      "circuit/voqc-benchmarks/barenco_tof_10.qasm";
  std::string result_qasm_filename =
      "circuit/voqc-benchmarks/barenco_tof_10_toffoli_flip.qasm";
  // Construct rules
  RuleParser toffoli_0({"ccz q0 q1 q2 = cx q1 q2; rz q2 -0.25pi; cx q0 q2; rz "
                        "q2 0.25pi; cx q1 q2; rz q2 -0.25pi; cx "
                        "q0 q2; cx q0 q1; rz q1 -0.25pi; cx q0 q1; rz q0 "
                        "0.25pi; rz q1 0.25pi; rz q2 0.25pi;"});
  RuleParser toffoli_1({"ccz q0 q1 q2 = cx q1 q2; rz q2 0.25pi; cx q0 q2; rz "
                        "q2 -0.25pi; cx q1 q2; rz q2 0.25pi; cx "
                        "q0 q2; cx q0 q1; rz q1 0.25pi; cx q0 q1; rz q0 "
                        "-0.25pi; rz q1 -0.25pi; rz q2 -0.25pi;"});
  // Construct contexts
  Context src_ctx({GateType::h, GateType::ccz, GateType::input_qubit,
                   GateType::input_param});
  Context dst_ctx({GateType::rz, GateType::cx, GateType::h,
                   GateType::input_qubit, GateType::input_param});
  Context union_ctx({GateType::ccz, GateType::rz, GateType::cx, GateType::h,
                     GateType::input_qubit, GateType::input_param});
  // Construct GraphXfers
  std::vector<Command> cmds;
  Command cmd;
  toffoli_0.find_convert_commands(&dst_ctx, GateType::ccz, cmd, cmds);
  TASOGraph::GraphXfer *xfer =
      TASOGraph::GraphXfer::create_single_gate_GraphXfer(&union_ctx, cmd, cmds);
  toffoli_1.find_convert_commands(&dst_ctx, GateType::ccz, cmd, cmds);
  TASOGraph::GraphXfer *xfer_inverse =
      TASOGraph::GraphXfer::create_single_gate_GraphXfer(&union_ctx, cmd, cmds);
  // Load qasm file
  QASMParser qasm_parser(&src_ctx);
  DAG *dag = nullptr;
  if (!qasm_parser.load_qasm(benchmark_filename, dag)) {
	std::cout << "Parser failed" << std::endl;
  }
  TASOGraph::Graph graph(&src_ctx, *dag);
  TASOGraph::Graph *new_graph =
      graph.toffoli_flip_greedy(GateType::rz, xfer, xfer_inverse);
  std::cout << "gate count after toffoli flip: " << new_graph->total_cost()
            << std::endl;
  new_graph->to_qasm(result_qasm_filename, false, false);
}