#include "quartz/parser/qasm_parser.h"
#include "quartz/tasograph/substitution.h"
#include "quartz/tasograph/tasograph.h"

using namespace quartz;

int main() {
  std::string benchmark_filename =
      "circuit/voqc-benchmarks/barenco_tof_10.qasm";
  std::string result_qasm_filename =
      "circuit/voqc-benchmarks/barenco_tof_10_toffoli_flip.qasm";
  // Construct rules
  // RuleParser toffoli_0(
  //     {"ccz q0 q1 q2 = cx q1 q2; rz q2 -0.25pi; cx q0 q2; rz "
  //      "q2 0.25pi; cx q1 q2; rz q2 -0.25pi; cx "
  //      "q0 q2; cx q0 q1; rz q1 -0.25pi; cx q0 q1; rz q0 "
  //      "0.25pi; rz q1 0.25pi; rz q2 0.25pi;"});
  // RuleParser toffoli_1({"ccz q0 q1 q2 = cx q1 q2; rz q2 0.25pi; cx q0 q2; rz
  // "
  //                       "q2 -0.25pi; cx q1 q2; rz q2 0.25pi; cx "
  //                       "q0 q2; cx q0 q1; rz q1 0.25pi; cx q0 q1; rz q0 "
  //                       "-0.25pi; rz q1 -0.25pi; rz q2 -0.25pi;"});
  auto rules = RuleParser::ccz_cx_t_rules();
  // Construct contexts
  ParamInfo param_info;
  Context src_ctx({GateType::h, GateType::ccz, GateType::input_qubit,
                   GateType::input_param},
                  &param_info);
  Context dst_ctx({GateType::t, GateType::tdg, GateType::cx, GateType::h,
                   GateType::input_qubit, GateType::input_param},
                  &param_info);
  Context union_ctx({GateType::ccz, GateType::t, GateType::tdg, GateType::cx,
                     GateType::h, GateType::input_qubit, GateType::input_param},
                    &param_info);
  // Construct GraphXfers
  std::vector<std::vector<Command>> cmds;
  std::vector<Command> cmd;
  rules.find_convert_commands(&dst_ctx, GateType::ccz, cmd, cmds);
  GraphXfer *xfer = GraphXfer::create_single_gate_GraphXfer(
      &src_ctx, &dst_ctx, &union_ctx, cmd[0], cmds[0]);
  GraphXfer *xfer_inverse = GraphXfer::create_single_gate_GraphXfer(
      &src_ctx, &dst_ctx, &union_ctx, cmd[1], cmds[1]);
  // Load qasm file
  QASMParser qasm_parser(&src_ctx);
  CircuitSeq *dag = nullptr;
  if (!qasm_parser.load_qasm(benchmark_filename, dag)) {
    std::cout << "Parser failed" << std::endl;
  }
  Graph graph(&src_ctx, dag);
  auto new_graph = graph.ccz_flip_t(&union_ctx);
  std::cout << "gate count after toffoli flip: " << new_graph->total_cost()
            << std::endl;
  new_graph->to_qasm(result_qasm_filename, false, false);
}
