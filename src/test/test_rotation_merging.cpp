#include "quartz/context/context.h"
#include "quartz/gate/gate_utils.h"
#include "quartz/parser/qasm_parser.h"
#include "quartz/tasograph/tasograph.h"

#include <iostream>

using namespace quartz;

int main() {
  std::string benchmark_filename =
      "circuit/voqc-benchmarks/barenco_tof_10.qasm";
  std::string result_qasm_filename =
      "circuit/voqc-benchmarks/barenco_tof_10_rotation_merging.qasm";
  ParamInfo param_info;
  Context src_ctx({GateType::input_param, GateType::input_qubit, GateType::ccz,
                   GateType::h},
                  &param_info);
  Context dst_ctx({GateType::input_param, GateType::input_qubit, GateType::h,
                   GateType::rz, GateType::cx},
                  &param_info);
  //   Context union_ctx({GateType::input_param, GateType::input_qubit,
  //   GateType::t,
  //                      GateType::tdg, GateType::cx, GateType::rz});
  Context union_ctx = union_contexts(&src_ctx, &dst_ctx);
  QASMParser qasm_parser(&union_ctx);
  CircuitSeq *dag = nullptr;
  if (!qasm_parser.load_qasm(benchmark_filename, dag)) {
    std::cout << "Parser failed" << std::endl;
  }

  Graph graph(&union_ctx, dag);
  RuleParser rule_parser(
      {"t q0 = rz q0 0.25pi",  // t -> rz
       "tdg q0 = rz q0 -0.25pi",
       "ccz q0 q1 q2 = cx q1 q2; rz q2 -0.25pi; cx q0 q2; rz "
       "q2 0.25pi; cx q1 q2; rz q2 -0.25pi; cx "
       "q0 q2; cx q0 q1; rz q1 -0.25pi; cx q0 q1; rz q0 "
       "0.25pi; rz q1 0.25pi; rz q2 0.25pi;"});
  auto newGraph =
      graph.context_shift(&src_ctx, &dst_ctx, &union_ctx, &rule_parser);
  newGraph->print_qubit_ops();
  //   newGraph->to_qasm("temp.qasm", /*print_result=*/true);
  newGraph->rotation_merging(GateType::rz);
  std::cout << newGraph->total_cost()
            << " gates in circuit after rotation merging." << std::endl;
  newGraph->to_qasm(result_qasm_filename, /*print_result=*/false, false);
}
