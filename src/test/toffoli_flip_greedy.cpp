#include "quartz/parser/qasm_parser.h"
#include "quartz/tasograph/substitution.h"
#include "quartz/tasograph/tasograph.h"

using namespace quartz;

int main() {
  const std::string input_fn = "../circuit/nam-benchmarks/barenco_tof_4.qasm";

  Context src_ctx({GateType::h, GateType::ccz, GateType::x, GateType::cx,
                   GateType::input_qubit, GateType::input_param});
  Context dst_ctx({GateType::h, GateType::rz, GateType::x, GateType::cx,
                   GateType::add, GateType::input_qubit,
                   GateType::input_param});
  auto union_ctx = union_contexts(&src_ctx, &dst_ctx);

  QASMParser qasm_parser(&src_ctx);
  DAG *dag = nullptr;
  if (!qasm_parser.load_qasm(input_fn, dag)) {
    std::cout << "Parser failed" << std::endl;
  }
  Graph graph(&src_ctx, dag);
}
