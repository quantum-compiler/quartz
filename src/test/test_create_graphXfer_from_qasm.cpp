#include "quartz/context/context.h"
#include "quartz/tasograph/substitution.h"
#include "quartz/tasograph/tasograph.h"

#include <string>

using namespace quartz;

int main() {
  ParamInfo param_info;
  Context ctx(
      {GateType::input_qubit, GateType::input_param, GateType::rz, GateType::z},
      &param_info);
  std::string src_str = "OPENQASM 2.0;\n"
                        "include \"qelib1.inc\";\n"
                        "qreg q[1];\n"
                        "rz(pi) q[0];";
  std::string dst_str = "OPENQASM 2.0;\n"
                        "include \"qelib1.inc\";\n"
                        "qreg q[1];\n"
                        "z q[0];\n";
  auto graph_xfer =
      GraphXfer::create_GraphXfer_from_qasm_str(&ctx, src_str, dst_str);
  std::cout << graph_xfer->src_str() << graph_xfer->dst_str() << std::endl;

  auto circuit = Graph::from_qasm_str(&ctx, src_str);

  std::vector<Op> all_ops;
  circuit->topology_order_ops(all_ops);
  auto op = all_ops[0];
  auto new_circuit = circuit->apply_xfer(graph_xfer, op);
  std::cout << new_circuit->to_qasm() << std::endl;
  return 0;
}
