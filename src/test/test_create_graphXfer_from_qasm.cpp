#include "quartz/context/context.h"
#include "quartz/tasograph/substitution.h"
#include "quartz/tasograph/tasograph.h"

#include <string>

using namespace quartz;

int main() {
  Context ctx({GateType::input_qubit, GateType::input_param, GateType::rz,
               GateType::z});
  std::string src_str = "OPENQASM 2.0;\n"
                        "include \"qelib1.inc\";\n"
                        "qreg q[1];\n"
                        "rz(pi/2) q[0];";
  std::string dst_str = "OPENQASM 2.0;\n"
                        "include \"qelib1.inc\";\n"
                        "qreg q[1];\n"
                        "z q[0];\n";
  auto graph_xfer =
      GraphXfer::create_GraphXfer_from_qasm_str(&ctx, src_str, dst_str);
  std::cout << graph_xfer->src_str() << graph_xfer->dst_str() << std::endl;

  return 0;
}