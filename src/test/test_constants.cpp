#include "quartz/context/context.h"
#include "quartz/gate/gate.h"
#include "quartz/parser/qasm_parser.h"

#include <cassert>

using namespace quartz;

int main() {
  ParamInfo param_info(0);
  Context ctx({GateType::rx}, 2, &param_info);

  QASMParser parser(&ctx);

  std::string str = "OPENQASM 2.0;\n"
                    "include \"qelib1.inc\";\n"
                    "qreg q[2];\n"
                    "rx(pi) q[0];\n"
                    "rx(3*pi) q[0];\n"
                    "rx(pi/3) q[1];\n"
                    "rx(2*pi/3) q[1];\n";

  CircuitSeq *seq2 = nullptr;
  parser.load_qasm_str(str, seq2);

  auto params = ctx.compute_parameters({});
  for (int i = 0; i < params.size(); ++i) {
    if (params[i] == 0) {
      std::cout << "Failed to initialize parameter " << i << "." << std::endl;
      assert(false);
    }
  }
}
