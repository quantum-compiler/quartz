#include "quartz/context/context.h"
#include "quartz/gate/gate.h"
#include "quartz/parser/qasm_parser.h"

#include <cassert>

using namespace quartz;

bool has_exprs(Context &ctx, CircuitSeq *seq) {
  for (auto id : seq->get_directly_used_param_indices()) {
    if (ctx.param_is_expression(id)) {
      return true;
    }
  }
  return false;
}

int main() {
  ParamInfo param_info(0);
  Context ctx({GateType::rx, GateType::ry, GateType::rz, GateType::cx, GateType::mult,
               GateType::pi}, 2, &param_info);

  QASMParser parser(&ctx);

  std::string str = "OPENQASM 2.0;\n"
                    "include \"qelib1.inc\";\n"
                    "qreg q[2];\n"
                    "rx(pi) q[0];\n"
                    "cx q[0], q[1];\n"
                    "ry(3*pi) q[0];\n"
                    "rz(pi/3) q[1];\n"
                    "cx q[1], q[0];\n"
                    "rx(2*pi/3) q[1];\n";

  CircuitSeq *seq1 = nullptr;
  parser.use_symbolic_pi(true);
  parser.load_qasm_str(str, seq1);

  CircuitSeq *seq2 = nullptr;
  parser.use_symbolic_pi(false);
  parser.load_qasm_str(str, seq2);

  std::string out1 = seq1->to_qasm_style_string(&ctx);
  std::string out2 = seq2->to_qasm_style_string(&ctx);

  std::cout << out1 << std::endl;
  assert(out1 == out2);

  if (!has_exprs(ctx, seq1)) {
    std::cout << "Parser did not use symbolic pi values." << std::endl;
    assert(false);
  }

  if (has_exprs(ctx, seq2)) {
    std::cout << "Parser failed to disable symbolic pi values." << std::endl;
    assert(false);
  }
}
