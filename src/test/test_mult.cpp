#include "quartz/circuitseq/circuitseq.h"
#include "quartz/context/context.h"
#include "quartz/gate/gate.h"

#include <cassert>

using namespace quartz;

int main() {
  ParamInfo param_info;
  Context ctx({GateType::rx, GateType::mult}, 1, &param_info);

  auto p0 = ctx.get_new_param_id(2.0);
  auto p1 = ctx.get_new_param_id(3.0);
  auto p2 = ctx.get_new_param_id(6.0);
  auto p3 =
      ctx.get_new_param_expression_id({p0, p1}, ctx.get_gate(GateType::mult));

  CircuitSeq dag1(1);
  dag1.add_gate({0}, {p2}, ctx.get_gate(GateType::rx), &ctx);

  CircuitSeq dag2(1);
  dag2.add_gate({0}, {p3}, ctx.get_gate(GateType::rx), &ctx);

  auto c1 = dag1.to_qasm_style_string(&ctx);
  auto c2 = dag2.to_qasm_style_string(&ctx);
  if (c1 != c2) {
    std::cout << "Failed to evaluate mult gate." << std::endl;
    assert(false);
  }

  // Working directory is cmake-build-debug/ here.
  system("python ../src/test/test_mult.py");
}
