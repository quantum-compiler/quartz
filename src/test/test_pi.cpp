#include "quartz/circuitseq/circuitseq.h"
#include "quartz/context/context.h"
#include "quartz/gate/gate.h"

#include <cassert>

using namespace quartz;

int main() {
  ParamInfo param_info(0);
  Context ctx({GateType::rx, GateType::pi}, 1, &param_info);

  auto p0 = ctx.get_new_param_id(PI / 2);
  auto p1 = ctx.get_new_param_id(2.0);
  auto p2 = ctx.get_new_param_expression_id({p1}, ctx.get_gate(GateType::pi));

  CircuitSeq dag1(1);
  dag1.add_gate({0}, {p0}, ctx.get_gate(GateType::rx), &ctx);

  CircuitSeq dag2(1);
  dag2.add_gate({0}, {p2}, ctx.get_gate(GateType::rx), &ctx);

  auto c1 = dag1.to_qasm_style_string(&ctx);
  auto c2 = dag2.to_qasm_style_string(&ctx);
  if (c1 != c2) {
    std::cout << "Failed to evaluate pi gate." << std::endl;
    assert(false);
  }

  // Working directory is cmake-build-debug/ here.
  system("python ../src/test/test_pi.py");
}
