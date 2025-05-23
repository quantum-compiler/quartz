#include "quartz/circuitseq/circuitseq.h"
#include "quartz/context/context.h"
#include "quartz/gate/gate.h"

#include <cassert>

using namespace quartz;

int main() {
  ParamInfo param_info;
  Context ctx({GateType::rx, GateType::pi}, 1, &param_info);

  auto p0 = ctx.get_new_param_id(PI / (ParamType)2);
  auto p1 = ctx.get_new_arithmetic_param_id((ParamType)2);
  auto p2 = ctx.get_new_param_expression_id({p1}, ctx.get_gate(GateType::pi));

  CircuitSeq dag1(1);
  dag1.add_gate({0}, {p0}, ctx.get_gate(GateType::rx), &ctx);

  CircuitSeq dag2(1);
  dag2.add_gate({0}, {p2}, ctx.get_gate(GateType::rx), &ctx);

  auto mat1 = dag1.get_matrix(&ctx);
  auto mat2 = dag2.get_matrix(&ctx);

  assert(mat1.size() == mat2.size());
  for (int i = 0; i < mat1.size(); ++i) {
    assert(mat1[i].size() == mat2[i].size());
    for (int j = 0; j < mat1[i].size(); ++j) {
      if (mat1[i][j] != mat2[i][j]) {
        std::cout << "Disagree at " << i << ", " << j << "." << std::endl;
        assert(false);
      }
    }
  }
  std::cout << dag1.to_string(/*line_number=*/true, &ctx) << std::endl;
  std::cout << dag1.to_qasm_style_string(&ctx) << std::endl;
  std::cout << dag2.to_string(/*line_number=*/true, &ctx) << std::endl;
  std::cout << dag2.to_qasm_style_string(&ctx) << std::endl;

  // Working directory is cmake-build-debug/ here.
  system("python ../src/test/test_pi.py");
}
