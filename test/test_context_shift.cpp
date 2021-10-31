#include "test_context_shift.h"

int main() {
  Context src_ctx({GateType::input_qubit, GateType::input_param, GateType::t,
                   GateType::tdg, GateType::h, GateType::cx});
  Context dst_ctx({GateType::input_qubit, GateType::input_param, GateType::u1,
                   GateType::u2, GateType::u3, GateType::cx});
  RuleParser rule_parser(
      {"t q0 = u1 q0 0.25pi", "h q0 = u2 q0 0 pi", "tdg q0 = u2 q0 -0.25pi"});
  test_context_shift("circuit/example-circuits/barenco_tof_3.qasm", &src_ctx,
                     &dst_ctx, &rule_parser);
}