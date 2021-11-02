#include "test_context_shift.h"

int main() {
  Context src_ctx({GateType::input_qubit, GateType::input_param, GateType::t,
                   GateType::tdg});
  Context dst_ctx({
      GateType::input_qubit,
      GateType::input_param,
      GateType::u1,
  });
  RuleParser rule_parser({
      "t q0 = u1 q0 0.25pi",    // t -> u1
      "h q0 = u2 q0 0 pi",      // h-> u2
      "tdg q0 = u1 q0 -0.25pi", // tdg -> u1
      "ccz q0 q1 q2 = cx q1 q2; tdg q2; cx q0 q2; t q2; cx q1 q2; tdg q2; cx "
      "q0 q2; cx q0 q1; tdg q1; cx q0 q1; t q0; t q1; t q2;" // ccx -> t, tdg,
                                                             // cx
  });
  test_context_shift("circuit/example-circuits/t_tdg_t.qasm", &src_ctx,
                     &dst_ctx, &rule_parser);
}