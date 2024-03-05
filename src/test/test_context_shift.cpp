#include "test_context_shift.h"

using namespace quartz;

void test_t_to_u1() {
  ParamInfo param_info;
  Context src_ctx({GateType::input_qubit, GateType::input_param, GateType::t,
                   GateType::tdg},
                  &param_info);
  Context dst_ctx({GateType::input_qubit, GateType::input_param, GateType::u1,
                   GateType::u2},
                  &param_info);
  auto union_ctx = union_contexts(&src_ctx, &dst_ctx);
  RuleParser rule_parser({
      "t q0 = u1 q0 0.25pi",     // t -> u1
      "h q0 = u2 q0 0 pi",       // h-> u2
      "tdg q0 = u1 q0 -0.25pi",  // tdg -> u1
  });
  test_context_shift("circuit/example-circuits/t_tdg_t.qasm", &src_ctx,
                     &dst_ctx, &union_ctx, &rule_parser);
}

void test_rz_to_t() {
  ParamInfo param_info;
  Context src_ctx({GateType::input_qubit, GateType::input_param, GateType::rz},
                  &param_info);
  Context dst_ctx({GateType::input_qubit, GateType::input_param, GateType::t,
                   GateType::tdg, GateType::s, GateType::sdg, GateType::z},
                  &param_info);
  auto union_ctx = union_contexts(&src_ctx, &dst_ctx);
  RuleParser rule_parser({"rz q0 0pi =", "rz q0 0.25pi = t q0",
                          "rz q0 -0.25pi = tdg q0", "rz q0 0.5pi = s q0",
                          "rz q0 -0.5pi = sdg q0", "rz q0 0.75pi = t q0; s q0",
                          "rz q0 -0.75pi = tdg q0; sdg q0", "rz q0 pi = z q0",
                          "rz q0 -pi = z q0"});
  test_context_shift("circuit/example-circuits/rz_multiples.qasm", &src_ctx,
                     &dst_ctx, &union_ctx, &rule_parser);
}

int main() {
  test_t_to_u1();
  test_rz_to_t();
  return 0;
}
