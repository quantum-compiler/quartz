#include "quartz/tasograph/tasograph.h"

using namespace quartz;

int main() {
  ParamInfo param_info;
  Context ctx({GateType::input_qubit, GateType::input_param, GateType::add,
               GateType::cx, GateType::rz, GateType::x, GateType::h},
              &param_info);
  //   Context ctx({GateType::input_qubit, GateType::input_param, GateType::add,
  //                GateType::cz, GateType::rz, GateType::x, GateType::rx1,
  //                GateType::rx3});
  //   Context ctx({GateType::input_qubit, GateType::input_param, GateType::add,
  //                GateType::rx1, GateType::x, GateType::rx3, GateType::rz,
  //                GateType::ry1, GateType::y, GateType::ry3, GateType::rxx1,
  //                GateType::rxx3});
  auto graph =
      Graph::from_qasm_file(&ctx, "../experiment/nam_rm_circs/gf2^6_mult.qasm");
  graph->to_qasm("test.qasm", false, false);
}
