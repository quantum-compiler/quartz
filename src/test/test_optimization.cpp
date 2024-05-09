#include "test_optimization.h"

#include "quartz/gate/gate_utils.h"

#include <iostream>

using namespace quartz;

int main() {
  ParamInfo param_info;
  Context ctx({GateType::input_qubit, GateType::input_param, GateType::cx,
               GateType::h, GateType::t, GateType::tdg, GateType::x,
               GateType::add},
              &param_info);
  test_optimization(&ctx, "circuit/example-circuits/barenco_tof_3.qasm",
                    "eccset/Clifford_T_5_3_complete_ECC_set.json",
                    /*timeout=*/10, "logs/barenco_tof_3_");
}
