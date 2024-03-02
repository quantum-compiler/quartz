#include "quartz/gate/gate_utils.h"
#include "test_optimization.h"

using namespace quartz;

#include <iostream>

int main() {
  ParamInfo param_info;
  Context ctx({GateType::input_qubit, GateType::input_param, GateType::h,
               GateType::t, GateType::cx, GateType::tdg},
              &param_info);
  //   test_optimization(&ctx, "circuit/example-circuits/voqc_fig5.qasm",
  //                     "cmake-build-debug/bfs_verified.json");
#ifdef __linux
  test_optimization(&ctx, "circuit/example-circuits/barenco_tof_3.qasm",
                    "cmake-build-debug/bfs_verified_simplified.json",
                    true /*use_simulated_annealing*/);
#else
  test_optimization(&ctx, "circuit/example-circuits/barenco_tof_3.qasm",
                    "cmake-build-debug/bfs_verified_simplified.json",
                    true /*use_simulated_annealing*/);
#endif
}
