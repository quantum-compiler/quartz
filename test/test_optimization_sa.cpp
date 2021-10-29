#include "test_optimization.h"
#include "../gate/gate_utils.h"

#include <iostream>

int main() {
  Context ctx({GateType::input_qubit, GateType::input_param, GateType::h,
               GateType::t, GateType::cx, GateType::tdg});
  //   test_optimization(&ctx, "circuit/example-circuits/voqc_fig5.qasm",
  //                     "cmake-build-debug/bfs_verified.json");
  test_optimization(&ctx,
                    "circuit/example-circuits/barenco_tof_3.qasm",
                    "cmake-build-debug/bfs_verified_simplified.json",
                    false/*use_simulated_annealing*/);
}
