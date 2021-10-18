#include "test_optimization.h"
#include "../gate/gate_utils.h"

#include <iostream>

int main() {
  Context ctx(all_supported_gates());
  test_optimization(&ctx, "circuit/example-circuits/barenco_tof_3_basic.qasm",
                    "cmake-build-debug/equivalences_sorted.json");
}