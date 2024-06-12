#include "test_optimization_steps.h"

#include "quartz/gate/gate_utils.h"
#include "quartz/verifier/verifier.h"

#include <filesystem>
#include <iostream>

using namespace quartz;

int main() {
  ParamInfo param_info;
  Context ctx({GateType::input_qubit, GateType::input_param, GateType::cx,
               GateType::h, GateType::t, GateType::tdg, GateType::x,
               GateType::add},
              &param_info);
  if (!std::filesystem::exists(kQuartzRootPath / "logs")) {
    std::filesystem::create_directory(kQuartzRootPath / "logs");
  }
  test_optimization(
      &ctx,
      (kQuartzRootPath / "circuit/example-circuits/barenco_tof_3.qasm")
          .string(),
      (kQuartzRootPath / "eccset/Clifford_T_5_3_complete_ECC_set.json")
          .string(),
      /*timeout=*/12, (kQuartzRootPath / "logs/barenco_tof_3_").string());
  Verifier verifier;
  bool verified = verifier.verify_transformation_steps(
      &ctx, (kQuartzRootPath / "logs/barenco_tof_3_").string(),
      /*verbose=*/true);
  if (verified) {
    std::cout << "All transformations are verified." << std::endl;
  } else {
    std::cout << "Some transformation is not verified." << std::endl;
  }
  return 0;
}
