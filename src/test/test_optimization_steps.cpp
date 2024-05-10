#include "test_optimization_steps.h"

#include "quartz/gate/gate_utils.h"

#include <filesystem>
#include <iostream>

using namespace quartz;

int main() {
  ParamInfo param_info;
  Context ctx({GateType::input_qubit, GateType::input_param, GateType::cx,
               GateType::h, GateType::t, GateType::tdg, GateType::x,
               GateType::add},
              &param_info);
  std::filesystem::path this_file_path(__FILE__);
  auto quartz_root_path =
      this_file_path.parent_path().parent_path().parent_path();
  if (!std::filesystem::exists(quartz_root_path / "logs")) {
    std::filesystem::create_directory(quartz_root_path / "logs");
  }
  test_optimization(
      &ctx, quartz_root_path / "circuit/example-circuits/barenco_tof_3.qasm",
      quartz_root_path / "eccset/Clifford_T_5_3_complete_ECC_set.json",
      /*timeout=*/10, quartz_root_path / "logs/barenco_tof_3_");
}
