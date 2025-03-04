#include "test_optimization_steps.h"

#include "quartz/gate/gate_utils.h"
#include "quartz/verifier/verifier.h"

#include <filesystem>
#include <iostream>

using namespace quartz;

void test_preprocessed() {
  ParamInfo param_info;
  Context ctx({GateType::input_qubit, GateType::input_param, GateType::cx,
               GateType::h, GateType::t, GateType::tdg, GateType::x,
               GateType::add},
              &param_info);
  if (!std::filesystem::exists(kQuartzRootPath / "logs")) {
    std::filesystem::create_directory(kQuartzRootPath / "logs");
  }
  auto xfers = GraphXfer::get_all_xfers_from_ecc(
      &ctx, (kQuartzRootPath / "eccset/Clifford_T_5_3_complete_ECC_set.json")
                .string());
  test_optimization(
      &ctx, nullptr,
      (kQuartzRootPath / "circuit/example-circuits/barenco_tof_3.qasm")
          .string(),
      xfers,
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
}

void test_ccz() {
  ParamInfo param_info;
  Context src_ctx({GateType::h, GateType::ccz, GateType::x, GateType::cx,
                   GateType::add, GateType::input_qubit, GateType::input_param},
                  &param_info);
  Context ctx({GateType::input_qubit, GateType::input_param, GateType::cx,
               GateType::h, GateType::rz, GateType::x, GateType::add},
              &param_info);
  Context union_ctx = union_contexts(&src_ctx, &ctx);
  if (!std::filesystem::exists(kQuartzRootPath / "logs")) {
    std::filesystem::create_directory(kQuartzRootPath / "logs");
  }
  auto xfers = GraphXfer::get_all_xfers_from_ecc(
      &ctx,
      (kQuartzRootPath / "eccset/Nam_5_3_complete_ECC_set.json").string());
  test_optimization(
      &src_ctx, &ctx,
      (kQuartzRootPath / "circuit/example-circuits/barenco_tof_3_ccz.qasm")
          .string(),
      xfers,
      /*timeout=*/12, (kQuartzRootPath / "logs/barenco_tof_3_ccz_").string());
  Verifier verifier;
  bool verified = verifier.verify_transformation_steps(
      &union_ctx, (kQuartzRootPath / "logs/barenco_tof_3_ccz_").string(),
      /*verbose=*/true);
  if (verified) {
    std::cout << "All transformations are verified." << std::endl;
  } else {
    std::cout << "Some transformation is not verified." << std::endl;
  }
}

int main() {
  test_ccz();
  return 0;
}
