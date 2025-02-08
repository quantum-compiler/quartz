#include "quartz/gate/gate_utils.h"
#include "quartz/verifier/verifier.h"
#include "test/test_optimization_steps.h"

#include <filesystem>
#include <iostream>

using namespace quartz;

const std::vector<std::string> kCircuitNames = {
    "tof_3",         "barenco_tof_3",  "mod5_4",      "tof_4",
    "tof_5",         "barenco_tof_4",  "mod_mult_55", "vbe_adder_3",
    "barenco_tof_5", "csla_mux_3",     "rc_adder_6",  "gf2^4_mult",
    "tof_10",        "mod_red_21",     "gf2^5_mult",  "csum_mux_9",
    "qcla_com_7",    "barenco_tof_10", "gf2^6_mult",  "qcla_adder_10",
    "gf2^7_mult",    "gf2^8_mult",     "qcla_mod_7",  "adder_8",
    "gf2^9_mult",    "gf2^10_mult"};
int main() {
  ParamInfo param_info;
  Context ctx({GateType::input_qubit, GateType::input_param, GateType::cx,
               GateType::h, GateType::rz, GateType::x, GateType::pi,
               GateType::mult, GateType::add},
              &param_info);
  EquivalenceSet eqs;
  // Load equivalent dags from file
  if (!eqs.load_json(
          &ctx,
          (kQuartzRootPath / "eccset/Nam_6_3_complete_ECC_set.json").string(),
          /*from_verifier=*/false)) {
    std::cout
        << "Failed to load equivalence file \""
        << (kQuartzRootPath / "eccset/Nam_6_3_complete_ECC_set.json").string()
        << "\"." << std::endl;
    assert(false);
  }
  if (!std::filesystem::exists(kQuartzRootPath / "benchmark-logs")) {
    std::filesystem::create_directory(kQuartzRootPath / "benchmark-logs");
  }
  Verifier verifier;
  auto xfers = GraphXfer::get_all_xfers_from_ecc(
      &ctx,
      (kQuartzRootPath / "eccset/Nam_6_3_complete_ECC_set.json").string());
  int verified_count = 0;
  int not_verified_count = 0;
  for (const auto &circuit : kCircuitNames) {
    freopen(((kQuartzRootPath / "benchmark-logs" / circuit).string() + ".log")
                .c_str(),
            "w", stdout);
    test_optimization(&ctx,
                      (kQuartzRootPath / "circuit/nam-benchmarks" /
                       (circuit + ".qasm.toffoli_flip"))
                          .string(),
                      xfers,
                      /*timeout=*/1,
                      (kQuartzRootPath / "benchmark-logs" / circuit).string() +
                          "_");
    bool verified = verifier.verify_transformation_steps(
        &ctx, (kQuartzRootPath / "benchmark-logs" / circuit).string() + "_",
        /*verbose=*/false);
    if (verified) {
      std::cout << "All transformations are verified." << std::endl;
      verified_count++;
    } else {
      std::cout << "Some transformation is not verified." << std::endl;
      not_verified_count++;
      std::cerr << circuit << " not verified." << std::endl;
    }
  }
  std::cerr << verified_count << " cases verified, " << not_verified_count
            << " cases not verified." << std::endl;
  return 0;
}
