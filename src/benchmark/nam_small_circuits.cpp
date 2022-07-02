#include "test/gen_ecc_set.h"
#include "quartz/parser/qasm_parser.h"
#include "quartz/tasograph/substitution.h"
#include <filesystem>

using namespace quartz;

const std::string ecc_set_prefix = "Nam_4_3_";

void benchmark_nam(const std::string &circuit_name) {
  std::string circuit_path = "circuit/nam-benchmarks/" + circuit_name + ".qasm";
  // Construct contexts
  Context src_ctx({GateType::h, GateType::ccz, GateType::x, GateType::cx,
                   GateType::input_qubit, GateType::input_param});
  Context dst_ctx({GateType::h, GateType::x, GateType::rz, GateType::add,
                   GateType::cx, GateType::input_qubit, GateType::input_param});
  auto union_ctx = union_contexts(&src_ctx, &dst_ctx);

  auto xfer_pair = GraphXfer::ccz_cx_rz_xfer(&union_ctx);
  // Load qasm file
  QASMParser qasm_parser(&src_ctx);
  DAG *dag = nullptr;
  if (!qasm_parser.load_qasm(circuit_path, dag)) {
    std::cout << "Parser failed" << std::endl;
  }
  //TODO
}

int main() {
  if (!std::filesystem::exists(ecc_set_prefix + "complete_ECC_set.json")) {
    std::cout << "Generating ECC set..." << std::endl;
    gen_ecc_set(
        {GateType::rz, GateType::h, GateType::cx, GateType::x,
         GateType::add}, ecc_set_prefix, true, 3, 2, 4);
    std::cout << "ECC set generated." << std::endl;
  }
  benchmark_nam("tof_3");  // 45 gates
  benchmark_nam("barenco_tof_3");  // 58 gates
  benchmark_nam("mod_mult_55");  // 119 gates
  benchmark_nam("vbe_adder_3");  // 150 gates
  benchmark_nam("gf2^4_mult");  // 225 gates
  return 0;
}
