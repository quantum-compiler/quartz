#include "quartz/tasograph/substitution.h"
#include "quartz/tasograph/tasograph.h"
#include "test/gen_ecc_set.h"

using namespace quartz;

int main(int argc, char **argv) {
  std::string input_fn =
      kQuartzRootPath.string() + "/circuit/nam_circs/barenco_tof_3.qasm";
  std::string circuit_name = "barenco_tof_3";
  std::string output_fn;
  std::string eqset_fn =
      kQuartzRootPath.string() + "/eccset/Nam_3_3_complete_ECC_set.json";

  if (argc >= 2) {
    assert(argv[1] != nullptr);
    input_fn = std::string(argv[1]);
    if (argc >= 3) {
      assert(argv[2] != nullptr);
      circuit_name = std::string(argv[2]);
    }
  }

  ParamInfo param_info;
  Context ctx({GateType::input_qubit, GateType::input_param, GateType::cx,
               GateType::h, GateType::rz, GateType::x, GateType::add},
              /*num_qubits=*/3, &param_info);

  EquivalenceSet eqs;
  // Load ECC set from file
  if (!eqs.load_json(&ctx, eqset_fn,
                     /*from_verifier=*/false)) {
    // generate ECC set
    gen_ecc_set(
        {GateType::rz, GateType::h, GateType::cx, GateType::x, GateType::add},
        kQuartzRootPath.string() + "/eccset/Nam_3_3_", true, false, 3, 2, 3);
    if (!eqs.load_json(&ctx, eqset_fn,
                       /*from_verifier=*/false)) {
      std::cout << "Failed to load equivalence file." << std::endl;
      assert(false);
    }
  }

  // Get xfer from the equivalent set
  std::vector<GraphXfer *> xfers = GraphXfer::get_all_xfers_from_eqs(&ctx, eqs);
  std::cout << "number of xfers: " << xfers.size() << std::endl;

  auto graph = Graph::from_qasm_file(&ctx, input_fn);
  assert(graph);

  auto graph_optimized = graph->optimize(xfers, graph->gate_count() * 1.05,
                                         circuit_name, "", true, nullptr, 10);
  std::cout << "Optimized graph:" << std::endl;
  std::cout << graph_optimized->to_qasm();
  return 0;
}
