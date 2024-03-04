#include "quartz/tasograph/substitution.h"
#include "quartz/tasograph/tasograph.h"
#include "test/gen_ecc_set.h"

using namespace quartz;

int main() {
  ParamInfo param_info(/*num_input_symbolic_params=*/2);
  Context ctx({GateType::input_qubit, GateType::input_param, GateType::cx,
               GateType::h, GateType::rz, GateType::x, GateType::add},
              /*num_qubits=*/3, &param_info);

  EquivalenceSet eqs;
  // Load ECC set from file
  if (!eqs.load_json(&ctx, "Nam_3_3_complete_ECC_set.json",
                     /*from_verifier=*/false)) {
    // generate ECC set
    gen_ecc_set(
        {GateType::rz, GateType::h, GateType::cx, GateType::x, GateType::add},
        "Nam_3_3_", true, false, 3, 2, 3);
    if (!eqs.load_json(&ctx, "Nam_3_3_complete_ECC_set.json",
                       /*from_verifier=*/false)) {
      std::cout << "Failed to load equivalence file." << std::endl;
      assert(false);
    }
  }

  auto graph = Graph::from_qasm_file(
      &ctx, "experiment/circs/nam_circs/barenco_tof_3.qasm");
  assert(graph);

  // Get xfer from the equivalent set
  auto ecc = eqs.get_all_equivalence_sets();
  std::vector<GraphXfer *> xfers;
  for (const auto &eqcs : ecc) {
    for (auto circ_0 : eqcs) {
      for (auto circ_1 : eqcs) {
        if (circ_0 != circ_1) {
          auto xfer = GraphXfer::create_GraphXfer(&ctx, circ_0, circ_1, true);
          if (xfer != nullptr) {
            xfers.push_back(xfer);
          }
        }
      }
    }
  }
  std::cout << "number of xfers: " << xfers.size() << std::endl;

  graph->optimize(xfers, graph->gate_count() * 1.05, "barenco_tof_3", "", true);
  return 0;
}
