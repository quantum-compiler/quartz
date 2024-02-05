#include "test/oracle.h"

#include "quartz/tasograph/substitution.h"
#include "quartz/tasograph/tasograph.h"
#include "test/gen_ecc_set.h"

using namespace quartz;

std::string optimize_(std::string s) {
  Context ctx({GateType::input_qubit, GateType::input_param, GateType::cx,
               GateType::h, GateType::rz, GateType::x, GateType::add},
              /*num_qubits=*/3, /*num_input_symbolic_params=*/2);

  auto graph = Graph::from_qasm_str(&ctx, s);
  assert(graph);
  EquivalenceSet eqs;
  eqs.load_json(&ctx, "Nam_3_3_complete_ECC_set.json",
                /*from_verifier=*/false);

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
  // std::cout << "number of xfers: " << xfers.size() << std::endl;

  auto new_graph = graph->optimize(xfers, graph->gate_count() * 1.05,
                                   "barenco_tof_3", "", false, nullptr, 1);
  return new_graph->to_qasm(false, false);
}
