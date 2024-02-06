#include "test/oracle.h"

#include "quartz/tasograph/substitution.h"
#include "quartz/tasograph/tasograph.h"
#include "test/gen_ecc_set.h"

using namespace quartz;

std::string optimize_(std::string circ_string, std::string cost_func,
                      std::string ecc_path, int timeout) {
  Context ctx({GateType::input_qubit, GateType::input_param, GateType::cx,
               GateType::h, GateType::rz, GateType::x, GateType::add},
              /*num_qubits=*/3, /*num_input_symbolic_params=*/2);

  auto graph = Graph::from_qasm_str(&ctx, circ_string);
  assert(graph);
  EquivalenceSet eqs;
  eqs.load_json(&ctx, ecc_path, false);

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
  std::function<float(Graph *)> cost_function;
  float init_cost;
  if (cost_func == "depth") {
    std::function<float(Graph *)> cost_function = [](Graph *graph) {
      return graph->cost_depth();
    };
    init_cost = graph->cost_depth();
  } else {
    std::function<float(Graph *)> cost_function = [](Graph *graph) {
      return graph->cost_gate_count();
    };
    init_cost = graph->cost_gate_count();
  }
  auto new_graph = graph->optimize(xfers, init_cost * 1.05, "barenco_tof_3", "",
                                   false, cost_function, timeout);
  return new_graph->to_qasm(false, false);
}
