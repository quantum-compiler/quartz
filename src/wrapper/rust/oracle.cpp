#include "oracle.h"

#include "quartz/tasograph/substitution.h"
#include "quartz/tasograph/tasograph.h"

using namespace quartz;

std::string optimize_(std::string circ_string, int n_qubits,
                      std::string cost_func, std::string ecc_path,
                      std::string gate_set, float timeout) {
  ParamInfo param_info(0);
  auto gate_set_vec = std::vector<GateType>();
  if (gate_set == "Nam") {
    gate_set_vec = {GateType::input_qubit, GateType::input_param, GateType::cx,
                    GateType::h,           GateType::rz,          GateType::x,
                    GateType::add};
  } else if (gate_set == "CliffordT") {
    gate_set_vec = {
        GateType::input_qubit, GateType::input_param, GateType::cx, GateType::h,
        GateType::rz,          GateType::x,           GateType::t};
  } else {
    std::cout << "Invalid gate set." << std::endl;
    assert(false);
  }
  auto ctx = Context(gate_set_vec, n_qubits, &param_info);

  auto graph = Graph::from_qasm_str(&ctx, circ_string);

  EquivalenceSet eqs;
  // Load equivalent dags from file
  if (!eqs.load_json(&ctx, ecc_path, false)) {
    std::cout << "Failed to load equivalence file." << std::endl;
    assert(false);
  }

  // Get xfer from the equivalent set
  auto ecc = eqs.get_all_equivalence_sets();
  std::vector<GraphXfer *> xfers;
  for (auto eqcs : ecc) {
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
  std::function<int(Graph *)> cost_function;
  if (cost_func == "Gate") {
    cost_function = [](Graph *graph) { return graph->total_cost(); };
  } else if (cost_func == "Depth") {
    cost_function = [](Graph *graph) { return graph->circuit_depth(); };
  } else if (cost_func == "Mixed") {
    cost_function = [](Graph *graph) {
      return graph->circuit_depth() + 0.1 * graph->total_cost();
    };
  } else {
    std::cout << "Invalid cost function." << std::endl;
    assert(false);
  }
  float init_cost = cost_function(graph.get());
  auto newgraph = graph->optimize(xfers, init_cost * 1.05, "barenco_tof_3", "",
                                  false, cost_function, timeout);
  return newgraph->to_qasm(false, false);
}