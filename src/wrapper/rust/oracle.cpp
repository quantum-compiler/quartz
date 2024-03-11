#include "oracle.h"

using namespace quartz;
std::shared_ptr<SuperContext> get_context_(const std::string gate_set,
                                           int n_qubits,
                                           const std::string ecc_path) {
  auto super_context =
      std::make_shared<SuperContext>(gate_set, n_qubits, ecc_path);
  return super_context;
}

std::string optimize_(std::string circ_string, std::string cost_func,
                      float timeout, std::shared_ptr<SuperContext> super_context

) {
  auto graph = Graph::from_qasm_str(&super_context->ctx, circ_string);

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
  auto newgraph =
      graph->optimize(super_context->xfers, init_cost * 1.05, "barenco_tof_3",
                      "", false, cost_function, timeout);
  return newgraph->to_qasm(false, false);
}