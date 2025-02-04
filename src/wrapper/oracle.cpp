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
                      std::string timeout_type, float timeout_value,
                      std::shared_ptr<SuperContext> super_context

) {
  auto graph = Graph::from_qasm_str(&super_context->ctx, circ_string);

  std::function<int(Graph *)> cost_function;
  std::vector<quartz::GraphXfer *> greedy_xfers;
  if (cost_func == "Gate") {
    cost_function = [](Graph *graph) { return graph->total_cost(); };
    greedy_xfers = super_context->xfers_greedy_gate;
  } else if (cost_func == "Depth") {
    cost_function = [](Graph *graph) { return graph->circuit_depth(); };
    greedy_xfers = super_context->xfers_greedy_gate;

  } else if (cost_func == "Mixed") {
    cost_function = [](Graph *graph) {
      return 10 * graph->circuit_depth() + graph->total_cost();
    };
    greedy_xfers = super_context->xfers_greedy_gate;

  } else {
    std::cout << "Invalid cost function." << std::endl;
    assert(false);
  }
  float timeout = 0;
  if (timeout_type == "PerSegment") {
    timeout = timeout_value;
  } else if (timeout_type == "PerGate") {
    timeout = timeout_value * graph->total_cost();
  } else {
    std::cout << "Invalid timeout type." << std::endl;
    assert(false);
  }

  float init_cost = cost_function(graph.get());
  auto start = std::chrono::steady_clock::now();
  auto graph_after_greedy =
      graph->greedy_optimize_with_xfer(greedy_xfers, false, cost_function);

  auto end = std::chrono::steady_clock::now();
  double remaining_time =
      timeout -
      std::chrono::duration_cast<std::chrono::milliseconds>(end - start)
              .count() /
          1000.0;
  if (remaining_time < 0.01) {
    return graph_after_greedy->to_qasm(false, false);
  } else {
    auto graph_after_search = graph_after_greedy->optimize(
        super_context->xfers, init_cost * 1.05, "barenco_tof_3", "", false,
        cost_function, remaining_time);
    return graph_after_search->to_qasm(false, false);
  }
}
std::string clifford_decomposition_(std::string circ) {
  auto param_info = ParamInfo(0);

  Context src_ctx(std::vector<GateType>{GateType::h, GateType::ccz, GateType::x,
                                        GateType::cx, GateType::add,
                                        GateType::s, GateType::sdg, GateType::t,
                                        GateType::tdg, GateType::input_qubit,
                                        GateType::input_param, GateType::rz},
                  3, &param_info);
  Context dst_ctx({GateType::h, GateType::x, GateType::rz, GateType::add,
                   GateType::cx, GateType::input_qubit, GateType::input_param},
                  3, &param_info);

  auto union_ctx = union_contexts(&src_ctx, &dst_ctx);

  auto graph = Graph::from_qasm_str(&src_ctx, circ);

  std::shared_ptr<Graph> newGraph;

  auto xfer_pair = GraphXfer::ccz_cx_rz_xfer(&src_ctx, &dst_ctx, &union_ctx);

  newGraph = graph->toffoli_flip_greedy(GateType::rz, xfer_pair.first,
                                        xfer_pair.second);
  return newGraph->to_qasm(false, false);
}

std::string rotation_merging_(std::string circ_string)

{
  auto param_info = ParamInfo(0);
  Context src_ctx(std::vector<GateType>{GateType::h, GateType::x, GateType::rz,
                                        GateType::add, GateType::cx,
                                        GateType::input_qubit,
                                        GateType::input_param},
                  3, &param_info);

  auto graph = Graph::from_qasm_str(&src_ctx, circ_string);
  graph->rotation_merging(GateType::rz);
  return graph->to_qasm(false, false);
}