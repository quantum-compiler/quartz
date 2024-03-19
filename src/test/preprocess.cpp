
#include "quartz/tasograph/substitution.h"
#include "quartz/tasograph/tasograph.h"

using namespace quartz;

void parse_args(char **argv, int argc, std::string &input_filename) {
  assert(argv[1] != nullptr);
  input_filename = std::string(argv[1]);
}

int main(int argc, char **argv) {
  std::string input_fn;

  parse_args(argv, argc, input_fn);
  std::cout << "input_fn: " << input_fn << std::endl;
  auto fn = input_fn.substr(input_fn.rfind('/') + 1);
  auto param_info = ParamInfo(0);
  Context src_ctx({GateType::h, GateType::ccz, GateType::x, GateType::cx,
                   GateType::add, GateType::input_qubit, GateType::input_param},
                  3, &param_info);
  Context dst_ctx({GateType::h, GateType::x, GateType::rz, GateType::add,
                   GateType::cx, GateType::input_qubit, GateType::input_param},
                  3, &param_info);
  Context dst2_ctx({GateType::h, GateType::x, GateType::t, GateType::tdg,
                    GateType::s, GateType::sdg, GateType::z, GateType::add,
                    GateType::cx, GateType::input_qubit, GateType::input_param},
                   3, &param_info);

  auto union_ctx = union_contexts(&src_ctx, &dst_ctx);
  auto union2_ctx = union_contexts(&dst_ctx, &dst2_ctx);

  auto xfer_pair = GraphXfer::ccz_cx_rz_xfer(&src_ctx, &dst_ctx, &union_ctx);
  // Load qasm file
  QASMParser qasm_parser(&src_ctx);
  CircuitSeq *dag = nullptr;
  if (!qasm_parser.load_qasm(input_fn, dag)) {
    std::cout << "Parser failed" << std::endl;
  }
  Graph graph(&src_ctx, dag);

  std::shared_ptr<Graph> graph_before_search;

  // Greedy toffoli flip
  graph_before_search = graph.toffoli_flip_greedy(GateType::rz, xfer_pair.first,
                                                  xfer_pair.second);
  // graph_before_search->to_qasm(input_fn + ".toffoli_flip", false, false);
  std::cout << "toffoli flip + rotation merging done" << std::endl;

  // O(n) rz -> t
  graph_before_search->context = &union2_ctx;
  auto graph_seq = graph_before_search->to_circuit_sequence();
  auto rz_to_t_seq = graph_seq->get_rz_to_t(&union2_ctx);
  auto newGraph = std::make_shared<Graph>(&dst2_ctx, rz_to_t_seq.get());
  // O(n^2) rz -> t

  //   std::cout << "rz -> t done" << std::endl;

  //   std::cout << "Optimization results of Quartz for " << fn
  //             << " on Clifford+T gate set."
  //             << " Gate count after optimization: " << newGraph->gate_count()
  //             << ", "
  //             << "T Count: "
  //             << newGraph->specific_gate_count(GateType::t) +
  //                    newGraph->specific_gate_count(GateType::tdg)
  //             << ", "
  //             << "Circuit depth: " << newGraph->circuit_depth();

  newGraph->to_qasm(input_fn + ".optimized", false, false);
  return 0;
}
