
#include "quartz/tasograph/substitution.h"
#include "quartz/tasograph/tasograph.h"

using namespace quartz;

void parse_args(char **argv, int argc, bool &rm_only, std::string &target_gates,
                std::string &input_filename) {
  assert(argv[1] != nullptr);
  assert(argv[2] != nullptr);
  assert(argv[3] != nullptr);
  rm_only = std::string(argv[1]) == "rmonly";
  target_gates = std::string(argv[2]);
  input_filename = std::string(argv[3]);
}

int main(int argc, char **argv) {
  std::string input_fn;
  bool rm_only;
  std::string target_gates;
  parse_args(argv, argc, rm_only, target_gates, input_fn);

  std::cout << "input_fn: " << input_fn << std::endl;
  auto fn = input_fn.substr(input_fn.rfind('/') + 1);
  auto param_info = ParamInfo(0);
  auto gates = std::vector<GateType>{};
  if (!rm_only) {
    gates = std::vector<GateType>{GateType::h,          GateType::ccz,
                                  GateType::x,          GateType::cx,
                                  GateType::add,        GateType::input_qubit,
                                  GateType::input_param};
  } else {
    gates = std::vector<GateType>{GateType::h,          GateType::x,
                                  GateType::rz,         GateType::add,
                                  GateType::cx,         GateType::input_qubit,
                                  GateType::input_param};
  }

  Context src_ctx(gates, 3, &param_info);
  Context dst_ctx({GateType::h, GateType::x, GateType::rz, GateType::add,
                   GateType::cx, GateType::input_qubit, GateType::input_param},
                  3, &param_info);
  Context dst2_ctx({GateType::h, GateType::x, GateType::t, GateType::tdg,
                    GateType::s, GateType::sdg, GateType::z, GateType::add,
                    GateType::cx, GateType::input_qubit, GateType::input_param},
                   3, &param_info);

  auto union_ctx = union_contexts(&src_ctx, &dst_ctx);
  auto union2_ctx = union_contexts(&dst_ctx, &dst2_ctx);

  // Load qasm file
  QASMParser qasm_parser(&src_ctx);
  CircuitSeq *dag = nullptr;
  if (!qasm_parser.load_qasm(input_fn, dag)) {
    std::cout << "Parser failed" << std::endl;
  }
  Graph graph(&src_ctx, dag);

  std::shared_ptr<Graph> newGraph;

  // Greedy toffoli flip
  if (!rm_only) {
    auto xfer_pair = GraphXfer::ccz_cx_rz_xfer(&src_ctx, &dst_ctx, &union_ctx);

    newGraph = graph.toffoli_flip_greedy(GateType::rz, xfer_pair.first,
                                         xfer_pair.second);
    // graph_before_search->to_qasm(input_fn + ".toffoli_flip", false, false);
    std::cout << "toffoli flip + rotation merging done" << std::endl;
  } else {
    graph.rotation_merging(GateType::rz);
    newGraph =
        std::make_shared<Graph>(&dst_ctx, graph.to_circuit_sequence().get());
  }
  // O(n) rz -> t
  if (target_gates == "clifford") {
    newGraph->context = &union2_ctx;
    auto graph_seq = newGraph->to_circuit_sequence();
    auto rz_to_t_seq = graph_seq->get_rz_to_t(&union2_ctx);
    auto newGraph = std::make_shared<Graph>(&dst2_ctx, rz_to_t_seq.get());
  }

  newGraph->to_qasm(input_fn + ".optimized", false, false);
  return 0;
}
