#include "../tasograph/tasograph.h"
#include "../tasograph/substitution.h"
#include "../parser/qasm_parser.h"

void parse_args(char **argv, int argc, bool &simulated_annealing, bool &early_stop,
                std::string &input_filename, std::string &output_filename,
                std::string &eqset_filename) {
  assert(argv[1] != nullptr);
  input_filename = std::string(argv[1]);
  output_filename = input_filename + ".optimized";
  for (int i = 2; i < argc; i++) {
	if (!std::strcmp(argv[i], "--sa")) {
	  simulated_annealing = true;
	  continue;
	}
	if (!std::strcmp(argv[i], "--simulated-annealing")) {
	  simulated_annealing = true;
	  continue;
	}
	if (!std::strcmp(argv[i], "--output")) {
	  output_filename = std::string(argv[++i]);
	  continue;
	}
	if (!std::strcmp(argv[i], "--eqset")) {
	  eqset_filename = std::string(argv[++i]);
	  continue;
	}
	if (!std::strcmp(argv[i], "--early-stop")) {
	  early_stop = true;
	  continue;
	}
  }
}

int main(int argc, char **argv) {
  std::string input_fn, output_fn;
  std::string eqset_fn = "bfs_verified_simplified.json";
  bool simulated_annealing = false;
  bool early_stop = false;
  parse_args(argv, argc, simulated_annealing, early_stop, input_fn, output_fn, eqset_fn);
  fprintf(stderr, "Input qasm file: %s\n", input_fn.c_str());

  // Construct contexts
  Context src_ctx({GateType::h, GateType::ccz, GateType::input_qubit,
                   GateType::input_param});
  Context dst_ctx({GateType::u1, GateType::u2, GateType::cx, GateType::add,
                   GateType::input_qubit, GateType::input_param});
  auto union_ctx = union_contexts(&src_ctx, &dst_ctx);

  // Construct GraphXfers for toffoli flip
  // Use this for voqc gate set(h, rz, x, cx)
  //   auto xfer_pair = TASOGraph::GraphXfer::ccz_cx_rz_xfer(&union_ctx);
  // Use this for ibmq gate set(u1, u2, u3, cx)
  auto xfer_pair = TASOGraph::GraphXfer::ccz_cx_u1_xfer(&union_ctx);
  // Load qasm file
  QASMParser qasm_parser(&src_ctx);
  DAG *dag = nullptr;
  if (!qasm_parser.load_qasm(input_fn, dag)) {
	std::cout << "Parser failed" << std::endl;
  }
  TASOGraph::Graph graph(&src_ctx, *dag);

  // Context shift
  RuleParser rule_parser({"h q0 = u2 q0 0 0.5pi;"});
  TASOGraph::Graph *graph_new_ctx = graph.context_shift(
      &src_ctx, &dst_ctx, &union_ctx, &rule_parser, /*ignore_toffoli*/ true);

  // Greedy toffoli flip
  TASOGraph::Graph *graph_before_search = graph_new_ctx->toffoli_flip_greedy(
      GateType::u1, xfer_pair.first, xfer_pair.second);
  std::cout << "gate count after toffoli flip: "
            << graph_before_search->total_cost() << std::endl;
  graph_before_search->to_qasm(input_fn + ".toffoli_flip", false, false);

  // Optimization
  TASOGraph::Graph *graph_after_search = graph_before_search->optimize(
      0.999, 0, false, &dst_ctx, eqset_fn, simulated_annealing, early_stop,
      /*rotation_merging_in_searching*/ true, GateType::u1);
  std::cout << "gate count after optimization: "
            << graph_after_search->total_cost() << std::endl;
  graph_after_search->to_qasm(output_fn, false, false);
}
