#include "../tasograph/tasograph.h"
#include "../tasograph/substitution.h"
#include "../parser/qasm_parser.h"

void parse_args(char **argv, int argc, bool &simulated_annealing,
                bool &early_stop, std::string &input_filename,
                std::string &output_filename, std::string &eqset_filename) {
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
  std::string eqset_fn = "build/bfs_verified_simplified_rigetti.json";
  bool simulated_annealing = false;
  bool early_stop = false;
  parse_args(argv, argc, simulated_annealing, early_stop, input_fn, output_fn,
             eqset_fn);
  fprintf(stderr, "Input qasm file: %s\n", input_fn.c_str());

  // Construct contexts
  Context src_ctx({GateType::h, GateType::ccz, GateType::cx, GateType::x,
                   GateType::input_qubit, GateType::input_param});
  Context dst_ctx({GateType::rz, GateType::h, GateType::cx, GateType::x,
                   GateType::add, GateType::input_qubit,
                   GateType::input_param});
  auto union_ctx = union_contexts(&src_ctx, &dst_ctx);

  // Construct GraphXfers for toffoli flip
  // Use this for voqc gate set(h, rz, x, cx)
  auto xfer_pair = TASOGraph::GraphXfer::ccz_cx_rz_xfer(&union_ctx);
  // Load qasm file
  QASMParser qasm_parser(&src_ctx);
  DAG *dag = nullptr;
  if (!qasm_parser.load_qasm(input_fn, dag)) {
	std::cout << "Parser failed" << std::endl;
  }
  TASOGraph::Graph graph(&src_ctx, *dag);

  // Greedy toffoli flip
  TASOGraph::Graph *new_graph = graph.toffoli_flip_greedy(
      GateType::rz, xfer_pair.first, xfer_pair.second);
  std::cout << "gate count after toffoli flip: " << new_graph->total_cost()
            << std::endl;

  // Convert cx to cz and merge h gates
  RuleParser cx_2_cz({"cx q0 q1 = h q1; cz q0 q1; h q1;"});
  Context cz_ctx({GateType::rz, GateType::h, GateType::x, GateType::cz,
                  GateType::add, GateType::input_qubit, GateType::input_param});
  auto union_ctx_0 = union_contexts(&cz_ctx, &dst_ctx);
  TASOGraph::Graph *graph_before_h_cz_merge = new_graph->context_shift(
      &dst_ctx, &cz_ctx, &union_ctx_0, &cx_2_cz, false);
  TASOGraph::Graph *graph_after_h_cz_merge = graph_before_h_cz_merge->optimize(
      0.999, 0, false, &union_ctx_0, "build/h_cz_merge.json",
      simulated_annealing, false, /*rotation_merging_in_searching*/ true,
      GateType::rz);
  graph_after_h_cz_merge->to_qasm(
      "circuit/voqc-benchmarks/after_h_cz_merge.qasm", false, false);

  // Shift the context to Rigetti Agave
  RuleParser rules({"h q0 = rx q0 pi; rz q0 0.5pi; rx q0 0.5pi; rz q0 -0.5pi;",
                    "x q0 = rx q0 pi;"});
  Context rigetti_ctx({GateType::rx, GateType::rz, GateType::cz, GateType::add,
                       GateType::input_qubit, GateType::input_param});
  auto union_ctx_1 = union_contexts(&rigetti_ctx, &union_ctx_0);
  TASOGraph::Graph *graph_rigetti = graph_after_h_cz_merge->context_shift(
      &cz_ctx, &rigetti_ctx, &union_ctx_1, &rules, false);

  // Optimization
  TASOGraph::Graph *graph_after_search = graph_rigetti->optimize(
      0.999, 0, false, &union_ctx_1, eqset_fn, simulated_annealing, early_stop,
      /*rotation_merging_in_searching*/ true, GateType::rz);
  std::cout << "gate count after optimization: "
            << graph_after_search->total_cost() << std::endl;
  graph_after_search->to_qasm(output_fn, false, false);
}
