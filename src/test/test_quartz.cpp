#include "quartz/parser/qasm_parser.h"
#include "quartz/tasograph/substitution.h"
#include "quartz/tasograph/tasograph.h"

using namespace quartz;

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
  std::string eqset_fn = "voqc_ecc.json";
  bool simulated_annealing = false;
  bool early_stop = false;
  parse_args(argv, argc, simulated_annealing, early_stop, input_fn, output_fn,
             eqset_fn);
  fprintf(stderr, "Input qasm file: %s\n", input_fn.c_str());

  // Construct contexts
  ParamInfo param_info;
  Context src_ctx({GateType::h, GateType::ccz, GateType::x, GateType::cx,
                   GateType::input_qubit, GateType::input_param},
                  &param_info);
  Context dst_ctx({GateType::h, GateType::x, GateType::rz, GateType::add,
                   GateType::cx, GateType::input_qubit, GateType::input_param},
                  &param_info);
  auto union_ctx = union_contexts(&src_ctx, &dst_ctx);

  // Construct GraphXfers for toffoli flip
  // Use this for voqc gate set(h, rz, x, cx)
  //   auto xfer_pair = TASOGraph::GraphXfer::ccz_cx_rz_xfer(&union_ctx);
  // Use this for ibmq gate set(u1, u2, u3, cx)
  auto xfer_pair = GraphXfer::ccz_cx_rz_xfer(&src_ctx, &dst_ctx, &union_ctx);
  // Load qasm file
  QASMParser qasm_parser(&src_ctx);
  CircuitSeq *dag = nullptr;
  if (!qasm_parser.load_qasm(input_fn, dag)) {
    std::cout << "Parser failed" << std::endl;
  }
  Graph graph(&src_ctx, dag);

  // Context shift
  //   RuleParser rule_parser({"h q0 = u2 q0 0 pi", "x q0 = u3 q0 0.5pi,
  //   0,-pi"}); RuleParser rule_parser({"h q0 = u2 q0 0 pi", "x q0 = u3 q0
  //   0.5pi, 0,-pi"}); Graph *graph_new_ctx =
  //       graph.context_shift(&src_ctx, &dst_ctx, &union_ctx, &rule_parser,
  //                           /*ignore_toffoli*/ true);

  // Greedy toffoli flip
  auto graph_before_search = graph.toffoli_flip_greedy(
      GateType::rz, xfer_pair.first, xfer_pair.second);
  std::cout << "gate count after toffoli flip: "
            << graph_before_search->total_cost() << std::endl;
  graph_before_search->to_qasm(input_fn + ".toffoli_flip", false, false);

  // Optimization
  auto graph_after_search = graph_before_search->optimize(
      &dst_ctx, eqset_fn, input_fn, /*print_message=*/
      true);
  std::cout << "gate count after optimization: "
            << graph_after_search->total_cost() << std::endl;
  graph_after_search->to_qasm(output_fn, false, false);
}
