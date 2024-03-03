#include "../quartz/parser/qasm_parser.h"
#include "../quartz/tasograph/substitution.h"
#include "../quartz/tasograph/tasograph.h"

using namespace quartz;

void parse_args(char **argv, int argc, bool &simulated_annealing,
                bool &early_stop, bool &disable_search,
                std::string &input_filename, std::string &output_filename,
                std::string &eqset_filename) {
  assert(argv[1] != nullptr);
  input_filename = std::string(argv[1]);
  early_stop = true;
  for (int i = 2; i < argc; i++) {
    if (!std::strcmp(argv[i], "--output")) {
      output_filename = std::string(argv[++i]);
      continue;
    }
    if (!std::strcmp(argv[i], "--eqset")) {
      eqset_filename = std::string(argv[++i]);
      continue;
    }
    if (!std::strcmp(argv[i], "--disable_search")) {
      disable_search = true;
      continue;
    }
  }
}

int main(int argc, char **argv) {
  std::string input_fn, output_fn;
  std::string eqset_fn = "../IBM_4_3_complete_ECC_set.json";
  bool simulated_annealing = false;
  bool early_stop = false;
  bool disable_search = false;
  parse_args(argv, argc, simulated_annealing, early_stop, disable_search,
             input_fn, output_fn, eqset_fn);
  auto fn = input_fn.substr(input_fn.rfind('/') + 1);
  // Construct contexts
  ParamInfo param_info;
  Context src_ctx({GateType::h, GateType::ccz, GateType::x, GateType::cx,
                   GateType::t, GateType::input_qubit, GateType::input_param},
                  &param_info);
  Context dst_ctx({GateType::u1, GateType::u2, GateType::u3, GateType::add,
                   GateType::cx, GateType::input_qubit, GateType::input_param},
                  &param_info);
  auto union_ctx = union_contexts(&src_ctx, &dst_ctx);

  // Construct GraphXfers for toffoli flip
  auto xfer_pair = GraphXfer::ccz_cx_u1_xfer(&src_ctx, &dst_ctx, &union_ctx);
  // Load qasm file
  QASMParser qasm_parser(&src_ctx);
  CircuitSeq *dag = nullptr;
  if (!qasm_parser.load_qasm(input_fn, dag)) {
    std::cout << "Parser failed" << std::endl;
  }
  Graph graph(&src_ctx, dag);

  auto start = std::chrono::steady_clock::now();
  // Context shift
  RuleParser rule_parser(
      {"h q0 = u2 q0 0 pi", "x q0 = u3 q0 pi 0 -pi", "t q0 = u1 q0 0.25pi"});
  auto graph_new_ctx = graph.context_shift(&src_ctx, &dst_ctx, &union_ctx,
                                           &rule_parser, /*ignore_toffoli*/
                                           true);

  // Greedy toffoli flip
  auto graph_before_search = graph_new_ctx->toffoli_flip_greedy(
      GateType::u1, xfer_pair.first, xfer_pair.second);

  auto end = std::chrono::steady_clock::now();
  if (disable_search) {
    std::cout << "Optimization results of Quartz for " << fn
              << " on IBMQ gate set."
              << " Gate count after optimization: "
              << graph_before_search->gate_count() << ", "
              << "Circuit depth: " << graph_before_search->circuit_depth()
              << ", "
              << (double)std::chrono::duration_cast<std::chrono::milliseconds>(
                     end - start)
                         .count() /
                     1000.0
              << " seconds." << std::endl;

    return 0;
  }

  // Optimization
  auto graph_after_search =
      graph_before_search->optimize(&dst_ctx, eqset_fn, fn, /*print_message=*/
                                    true);
  end = std::chrono::steady_clock::now();
  std::cout << "Optimization results of Quartz for " << fn
            << " on IBMQ gate set."
            << " Gate count after optimization: "
            << graph_after_search->gate_count() << ", "
            << "Circuit depth: " << graph_after_search->circuit_depth() << ", "
            << (double)std::chrono::duration_cast<std::chrono::milliseconds>(
                   end - start)
                       .count() /
                   1000.0
            << " seconds." << std::endl;

  graph_after_search->to_qasm(output_fn, false, false);
}
