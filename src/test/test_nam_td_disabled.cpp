#include "quartz/parser/qasm_parser.h"
#include "quartz/tasograph/substitution.h"
#include "quartz/tasograph/tasograph.h"

using namespace quartz;

void parse_args(char **argv, int argc, bool &simulated_annealing,
                bool &early_stop, std::string &input_filename,
                std::string &output_filename, std::string &eqset_filename) {
  assert(argv[1] != nullptr);
  input_filename = std::string(argv[1]);
  early_stop = false;
  for (int i = 2; i < argc; i++) {
    if (!std::strcmp(argv[i], "--output")) {
      output_filename = std::string(argv[++i]);
      break;
    }
  }
}

int main(int argc, char **argv) {
  std::string input_fn, output_fn;
  std::string eqset_fn = "../Nam_5_3_complete_ECC_set.json";
  bool simulated_annealing = false;
  bool early_stop = false;
  parse_args(argv, argc, simulated_annealing, early_stop, input_fn, output_fn,
             eqset_fn);

  // Construct contexts
  ParamInfo param_info;
  Context src_ctx({GateType::h, GateType::ccz, GateType::x, GateType::cx,
                   GateType::input_qubit, GateType::input_param},
                  &param_info);
  Context dst_ctx({GateType::h, GateType::x, GateType::rz, GateType::add,
                   GateType::cx, GateType::input_qubit, GateType::input_param},
                  &param_info);
  auto union_ctx = union_contexts(&src_ctx, &dst_ctx);

  auto xfer_pair = GraphXfer::ccz_cx_rz_xfer(&src_ctx, &dst_ctx, &union_ctx);
  // Load qasm file
  QASMParser qasm_parser(&src_ctx);
  CircuitSeq *dag = nullptr;
  if (!qasm_parser.load_qasm(input_fn, dag)) {
    std::cout << "Parser failed" << std::endl;
  }
  Graph graph(&src_ctx, dag);

  auto start = std::chrono::steady_clock::now();
  // Greedy toffoli flip
  std::vector<int> trace{0, 0, 0, 0};
  graph.toffoli_flip_greedy_with_trace(GateType::rz, xfer_pair.first,
                                       xfer_pair.second, trace);
  auto graph_before_search = graph.toffoli_flip_by_instruction(
      GateType::rz, xfer_pair.first, xfer_pair.second, trace);
  // graph_before_search->to_qasm(input_fn + ".toffoli_flip", false, false);

  // Optimization
  auto fn = input_fn.substr(input_fn.rfind('/') + 1);
  auto graph_after_search =
      graph_before_search->optimize(&dst_ctx, eqset_fn, fn, /*print_message=*/
                                    true);
  auto end = std::chrono::steady_clock::now();
  std::cout << "Optimization results of Quartz for " << fn
            << " on Nam's gate set."
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
