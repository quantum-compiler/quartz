#include "quartz/parser/qasm_parser.h"
#include "quartz/tasograph/substitution.h"
#include "quartz/tasograph/tasograph.h"

using namespace quartz;

void parse_args(char **argv, int argc, bool &simulated_annealing,
                bool &early_stop, std::string &input_filename,
                std::string &output_filename, std::string &eqset_filename) {
  assert(argv[1] != nullptr);
  input_filename = std::string(argv[1]);
  early_stop = true;
  for (int i = 2; i < argc; i++) {
    if (!std::strcmp(argv[i], "--output")) {
      output_filename = std::string(argv[++i]);
      break;
    }
  }
}

int main(int argc, char **argv) {
  std::string input_fn, output_fn;
  std::string eqset_fn = "../Rigetti_5_3_complete_ECC_set.json";
  bool simulated_annealing = false;
  bool early_stop = false;
  parse_args(argv, argc, simulated_annealing, early_stop, input_fn, output_fn,
             eqset_fn);
  auto fn = input_fn.substr(input_fn.rfind('/') + 1);

  // Construct contexts
  Context src_ctx({GateType::h, GateType::ccz, GateType::cx, GateType::x,
                   GateType::input_qubit, GateType::input_param});
  Context dst_ctx({GateType::rz, GateType::h, GateType::cx, GateType::x,
                   GateType::add, GateType::input_qubit,
                   GateType::input_param});
  auto union_ctx = union_contexts(&src_ctx, &dst_ctx);

  // Construct GraphXfers for toffoli flip
  auto xfer_pair = GraphXfer::ccz_cx_rz_xfer(&union_ctx);
  // Load qasm file
  QASMParser qasm_parser(&src_ctx);
  CircuitSeq *dag = nullptr;
  if (!qasm_parser.load_qasm(input_fn, dag)) {
    std::cout << "Parser failed" << std::endl;
  }
  Graph graph(&src_ctx, dag);

  auto start = std::chrono::steady_clock::now();
  // Greedy toffoli flip
  std::vector<int> trace;
  graph.toffoli_flip_greedy_with_trace(GateType::rz, xfer_pair.first,
                                       xfer_pair.second, trace);
  auto new_graph = graph.toffoli_flip_by_instruction(
      GateType::rz, xfer_pair.first, xfer_pair.second, trace);
  // Convert cx to cz and merge h gates
  RuleParser cx_2_cz({"cx q0 q1 = h q1; cz q0 q1; h q1;"});
  Context cz_ctx({GateType::rz, GateType::h, GateType::x, GateType::cz,
                  GateType::add, GateType::input_qubit, GateType::input_param});
  auto union_ctx_0 = union_contexts(&cz_ctx, &dst_ctx);
  auto graph_before_h_cz_merge = new_graph->context_shift(
      &dst_ctx, &cz_ctx, &union_ctx_0, &cx_2_cz, false);
  auto graph_after_h_cz_merge = graph_before_h_cz_merge->optimize(
      &union_ctx_0, "../H_CZ_2_2_complete_ECC_set_modified.json",
      fn, /*print_message=*/
      true);
  graph_after_h_cz_merge->to_qasm(
      "circuit/voqc-benchmarks/after_h_cz_merge.qasm", false, false);

  // Shift the context to Rigetti Agave
  RuleParser rules({"h q0 = rx q0 pi; rz q0 0.5pi; rx q0 0.5pi; rz q0 -0.5pi;",
                    "x q0 = rx q0 pi;"});
  Context rigetti_ctx({GateType::rx, GateType::rz, GateType::cz, GateType::add,
                       GateType::input_qubit, GateType::input_param});
  auto union_ctx_1 = union_contexts(&rigetti_ctx, &union_ctx_0);
  auto graph_rigetti = graph_after_h_cz_merge->context_shift(
      &cz_ctx, &rigetti_ctx, &union_ctx_1, &rules, false);

  // Optimization
  auto graph_after_search =
      graph_rigetti->optimize(&union_ctx_1, eqset_fn, fn, /*print_message=*/
                              true);
  auto end = std::chrono::steady_clock::now();
  std::cout << "Optimization results of Quartz for " << fn
            << " on Rigetti gate set."
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
