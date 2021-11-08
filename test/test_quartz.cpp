#include "../tasograph/tasograph.h"
#include "../tasograph/substitution.h"
#include "../parser/qasm_parser.h"

void parse_args(char** argv, int argc,
                bool &simulated_annealing,
                std::string& input_filename,
                std::string& output_filename,
                std::string& eqset_filename)
{
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
  }
}

int main(int argc, char** argv) {
  std::string input_fn, output_fn;
  std::string eqset_fn = "bfs_verified_simplified.json";
  bool simulated_annealing = false;
  parse_args(argv, argc, simulated_annealing, input_fn, output_fn,
             eqset_fn);
  fprintf(stderr, "Input qasm file: %s\n", input_fn.c_str());
  // Construct rules
  RuleParser toffoli_0({"ccz q0 q1 q2 = cx q1 q2; rz q2 -0.25pi; cx q0 q2; rz "
                        "q2 0.25pi; cx q1 q2; rz q2 -0.25pi; cx "
                        "q0 q2; cx q0 q1; rz q1 -0.25pi; cx q0 q1; rz q0 "
                        "0.25pi; rz q1 0.25pi; rz q2 0.25pi;"});
  RuleParser toffoli_1({"ccz q0 q1 q2 = cx q1 q2; rz q2 0.25pi; cx q0 q2; rz "
                        "q2 -0.25pi; cx q1 q2; rz q2 0.25pi; cx "
                        "q0 q2; cx q0 q1; rz q1 0.25pi; cx q0 q1; rz q0 "
                        "-0.25pi; rz q1 -0.25pi; rz q2 -0.25pi;"});
  // Construct contexts
  Context src_ctx({GateType::h, GateType::ccz, GateType::input_qubit,
                   GateType::input_param});
  Context dst_ctx({GateType::rz, GateType::cx, GateType::h, GateType::add,
                   GateType::input_qubit, GateType::input_param});
  Context union_ctx({GateType::ccz, GateType::rz, GateType::cx, GateType::h,
                     GateType::input_qubit, GateType::input_param});
  // Construct GraphXfers
  std::vector<Command> cmds;
  Command cmd;
  toffoli_0.find_convert_commands(&dst_ctx, GateType::ccz, cmd, cmds);
  TASOGraph::GraphXfer *xfer =
      TASOGraph::GraphXfer::create_single_gate_GraphXfer(&union_ctx, cmd, cmds);
  toffoli_1.find_convert_commands(&dst_ctx, GateType::ccz, cmd, cmds);
  TASOGraph::GraphXfer *xfer_inverse =
      TASOGraph::GraphXfer::create_single_gate_GraphXfer(&union_ctx, cmd, cmds);
  // Load qasm file
  QASMParser qasm_parser(&src_ctx);
  DAG *dag = nullptr;
  if (!qasm_parser.load_qasm(input_fn, dag)) {
	std::cout << "Parser failed" << std::endl;
  }
  TASOGraph::Graph graph(&src_ctx, *dag);
  TASOGraph::Graph *graph_before_search =
      graph.toffoli_flip_greedy(GateType::rz, xfer, xfer_inverse);
  std::cout << "gate count after toffoli flip: " << graph_before_search->total_cost()
            << std::endl;
  graph_before_search->to_qasm(input_fn + ".toffoli_flip", false, false);
  TASOGraph::Graph *graph_after_search =
      graph_before_search->optimize(0.999, 0, false, &dst_ctx, eqset_fn, simulated_annealing);
  graph_after_search->to_qasm(output_fn, false, false);
}
