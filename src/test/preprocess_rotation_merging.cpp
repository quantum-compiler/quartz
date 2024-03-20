
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

  Context ctx({GateType::h, GateType::x, GateType::rz, GateType::add,
               GateType::cx, GateType::input_qubit, GateType::input_param},
              3, &param_info);

  // Load qasm file
  QASMParser qasm_parser(&ctx);
  CircuitSeq *dag = nullptr;
  if (!qasm_parser.load_qasm(input_fn, dag)) {
    std::cout << "Parser failed" << std::endl;
  }

  Graph graph(&ctx, dag);
  graph.rotation_merging(GateType::rz);
  graph.to_qasm(input_fn + ".optimized", false, false);
  return 0;
}
