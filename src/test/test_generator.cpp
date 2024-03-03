#include "test_generator.h"

#include "quartz/parser/qasm_parser.h"
#include "quartz/tasograph/substitution.h"

using namespace quartz;

int main() {
  ParamInfo param_info(/*num_input_symbolic_params=*/3);
  Context ctx({GateType::x, GateType::y, GateType::cx, GateType::h}, 3,
              &param_info);
  Generator gen(&ctx);
  Dataset dataset;
  EquivalenceSet ecc;
  gen.generate(3 /*num_qubits*/, 3 /*max_num_gates*/, &dataset,
               /*invoke_python_verifier=*/false, &ecc,
               /*unique_parameters=*/false, /*verbose=*/true);
  std::cout << "ECC generated." << std::endl;
  for (const auto &it : dataset.dataset) {
    bool is_first = true;
    CircuitSeq *first_dag = NULL;
    for (auto &dag : it.second) {
      if (is_first) {
        first_dag = dag.get();
        is_first = false;
      } else {
        GraphXfer xfer(&ctx, &ctx, &ctx, first_dag, dag.get());
      }
    }
  }
  return 0;
}
