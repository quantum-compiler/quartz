#include "test_generator.h"
#include "../tasograph/substitution.h"
#include "../parser/qasm_parser.h"

int main() {
  Context ctx({GateType::x, GateType::y, GateType::cx, GateType::h});
  Generator gen(&ctx);
  Dataset dataset;
  gen.generate_dfs(3/*num_qubits*/,
                   3/*max_num_input_parameters*/,
                   3/*max_num_gates*/,
                   dataset,
                   true/*restrict_search_space*/);
  for (const auto& it : dataset.dataset) {
    bool is_first = true;
    DAG* first_dag = NULL;
    for (auto& dag : it.second) {
      if (is_first) {
        first_dag = dag.get();
        is_first = false;
      } else {
        TASOGraph::GraphXfer xfer(&ctx, first_dag, dag.get());
      }
    }
  }

  QASMParser parser(&ctx);
  DAG* dag = NULL;
  parser.load_qasm("test.qasm", dag);
  return 0;
}
