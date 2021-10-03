#include "test_generator.h"
#include "../tasograph/substitution.h"

int main() {
  Context ctx({GateType::rx, GateType::ry, GateType::rz,
                                    GateType::cx});
  Generator gen(&ctx);
  Dataset dataset;
  gen.generate(3/*num_qubits*/,
               3/*max_num_input_parameters*/,
               3/*max_num_gates*/,
               dataset);
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
  test_generator(/*support_gates=*/{GateType::rx, GateType::ry, GateType::rz,
                                    GateType::cx}, /*num_qubits=*/
                                   3, /*max_num_input_parameters=*/
                                   3, /*max_num_gates=*/
                                   3, /*verbose=*/
                                   true, /*save_file_name=*/
                                   "data.json", /*count_minimal_representations=*/
                                   true);
  return 0;
}
