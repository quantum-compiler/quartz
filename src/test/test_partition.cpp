#include "quartz/context/context.h"
#include "quartz/tasograph/substitution.h"
#include "quartz/tasograph/tasograph.h"

using namespace quartz;

int main() {
  ParamInfo param_info;
  Context ctx({GateType::input_qubit, GateType::input_param, GateType::h,
               GateType::cx, GateType::x, GateType::rz, GateType::add,
               GateType::neg},
              &param_info);
  auto graph = Graph::from_qasm_file(
      &ctx, "../experiment/circs/scalability_study/adder_64/adder_64.qasm");
  auto subgraphs = graph->topology_partition(512);
  std::cout << "number of subgraphs: " << subgraphs.size() << std::endl;
  for (auto subgraph : subgraphs) {
    std::cout << subgraph->to_qasm() << std::endl;
  }
  for (int i = 0; i < subgraphs.size(); i++) {
    subgraphs[i]->to_qasm("../experiment/circs/scalability_study/adder_64/" +
                              std::to_string(i) + ".qasm",
                          false, false);
  }
}
