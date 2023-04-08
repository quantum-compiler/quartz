#include "quartz/context/context.h"
#include "quartz/tasograph/substitution.h"
#include "quartz/tasograph/tasograph.h"

using namespace quartz;

int main() {
  Context ctx({GateType::input_qubit, GateType::input_param, GateType::h,
               GateType::cx, GateType::x, GateType::rz, GateType::add,
               GateType::neg});
  auto graph = Graph::from_qasm_file(
      &ctx, "../experiment/circs/nam_circs/barenco_tof_3.qasm");
  auto subgraphs = graph->topology_partition(1);
  std::cout << "number of subgraphs: " << subgraphs.size() << std::endl;
  for (auto subgraph : subgraphs) {
    std::cout << subgraph->to_qasm() << std::endl;
  }
}