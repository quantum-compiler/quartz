#include "quartz/tasograph/tasograph.h"

using namespace quartz;

int main() {
  Context ctx({GateType::input_qubit, GateType::input_param, GateType::h,
               GateType::cx, GateType::rz, GateType::x});
  auto graph = Graph::from_qasm_file(
      &ctx, "../experiment/t_tdg_h_cx_toffoli_flip_dataset/"
            "barenco_tof_3.qasm.toffoli_flip");
  graph->to_qasm("a.qasm", false, false);
}
