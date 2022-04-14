#include "test_optimization.h"
#include "quartz/gate/gate_utils.h"

#include <iostream>
#include <filesystem>
using std::filesystem::current_path;
using namespace quartz;

int main() {
	Context ctx({GateType::input_qubit, GateType::input_param, GateType::cx,
	             GateType::h, GateType::s, GateType::t, GateType::tdg,
	             GateType::x, GateType::add, GateType::z, GateType::rz});
	//   test_optimization(&ctx, "circuit/example-circuits/voqc_fig5.qasm",
	//                     "cmake-build-debug/bfs_verified.json");
  std::cout << "Current working directory: " << current_path() << std::endl;
	test_optimization(&ctx, "circuit/example-circuits/barenco_tof_3.qasm",
	                  "Nam_3_4_complete_ECC_set.json",
	                  false /*use_simulated_annealing*/);
}