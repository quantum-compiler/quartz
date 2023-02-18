#include "../sim/circuit.h"
#include "../sim/simulator.h"
#include "legion.h"

namespace sim {
class DSConfig {
  DSConfig();
  void parse_args(char **argv, int argc);
  DataType data_type;
  Legion::coord_t gpus_per_node, cpus_per_node, num_nodes;
  Legion::coord_t num_local_qubits, num_all_qubits, num_state_vectors_on_gpu;
};

class DistributedSimulator {
public:
  DistributedSimulator(const DSConfig& config);
private:
  DSConfig config;
  std::vector<std::pair<LogicalRegion, LogicalPartition> > cpu_state_vectors;
  std::vector<std::pair<LogicalRegion, LogicalPartition> > gpu_state_vectors;
};

};
