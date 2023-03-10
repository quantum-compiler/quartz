#include "cuda_helper.h"
#include "simgate.h"
#include "circuit.h"
#include "mapper.h"

namespace sim {

class DSConfig {
public:
  DSConfig();
  void parse_args(char **argv, int argc);
  DataType state_vec_data_type;
  Legion::coord_t gpus_per_node, cpus_per_node, num_nodes;
  Legion::coord_t num_local_qubits, num_all_qubits, num_state_vectors_on_gpu;
  Legion::Context lg_ctx;
  Legion::Runtime *lg_hlr;
};

struct GateInfo {
  FusedGate* fgates = nullptr;
  // int* num_fused = nullptr;
  // KernelGate** kgates = nullptr;
  // KernelGate* kgates = nullptr;
  // int* num_shm = nullptr;
  // qindex* active_qubits_logical = nullptr;
  int num_tasks = 0;
  int fused_idx = 0;
  int shm_idx = 0;
  SimGateType tasks[MAX_BATCHED_TASKS];
  unsigned permutation[MAX_QUBIT];
};

class DistributedSimulator {
public:
  struct DSHandler {
    custatevecHandle_t statevec;
    //cudnnHandle_t dnn;
    //cublasHandle_t blas;
    void *workSpace;
    size_t workSpaceSize;
    ncclComm_t ncclComm;
    DataType vecDataType;
    Legion::coord_t num_local_qubits;
  };
public:
  DistributedSimulator(const DSConfig &config, const qcircuit::Circuit<qreal> &circuit);
  static DSHandler cuda_init_task(Legion::Task const *task,
                             std::vector<Legion::PhysicalRegion> const &regions,
                             Legion::Context ctx,
                             Legion::Runtime *runtime);
  static void sv_init_task(Legion::Task const *task,
                           std::vector<Legion::PhysicalRegion> const &regions,
                           Legion::Context ctx,
                           Legion::Runtime *runtime);
  static void sv_comp_task(Legion::Task const *task,
                           std::vector<Legion::PhysicalRegion> const &regions,
                           Legion::Context ctx,
                           Legion::Runtime *runtime);
  bool create_regions();
  void init_devices();
  bool init_state_vectors();
  bool run();

  bool apply_gates(const GateInfo &info);
  
private:
  DSConfig config;
  qcircuit::Circuit<qreal> circuit;
  Legion::IndexSpace parallel_is;
  std::vector<std::pair<Legion::LogicalRegion, Legion::LogicalPartition> > cpu_state_vectors;
  std::vector<std::pair<Legion::LogicalRegion, Legion::LogicalPartition> > gpu_state_vectors;
  DSHandler handlers[MAX_NUM_WORKERS];
  // Task Info
  unsigned permutation[MAX_QUBIT];
};

void top_level_task(Legion::Task const *task,
                             std::vector<Legion::PhysicalRegion> const &regions,
                             Legion::Context ctx,
                             Legion::Runtime *runtime);

}; // namespace sim
