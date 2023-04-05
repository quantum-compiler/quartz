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
  int batch_id = 0;
  int use_buffer;
  FusedGate* fgates = nullptr;
  int fused_idx = 0;
  KernelGate* kgates = nullptr;
  int shm_idx = 0;
  qindex active_qubits_logical[MAX_BATCHED_TASKS];
  unsigned permutation[MAX_QUBIT];
  unsigned pos[MAX_QUBIT];
  SimGateType task_map[MAX_BATCHED_TASKS];
  int task_num_gates[MAX_BATCHED_TASKS];
  int num_tasks = 0;
  int nLocalSwaps = 0;
  int local_swap[MAX_QUBIT];
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
    Legion::coord_t num_all_qubits;
    int chunk_id;
    unsigned* threadBias;
    int* loIdx_device;
    int* shiftAt_device;
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
  static void shuffle_task(Legion::Task const *task,
                           std::vector<Legion::PhysicalRegion> const &regions,
                           Legion::Context ctx,
                           Legion::Runtime *runtime);
  static void store_task(Legion::Task const *task,
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
  std::map<int, Legion::IndexSpace> stage_parallel_is;
  std::vector<std::pair<Legion::LogicalRegion, Legion::LogicalPartition> > cpu_state_vectors;
  std::vector<std::pair<Legion::LogicalRegion, Legion::LogicalPartition> > gpu_state_vectors;
  std::vector<std::map<int, std::vector<Legion::LogicalPartition>>> cpu_sv_shuffle_lp;
  std::vector<std::map<int, std::vector<Legion::LogicalPartition>>> gpu_sv_shuffle_lp;
  DSHandler handlers[MAX_NUM_WORKERS];
  // Task Info
  unsigned permutation[MAX_QUBIT];
  unsigned pos[MAX_QUBIT];
};

KernelGate getGate(const KernelGate& gate, int part_id, qindex relatedLogicQb, const std::map<int, int>& toID, const unsigned* pos, unsigned n_local);

KernelGateType toU(KernelGateType type);

void top_level_task(Legion::Task const *task,
                             std::vector<Legion::PhysicalRegion> const &regions,
                             Legion::Context ctx,
                             Legion::Runtime *runtime);

}; // namespace sim
