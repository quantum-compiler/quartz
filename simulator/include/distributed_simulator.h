//#include "../sim/circuit.h"
//#include "../sim/simulator.h"
#include "legion.h"
#include "simulator_const.h"
#include "cuda_helper.h"
#include "sim/simgate.h"
#include <custatevec.h>
#include <nccl.h>

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

class GateInfo {
public:
  int num_targets, num_controls;
  int permutation[MAX_NUM_QUBITS], target[MAX_NUM_QUBITS];
  // FIXME: currently we send matrix_data to devices for each compute task
  // Should precompute all matrices and send them to devices before computation
  // starts
  char matrix_data[MAX_GATE_MATRIX_SIZE];
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
  DistributedSimulator(const DSConfig &config);
  static void top_level_task(Legion::Task const *task,
                             std::vector<Legion::PhysicalRegion> const &regions,
                             Legion::Context ctx,
                             Legion::Runtime *runtime);
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
  template<typename DT>
  bool set_gate_info(const std::vector<Gate<DT> > &gates, GateInfo& info);
  template<typename DT>
  bool apply_gates(const std::vector<Gate<DT> > &gates);
private:
  DSConfig config;
  Legion::IndexSpace parallel_is;
  std::vector<std::pair<Legion::LogicalRegion, Legion::LogicalPartition> > cpu_state_vectors;
  std::vector<std::pair<Legion::LogicalRegion, Legion::LogicalPartition> > gpu_state_vectors;
  DSHandler handlers[MAX_NUM_WORKERS];
};

}; // namespace sim
