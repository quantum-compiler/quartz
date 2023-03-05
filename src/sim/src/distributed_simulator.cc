#include "distributed_simulator.h"

using namespace sim;
using namespace Legion;
// declare Legion names
using Legion::ArgumentMap;
using Legion::Context;
using Legion::coord_t;
using Legion::Domain;
using Legion::FutureMap;
using Legion::IndexLauncher;
using Legion::InlineLauncher;
using Legion::PhysicalRegion;
using Legion::Predicate;
using Legion::Rect;
using Legion::RegionRequirement;
using Legion::Runtime;
using Legion::Task;
using Legion::TaskArgument;
using Legion::TaskLauncher;

using quartz::GateType;

DistributedSimulator::DistributedSimulator(const DSConfig &_config, const qcircuit::Circuit<qreal> &_circuit)
    : config(_config), circuit(_circuit) {}

void DistributedSimulator::init_devices() {
  ArgumentMap argmap;
  Context ctx = config.lg_ctx;
  Runtime *runtime = config.lg_hlr;
  // Init CUDA library on each worker
  IndexLauncher initLauncher(
      CUDA_INIT_TASK_ID, this->parallel_is, TaskArgument(&config, sizeof(DSConfig)), argmap,
      Predicate::TRUE_PRED, false /*must*/, 0 /*mapper_id*/);
  FutureMap fm = runtime->execute_index_space(ctx, initLauncher);
  fm.wait_all_results();
  Domain domain = runtime->get_index_space_domain(ctx, this->parallel_is);
  int idx = 0;
  for (Domain::DomainPointIterator it(domain); it; it++) {
    handlers[idx++] = fm.get_result<DSHandler>(*it);
  }
}

bool DistributedSimulator::create_regions() {
  Context ctx = config.lg_ctx;
  Runtime* runtime = config.lg_hlr;
  // Creating index spaces / partitions
  coord_t total_gpus = config.gpus_per_node * config.num_nodes;
  Rect<1> task_rect(Point<1>(0), Point<1>(total_gpus - 1));
  this->parallel_is = runtime->create_index_space(ctx, task_rect);
  Rect<1> state_vec_rect(Point<1>(0),
                         total_gpus * (1 << config.num_local_qubits) - 1);
  IndexSpaceT<1> is = runtime->create_index_space(ctx, state_vec_rect);
  Transform<1, 1> transform;
  Point<1> ext_hi;
  ext_hi[0] = (state_vec_rect.hi[0] - state_vec_rect.lo[0] + total_gpus) /
                  total_gpus -
              1;
  Rect<1> extent(Point<1>::ZEROES(), ext_hi);
  transform[0][0] = extent.hi[0] - extent.lo[0] + 1;
  IndexPartition ip = runtime->create_partition_by_restriction(
      ctx, is, parallel_is, transform, extent);
  assert(runtime->is_index_partition_disjoint(ctx, ip));
  assert(runtime->is_index_partition_complete(ctx, ip));
  // Creating logical spaces / partitions
  FieldSpace fs = runtime->create_field_space(ctx);
  FieldAllocator allocator = runtime->create_field_allocator(ctx, fs);
  switch (config.state_vec_data_type) {
    case DT_FLOAT:
      allocator.allocate_field(sizeof(cuFloatComplex), FID_DATA);
      break;
    case DT_DOUBLE:
      allocator.allocate_field(sizeof(cuDoubleComplex), FID_DATA);
      break;
    default:
      assert(false);
  }
  // currently assume that 2^num_all_qubits is a multiplier of
  // 2^num_local_qubits * total_gpus
  assert((1 << (size_t)config.num_all_qubits) % (state_vec_rect.hi[0] + 1) ==
         0);
  size_t num_state_vectors =
      (1 << (size_t)config.num_all_qubits) / (state_vec_rect.hi[0] + 1);
  for (size_t i = 0; i < num_state_vectors; i++) {
    LogicalRegion lr = runtime->create_logical_region(ctx, is, fs);
    LogicalPartition lp = runtime->get_logical_partition(ctx, lr, ip);
    cpu_state_vectors.push_back(std::make_pair(lr, lp));
  }
  for (size_t i = 0; i < config.num_state_vectors_on_gpu; i++) {
    LogicalRegion lr = runtime->create_logical_region(ctx, is, fs);
    LogicalPartition lp = runtime->get_logical_partition(ctx, lr, ip);
    gpu_state_vectors.push_back(std::make_pair(lr, lp));
  }
  printf("num_state_vectors: %d\n", num_state_vectors);
  return true;
}

bool DistributedSimulator::init_state_vectors() {
  // initialize all cpu_state_vectors
  Context ctx = config.lg_ctx;
  Runtime* runtime = config.lg_hlr;
  for (size_t i = 0; i < cpu_state_vectors.size(); i++) {
    ArgumentMap argmap;
    IndexLauncher launcher(
        SV_INIT_TASK_ID, parallel_is, TaskArgument(nullptr, 0),
        argmap, Predicate::TRUE_PRED, false /*must*/, 0 /*mapper_id*/);
    launcher.add_region_requirement(
        RegionRequirement(cpu_state_vectors[i].second, 0 /*projection ID*/, WRITE_ONLY,
                          EXCLUSIVE, cpu_state_vectors[i].first, MAP_TO_ZC_MEMORY));
    launcher.add_field(0, FID_DATA);
    FutureMap fm = runtime->execute_index_space(ctx, launcher);
    fm.wait_all_results();
  }
  // initialize all gpu_state_vectors
  for (size_t i = 0; i < gpu_state_vectors.size(); i++) {
    ArgumentMap argmap;
    IndexLauncher launcher(
        SV_INIT_TASK_ID, parallel_is, TaskArgument(nullptr, 0),
        argmap, Predicate::TRUE_PRED, false /*must*/, 0 /*mapper_id*/);
    launcher.add_region_requirement(
        RegionRequirement(gpu_state_vectors[i].second, 0 /*projection ID*/, WRITE_ONLY,
                          EXCLUSIVE, gpu_state_vectors[i].first));
    launcher.add_field(0, FID_DATA);
    FutureMap fm = runtime->execute_index_space(ctx, launcher);
    fm.wait_all_results();
  }
  return true;
}

bool DistributedSimulator::run() {
  create_regions();
  init_devices();
  
  init_state_vectors();

  for (int i = 0; i < config.num_all_qubits; i++) {
    permutation[i] = i;
  }
  
  int fused_idx = 0;
  int shm_idx = 0;
  for (int i = 0; i < circuit.task_map.size(); i++) {
    if (circuit.task_map[i] == FUSED) {
      GateInfo info;
      for (int i = 0; i < config.num_all_qubits; i++) {
        info.permutation[i] = permutation[i];
      }
      info.num_batched_gates = circuit.fused_gates[fused_idx].size();
      info.gtype = FUSED;
      info.gates = circuit.fused_gates[fused_idx].data();
      apply_gates(info);
      fused_idx++;
    }
    else if (circuit.task_map[i] == SHM) {
      qindex active_logical_qs = 0;
      // TODO: convert active physical qubits to logical qubits
      GateInfo info;
      for (int i = 0; i < config.num_all_qubits; i++) {
        info.permutation[i] = permutation[i];
      }
      info.num_batched_gates = circuit.shm_gates[shm_idx].size();
      info.gtype = SHM;
      info.kgates = circuit.shm_gates[shm_idx].data();
      info.active_qubits_logical = circuit.active_physic_qs[shm_idx];
      apply_gates(info);
      shm_idx++;
    }
    else if (circuit.task_map[i] == SHUFFLE) {
      // TODO: implement shuffle
    }

  }
  
  return true;
}

bool DistributedSimulator::apply_gates(const GateInfo &info) {
  Context ctx = config.lg_ctx;
  Runtime* runtime = config.lg_hlr;
  for (size_t i = 0; i < 1; i++) {
    std::pair<LogicalRegion, LogicalPartition> cpu_sv = cpu_state_vectors[i];
    std::pair<LogicalRegion, LogicalPartition> gpu_sv =
        gpu_state_vectors[i % gpu_state_vectors.size()];
    // Step 1: move a state vector from DRAM to GPU memory
    {
      IndexCopyLauncher launcher(parallel_is);
      launcher.add_copy_requirements(
          RegionRequirement(cpu_sv.second, 0 /*projection ID*/, READ_ONLY,
                            EXCLUSIVE, cpu_sv.first, MAP_TO_ZC_MEMORY),
          RegionRequirement(gpu_sv.second, 0 /*projection ID*/, WRITE_DISCARD,
                            EXCLUSIVE, gpu_sv.first));
      launcher.add_src_field(0, FID_DATA);
      launcher.add_dst_field(0, FID_DATA);
      runtime->issue_copy_operation(ctx, launcher);
    }
    // Step 2: launch all gate kernels
    // {
    //   // GateInfo info;
    //   // set_gate_info(gates, info);
    //   printf("size: %d, num: %d, %p, %p\n",sizeof(GateInfo), info.num_batched_gates, info.kgates, info);
    //   ArgumentMap argmap;
    //   //FIXME: set_argmap
    //   IndexLauncher launcher(
    //       GATE_COMP_TASK_ID, parallel_is, TaskArgument(&info, sizeof(GateInfo)),
    //       argmap, Predicate::TRUE_PRED, false /*must*/, 0 /*mapper_id*/);
    //   launcher.add_region_requirement(
    //       RegionRequirement(gpu_sv.second, 0 /*projection ID*/, READ_WRITE,
    //                         EXCLUSIVE, gpu_sv.first));
    //   launcher.add_field(0, FID_DATA);
    //   runtime->execute_index_space(ctx, launcher);
    // }
    // Step 3: move a state vector from GPU memory to DRAM
    // {
    //   IndexCopyLauncher launcher(parallel_is);
    //   launcher.add_copy_requirements(
    //       RegionRequirement(
    //           gpu_state_vectors[i % gpu_state_vectors.size()].second,
    //           0 /*projection ID*/, READ_ONLY, EXCLUSIVE,
    //           gpu_state_vectors[i % gpu_state_vectors.size()].first),
    //       RegionRequirement(cpu_state_vectors[i].second, 0 /*projection ID*/,
    //                         WRITE_DISCARD, EXCLUSIVE, cpu_state_vectors[i].first,
    //                         MAP_TO_ZC_MEMORY));
    //   launcher.add_src_field(0, FID_DATA);
    //   launcher.add_dst_field(0, FID_DATA);
    //   runtime->issue_copy_operation(ctx, launcher);
    // }
  }
  return true;
}
