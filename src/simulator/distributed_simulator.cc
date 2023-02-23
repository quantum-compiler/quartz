#include "distributed_simulator.h"

using namespace sim;
using namespace Legion;
using Legion::coord_t;
using quartz::GateType;

DistributedSimulator::DistributedSimulator(const DSConfig &_config)
    : config(_config) {}

void DistributedSimulator::init_devices() {
  ArgumentMap argmap;
  Context ctx = config.lg_ctx;
  Runtime *runtime = config.lg_hlr;
  // Init CUDA library on each worker
  IndexLauncher initLauncher(
      CUDA_INIT_TASK_ID, task_is, TaskArgument(config, sizeof(DSConfig)), argmap,
      Predicate::TRUE_PRED, false /*must*/, 0 /*mapper_id*/);
  FutureMap fm = runtime->execute_index_space(ctx, initLauncher);
  fm.wait_all_results();
  int idx = 0;
  for (PointInRectIterator<1> it(task_rect); it(); it++) {
    handlers[idx++] = fm.get_result<DSHandler>(*it);
  }
}

/*static*/
void DistributedSimulator::top_level_task(
    Task const *task, std::vector<PhysicalRegion> const &regions, Context ctx,
    Runtime *runtime) {
  DSConfig config;
  quartz::Context ctx({GateType::input_qubit, GateType::input_param,
                       GateType::h, GateType::x, GateType::ry, GateType::u2,
                       GateType::u3, GateType::cx, GateType::cz, GateType::cp,
                       GateType::swap});
  auto seq = quartz::CircuitSeq::from_qasm_file(
      &ctx, std::string("/home/ubuntu/quartz/circuit/MQTBench_") +
                std::to_string(config.num_all_qubits) + "q/" + circuit_file +
                "_indep_qiskit_" + std::to_string(config.num_all_qubits) +
                ".qasm");
  sim::qcircuit::Circuit<double> circuit(config.num_all_qubits,
                                         config.num_local_qubits);
  circuit.compile(seq.get(), &ctx);
  DistributedSimulator simulator(config);
  simulator.init_devices();
  simulator.create_regions();
  simulator.init_state_vectors();
  // Start simulation
  std::vector<unsigned> init_perm, permutation;
  for (int i = 0; i < config.num_all_qubits; i++) {
    init_perm.push_back(i);
    permutation.push_back(i);
  }
  int index = 0;
  while (index < gates.size()) {
    // Collect all non-shuffle gate
    std::vector<Gate<DT>> gate_batch;
    while (index < gates.size()) {
      if (gate.gtype == SHUFFLE)
        break;
      else
        gate_batch.push_back(gates[index++]);
    }
    simulator.apply_gates(gate_batch);
    if (gates[index].gtype == SHUTTLE) {
      // TODO: implement shuffle
      index++;
    }
  }
}

bool DistributedSimulator::create_regions() {
  // Creating index spaces / partitions
  coord_t total_gpus = config.gpus_per_node * config.num_nodes;
  Rect<1> task_rect(Point<1>(0), Point<1>(total_gpus - 1));
  simulator.parallel_is = runtime->create_index_space(ctx, task_rect);
  Rect<1> state_vec_rect(Point<1>(0),
                         total_gpus * (1 << config.num_local_qubits));
  IndexSpaceT<NDIM> is = runtime->create_index_space(ctx, rect);
  tensor->region = runtime->create_logical_region(ctx, is, fs);
  Transform<1, 1> transform;
  Point<1> ext_hi;
  ext_hi[i] = (state_vec_rect.hi[0] - state_vector_rect.lo[0] + total_gpus) /
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
  switch (config->data_type) {
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
  return true;
}

bool DistributedSimulator::init_state_vectors() {
  // initialize all cpu_state_vectors
  for (size_t i = 0; i < cpu_state_vectors.size(); i++) {
    IndexLauncher launcher
  }
}

template <typename DT>
bool DistributedSimulator::apply_gates(const std::vector<Gate<DT>> &gates) {
  for (size_t i = 0; i < cpu_state_vectors.size(); i++) {
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
      launcher.add_src_field(0, FT_DATA);
      launcher.add_dst_field(0, FT_DATA);
      runtime->issue_copy_operation(ctx, launcher);
    }
    // Step 2: launch all gate kernels
    {
      GateInfo info(*this, gates[i]);
      IndexLauncher launcher(
          GATE_COMP_TASK_ID, parallel_is, TaskArgument(&info, sizeof(GateInfo)),
          argmap, Predicate::TRUE_PRED, false /*must*/, 0 /*mapper_id*/);
      launcher.add_region_requirement(
          RegionRequirement(gpu_sv.second, 0 /*projection ID*/, READ_WRITE,
                            EXCLUSIVE, gpu_sv.first));
    }
    // Step 3: move a state vector from GPU memory to DRAM
    {
      IndexCopyLauncher launcher(parallel_is);
      launcher.add_copy_requirements(
          RegionRequirement(
              gpu_state_vectors[i % gpu_state_vectors.size()].second,
              0 /*projection ID*/, WRITE_DISCARD, EXCLUSIVE,
              gpu_state_vectors[i % gpu_state_vectors.size()].first),
          RegionRequirement(cpu_state_vectors[i].second, 0 /*projection ID*/,
                            READ_ONLY, EXCLUSIVE, cpu_state_vectors[i].first,
                            MAP_TO_ZC_MEMORY));
      launcher.add_src_field(0, FT_DATA);
      launcher.add_dst_field(0, FT_DATA);
      runtime->issue_copy_operation(ctx, launcher);
    }
  }
}

int main(int argc, char **argv) {
  // This needs to be set, otherwise NCCL will try to use group kernel launches,
  // which are not compatible with the Realm CUDA hijack.
  setenv("NCCL_LAUNCH_MODE", "PARALLEL", true);

  Runtime::set_top_level_task_id(TOP_LEVEL_TASK_ID);
  {
    TaskVariantRegistrar registrar(TOP_LEVEL_TASK_ID, "top_level");
    registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
    registrar.set_replicable();
    Runtime::preregister_task_variant<DistributedSimulator::top_level_task>(
        registrar, "top_level");
  }
  {
    TaskVariantRegistrar registrar(CUDA_INIT_TASK_ID, "CUDA Init");
    registrar.add_constraint(ProcessorConstraint(Processor::TOC_PROC));
    registrar.set_leaf();
    Runtime::preregister_task_variant<DistributedSimulator::cuda_init_task>(
        registrar, "CUDA_INIT_TASK");
  }
  {
    TaskVariantRegistrar registrar(GATE_COMP_TASK_ID, "Gate Compute");
    registrar.add_constraint(ProcessorConstraint(Processor::TOC_PROC));
    registrar.set_leaf();
    Runtime::preregister_task_variant<DistributedSimulator::compute_task>(
        registrar, "Gate Compute Task");
  }

  Runtime::add_registration_callback(FFMapper::update_mappers);
  return Runtime::start(argc, argv);
}
