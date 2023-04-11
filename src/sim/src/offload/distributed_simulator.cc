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
      Predicate::TRUE_PRED, false /*must*/, 0 /*mapper_id*/, PreservedIDs::DataParallelism_GPU);
  FutureMap fm = runtime->execute_index_space(ctx, initLauncher);
  fm.wait_all_results();
  Domain domain = runtime->get_index_space_domain(ctx, this->parallel_is);
  int idx = 0;
  for (Domain::DomainPointIterator it(domain); it; it++) {
    handlers[idx++] = fm.get_result<DSHandler>(*it);
  }
}

bool DistributedSimulator::create_regions() {
  printf("Creating regions...\n");
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
    case DT_FLOAT_COMPLEX:
      allocator.allocate_field(sizeof(cuFloatComplex), FID_DATA);
      break;
    case DT_DOUBLE_COMPLEX:
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
  // on cpu we also have a buffer of the sv for shuffle
  for (size_t i = 0; i < 2 * num_state_vectors; i++) {
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


  // create index partitions for shuffles:
  std::map<int, std::vector<IndexPartition>> shuffle_ips;
  std::map<int, std::vector<IndexPartition>> shuffle_ips_gpu;
  for (int i = 0; i <= circuit.global_swap_record.size(); i++) {
    int n_swap = 0;
    unsigned global_swap_mask = i == circuit.global_swap_record.size() ? 0 : circuit.global_swap_record[i];
    for (int j = 0; j < config.num_all_qubits - config.num_local_qubits; j++) {
      if(global_swap_mask >> j & 1) {
        n_swap++;
      }     
    }

    int n_partition = 1 << n_swap;

    int parallel_degree = n_partition < total_gpus ? n_partition : total_gpus;
    // task is
    if (stage_parallel_is.find(parallel_degree) == stage_parallel_is.end()) {
      Rect<1> task(Point<1>(0), Point<1>(parallel_degree - 1));
      this->stage_parallel_is[parallel_degree] = runtime->create_index_space(ctx, task);
    }
    // gpu sv
    if (shuffle_ips_gpu.find(parallel_degree) == shuffle_ips_gpu.end()){
      std::vector<IndexPartition> shuffle_ip;
      Point<1> extent_hi;
      Point<1> extent_lo;
      for (size_t k = 0; k < total_gpus / parallel_degree; k++) {
        extent_lo[0] = k * parallel_degree * transform[0][0];
        extent_hi[0] = extent_lo[0] + transform[0][0] - 1;
        Rect<1> shuffle_extent(extent_lo, extent_hi);
        IndexPartition s_ip = runtime->create_partition_by_restriction(
            ctx, is, stage_parallel_is.at(parallel_degree), transform, shuffle_extent);
        shuffle_ip.push_back(s_ip);
      }
      shuffle_ips_gpu[parallel_degree] = shuffle_ip;
    }
    // cpu sv
    if (shuffle_ips.find(n_partition) == shuffle_ips.end()){
      std::vector<IndexPartition> shuffle_ip;
      Point<1> extent_hi;
      Point<1> extent_lo;
      extent_hi[0] = (ext_hi[0] + n_partition) / n_partition - 1;
      extent_lo[0] = 0;
      Transform<1, 1> trans;
      trans[0][0] = extent_hi[0] + 1;
      for (size_t k = 0; k < n_partition / total_gpus; k++) {
        extent_lo[0] = k * total_gpus * trans[0][0];
        extent_hi[0] = extent_lo[0] + (ext_hi[0] +  n_partition) / n_partition - 1;
        for (int j = 0; j < total_gpus; j++) {
          Rect<1> shuffle_extent(extent_lo, extent_hi);
          IndexPartition s_ip = runtime->create_partition_by_restriction(
              ctx, is, stage_parallel_is.at(parallel_degree), trans, shuffle_extent);
          shuffle_ip.push_back(s_ip);
          extent_lo[0] += transform[0][0];
          extent_hi[0] += transform[0][0];
        }

      }
      shuffle_ips[n_partition] = shuffle_ip;
    }
  }

  // create logical partition for all cpu logical region:
  for (size_t i = 0; i < 2*num_state_vectors; i++) {
    LogicalRegion lr = cpu_state_vectors[i].first;
    std::map<int, std::vector<LogicalPartition>> cpu_shuffle_map;
    for(auto &t : shuffle_ips) {
      std::vector<LogicalPartition> s_lps;
      for (auto &s_ip : t.second) {
        LogicalPartition lp = runtime->get_logical_partition(ctx, lr, s_ip);
        s_lps.push_back(lp);
      }
      cpu_shuffle_map[t.first] = s_lps;
    }
    cpu_sv_shuffle_lp.push_back(cpu_shuffle_map);
  }

  // create logical partition for all cpu logical region:
  for (size_t i = 0; i < config.num_state_vectors_on_gpu; i++) {
    LogicalRegion lr = gpu_state_vectors[i].first;
    std::map<int, std::vector<LogicalPartition>> gpu_shuffle_map;
    for(auto &t : shuffle_ips_gpu) {
      std::vector<LogicalPartition> s_lps;
      for (auto &s_ip : t.second) {
        LogicalPartition lp = runtime->get_logical_partition(ctx, lr, s_ip);
        s_lps.push_back(lp);
      }
      gpu_shuffle_map[t.first] = s_lps;
    }
    gpu_sv_shuffle_lp.push_back(gpu_shuffle_map);
  }

  return true;
}

bool DistributedSimulator::init_state_vectors() {
  // initialize all cpu_state_vectors
  Context ctx = config.lg_ctx;
  Runtime* runtime = config.lg_hlr;
  for (size_t i = 0; i < cpu_state_vectors.size(); i++) {
    ArgumentMap argmap;
    IndexLauncher launcher(
        CPU_SV_INIT_TASK_ID, parallel_is, TaskArgument(nullptr, 0),
        argmap, Predicate::TRUE_PRED, false /*must*/, 0 /*mapper_id*/, PreservedIDs::DataParallelism_CPU);
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
        GPU_SV_INIT_TASK_ID, parallel_is, TaskArgument(nullptr, 0),
        argmap, Predicate::TRUE_PRED, false /*must*/, 0 /*mapper_id*/, PreservedIDs::DataParallelism_GPU);
    launcher.add_region_requirement(
        RegionRequirement(gpu_state_vectors[i].second, 0 /*projection ID*/, WRITE_ONLY,
                          EXCLUSIVE, gpu_state_vectors[i].first));
    launcher.add_field(0, FID_DATA);
    FutureMap fm = runtime->execute_index_space(ctx, launcher);
    fm.wait_all_results();
  }
  printf("Init done..\n");
  return true;
}

bool DistributedSimulator::run() {
  printf("Simulator (offloading) is running...\n");
  create_regions();
  init_devices();
  
  init_state_vectors();

  for (int i = 0; i < config.num_all_qubits; i++) {
    permutation[i] = circuit.init_permutation[i];
    pos[circuit.init_permutation[i]] = i;
  }

  FusedGate* fgates = (FusedGate*) malloc(MAX_FUSED_GATE*sizeof(FusedGate));
  printf("sizeof FusedGate: %lld\n", sizeof(FusedGate)/1024);
  // KernelGate* kgates = (KernelGate*) malloc(MAX_GATE*sizeof(KernelGate));
  KernelGate* kgates = nullptr;
  
  int fused_idx = 0;
  int shm_idx = 0;
  int shuffle_idx = 0;
  int current_task = 0;
  int batch_start_fused = 0;
  int batch_start_shm = 0;
  int num_batched_task = 0;
  int num_task_shm = 0;
  int batch_id = 0;
  int use_buffer = 0;
  // FusedGate* fgates = (FusedGate*) malloc(4*sizeof(FusedGate));
  // for (int j = 0; j < 4; j++) {
  //   FusedGate gate(circuit.gates[j+1]);
  //   printf("num target: %d\n", gate.num_target);
  //   fgates[j] = gate;
  // }
  // // GateInfo info{circuit.fused_gates_.data(), circuit.num_fused.data(), circuit.shm_gates_.data(), circuit.num_shm.data(), circuit.active_physic_qs.data()};
  // GateInfo info{fgates};
  // apply_gates(info);
  // printf("hhhhh\n");
  
  while (current_task < circuit.task_map.size()) {
    printf("batch id %d\n", batch_id);
    GateInfo info;
    info.batch_id = batch_id;
    info.use_buffer = use_buffer;
    info.fgates = fgates;
    info.kgates = kgates;
    info.fused_idx = fused_idx;
    info.shm_idx = shm_idx;
    num_batched_task = 0;
    for (int i = current_task; i < circuit.task_map.size(); i++) {
      if (circuit.task_map[i] != SHUFFLE) {
        if (circuit.task_map[i] == FUSED) {
          printf("fused: %d\n", fused_idx);
          FusedGate gate(circuit.gates[fused_idx+shuffle_idx]);
          fgates[fused_idx] = gate;
          fused_idx++;
          info.task_map[num_batched_task] = FUSED;
          info.task_num_gates[num_batched_task] = 1;
          num_batched_task++;
          current_task++;
        }
        else if (circuit.task_map[i] == SHM) {
          printf("shm: %d\n", shm_idx);
          for (int j = 0; j < circuit.shm_gates[num_task_shm].size(); j++) {
            kgates[shm_idx] = circuit.shm_gates[num_task_shm][j];
            shm_idx++;
          }
          info.task_map[num_batched_task] = SHM;
          info.task_num_gates[num_batched_task] = circuit.shm_gates[num_task_shm].size();
          info.active_qubits_logical[num_batched_task] = circuit.active_logical_qs[num_task_shm];
          num_batched_task++;
          num_task_shm++;
          current_task++;
        }
      }
      
      if (circuit.task_map[i] == SHUFFLE || i == (circuit.task_map.size() - 1)) {
        // launch previous batched gates if meet a shuffle op or meet the last task:
        for (int j = 0; j < config.num_all_qubits; j++) {
          info.permutation[j] = circuit.permutation_record[shuffle_idx][j];
          info.pos[j] = circuit.pos_record[shuffle_idx][j];
        }
        info.num_tasks = num_batched_task;
        info.nLocalSwaps = circuit.task_map[i] == SHUFFLE ? circuit.local_swap_record[shuffle_idx].size() : 0;
        for (int j = 0; j < info.nLocalSwaps; j++) {
          info.local_swap[j*2] = circuit.local_swap_record[shuffle_idx][j].x;
          info.local_swap[j*2+1] = circuit.local_swap_record[shuffle_idx][j].y;
        }
        apply_gates(info);
        if (circuit.task_map[i] == SHUFFLE) {
          current_task++;
          shuffle_idx++;
          batch_id++;
          use_buffer = 1 - use_buffer;
        }
        break;
      }
    }
  }
  
  return true;
}

bool DistributedSimulator::apply_gates(const GateInfo &info) {
  printf("Apply gates batch %d\n", info.batch_id);
  Context ctx = config.lg_ctx;
  Runtime* runtime = config.lg_hlr;
  int use_buffer = info.use_buffer;
  int num_sv_cpu = cpu_state_vectors.size() / 2;
  for (size_t i = 0; i < num_sv_cpu; i++) {
    std::pair<LogicalRegion, LogicalPartition> cpu_sv = cpu_state_vectors[i];
    std::pair<LogicalRegion, LogicalPartition> gpu_sv =
        gpu_state_vectors[i % gpu_state_vectors.size()];
    std::vector<int> chunk_id;
    // Step 1: move a state vector from DRAM to GPU memory
    {
      if (info.batch_id == 0) {
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
      else {
        
        int n_swap = 0;
        unsigned global_swap_mask = circuit.global_swap_record[info.batch_id-1];
        int n_global = config.num_all_qubits - config.num_local_qubits;
        for (int j = 0; j < n_global; j++) {
          if(global_swap_mask >> j & 1) {
            n_swap++;
          }     
        }
        int n_partition = 1 << n_swap;
        int total_gpu = config.gpus_per_node * config.num_nodes;
        int parallel_degree = n_partition < total_gpu ? n_partition : total_gpu;
        MachineView view;
        view.device_type = MachineView::GPU;
        view.ndims = 1;
        view.dim[0] = parallel_degree;
        view.stride[0] = 1;
        view.start_device_id = 0;

        int num_all2all_groups = total_gpu / parallel_degree;
        int num_batches = n_partition / parallel_degree;
        int cur_all2all_group = 0;
        int cur_batch = 0;
        if (total_gpu > n_partition) {
          cur_all2all_group = i * num_all2all_groups;
        }
        else {
          cur_all2all_group = i / num_batches;
          cur_batch = i % num_batches;
        }
        int idx = 0;
        for (auto &lp : gpu_sv_shuffle_lp[i % gpu_state_vectors.size()].at(parallel_degree)) {
          int field_idx = 0;
          ArgumentMap argmap;
          Domain domain = runtime->get_index_space_domain(ctx, stage_parallel_is.at(parallel_degree));
          
          for (Domain::DomainPointIterator it(domain); it; it++) {
            unsigned my_chunk_idx = 0;
            unsigned pos1 = 0;
            unsigned pos2 = 0;
            unsigned k_ = idx + cur_batch * parallel_degree;
            unsigned q = 0;

            while (q <= n_global) {
              if ((global_swap_mask >> q) & 1) {
                my_chunk_idx |= ((k_ >> pos1) & 1) << q;
                ++pos1;
              }
              else {
                my_chunk_idx |= ((cur_all2all_group >> pos2) & 1) << q;
                ++pos2;
              }
              q++;
            }
            handlers[idx].chunk_id = my_chunk_idx;
            chunk_id.push_back(my_chunk_idx);
            argmap.set_point(*it, TaskArgument(&handlers[idx++], sizeof(DSHandler)));
          }
          IndexLauncher launcher(
              SHUFFLE_TASK_ID, stage_parallel_is.at(parallel_degree), TaskArgument(NULL, 0),
              argmap, Predicate::TRUE_PRED, false /*must*/, 0 /*mapper_id*/, view.hash());
          launcher.add_region_requirement(
              RegionRequirement(lp, 0 /*projection ID*/, WRITE_ONLY,
                                EXCLUSIVE, gpu_sv.first));
          launcher.add_field(field_idx++, FID_DATA);
          
          // add cpu sv as input
          for (int k = 0; k < n_partition; k++) {
            unsigned cpu_lp_idx = 0;
            unsigned pos1 = 0;
            unsigned pos2 = 0;
            unsigned k_ = k;
            unsigned q = 0;

            while (q <= n_global) {
              if ((global_swap_mask >> q) & 1) {
                cpu_lp_idx |= ((k_ >> pos1) & 1) << q;
                ++pos1;
              }
              else {
                cpu_lp_idx |= ((cur_all2all_group >> pos2) & 1) << q;
                ++pos2;
              }
              q++;
            }

            std::vector<LogicalPartition> s_lp = cpu_sv_shuffle_lp[cpu_lp_idx/total_gpu+num_sv_cpu*use_buffer].at(n_partition);
            printf("cpu lp part: %d, %d\n", cpu_lp_idx, cur_batch*total_gpu);
            launcher.add_region_requirement(
                RegionRequirement(s_lp[cur_batch*total_gpu], 0 /*projection ID*/, READ_ONLY,
                                  EXCLUSIVE, cpu_state_vectors[cpu_lp_idx/total_gpu+num_sv_cpu*use_buffer].first, MAP_TO_ZC_MEMORY));
            launcher.add_field(field_idx++, FID_DATA);
          }
          runtime->execute_index_space(ctx, launcher);

          // apply gates
          view.start_device_id += parallel_degree;
          cur_all2all_group++;
        } 
      }
    }
    // Step 2: launch all gate kernels
    {
      ArgumentMap argmap;
      Domain domain = runtime->get_index_space_domain(ctx, parallel_is);
      int idx = 0;
      for (Domain::DomainPointIterator it(domain); it; it++) {
        handlers[idx].chunk_id = info.batch_id == 0? idx+i*(domain.hi()[0]+1) : chunk_id[idx];
        argmap.set_point(*it, TaskArgument(&handlers[idx++], sizeof(DSHandler)));
      }
      IndexLauncher launcher(
          GATE_COMP_TASK_ID, parallel_is, TaskArgument(&info, sizeof(GateInfo)),
          argmap, Predicate::TRUE_PRED, false /*must*/, 0 /*mapper_id*/, PreservedIDs::DataParallelism_GPU);
      launcher.add_region_requirement(
          RegionRequirement(gpu_sv.second, 0 /*projection ID*/, READ_WRITE,
                            EXCLUSIVE, gpu_sv.first));
      launcher.add_field(0, FID_DATA);
      runtime->execute_index_space(ctx, launcher);
    }
    // Step 3: move a state vector from GPU memory to DRAM
    {
      if (info.batch_id == 0) {
        IndexCopyLauncher launcher(parallel_is);
        launcher.add_copy_requirements(
            RegionRequirement(
                gpu_state_vectors[i % gpu_state_vectors.size()].second,
                0 /*projection ID*/, READ_ONLY, EXCLUSIVE,
                gpu_state_vectors[i % gpu_state_vectors.size()].first),
            RegionRequirement(cpu_state_vectors[i+num_sv_cpu*(1-use_buffer)].second, 0 /*projection ID*/,
                              WRITE_DISCARD, EXCLUSIVE, cpu_state_vectors[i+num_sv_cpu*(1-use_buffer)].first,
                              MAP_TO_ZC_MEMORY));
        launcher.add_src_field(0, FID_DATA);
        launcher.add_dst_field(0, FID_DATA);
        runtime->issue_copy_operation(ctx, launcher);
      }
      else {
        int n_swap = 0;
        unsigned global_swap_mask = circuit.global_swap_record[info.batch_id-1];
        int n_global = config.num_all_qubits - config.num_local_qubits;
        for (int j = 0; j < n_global; j++) {
          if(global_swap_mask >> j & 1) {
            n_swap++;
          }     
        }
        int n_partition = 1 << n_swap;
        int total_gpu = config.gpus_per_node * config.num_nodes;
        int parallel_degree = n_partition < total_gpu ? n_partition : total_gpu;
        MachineView view;
        view.device_type = MachineView::GPU;
        view.ndims = 1;
        view.dim[0] = 1;
        view.stride[0] = 1;
        view.start_device_id = 0;

        int gpu_sv_id = 0;

        for (auto &lp : gpu_sv_shuffle_lp[i % gpu_state_vectors.size()].at(1)) {
          int field_idx = 0;
          ArgumentMap argmap;
          Domain domain = runtime->get_index_space_domain(ctx, stage_parallel_is.at(1));
          int idx = 0;
          for (Domain::DomainPointIterator it(domain); it; it++) {
            argmap.set_point(*it, TaskArgument(&handlers[idx++], sizeof(DSHandler)));
          }
          IndexLauncher launcher(
              STORE_TASK_ID, stage_parallel_is.at(1), TaskArgument(NULL, 0),
              argmap, Predicate::TRUE_PRED, false /*must*/, 0 /*mapper_id*/, view.hash());
          launcher.add_region_requirement(
              RegionRequirement(lp, 0 /*projection ID*/, READ_ONLY,
                                EXCLUSIVE, gpu_sv.first));
          launcher.add_field(field_idx++, FID_DATA);
          
          int my_chunk_id = chunk_id[gpu_sv_id];
          // add cpu sv as output
          printf("my chunk id %d\n", my_chunk_id);
          std::vector<LogicalPartition> s_lp = cpu_sv_shuffle_lp[my_chunk_id/total_gpu+num_sv_cpu*(1-use_buffer)].at(1);
          launcher.add_region_requirement(
              RegionRequirement(s_lp[my_chunk_id%total_gpu], 0 /*projection ID*/, WRITE_ONLY,
                                EXCLUSIVE, cpu_state_vectors[my_chunk_id/total_gpu+num_sv_cpu*(1-use_buffer)].first, MAP_TO_ZC_MEMORY));
          launcher.add_field(field_idx++, FID_DATA);
          
          runtime->execute_index_space(ctx, launcher);
          view.start_device_id = (view.start_device_id + 1) % total_gpu;
          gpu_sv_id++;
          printf("ok\n");
        }
        
      }


    }
  }
  return true;
}
