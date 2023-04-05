#include "distributed_simulator.h"
#include "kernel.h"

using namespace sim;
using namespace Legion;

DistributedSimulator::DSHandler
DistributedSimulator::cuda_init_task(Task const *task,
                                     std::vector<PhysicalRegion> const &regions,
                                     Context ctx, Runtime *runtime) {
  cudaStream_t stream;
  cudaStreamCreate(&stream);
  DSConfig const *config = (DSConfig *)task->args;
  DSHandler handle;
  handle.workSpaceSize = (size_t)4 * 1024 * 1024 * 1024; // 1GB work space
  handle.num_local_qubits = config->num_local_qubits;
  handle.num_all_qubits = config->num_all_qubits;
  printf("Num_local_qubits = %lld\n", handle.num_local_qubits);
  custatevecCreate(&handle.statevec);
  {
    // allocate memory for workspace
    Memory gpu_mem = Machine::MemoryQuery(Machine::get_machine())
                         .only_kind(Memory::GPU_FB_MEM)
                         .best_affinity_to(task->target_proc)
                         .first();
    Realm::Rect<1, coord_t> bounds(
        Realm::Point<1, coord_t>(0),
        Realm::Point<1, coord_t>(handle.workSpaceSize - 1));
    std::vector<size_t> field_sizes;
    field_sizes.push_back(sizeof(char));
    Realm::RegionInstance workspaceInst;
    Realm::RegionInstance::create_instance(workspaceInst, gpu_mem, bounds,
                                           field_sizes, 0,
                                           Realm::ProfilingRequestSet())
        .wait();
    handle.workSpace = workspaceInst.pointer_untyped(0, sizeof(char));
  }
  handle.ncclComm = nullptr;
  handle.vecDataType = config->state_vec_data_type;
  cudaMalloc(&handle.threadBias, sizeof(qindex) << THREAD_DEP);
  // for SHM method
  // initControlIdx
  int loIdx_host[10][10][128];
  int shiftAt_host[10][10];
  cudaMalloc(&handle.loIdx_device, sizeof(loIdx_host));
  cudaMalloc(&handle.shiftAt_device, sizeof(shiftAt_host));

  for (int i = 0; i < 128; i++)
    loIdx_host[0][0][i] = (i << 1) ^ ((i & 4) >> 2);

  for (int i = 0; i < 128; i++)
    loIdx_host[1][1][i] = (((i >> 4) << 5) | (i & 15)) ^ ((i & 2) << 3);

  for (int i = 0; i < 128; i++)
    loIdx_host[2][2][i] = (((i >> 5) << 6) | (i & 31)) ^ ((i & 4) << 3);

  for (int q = 3; q < 10; q++)
    for (int i = 0; i < 128; i++)
      loIdx_host[q][q][i] = ((i >> q) << (q + 1)) | (i & ((1 << q) - 1));

  for (int c = 0; c < 10; c++) {
    for (int t = 0; t < 10; t++) {
      if (c == t)
        continue;
      std::vector<int> a[8];
      for (int i = 0; i < 1024; i++) {
        int p = i ^ ((i >> 3) & 7);
        if ((p >> c & 1) && !(p >> t & 1)) {
          a[i & 7].push_back(i);
        }
      }
      for (int i = 0; i < 8; i++) {
        if (a[i].size() == 0) {
          for (int j = i + 1; j < 8; j++) {
            if (a[j].size() == 64) {
              std::vector<int> tmp = a[j];
              a[j].clear();
              for (int k = 0; k < 64; k += 2) {
                a[i].push_back(tmp[k]);
                a[j].push_back(tmp[k + 1]);
              }
              break;
            }
          }
        }
      }
      for (int i = 0; i < 128; i++)
        loIdx_host[c][t][i] = a[i & 7][i / 8];
    }
  }
  cudaMemcpyAsync(handle.loIdx_device, loIdx_host[0][0], sizeof(loIdx_host),
                  cudaMemcpyHostToDevice, stream);
  cudaMemcpyAsync(handle.shiftAt_device, shiftAt_host[0], sizeof(shiftAt_host),
                  cudaMemcpyHostToDevice, stream);

  return handle;
}

void DistributedSimulator::sv_init_task(
    Task const *task, std::vector<PhysicalRegion> const &regions, Context ctx,
    Runtime *runtime) {
  // TODO: implement this function
  printf("SV Init...\n");
  return;
}

void DistributedSimulator::shuffle_task(
    Task const *task, std::vector<PhysicalRegion> const &regions, Context ctx,
    Runtime *runtime) {
  cudaStream_t stream;
  cudaStreamCreate(&stream);
  DSHandler const *handler = (DSHandler *)task->local_args;
  GenericTensorAccessorW gpu_state_vector = helperGetGenericTensorAccessorWO(
      handler->vecDataType, regions[0], task->regions[0], FID_DATA, ctx, runtime);
  GenericTensorAccessorR cpu_sv[MAX_CPU_SV_INPUT];
  for (int i = 0; i < regions.size() - 1; i++) {
    cpu_sv[i] = helperGetGenericTensorAccessorRO(
      handler->vecDataType, regions[i+1], task->regions[i+1], FID_DATA, ctx, runtime);
  }
  Domain domain = runtime->get_index_space_domain(
      ctx, task->regions[1].region.get_index_space());

  for (int i = 0; i < regions.size() - 1; i++) {
    cudaMemcpyAsync((void*)(gpu_state_vector.get_void_ptr() + i * sizeof(handler->vecDataType) * domain.get_volume()),
              cpu_sv[i].get_void_ptr(),
              sizeof(handler->vecDataType) * domain.get_volume(),
              cudaMemcpyHostToDevice, stream);
  }
  return;
}

void DistributedSimulator::store_task(
    Task const *task, std::vector<PhysicalRegion> const &regions, Context ctx,
    Runtime *runtime) {
  cudaStream_t stream;
  cudaStreamCreate(&stream);
  DSHandler const *handler = (DSHandler *)task->local_args;
  GenericTensorAccessorR gpu_state_vector = helperGetGenericTensorAccessorRO(
      handler->vecDataType, regions[0], task->regions[0], FID_DATA, ctx, runtime);
  GenericTensorAccessorW cpu_sv = helperGetGenericTensorAccessorWO(
    handler->vecDataType, regions[1], task->regions[1], FID_DATA, ctx, runtime);

  Domain domain = runtime->get_index_space_domain(
      ctx, task->regions[0].region.get_index_space());
  
  cudaMemcpyAsync(cpu_sv.get_void_ptr(),
            gpu_state_vector.get_void_ptr(),
            sizeof(handler->vecDataType) * domain.get_volume(),
            cudaMemcpyDeviceToHost, stream);
  return;
}

void DistributedSimulator::sv_comp_task(
    Task const *task, std::vector<PhysicalRegion> const &regions, Context ctx,
    Runtime *runtime) {
  GateInfo const *info = (GateInfo *)task->args;
  DSHandler const *handler = (DSHandler *)task->local_args;
  assert(handler->vecDataType == DT_FLOAT_COMPLEX || handler->vecDataType == DT_DOUBLE_COMPLEX);
  cudaDataType_t data_type = handler->vecDataType == DT_FLOAT_COMPLEX ? CUDA_C_32F : CUDA_C_64F;
  custatevecComputeType_t compute_type = handler->vecDataType == DT_FLOAT_COMPLEX ? CUSTATEVEC_COMPUTE_32F : CUSTATEVEC_COMPUTE_64F;
  GenericTensorAccessorW state_vector = helperGetGenericTensorAccessorWO(
      handler->vecDataType, regions[0], task->regions[0], FID_DATA, ctx, runtime);

  cudaStream_t stream;
  cudaStreamCreate(&stream);
  unsigned const nIndexBits = handler->num_local_qubits;
  int fused_idx = 0;
  int shm_idx = 0;
  int fused_start = info->fused_idx;
  int shm_start = info->shm_idx;
  FusedGate* fgates = info->fgates;

  for (int task_id = 0; task_id < info->num_tasks; task_id++) {
    if (info->task_map[task_id] == FUSED) {
      unsigned const nTargets = fgates[fused_idx+fused_start].num_target;
      unsigned const nControls = 0;
      int const adjoint = 0;
      std::vector<int> targets;
      std::vector<int> controls;

      for (int k = 0; k < nTargets; k++) {
        targets.push_back(fgates[fused_idx+fused_start].target[k]);
      }

      qComplex* mat = fgates[fused_idx+fused_start].matrix;

      //   // apply gate
      custatevecApplyMatrix(
          /* custatevecHandle_t */ handler->statevec,
          /* void* */ state_vector.get_void_ptr(),
          /* cudaDataType_t */ data_type,
          /* const uint32_t */ nIndexBits,
          /* const void* */ (void*) &mat[handler->chunk_id*(1<<MAX_KERNEL_SIZE)],
          /* cudaDataType_t */ data_type,
          /* custatevecMatrixLayout_t */ CUSTATEVEC_MATRIX_LAYOUT_ROW,
          /* const int32_t */ adjoint,
          /* const int32_t* */ targets.data(),
          /* const uint32_t */ nTargets,
          /* const int32_t* */ controls.data(),
          /* const int32_t* */ nullptr,
          /* const uint32_t */ nControls,
          /* custatevecComputeType_t */ compute_type,
          /* void* */ handler->workSpace,
          /* size_t */ handler->workSpaceSize);

      fused_idx++;
    }
    else if (info->task_map[task_id] == SHM) {
      unsigned n_local = handler->num_local_qubits;
      unsigned n_qubits = handler->num_all_qubits;
      unsigned blockHot, enumerate;
      qindex relatedQubits = 0;
      qindex logicQubitset = info->active_qubits_logical[task_id];
      for (int i = 0; i < handler->num_all_qubits; i++) {
        if ((logicQubitset >> i) & 1) {
          relatedQubits |= qindex(1) << info->pos[i];
        }
      }
      enumerate = relatedQubits;
      blockHot = (qindex(1) << n_local) - 1 - enumerate;
      qindex threadHot = 0;
      for (int i = 0; i < THREAD_DEP; i++) {
        qindex x = enumerate & (-enumerate);
        threadHot += x;
        enumerate -= x;
      }
      unsigned int hostThreadBias[1 << THREAD_DEP];
      assert((threadHot | enumerate) == relatedQubits);
      for (qindex i = (1 << THREAD_DEP) - 1, j = threadHot; i >= 0;
          i--, j = threadHot & (j - 1)) {
        hostThreadBias[i] = j;
      }

      checkCUDA(cudaMemcpyAsync(handler->threadBias, hostThreadBias,
                                      sizeof(hostThreadBias),
                                      cudaMemcpyHostToDevice, stream));

      std::map<int, int> qubit_group_map;
      int shm = 0;
      int local = 0;
      int global = 0; 
      for (int i = 0; i < n_local; i++) {
        if (relatedQubits & (qindex(1) << i)) {
            qubit_group_map[info->permutation[i]] = shm++;
        } else {
            qubit_group_map[info->permutation[i]] = local++;
        }
      }
      for (int i = n_local; i < n_qubits; i++)
          qubit_group_map[info->permutation[i]] = global++;

      // now we 
      // 1. reset all the gates' target/control qubit to group qubit id
      // 2. generate per-device schedule
      int num_kgates = info->task_num_gates[task_id];
      KernelGate* kgates = info->kgates;
      KernelGate* kernelgates = &kgates[shm_idx+shm_start];
      KernelGate hostGates[num_kgates];
      int* loIdx_device = handler->loIdx_device;
      int* shiftAt_device = handler->shiftAt_device;
      assert(num_kgates < MAX_GATE);
      for (size_t i = 0; i < num_kgates; i++) {
          hostGates[i] = getGate(kernelgates[i], handler->chunk_id, logicQubitset, qubit_group_map, info->pos, n_local);
      }

      qindex gridDim = (qindex(1) << n_local) >> SHARED_MEM_SIZE;
      
      copyGatesToSymbol(hostGates, num_kgates, stream, 0);
      LegionApplyGatesSHM(gridDim, (qComplex *)state_vector.get_void_ptr(), handler->threadBias, n_local,
                    num_kgates, blockHot, enumerate, stream, loIdx_device, shiftAt_device);

      shm_idx += num_kgates;

    }
  }
  
  // do local transpose for next stage
  int const maskLen = 0;
  int maskBitString[] = {};
  int maskOrdering[] = {};
  std::vector<int2> LocalIndexBitSwaps;
  for (int i = 0; i < info->nLocalSwaps; i++) {
    int2 swap;
    swap.x = info->local_swap[i*2];
    swap.y = info->local_swap[i*2+1];
    LocalIndexBitSwaps.push_back(swap);
  }
  custatevecSwapIndexBits(
          handler->statevec, state_vector.get_void_ptr(), data_type, nIndexBits, LocalIndexBitSwaps.data(),
          info->nLocalSwaps, maskBitString, maskOrdering, maskLen);

  return;
  
}

#define IS_SHARE_QUBIT(logicIdx) ((relatedLogicQb >> logicIdx & 1) > 0)
#define IS_LOCAL_QUBIT(logicIdx) (pos[logicIdx] < n_local)
#define IS_HIGH_PART(part_id, logicIdx) ((part_id >> (pos[logicIdx] - n_local) & 1) > 0)

KernelGate sim::getGate(const KernelGate& gate, int part_id, qindex relatedLogicQb, const std::map<int, int>& toID, const unsigned* pos, unsigned n_local) {
    qComplex mat_[2][2] = {make_qComplex(gate.r00, gate.i00), make_qComplex(gate.r01, gate.i01), make_qComplex(gate.r10, gate.i10), make_qComplex(gate.r11, gate.i11)};
    if (gate.controlQubit2 != -1) { // 2 control-qubit
        // Assume no CC-Diagonal
        int c1 = gate.controlQubit;
        int c2 = gate.controlQubit2;
        if (IS_LOCAL_QUBIT(c2) && !IS_LOCAL_QUBIT(c1)) {
            int c = c1; c1 = c2; c2 = c;
        }
        if (IS_LOCAL_QUBIT(c1) && IS_LOCAL_QUBIT(c2)) { // CCU(c1, c2, t)
            if (IS_SHARE_QUBIT(c2) && !IS_SHARE_QUBIT(c1)) {
                int c = c1; c1 = c2; c2 = c;
            }
            return KernelGate(
                gate.type,
                toID.at(c2), 1 - IS_SHARE_QUBIT(c2),
                toID.at(c1), 1 - IS_SHARE_QUBIT(c1),
                toID.at(gate.targetQubit), 1 - IS_SHARE_QUBIT(gate.targetQubit),
                mat_
            );
        } else if (IS_LOCAL_QUBIT(c1) && !IS_LOCAL_QUBIT(c2)) {
            if (IS_HIGH_PART(part_id, c2)) { // CU(c1, t)
              KernelGateType new_type;
              switch (gate.type) {
                case KernelGateType::CCX:
                  new_type = KernelGateType::CNOT;
                  break;
                default:
                    assert(false);
              }   
              return KernelGate(
                  new_type,
                  toID.at(c1), 1 - IS_SHARE_QUBIT(c1),
                  toID.at(gate.targetQubit), 1 - IS_SHARE_QUBIT(gate.targetQubit),
                  mat_
              );
            } else { // ID(t)
                return KernelGate::ID();
            }
        } else { // !IS_LOCAL_QUBIT(c1) && !IS_LOCAL_QUBIT(c2)
            if (IS_HIGH_PART(part_id, c1) && IS_HIGH_PART(part_id, c2)) { // U(t)
                return KernelGate(
                    toU(gate.type),
                    toID.at(gate.targetQubit), 1 - IS_SHARE_QUBIT(gate.targetQubit),
                    mat_
                );
            } else { // ID(t)
                return KernelGate::ID();
            }
        }
    } else if (gate.controlQubit != -1) {
        int c = gate.controlQubit, t = gate.targetQubit;
        if (IS_LOCAL_QUBIT(c) && IS_LOCAL_QUBIT(t)) { // CU(c, t)
            return KernelGate(
                gate.type,
                toID.at(c), 1 - IS_SHARE_QUBIT(c),
                toID.at(t), 1 - IS_SHARE_QUBIT(t),
                mat_
            );
        } else if (IS_LOCAL_QUBIT(c) && !IS_LOCAL_QUBIT(t)) { // U(c)
            switch (gate.type) {
                case KernelGateType::CZ: {
                    if (IS_HIGH_PART(part_id, t)) {
                        return KernelGate(
                            KernelGateType::Z,
                            toID.at(c), 1 - IS_SHARE_QUBIT(c),
                            mat_
                        );
                    } else {
                        return KernelGate::ID();
                    }
                }
                case KernelGateType::CU1: {
                    if (IS_HIGH_PART(part_id, t)) {
                        return KernelGate(
                            KernelGateType::U1,
                            toID.at(c), 1 - IS_SHARE_QUBIT(c),
                            mat_
                        );
                    } else {
                        return KernelGate::ID();
                    }
                }
                case KernelGateType::CRZ: { // GOC(c)
                    qComplex mat[2][2] = {make_qComplex(1, 0), make_qComplex(0,0), make_qComplex(0,0), IS_HIGH_PART(part_id, t) ? mat_[1][1]: mat_[0][0]};
                    return KernelGate(
                        KernelGateType::GOC,
                        toID.at(c), 1 - IS_SHARE_QUBIT(c),
                        mat
                    );
                }
                default: {
                    assert(false);
                }
            }
        } else if (!IS_LOCAL_QUBIT(c) && IS_LOCAL_QUBIT(t)) {
            if (IS_HIGH_PART(part_id, c)) { // U(t)
                return KernelGate(
                    toU(gate.type),
                    toID.at(t), 1 - IS_SHARE_QUBIT(t),
                    mat_
                );
            } else {
                return KernelGate::ID();
            }
        } else { // !IS_LOCAL_QUBIT(c) && !IS_LOCAL_QUBIT(t)
            assert(gate.type == KernelGateType::CZ || gate.type == KernelGateType::CU1 || gate.type == KernelGateType::CRZ);
            if (IS_HIGH_PART(part_id, c)) {
                switch (gate.type) {
                    case KernelGateType::CZ: {
                        if (IS_HIGH_PART(part_id, t)) {
                            qComplex mat[2][2] = {make_qComplex(-1, 0), make_qComplex(0,0), make_qComplex(0,0), make_qComplex(-1, 0)};
                            return KernelGate(
                                KernelGateType::GZZ,
                                0, 0,
                                mat
                            );
                        } else {
                            return KernelGate::ID();
                        }
                    }
                    case KernelGateType::CU1: {
                        if (IS_HIGH_PART(part_id, t)) {
                            qComplex mat[2][2] = {mat_[1][1], make_qComplex(0,0), make_qComplex(0,0), mat_[1][1]};
                            return KernelGate(
                                KernelGateType::GCC,
                                0, 0,
                                mat
                            );
                        }
                    }
                    case KernelGateType::CRZ: {
                        qComplex val = IS_HIGH_PART(part_id, t) ? mat_[1][1]: mat_[0][0];
                        qComplex mat[2][2] = {val, make_qComplex(0,0), make_qComplex(0,0), val};
                        return KernelGate(
                            KernelGateType::GCC,
                            0, 0,
                            mat
                        );
                    }
                    default: {
                        assert(false);
                    }
                }
            } else {
                return KernelGate::ID();
            }
        }
    } else {
        int t = gate.targetQubit;
        if (!IS_LOCAL_QUBIT(t)) { // GCC(t)
            switch (gate.type) {
                case KernelGateType::U1: {
                    if (IS_HIGH_PART(part_id, t)) {
                        qComplex val = mat_[1][1];
                        qComplex mat[2][2] = {val, make_qComplex(0,0), make_qComplex(0,0), val};
                        return KernelGate(KernelGateType::GCC, 0, 0, mat);
                    } else {
                        return KernelGate::ID();
                    }
                }
                case KernelGateType::Z: {
                    if (IS_HIGH_PART(part_id, t)) {
                        qComplex mat[2][2] = {make_qComplex(-1, 0), make_qComplex(0,0), make_qComplex(0,0), make_qComplex(-1, 0)};
                        return KernelGate(KernelGateType::GZZ, 0, 0, mat);
                    } else {
                        return KernelGate::ID();
                    }
                }
                case KernelGateType::S: {
                    if (IS_HIGH_PART(part_id, t)) {
                        qComplex val = make_qComplex(0, 1);
                        qComplex mat[2][2] = {val, make_qComplex(0,0), make_qComplex(0,0), val};
                        return KernelGate(KernelGateType::GII, 0, 0, mat);
                    } else {
                        return KernelGate::ID();
                    }
                }
                case KernelGateType::SDG: {
                    // FIXME
                    if (IS_HIGH_PART(part_id, t)) {
                        qComplex val = make_qComplex(0, -1);
                        qComplex mat[2][2] = {val, make_qComplex(0,0), make_qComplex(0,0), val};
                        return KernelGate(KernelGateType::GCC, 0, 0, mat);
                    } else {
                        return KernelGate::ID();
                    }
                }
                case KernelGateType::T: {
                    if (IS_HIGH_PART(part_id, t)) {
                        qComplex val = mat_[1][1];
                        qComplex mat[2][2] = {val, make_qComplex(0,0), make_qComplex(0,0), val};
                        return KernelGate(KernelGateType::GCC, 0, 0, mat);
                    } else {
                        return KernelGate::ID();
                    }
                }
                case KernelGateType::TDG: {
                    if (IS_HIGH_PART(part_id, t)) {
                        qComplex val = mat_[1][1];
                        qComplex mat[2][2] = {val, make_qComplex(0,0), make_qComplex(0,0), val};
                        return KernelGate(KernelGateType::GCC, 0, 0, mat);
                    } else {
                        return KernelGate::ID();
                    }
                }
                case KernelGateType::RZ: {
                    qComplex val = IS_HIGH_PART(part_id, t) ? mat_[1][1]: mat_[0][0];
                    qComplex mat[2][2] = {val, make_qComplex(0,0), make_qComplex(0,0), val};
                    return KernelGate(KernelGateType::GCC, 0, 0, mat);
                }
                case KernelGateType::ID: {
                    return KernelGate::ID();
                }
                default: {
                    assert(false);
                }
            }
        } else { // IS_LOCAL_QUBIT(t) -> U(t)
            return KernelGate(gate.type, toID.at(t), 1 - IS_SHARE_QUBIT(t), mat_);
        }
    }
}

KernelGateType sim::toU(KernelGateType type) {
    switch (type) {
      case KernelGateType::CCX:
        return KernelGateType::X;
      case KernelGateType::CNOT:
        return KernelGateType::X;
      case KernelGateType::CY:
        return KernelGateType::Y;
      case KernelGateType::CZ:
        return KernelGateType::Z;
      case KernelGateType::CRX:
        return KernelGateType::RX;
      case KernelGateType::CRY:
        return KernelGateType::RY;
      case KernelGateType::CU1:
        return KernelGateType::U1;
      case KernelGateType::CRZ:
        return KernelGateType::RZ;
      default:
          assert(false);
    }
}
