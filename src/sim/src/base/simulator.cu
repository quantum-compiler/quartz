#include <assert.h>
#include <fstream>
#include <iostream>
#include <string>
#include <unistd.h>
#include <vector>

#include "kernel.h"
#include "simulator.h"

namespace sim {
// only support applying gates to local qubits, TODO: support batched gates application
template <typename DT>
bool SimulatorCuQuantum<DT>::ApplyGate(Gate<DT> &gate, int device_id) {
  HANDLE_CUDA_ERROR(cudaSetDevice(devices[device_id]));
  void *extraWorkspace = nullptr;
  size_t extraWorkspaceSizeInBytes = 0;
  unsigned const nIndexBits = n_local;
  unsigned const nTargets = gate.num_target;
  unsigned const nControls = gate.num_control;
  int const adjoint = 0;

  cudaDataType_t data_type = cuDT;
  custatevecComputeType_t compute_type = cuCompute;

  // TODO: get target & control qubit idx from current perm[]
  std::vector<int> targets;
  std::vector<int> controls;

  // TODO: check if targets should be ordered
  printf("Targets: [");
  for (int i = 0; i < gate.target.size(); i++) {
    // auto it = find(permutation.begin(), permutation.end(), gate.target[i]);
    // assert(it != permutation.end());
    // int idx = it - permutation.begin();
    targets.push_back(pos[gate.target[i]]);
    printf("(%d, %d) ", gate.target[i], pos[gate.target[i]]);
  }
  printf("]\n");

  for (int i = 0; i < gate.control.size(); i++) {
    // auto it = find(permutation.begin(), permutation.end(), gate.control[i]);
    // assert(it != permutation.end());
    // int idx = it - permutation.begin();
    controls.push_back(pos[gate.control[i]]);
  }

  // check the size of external workspace
  HANDLE_ERROR(custatevecApplyMatrixGetWorkspaceSize(
      /* custatevecHandle_t */ handle_[device_id],
      /* cudaDataType_t */ data_type,
      /* const uint32_t */ nIndexBits,
      /* const void* */ gate.matrix.data(),
      /* cudaDataType_t */ data_type,
      /* custatevecMatrixLayout_t */ CUSTATEVEC_MATRIX_LAYOUT_ROW,
      /* const int32_t */ adjoint,
      /* const uint32_t */ nTargets,
      /* const uint32_t */ nControls,
      /* custatevecComputeType_t */ compute_type,
      /* size_t* */ &extraWorkspaceSizeInBytes));

  // allocate external workspace if necessary
  if (extraWorkspaceSizeInBytes > 0) {
    HANDLE_CUDA_ERROR(cudaMalloc(&extraWorkspace, extraWorkspaceSizeInBytes));
  }

  // apply gate
  HANDLE_ERROR(custatevecApplyMatrix(
      /* custatevecHandle_t */ handle_[device_id],
      /* void* */ d_sv[device_id],
      /* cudaDataType_t */ data_type,
      /* const uint32_t */ nIndexBits,
      /* const void* */ gate.matrix.data(),
      /* cudaDataType_t */ data_type,
      /* custatevecMatrixLayout_t */ CUSTATEVEC_MATRIX_LAYOUT_ROW,
      /* const int32_t */ adjoint,
      /* const int32_t* */ targets.data(),
      /* const uint32_t */ nTargets,
      /* const int32_t* */ controls.data(),
      /* const int32_t* */ nullptr,
      /* const uint32_t */ nControls,
      /* custatevecComputeType_t */ compute_type,
      /* void* */ extraWorkspace,
      /* size_t */ extraWorkspaceSizeInBytes));

  if (extraWorkspaceSizeInBytes)
    HANDLE_CUDA_ERROR(cudaFree(extraWorkspace));

  return true;
}

template <typename DT>
bool SimulatorCuQuantum<DT>::ApplyKernelGates(
    std::vector<KernelGate> &kernelgates, qindex logicQubitset) {

  // test SHM
  // initialize blockHot (physical non-activate qubuits mask), enumerate,
  // threadBias
  unsigned blockHot, enumerate;
  qindex relatedQubits = 0;
  for (int i = 0; i < n_local; i++) {
    if (logicQubitset >> i & 1) {
      // auto it = find(permutation.begin(), permutation.end(), i);
      // assert(it != permutation.end());
      // int idx = it - permutation.begin();
      relatedQubits |= qindex(1) << pos[i];
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

  for (int i = 0; i < n_devices; i++) {
    HANDLE_CUDA_ERROR(cudaMemcpyAsync(threadBias[i], hostThreadBias,
                                      sizeof(hostThreadBias),
                                      cudaMemcpyHostToDevice, s[i]));
  }

  // get the current logic_qubit <-> qubit_group_idx map:
  // example 6-qubit circuit: 5 local [], 4 shm '', 1 global {}: ['3', '5', '4', 0, '1'] {2}
  // map 3 -> 0 in shm
  //     5 -> 1 in shm
  // .   4 -> 2 in shm
  //     0 -> 0 in local-shm
  //     1 -> 3 in shm
  //     2 -> 0 in global
  std::map<int, int> qubit_group_map;
  int shm = 0;
  int local = 0;
  int global = 0; 
  for (int i = 0; i < n_local; i++) {
    if (relatedQubits & (qindex(1) << i)) {
        qubit_group_map[permutation[i]] = shm++;
    } else {
        qubit_group_map[permutation[i]] = local++;
    }
  }
  for (int i = n_local; i < n_qubits; i++)
      qubit_group_map[permutation[i]] = global++;

  // now we 
  // 1. reset all the gates' target/control qubit to group qubit id
  // 2. generate per-device schedule
  KernelGate hostGates[n_devices * kernelgates.size()];
    assert(kernelgates.size() < MAX_GATE);
    #pragma omp parallel for num_threads(n_devices)
    for (int k = 0; k < n_devices; k++) {
        int myncclrank = device_phy_to_logical.at(myRank * n_devices + k);
        for (size_t i = 0; i < kernelgates.size(); i++) {
           hostGates[k * kernelgates.size() + i] = getGate(kernelgates[i], myncclrank, logicQubitset, qubit_group_map);
        }
    }

  qindex gridDim = (qindex(1) << n_local) >> SHARED_MEM_SIZE;
  for (int k = 0; k < n_devices; k++) {
    HANDLE_CUDA_ERROR(cudaSetDevice(devices[k]));
    // now we have per-device gates
    copyGatesToSymbol(hostGates, kernelgates.size(), s[k], k);
    ApplyGatesSHM(gridDim, (qComplex *)d_sv[k], threadBias[k], n_local,
                  kernelgates.size(), blockHot, enumerate, s[k], k);
  }

  return true;
}

template <typename DT>
bool SimulatorCuQuantum<DT>::ApplyShuffle(Gate<DT> &gate) {
  std::vector<int2> GlobalIndexBitSwaps;
  std::vector<int2> LocalIndexBitSwaps;

  int nGlobalSwaps = n_global;
  int nLocalSwaps = 0;
  printf("Before Perm: [");
  for (int i = 0; i < n_local + n_global; i++) {
    printf("%d,", permutation[i]);
  }
  printf("]\n");

  std::vector<int> new_global_pos;
  int num_swaps = 0;
  for (int i = 0; i < n_global; i++) {
    new_global_pos.push_back(pos[gate.target[i + n_local]]);
    if(pos[gate.target[i + n_local]] < n_local) num_swaps++;
  }
  std::sort(new_global_pos.begin(), new_global_pos.end());
  
  unsigned local_mask = 0;
  unsigned global_mask = 0;
  int j1 = 0;
  for (int i = n_global - 1; i >= 0; i--) {
    if(new_global_pos[i] >= n_local) {
      global_mask |= 1 << (new_global_pos[i] - n_local);
      nGlobalSwaps--;
    }
    else {
      // for cuQuantum-based comm (~global_mask < n_devices)
      for (int j = j1; j < num_swaps; j++) {
          if(!(global_mask >> j & 1)) {
            int2 swap;
            swap.x = new_global_pos[i];
            swap.y = n_local + j;
            GlobalIndexBitSwaps.push_back(swap);
            j1 = j + 1;
            break;
          }
      }
      
      // for nccl-based comm, local transpose
      if(new_global_pos[i] >= (n_local - num_swaps)) {
        local_mask |= 1 << (new_global_pos[i] - n_local + num_swaps);
      }
      else {
        nLocalSwaps++;
        for (int j = num_swaps - 1; j >= 0; j--) {
          if(!(local_mask >> j & 1)) {
            int2 swap;
            swap.x = new_global_pos[i];
            swap.y = n_local - num_swaps + j;
            LocalIndexBitSwaps.push_back(swap);
            local_mask |= 1 << j;
            break;
          }
        }
      }
    }
  }

  printf("Current Perm: [");
  for (int i = 0; i < n_local + n_global; i++) {
    printf("%d,", permutation[i]);
  }
  printf("]\n");

  printf("Shuffle: [");
  for (int i = 0; i < n_local + n_global; i++) {
    printf("%d,", gate.target[i]);
  }
  printf("]\n");

  if (nGlobalSwaps == 0)
    return true;

  cudaDataType_t data_type = cuDT;
  // move to class
  const custatevecDeviceNetworkType_t deviceNetworkType =
      CUSTATEVEC_DEVICE_NETWORK_TYPE_SWITCH;

  int const maskLen = 0;
  int maskBitString[] = {};
  int maskOrdering[] = {};

  // global qubit swap within a node (currently disabled since we want to keep consistent witht the DP on qubit layout)
  // if ((~global_mask) + 1 <= n_devices) {
  if (false) {
    printf("Using cuQuantum for swaps within a node %d\n",
           n_global_within_node);
    // need to perm d_sv according to device_phy_to_logical map
    void** d_sv_;
    for (int i = 0; i < n_devices; i++) {
      unsigned myncclrank = device_phy_to_logical.at(myRank * n_devices + i);
      int i_new = myncclrank & (1 << (n_global_within_node - 1));
      d_sv_[i] = d_sv[i_new]; 
    }
    HANDLE_ERROR(custatevecMultiDeviceSwapIndexBits(
        handle_, n_devices, (void **)d_sv_, data_type, n_global_within_node,
        n_local, GlobalIndexBitSwaps.data(), nGlobalSwaps, maskBitString,
        maskOrdering, maskLen, deviceNetworkType));
    // update perm
    for (int i = 0; i < nGlobalSwaps; i++) {
      std::swap(pos[permutation[GlobalIndexBitSwaps[i].x]], pos[permutation[GlobalIndexBitSwaps[i].y]]);
      std::swap(permutation[GlobalIndexBitSwaps[i].x], permutation[GlobalIndexBitSwaps[i].y]);
    }
    //print layout
    printf("After global Perm: [");
    for (int i = 0; i < n_local + n_global; i++) {
      printf("%d,", permutation[i]);
    }
    printf("]\n");
  } else { // else transpose + all2all + update curr perm
    printf("Using NCCL for cross-node shuffle\n");
    
    // local bit swap
    for (int i = 0; i < n_devices; i++) {
      if (nLocalSwaps == 0)
        break;
      HANDLE_CUDA_ERROR(cudaSetDevice(devices[i]));
      HANDLE_ERROR(custatevecSwapIndexBits(
          handle_[i], d_sv[i], data_type, n_local, LocalIndexBitSwaps.data(),
          nLocalSwaps, maskBitString, maskOrdering, maskLen));
    }
    // update perm/pos
    for (int i = 0; i < nLocalSwaps; i++) {
      std::swap(pos[permutation[LocalIndexBitSwaps[i].x]], pos[permutation[LocalIndexBitSwaps[i].y]]);
      std::swap(permutation[LocalIndexBitSwaps[i].x], permutation[LocalIndexBitSwaps[i].y]);
    }
    //print layout
    printf("After local Perm: [");
    for (int i = 0; i < n_local + n_global; i++) {
      printf("%d,", permutation[i]);
    }
    printf("]\n");

    // update perm/pos after global
    int idx = 0;
    for (int i = 0; i < n_global; i++) {
      if((~global_mask) >> i & 1) {
        std::swap(pos[permutation[idx+n_local-num_swaps]], pos[permutation[i+n_local]]);
        std::swap(permutation[idx+n_local-num_swaps], permutation[i+n_local]);
        idx++;
      }     
    }
    //print layout
    printf("After global Perm: [");
    for (int i = 0; i < n_local + n_global; i++) {
      printf("%d,", permutation[i]);
    }
    printf("]\n");

    int sendsize = subSvSize / (1 << nGlobalSwaps);
    for (int i = 0; i < n_devices; i++) {
      HANDLE_CUDA_ERROR(cudaSetDevice(devices[i]));
      HANDLE_CUDA_ERROR(
          cudaMalloc(&recv_buf[i], subSvSize * sizeof(cuDoubleComplex)));
    }

    cudaEvent_t t_start[MAX_DEVICES], t_end[MAX_DEVICES];
    for (int i = 0; i < n_devices; ++i) {
      // cudaEventCreate(&t_start[i]);
      // cudaEventCreate(&t_end[i]);
      HANDLE_CUDA_ERROR(cudaSetDevice(devices[i]));
      // cudaEventCreateWithFlags(&t_start[i], cudaEventBlockingSync);
      // cudaEventCreateWithFlags(&t_end[i], cudaEventBlockingSync);
      cudaEventCreate(&t_start[i]);
      cudaEventCreate(&t_end[i]);
      cudaEventRecord(t_start[i], s[i]);
    }

    NCCLCHECK(ncclGroupStart());
    for (int i = 0; i < n_devices; ++i) {
      unsigned myncclrank = device_phy_to_logical.at(myRank * n_devices + i);
      all2all(d_sv[i], sendsize, ncclDouble, recv_buf[i], sendsize, ncclDouble,
              comms[i], s[i], ~global_mask, myncclrank);
    }
    NCCLCHECK(ncclGroupEnd());

    for (int i = 0; i < n_devices; i++) {
      HANDLE_CUDA_ERROR(cudaStreamSynchronize(s[i]));
      // profiling
      cudaEventRecord(t_end[i], s[i]);
      HANDLE_CUDA_ERROR(cudaEventSynchronize(t_end[i]));
      float elapsed = 0;
      HANDLE_CUDA_ERROR(cudaEventElapsedTime(&elapsed, t_start[i], t_end[i]));
      cudaEventDestroy(t_start[i]);
      cudaEventDestroy(t_end[i]);
      printf("[NCCL Rank %d] Shuffle Time: %.2fms\n", device_phy_to_logical.at(myRank * n_devices + i),
             elapsed);
    }

    for (int i = 0; i < n_devices; i++) {
      HANDLE_CUDA_ERROR(cudaFree(recv_buf[i]));
    }
  }

  return true;
}

// create sv handles and statevectors for each device on single node
template <typename DT>
bool SimulatorCuQuantum<DT>::InitStateSingle(
    std::vector<unsigned> const &init_perm) {
  int size = init_perm.size();
  for (int i = 0; i < size; i++) {
    permutation.push_back(init_perm[i]);
  }

  if (is_float)
    using statevector_t = cuComplex;
  else
    using statevector_t = cuDoubleComplex;

  int const subSvSize = (1 << n_local);

  int nDevices;
  HANDLE_CUDA_ERROR(cudaGetDeviceCount(&nDevices));
  nDevices = min(nDevices, n_devices);
  printf("Simulating on %d devices\n", nDevices);
  for (int i = 0; i < nDevices; i++) {
    devices[i] = i;
  }

  // check if device ids do not duplicate
  for (int i0 = 0; i0 < nDevices - 1; i0++) {
    for (int i1 = i0 + 1; i1 < nDevices; i1++) {
      if (devices[i0] == devices[i1]) {
        printf("device id %d is defined more than once.\n", devices[i0]);
        return EXIT_FAILURE;
      }
    }
  }

  // enable P2P access
  for (int i0 = 0; i0 < nDevices; i0++) {
    HANDLE_CUDA_ERROR(cudaSetDevice(devices[i0]));
    for (int i1 = 0; i1 < nDevices; i1++) {
      if (i0 == i1) {
        continue;
      }
      int canAccessPeer;
      HANDLE_CUDA_ERROR(
          cudaDeviceCanAccessPeer(&canAccessPeer, devices[i0], devices[i1]));
      if (canAccessPeer == 0) {
        printf("P2P access between device id %d and %d is unsupported.\n",
               devices[i0], devices[i1]);
        return EXIT_SUCCESS;
      }
      HANDLE_CUDA_ERROR(cudaDeviceEnablePeerAccess(devices[i1], 0));
    }
  }

  // define which device stores each sub state vector
  int subSvLayout[nDevices];
  for (int iSv = 0; iSv < nDevices; iSv++) {
    subSvLayout[iSv] = devices[iSv % nDevices];
  }

  printf("The following devices will be used in this sample: \n");
  for (int iSv = 0; iSv < nDevices; iSv++) {
    printf("  sub-SV #%d : device id %d\n", iSv, subSvLayout[iSv]);
  }

  for (int iSv = 0; iSv < nDevices; iSv++) {
    HANDLE_CUDA_ERROR(cudaSetDevice(subSvLayout[iSv]));
    HANDLE_CUDA_ERROR(
        cudaMalloc(&d_sv[iSv], subSvSize * sizeof(cuDoubleComplex)));
    // TODO: add sv init
  }

  for (int i = 0; i < nDevices; i++) {
    HANDLE_CUDA_ERROR(cudaSetDevice(devices[i]));
    HANDLE_ERROR(custatevecCreate(&handle_[i]));
  }

  return true;
}

// init for multinode setting
template <typename DT>
bool SimulatorCuQuantum<DT>::InitStateMulti(
    std::vector<unsigned> const &init_perm) {

  MPICHECK(MPI_Comm_rank(MPI_COMM_WORLD, &myRank));
  MPICHECK(MPI_Comm_size(MPI_COMM_WORLD, &nRanks));
  printf("Num ranks: %d, myrank: %d\n", nRanks, myRank);
  // assert((1<<n_global)== nRanks*n_devices);

  unsigned x = n_devices;
  while (x >>= 1)
    n_global_within_node++;
  subSvSize = (1 << n_local);
  int size = init_perm.size();
  pos.resize(size);
  for (int i = 0; i < size; i++) {
    permutation.push_back(init_perm[i]);
    pos[init_perm[i]] = i;
  }

  for (int i = 0; i < n_devices; i++) {
    devices[i] = i;
  }

  for (int i0 = 0; i0 < n_devices; i0++) {
    HANDLE_CUDA_ERROR(cudaSetDevice(devices[i0]));
    for (int i1 = 0; i1 < n_devices; i1++) {
      if (i0 == i1) {
        continue;
      }
      int canAccessPeer;
      HANDLE_CUDA_ERROR(
          cudaDeviceCanAccessPeer(&canAccessPeer, devices[i0], devices[i1]));
      if (canAccessPeer == 0) {
        printf("P2P access between device id %d and %d is unsupported.\n",
               devices[i0], devices[i1]);
        continue;
      }
      HANDLE_CUDA_ERROR(cudaDeviceEnablePeerAccess(devices[i1], 0));
    }
  }

  for (int i = 0; i < n_devices; i++) {
    HANDLE_CUDA_ERROR(cudaSetDevice(devices[i]));
    HANDLE_CUDA_ERROR(
        cudaMalloc(&d_sv[i], subSvSize * sizeof(cuDoubleComplex)));
    HANDLE_ERROR(custatevecCreate(&handle_[i]));
    HANDLE_CUDA_ERROR(cudaStreamCreate(&s[i]));
  }

  // for SHM method
  initControlIdx(n_devices, s);
  threadBias.resize(n_devices);
  for (int i = 0; i < n_devices; i++) {
    HANDLE_CUDA_ERROR(cudaSetDevice(devices[i]));
    HANDLE_CUDA_ERROR(cudaMalloc(&threadBias[i], sizeof(qindex) << THREAD_DEP));
  }

  if (myRank == 0)
    ncclGetUniqueId(&id);
  MPICHECK(MPI_Bcast((void *)&id, sizeof(id), MPI_BYTE, 0, MPI_COMM_WORLD));

  // init NCCL
  NCCLCHECK(ncclGroupStart());
  for (int i = 0; i < n_devices; i++) {
    HANDLE_CUDA_ERROR(cudaSetDevice(devices[i]));
    NCCLCHECK(ncclCommInitRank(&comms[i], nRanks * n_devices, id,
                               myRank * n_devices + i));
  }
  NCCLCHECK(ncclGroupEnd());

  // init logical <-> device map
  for (int i = 0; i < nRanks * n_devices; i++) {
    device_logical_to_phy[i] = i;
    device_phy_to_logical[i] = i;
  }


  // for profiling TODO: considering using openmp for this parts
  for (int i = 0; i < n_devices; ++i) {
    // cudaEventCreate(&start[i]);
    // cudaEventCreate(&end[i]);
    HANDLE_CUDA_ERROR(cudaSetDevice(devices[i]));
    // cudaEventCreateWithFlags(&start[i], cudaEventBlockingSync);
    // cudaEventCreateWithFlags(&end[i], cudaEventBlockingSync);
    cudaEventCreate(&start[i]);
    cudaEventCreate(&end[i]);
    cudaEventRecord(start[i], s[i]);
  }

  return true;
}

template <typename DT> bool SimulatorCuQuantum<DT>::Destroy() {

  // profiling
  for (int i = 0; i < n_devices; i++) {
    cudaEventRecord(end[i], s[i]);
    HANDLE_CUDA_ERROR(cudaEventSynchronize(end[i]));
    float elapsed = 0;
    HANDLE_CUDA_ERROR(cudaEventElapsedTime(&elapsed, start[i], end[i]));
    cudaEventDestroy(start[i]);
    cudaEventDestroy(end[i]);
    printf("[NCCL Rank %d] Total Simulation Time: %.2fms\n",
           myRank * n_devices + i, elapsed);
  }

  for (int i = 0; i < n_devices; i++) {
    HANDLE_CUDA_ERROR(cudaSetDevice(devices[i]));
    HANDLE_ERROR(custatevecDestroy(handle_[i]));
    HANDLE_CUDA_ERROR(cudaFree(d_sv[i]));
    ncclCommDestroy(comms[i]);
  }

  HANDLE_CUDA_ERROR(cudaDeviceSynchronize());

  printf("[MPI Rank %d]: Destroyed everthing!\n", myRank);

  return true;
}

// private
template <typename DT>
ncclResult_t SimulatorCuQuantum<DT>::all2all(
    void *sendbuff, size_t sendcount, ncclDataType_t senddatatype,
    void *recvbuff, size_t recvcount, ncclDataType_t recvdatatype,
    ncclComm_t comm, cudaStream_t stream, unsigned mask, unsigned myncclrank) {
  ncclGroupStart();
  int ncclnRanks;
  ncclCommCount(comm, &ncclnRanks);
  printf("NCCL comm nRanks: %d, i am %d\n", ncclnRanks, myncclrank);
  for (int i = 0; i < subSvSize / sendcount; ++i) {
    unsigned peer_idx = 0;
    unsigned pos = 0;
    unsigned i_ = i;
    unsigned q = 0;

    while ((1 << q) <= ncclnRanks) {
      if ((mask >> q) & 1) {
        peer_idx |= ((i_ >> pos) & 1) << q;
        ++pos;
      }
      q++;
    }
    peer_idx |= (myncclrank & (~mask));

    printf("I am %d, mask %d, I am sending to %d\n", myncclrank, mask,
           peer_idx);
    
    unsigned peer_phy = device_logical_to_phy.at(peer_idx);

    auto a = NCCLSendrecv(static_cast<std::byte *>(sendbuff) +
                              i * ncclTypeSize(senddatatype) * sendcount,
                          sendcount, senddatatype, peer_phy,
                          static_cast<std::byte *>(recvbuff) +
                              i * ncclTypeSize(recvdatatype) * recvcount,
                          recvcount, comm, stream);
    if (a)
      return a;
  }
  ncclGroupEnd();
  return ncclSuccess;
}

template <typename DT>
ncclResult_t SimulatorCuQuantum<DT>::NCCLSendrecv(
    void *sendbuff, size_t sendcount, ncclDataType_t datatype, int peer,
    void *recvbuff, size_t recvcount, ncclComm_t comm, cudaStream_t stream) {
  ncclGroupStart();
  auto a = ncclSend(sendbuff, sendcount, datatype, peer, comm, stream);
  auto b = ncclRecv(recvbuff, recvcount, datatype, peer, comm, stream);
  ncclGroupEnd();
  if (a || b) {
    if (a)
      return a;
    return b;
  }
  return ncclSuccess;
}

// below is from HyQuas

#define IS_SHARE_QUBIT(logicIdx) ((relatedLogicQb >> logicIdx & 1) > 0)
#define IS_LOCAL_QUBIT(logicIdx) (pos[logicIdx] < n_local)
#define IS_HIGH_PART(part_id, logicIdx) ((part_id >> (pos[logicIdx] - n_local) & 1) > 0)

template <typename DT>
KernelGate SimulatorCuQuantum<DT>::getGate(const KernelGate& gate, int part_id, qindex relatedLogicQb, const std::map<int, int>& toID) const {
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

template <typename DT>
KernelGateType SimulatorCuQuantum<DT>::toU(KernelGateType type) {
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

template class SimulatorCuQuantum<double>;
template class SimulatorCuQuantum<float>;

} // namespace sim
