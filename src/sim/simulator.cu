#include <assert.h>
#include <fstream>
#include <iostream>
#include <string>
#include <unistd.h>
#include <vector>

#include "simulator.h"
#include "kernel.h"

namespace sim {
// only support applying gates to local qubits
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
    auto it = find(permutation.begin(), permutation.end(), gate.target[i]);
    assert(it != permutation.end());
    int idx = it - permutation.begin();
    targets.push_back(idx);
    printf("(%d, %d) ", gate.target[i], idx);
  }
  printf("]\n");

  for (int i = 0; i < gate.control.size(); i++) {
    auto it = find(permutation.begin(), permutation.end(), gate.control[i]);
    assert(it != permutation.end());
    int idx = it - permutation.begin();
    controls.push_back(idx);
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
bool SimulatorCuQuantum<DT>::ApplyKernelGates(std::vector<KernelGate> &kernelgates, qindex logicQubitset) {

  // test SHM
  // initialize blockHot (physical non-activate qubuits mask), enumerate, threadBias
  unsigned blockHot, enumerate;
  qindex relatedQubits = 0;
  for (int i = 0; i < n_local; i++) {
    if (logicQubitset >> i & 1){
        auto it = find(permutation.begin(), permutation.end(), i);
        assert(it != permutation.end());
        int idx = it - permutation.begin();
        relatedQubits |= qindex(1) << idx;
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
  for (qindex i = (1 << THREAD_DEP) - 1, j = threadHot; i >= 0; i--, j = threadHot & (j - 1)) {
      hostThreadBias[i] = j;
  }
  for (int i = 0; i < n_devices; i++) {
      HANDLE_CUDA_ERROR(cudaMemcpyAsync(threadBias[i], hostThreadBias, sizeof(hostThreadBias), cudaMemcpyHostToDevice, s[i]));
  }

  qindex gridDim = (qindex(1) << n_local) >> SHARED_MEM_SIZE;
  for (int k = 0; k < n_devices; k++) {
      HANDLE_CUDA_ERROR(cudaSetDevice(devices[k]));
      // currently all the GPU executes same set of gates; TODO: per-gpu schedule
      copyGatesToSymbol(kernelgates.data(), kernelgates.size(), s[k], 0);
      ApplyGatesSHM(gridDim, (qComplex*)d_sv[k], threadBias[k], n_local, kernelgates.size(), blockHot, enumerate, s[k], k);
  }

  return true;

}

template <typename DT>
bool SimulatorCuQuantum<DT>::ApplyShuffle(Gate<DT> &gate) {
  std::vector<int2> GlobalIndexBitSwaps;
  std::vector<int2> LocalIndexBitSwaps;

  // currently only global
  int nGlobalSwaps = 0;
  int nLocalSwaps = 0;
  int maxGlobal = 0;
  printf("Before Perm: [");
  for (int i = 0; i < n_local + n_global; i++) {
    printf("%d,", permutation[i]);
  }
  printf("]\n");
  
  std::vector<int> from_idx;
  std::vector<int> to;
  for (int i = 0; i < n_global; i++) {
    auto it = find(gate.target.begin() + n_local, gate.target.end(),
                   permutation[n_local + i]);
    if (it == gate.target.end()) {
      from_idx.push_back(n_local + i);
    }
    auto it2 = find(permutation.begin() + n_local, permutation.end(),
                    gate.target[n_local + i]);
    if (it2 == permutation.end()) {
      to.push_back(gate.target[n_local + i]);
    }
  }
  assert(from_idx.size() == to.size());

  for (int i = 0; i < to.size(); i++) {
    int2 swap;
    auto it = find(permutation.begin(), permutation.end(), to[i]);
    assert(it != permutation.end());
    int idx = it - permutation.begin();
    swap.x = idx;
    swap.y = from_idx[i];
    GlobalIndexBitSwaps.push_back(swap);
    nGlobalSwaps++;
    maxGlobal = maxGlobal > from_idx[i] ? maxGlobal : from_idx[i];
    // update perm
    permutation[idx] = permutation[swap.y];
    permutation[swap.y] = to[i];
    printf("(%d, %d)\n", idx, swap.y);
  }
  maxGlobal -= n_local;

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

  // global bit swap within a node
  if ((1 << (maxGlobal + 1)) <= n_devices) {
    printf("Using cuQuantum for swaps within a node %d\n",
           n_global_within_node);
    HANDLE_ERROR(custatevecMultiDeviceSwapIndexBits(
        handle_, n_devices, (void **)d_sv, data_type, n_global_within_node,
        n_local, GlobalIndexBitSwaps.data(), nGlobalSwaps, maskBitString,
        maskOrdering, maskLen, deviceNetworkType));
  } else { // else transpose + all2all + update curr perm
    printf("Using NCCL for cross-node shuffle\n");
    // get curr highest nGlobalSwaps qubits
    std::vector<int> high_bits;
    for (int i = 0; i < nGlobalSwaps; i++) {
      high_bits.push_back(n_local - nGlobalSwaps + i);
    }
    for (int i = 0; i < nGlobalSwaps; i++) {
      auto it =
          find(high_bits.begin(), high_bits.end(), GlobalIndexBitSwaps[i].x);
      if (it != high_bits.end()) {
        high_bits.erase(it);
        continue;
      }
      int2 swap;
      swap.x = GlobalIndexBitSwaps[i].x;
      swap.y = high_bits.back();
      high_bits.pop_back();
      LocalIndexBitSwaps.push_back(swap);
      nLocalSwaps++;
      // update perm
      unsigned temp = permutation[swap.x];
      permutation[swap.x] = permutation[swap.y];
      permutation[swap.y] = temp;
      printf("(%d, %d)\n", swap.x, swap.y);
    }
    // local bit swap
    for (int i = 0; i < n_devices; i++) {
      if (nLocalSwaps == 0)
        break;
      HANDLE_CUDA_ERROR(cudaSetDevice(devices[i]));
      HANDLE_ERROR(custatevecSwapIndexBits(
          handle_[i], d_sv[i], data_type, n_local, LocalIndexBitSwaps.data(),
          nLocalSwaps, maskBitString, maskOrdering, maskLen));
    }

    int sendsize = subSvSize / (1 << GlobalIndexBitSwaps.size());
    printf("MyRank %d, sendsize %d\n", myRank, sendsize);
    for (int i = 0; i < n_devices; i++) {
      HANDLE_CUDA_ERROR(cudaSetDevice(devices[i]));
      HANDLE_CUDA_ERROR(
          cudaMalloc(&recv_buf[i], subSvSize * sizeof(cuDoubleComplex)));
    }

    unsigned mask = 0;
    for (int i = 0; i < nGlobalSwaps; i++) {
      mask |= (1 << (GlobalIndexBitSwaps[i].y - n_local));
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
      unsigned myncclrank = myRank * n_devices + i;
      all2all(d_sv[i], sendsize, ncclDouble, recv_buf[i], sendsize, ncclDouble,
              comms[i], s[i], mask, myncclrank);
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
      printf("[NCCL Rank %d] Shuffle Time: %.2fms\n", myRank * n_devices + i,
             elapsed);
    }

    for (int i = 0; i < n_devices; i++) {
      HANDLE_CUDA_ERROR(cudaFree(recv_buf[i]));
    }
  }

  // all2all
  // int sendsize = subSvSize / (1 << GlobalIndexBitSwaps.size());
  // printf("MyRank %d, sendsize %d\n", myRank, sendsize);
  // for (int i = 0; i < n_devices; i++) {
  //   HANDLE_CUDA_ERROR(cudaSetDevice(i));
  //   HANDLE_CUDA_ERROR(
  //       cudaMalloc(&recv_buf[i], subSvSize * sizeof(cuDoubleComplex)));
  // }

  // unsigned mask = 0;

  // NCCLCHECK(ncclGroupStart());
  // for(int i = 0; i < n_devices; ++i){
  //   unsigned myncclrank =  myRank * n_devices + i;
  //   all2all(d_sv[i], sendsize, ncclFloat, recv_buf[i], sendsize, ncclFloat,
  //   comms[i], s[i], mask, myncclrank);
  // }
  // NCCLCHECK(ncclGroupEnd());

  // for (int i = 0; i < n_devices; i++)
  //   HANDLE_CUDA_ERROR(cudaStreamSynchronize(s[i]));

  // for (int i = 0; i < n_devices; i++) {
  //    HANDLE_CUDA_ERROR(cudaFree(recv_buf[i]));
  // }

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
  for (int i = 0; i < size; i++) {
    permutation.push_back(init_perm[i]);
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

    auto a = NCCLSendrecv(static_cast<std::byte *>(sendbuff) +
                              i * ncclTypeSize(senddatatype) * sendcount,
                          sendcount, senddatatype, peer_idx,
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

template class SimulatorCuQuantum<double>;
template class SimulatorCuQuantum<float>;

} // namespace sim
