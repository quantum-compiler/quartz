#include <assert.h>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "simulator.h"

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
bool SimulatorCuQuantum<DT>::ApplyShuffle(Gate<DT> &gate) {
  std::vector<int2> GlobalIndexBitSwaps;
  std::vector<int2> LocalIndexBitSwaps;

  // currently only global
  int nGlobalSwaps = 0;
  int nLocalSwaps = 0;
  int maxGlobal = 0;
  for (int i = 0; i < n_global; i++) {
    int2 swap;
    if (gate.target[n_local + i] == permutation[n_local + i])
      continue;
    auto it =
        find(permutation.begin(), permutation.end(), gate.target[n_local + i]);
    assert(it != permutation.end());
    int idx = it - permutation.begin();
    swap.x = idx;
    swap.y = n_local + i;
    GlobalIndexBitSwaps.push_back(swap);
    nGlobalSwaps++;
    maxGlobal = maxGlobal > i ? maxGlobal : i;
    // update perm
    permutation[idx] = permutation[n_local + i];
    permutation[n_local + i] = gate.target[n_local + i];
    printf("(%d, %d)\n", idx, n_local + i);
  }
  printf("Current Perm: [");
  for (int i = 0; i < n_local; i++) {
    printf("%d,", permutation[i]);
  }
  printf("]\n");

  printf("Shuffle: [");
  for (int i = 0; i < n_local; i++) {
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
  if (1 << (maxGlobal - n_local) < n_devices) {
    HANDLE_ERROR(custatevecMultiDeviceSwapIndexBits(
        handle_, n_devices, (void **)d_sv, data_type, n_global, n_local,
        GlobalIndexBitSwaps.data(), nGlobalSwaps, maskBitString, maskOrdering,
        maskLen, deviceNetworkType));
  } else { // else transpose + all2all + update curr perm
    // get curr highest nGlobalSwaps qubits
    for (int i = 0; i < nGlobalSwaps; i++) {
      int2 swap;
      swap.x = GlobalIndexBitSwaps[i].x;
      swap.y = n_local - nGlobalSwaps + i;
      LocalIndexBitSwaps.push_back(swap);
      nLocalSwaps++;
      // update perm
      unsigned temp = permutation[swap.x];
      permutation[swap.x] = permutation[swap.y];
      permutation[swap.y] = temp;
      printf("(%d, %d)\n", swap.x, swap.y);
    }
    // local bit swap
    for (int i = 0; i < n_device; i++) {
      HANDLE_CUDA_ERROR(cudaSetDevice(devices[i]));
      HANDLE_ERROR(custatevecSwapIndexBits(
          handle_[i], d_sv[i], compute_type, n_local, LocalIndexBitSwaps,
          nLocalSwaps, maskBitString, maskOrdering, maskLen));
    }

    int sendsize = subSvSize / (1 << GlobalIndexBitSwaps.size());
    printf("MyRank %d, sendsize %d\n", myRank, sendsize);
    for (int i = 0; i < n_devices; i++) {
      HANDLE_CUDA_ERROR(cudaSetDevice(i));
      HANDLE_CUDA_ERROR(
          cudaMalloc(&recv_buf[i], subSvSize * sizeof(cuDoubleComplex)));
    }

    unsigned mask = 0;

    NCCLCHECK(ncclGroupStart());
    for (int i = 0; i < n_devices; ++i) {
      unsigned myncclrank = myRank * n_devices + i;
      all2all(d_sv[i], sendsize, ncclFloat, recv_buf[i], sendsize, ncclFloat,
              comms[i], s[i], mask, myncclrank);
    }
    NCCLCHECK(ncclGroupEnd());

    for (int i = 0; i < n_devices; i++)
      HANDLE_CUDA_ERROR(cudaStreamSynchronize(s[i]));

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

  subSvSize = (1 << n_local);
  int size = init_perm.size();
  for (int i = 0; i < size; i++) {
    permutation.push_back(init_perm[i]);
  }

  for (int i = 0; i < n_devices; i++) {
    HANDLE_CUDA_ERROR(cudaSetDevice(i));
    HANDLE_CUDA_ERROR(
        cudaMalloc(&d_sv[i], subSvSize * sizeof(cuDoubleComplex)));
    HANDLE_ERROR(custatevecCreate(&handle_[i]));
    HANDLE_CUDA_ERROR(cudaStreamCreate(&s[i]));
  }

  if (myRank == 0)
    ncclGetUniqueId(&id);
  MPICHECK(MPI_Bcast((void *)&id, sizeof(id), MPI_BYTE, 0, MPI_COMM_WORLD));

  // init NCCL
  NCCLCHECK(ncclGroupStart());
  for (int i = 0; i < n_devices; i++) {
    HANDLE_CUDA_ERROR(cudaSetDevice(i));
    NCCLCHECK(ncclCommInitRank(&comms[i], nRanks * n_devices, id,
                               myRank * n_devices + i));
  }
  NCCLCHECK(ncclGroupEnd());
}

template <typename DT> bool SimulatorCuQuantum<DT>::Destroy() {
  for (int i = 0; i < n_devices; i++) {
    HANDLE_CUDA_ERROR(cudaSetDevice(devices[i]));
    HANDLE_ERROR(custatevecDestroy(handle_[i]));
    HANDLE_CUDA_ERROR(cudaFree(d_sv[i]));
    ncclCommDestroy(comms[i]);
  }

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
  printf("NCCL comm nRanks: %d, i am %d\n", nRanks, myncclrank);
  for (int i = 0; i < subSvSize / sendcount; ++i) {
    unsigned peer_idx = 0;

    peer_idx = (myncclrank & (~mask)) | i;

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
