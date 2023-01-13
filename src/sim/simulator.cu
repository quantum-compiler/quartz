#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include <assert.h>

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

  for(int i = 0; i < gate.target.size(); i++){
    auto it =
        find(permutation.begin(), permutation.end(), gate.target[i]);
    assert(it != permutation.end());
    int idx = it - permutation.begin();
    targets.push_back(idx);
  }

  for(int i = 0; i < gate.control.size(); i++){
    auto it =
        find(permutation.begin(), permutation.end(), gate.control[i]);
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
  // std::vector<int2> GlobalIndexBitSwaps;

  // currently only global
  int nGlobalSwaps = 0;
  for (int i = 0; i < n_global; i++) {
    int2 swap;
    if (gate.target[n_global + i] == permutation[n_global + i])
      continue;
    auto it =
        find(permutation.begin(), permutation.end(), gate.target[n_global + i]);
    assert(it != permutation.end());
    int idx = it - permutation.begin();
    swap.x = idx;
    swap.y = n_global + i;
    GlobalIndexBitSwaps.push_back(swap);
    nGlobalSwaps++;
    // update perm
    permutation[idx] = permutation[n_global + i];
    permutation[n_global + i] = gate.target[n_global + i];
  }

  cudaDataType_t data_type = cuDT;
  // move to class
  const custatevecDeviceNetworkType_t deviceNetworkType =
      CUSTATEVEC_DEVICE_NETWORK_TYPE_SWITCH;

  int const maskLen = 0;
  int maskBitString[] = {};
  int maskOrdering[] = {};

  // // local bit swap
  // for (int i = 0; i < n_device; i++)
  // {
  //     HANDLE_CUDA_ERROR(cudaSetDevice(devices[i]));
  //     HANDLE_ERROR(custatevecSwapIndexBits(
  //         handle_[i], d_sv[i], compute_type, n_local, LocallIndexBitSwaps,
  //         nLocalSwaps, maskBitString, maskOrdering, maskLen));
  // }

  // global bit swap
  HANDLE_ERROR(custatevecMultiDeviceSwapIndexBits(handle_,
                                                  n_devices,
                                                  (void **)d_sv,
                                                  data_type,
                                                  n_global,
                                                  n_local,
                                                  GlobalIndexBitSwaps.data(),
                                                  nGlobalSwaps,
                                                  maskBitString,
                                                  maskOrdering,
                                                  maskLen,
                                                  deviceNetworkType));

  return true;
}

// create sv handles and statevectors for each device
template <typename DT>
bool SimulatorCuQuantum<DT>::InitState(std::vector<unsigned> const &init_perm) {
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
  for (int i = 0; i < nDevices; i++)
  {
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
               devices[i0],
               devices[i1]);
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
    HANDLE_CUDA_ERROR(cudaMalloc(&d_sv[iSv], subSvSize * sizeof(cuDoubleComplex)));
    // TODO: add sv init
  }

  for (int i = 0; i < nDevices; i++) {
    HANDLE_CUDA_ERROR(cudaSetDevice(devices[i]));
    HANDLE_ERROR(custatevecCreate(&handle_[i]));
  }

  return true;
}

template <typename DT>
bool SimulatorCuQuantum<DT>::Destroy() {
  for (int i = 0; i < n_devices; i++) {
    HANDLE_CUDA_ERROR(cudaSetDevice(devices[i]));
    HANDLE_ERROR(custatevecDestroy(handle_[i]));
    HANDLE_CUDA_ERROR(cudaFree(d_sv[i]));
  }

  return true;
}

template class SimulatorCuQuantum<double>;
template class SimulatorCuQuantum<float>;

} // namespace sim