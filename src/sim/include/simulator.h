#ifndef _SIMULATOR_H_
#define _SIMULATOR_H_

#include <cmath>
#include <complex>
#include <cstdint>
#include <type_traits>
#include <vector>
#include <map>

#include "simgate.h"

#define MPICHECK(cmd)                                                          \
  do {                                                                         \
    int e = cmd;                                                               \
    if (e != MPI_SUCCESS) {                                                    \
      printf("Failed: MPI error %s:%d '%d'\n", __FILE__, __LINE__, e);         \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  } while (0)

#define NCCLCHECK(cmd)                                                         \
  do {                                                                         \
    ncclResult_t r = cmd;                                                      \
    if (r != ncclSuccess) {                                                    \
      printf("Failed, NCCL error %s:%d '%s'\n", __FILE__, __LINE__,            \
             ncclGetErrorString(r));                                           \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  } while (0)

static __inline__ int ncclTypeSize(ncclDataType_t type) {
  switch (type) {
  case ncclInt8:
  case ncclUint8:
    return 1;
  case ncclFloat16:
#if defined(__CUDA_BF16_TYPES_EXIST__)
  case ncclBfloat16:
#endif
    return 2;
  case ncclInt32:
  case ncclUint32:
  case ncclFloat32:
    return 4;
  case ncclInt64:
  case ncclUint64:
  case ncclFloat64:
    return 8;
  default:
    return -1;
  }
}

namespace sim {
template <typename DT> class SimulatorCuQuantum {
private:
  static constexpr auto is_float = std::is_same<DT, float>::value;

public:
  static constexpr auto cuDT = is_float ? CUDA_C_32F : CUDA_C_64F;
  static constexpr auto cuCompute =
      is_float ? CUSTATEVEC_COMPUTE_32F : CUSTATEVEC_COMPUTE_64F;

  SimulatorCuQuantum(unsigned nlocal, unsigned nglobal, int ndevices, int myrank, int nranks)
      : n_qubits(nlocal + nglobal), n_local(nlocal), n_global(nglobal),
        n_devices(ndevices), myRank(myrank), nRanks(nranks) {}

  // for simulation
  bool InitStateSingle(std::vector<unsigned> const &init_perm);
  bool InitStateMulti(std::vector<unsigned> const &init_perm);
  bool ApplyGate(Gate<DT> &gate, int device_id);
  bool ApplyKernelGates(std::vector<KernelGate> &kernelgates,
                        qindex logicQubitset);
  bool ApplyShuffle(Gate<DT> &gate);
  bool Destroy(bool dump_results);

private:
  ncclResult_t all2all(void *sendbuff, size_t sendcount,
                       ncclDataType_t senddatatype, void *recvbuff,
                       size_t recvcount, ncclDataType_t recvdatatype,
                       ncclComm_t comm, cudaStream_t stream, unsigned mask,
                       unsigned myncclrank);
  ncclResult_t NCCLSendrecv(void *sendbuff, size_t sendcount,
                            ncclDataType_t datatype, int peer, void *recvbuff,
                            size_t recvcount, ncclComm_t comm,
                            cudaStream_t stream);
  // from HyQuas
  KernelGate getGate(const KernelGate& gate, int part_id, qindex relatedLogicQb, const std::map<int, int>& toID) const;
  static KernelGateType toU(KernelGateType type);

public:
  custatevecHandle_t handle_[MAX_DEVICES];
  void *d_sv[MAX_DEVICES];
  void *h_sv[MAX_DEVICES];
  // num_devices per node
  int n_devices;
  int subSvSize;
  unsigned n_qubits, n_local, n_global;
  unsigned n_global_within_node = 0;
  int devices[MAX_DEVICES];
  // this will be changed after each shuffle operation
  std::vector<unsigned> permutation;
  std::vector<unsigned> pos;
  cudaStream_t s[MAX_DEVICES];
  // physical id = myNcclRank; this will be changed when we encouter X gates targeting global qubit
  std::map<unsigned, unsigned> device_logical_to_phy;
  std::map<unsigned, unsigned> device_phy_to_logical;

private:
  // nccl, mpi related info
  int myRank, nRanks = 0;
  void *recv_buf[MAX_DEVICES];
  ncclUniqueId id;
  ncclComm_t comms[MAX_DEVICES];
  // timing metrics
  cudaEvent_t start[MAX_DEVICES], end[MAX_DEVICES];
  // for SHM method
  std::vector<unsigned int *> threadBias;
  // for log
  std::vector<float> shuffle_time;
};

} // namespace sim

#endif // _SIMULATOR_H_
