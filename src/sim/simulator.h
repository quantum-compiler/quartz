#ifndef _SIMULATOR_H_
#define _SIMULATOR_H_

#include <cmath>
#include <complex>
#include <cstdint>
#include <type_traits>
#include <vector>

#include <cuComplex.h>        // cuDoubleComplex
#include <cuda_runtime_api.h> // cudaMalloc, cudaMemcpy, etc.
#include <custatevec.h>       // custatevecApplyMatrix

#include "simgate.h"
#include "helper.h" // HANDLE_ERROR, HANDLE_CUDA_ERROR

#define MAX_DEVICES 4
#define MAX_QUBIT 30

namespace sim {
template <typename DT>
class SimulatorCuQuantum {
private:
  static constexpr auto is_float = std::is_same<DT, float>::value;

public:
  static constexpr auto cuDT = is_float ? CUDA_C_32F : CUDA_C_64F;
  static constexpr auto cuCompute =
      is_float ? CUSTATEVEC_COMPUTE_32F : CUSTATEVEC_COMPUTE_64F;

  SimulatorCuQuantum(unsigned nlocal, unsigned nglobal, int ndevices)
      : n_qubits(nlocal + nglobal), n_local(nlocal), n_global(nglobal),
        n_devices(ndevices) {}

  // for simulation
  bool InitState(std::vector<unsigned> const &init_perm);
  bool ApplyGate(Gate<DT> &gate, int device_id);
  bool ApplyShuffle(Gate<DT> &gate);
  bool Destroy();

  custatevecHandle_t handle_[MAX_DEVICES];
  void *d_sv[MAX_DEVICES];
  int n_devices;
  unsigned n_qubits, n_local, n_global;
  int devices[MAX_DEVICES];
  std::vector<unsigned> permutation;
};

} // namespace sim

#endif // _SIMULATOR_H_