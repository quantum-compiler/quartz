#ifndef _CIRCUIT_H_
#define _CIRCUIT_H_

#include <vector>

#include "quartz/pybind/pybind.h"
#include "quartz/simulator/schedule.h"
#include "simgate.h"
#include "simulator.h"

#define MAXR 15

namespace sim {
namespace qcircuit {
template <typename DT> class Circuit {
  using M = std::vector<std::complex<DT>>;

public:
  Circuit(std::vector<unsigned> const &perm, unsigned num_local);
  Circuit(unsigned nqubits, unsigned num_local);
  Circuit(unsigned nqubits, unsigned num_local, int myrank, int nranks);

  // APIs for loading circuit from file
  bool load_circuit_from_file(std::string const &filename);

  // APIs for generating circuit from quartz CircuitSeq
  bool compile(quartz::CircuitSeq *seq, quartz::Context *ctx,
               quartz::PythonInterpreter *interpreter, int ndevices, bool use_ilp);

  // APIs for creating gates, currently just read from files

  // API for compile, can move ILP or fusion DP later here
  // void compile();

  // API for running simulation
  void simulate(bool use_mpi);

private:
  // called inside compile
  bool FuseGates(const quartz::Kernel &kernel, quartz::Context *ctx);
  void MatMul(unsigned mask, unsigned n_fused, M &res_mat, const M &m1,
              unsigned m_size);
  void MatShuffle(M &res_mat, unsigned n_qubit, const std::vector<int> &perm);
  void update_layout(std::vector<int> local_qubits);
  bool getMat_per_device(quartz::Context *ctx, int part_id, quartz::Gate* gate, std::vector<int> qubit_indices, std::vector<ParamType>& params, std::vector<ComplexType> &res, unsigned &mask, std::vector<int>& perm);
  
  KernelGateType toKernel(quartz::GateType type);

public:
  unsigned num_qubits;
  unsigned n_local, n_global;
  int n_devices;
  int myRank, nRanks;
  
  // states info kept for doing compilation
  std::vector<unsigned> permutation;
  std::vector<unsigned> pos;
  std::vector<bool> local_mask;
  std::map<int, int> qubit_group_map_fusion;

  /* set in compile() */
  std::vector<sim::Gate<DT>> gates;
  std::vector<std::vector<sim::Gate<DT>>> gates_per_device;
  // batched gates
  std::vector<std::vector<sim::Gate<qreal>>> fused_gates;
  std::vector<std::vector<KernelGate>> shm_gates;
  std::vector<qindex> active_logical_qs;
  std::vector<SimGateType> task_map;
};
} // namespace qcircuit

} // namespace sim

#endif // _CIRCUIT_H_
