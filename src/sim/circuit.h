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

  // APIs for loading circuit from file
  bool load_circuit_from_file(std::string const &filename);

  // APIs for generating circuit from quartz CircuitSeq
  bool compile(quartz::CircuitSeq *seq, quartz::Context *ctx,
               quartz::PythonInterpreter *interpreter, bool use_ilp);

  // APIs for creating gates, currently just read from files

  // API for compile, can move ILP or fusion DP later here
  // void compile();

  // API for running simulation
  void simulate(int ndevices, bool use_mpi);

private:
  // called inside load_circuit_from_file
  bool parse_gate(std::stringstream &ss, std::string const &gate_name);
  void MatMul(unsigned mask, unsigned n_fused, M &res_mat, const M &m1,
              unsigned m_size);
  void MatShuffle(M &res_mat, unsigned n_qubit, const std::vector<int> &perm);

  // called inside compile
  std::vector<std::complex<DT>> FuseGates(const quartz::CircuitSeq &seq,
                                          const std::vector<int> &qubits,
                                          quartz::Context *ctx);

public:
  unsigned num_qubits;
  unsigned n_local, n_global;

  unsigned permutation[MAX_QUBIT];
  std::vector<sim::Gate<DT>> gates;
};
} // namespace qcircuit

} // namespace sim

#endif // _CIRCUIT_H_
