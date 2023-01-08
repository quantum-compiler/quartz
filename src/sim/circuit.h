#ifndef _CIRCUIT_H_
#define _CIRCUIT_H_

#include <vector>

#include "simgate.h"
#include "simulator.h"
#include "quartz/simulator/schedule.h"

#define MAXR 15

namespace sim
{
  namespace qcircuit
  {
    template <typename DT>
    class Circuit
    {
    public:
      Circuit(std::vector<unsigned> const &perm, unsigned num_local);
      Circuit(unsigned nqubits, unsigned num_local);

      // APIs for loading circuit from file
      bool load_circuit_from_file(std::string const &filename);

      // APIs for generating circuit from quartz CircuitSeq
      bool compile(quartz::CircuitSeq *seq, quartz::Context *ctx);

      // APIs for creating gates, currently just read from files

      // API for compile, can move ILP or fusion DP later here
      // void compile();

      // API for running simulation
      void simulate(int ndevices);

    private:
      // called inside load_circuit_from_file
      bool parse_gate(std::stringstream &ss, std::string const &gate_name);

      // called inside compile
      std::vector<std::complex<DT>> FuseGates(const quartz::CircuitSeq& seq, const std::vector<int>& qubits, quartz::Context *ctx);

    public:
      unsigned num_qubits;
      unsigned n_local, n_global;

      unsigned permutation[MAX_QUBIT];
      std::vector<sim::Gate<DT>> gates;
    };


    template <typename DT>
    using M = std::vector<std::complex<DT>>;

    template <typename DT>
    void MatMul(
        unsigned mask, unsigned n_fused, M<DT>& res_mat, const M<DT>& m1, unsigned m_size) {
        // expand m1
        unsigned n1 = unsigned{1} << m_size;
        unsigned n = unsigned{1} << n_fused;
        // std::vector<std::complex<DT>> res_mat;
        // res_mat.resize(n*n);
        // for (unsigned i = 0; i < n; ++i) {
        //   res_mat[(n * i + i)] = std::complex<DT> (1, 0);
        // }
        std::vector<std::complex<DT>> temp_mat = res_mat;

        for (unsigned i = 0; i < n; ++i) {
          unsigned i_ = i;
          unsigned row_m1 = 0;
          unsigned pos = 0;
          for (unsigned q = 0; q < n; ++q) {
            if ((mask >> q) & 1) {
              row_m1 |= ((i_ >> q) & 1) << pos;
              ++pos;
            }
          }

          for (unsigned j = 0; j < n; ++j) {
            std::complex<DT> re = std::complex<DT> (0, 0);
            for (unsigned k = 0; k < n1; ++k) {
              // row res_mat
              unsigned row_res = 0;
              unsigned k_ = k;
              pos = 0;
              for (unsigned q = 0; q < n; ++q) {
                if ((mask >> q) & 1) {
                  row_res |= ((k_ >> pos) & 1) << q;
                  ++pos;
                }
              }
              std::complex<DT> v1 = m1[(n1 * row_m1 + k)];
              std::complex<DT> v2 = temp_mat[(n * row_res + j)];

              re += v1 * v2;
            }

            res_mat[(n * i + j)] = re;
          }
        }
    }
  } // namespace qcircuit

} // namespace sim

#endif // _CIRCUIT_H_