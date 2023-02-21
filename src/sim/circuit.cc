#include <algorithm>
#include <assert.h>
#include <fstream>
#include <iostream>
#include <string>
#include <unordered_map>
#include <vector>

#include "circuit.h"
#include "quartz/gate/gate_utils.h"
#include "quartz/parser/qasm_parser.h"
#include "quartz/tasograph/tasograph.h"

namespace quartz {

// check type
int num_iterations_by_heuristics(CircuitSeq *seq, int num_local_qubits,
                                 std::vector<std::vector<bool>> &local_qubits) {
  int num_qubits = seq->get_num_qubits();
  std::unordered_map<CircuitGate *, bool> executed;
  // No initial configuration -- all qubits are global.
  std::vector<bool> local_qubit(num_qubits, false);
  int num_iterations = 0;
  while (true) {
    bool all_done = true;
    std::vector<bool> executable(num_qubits, true);
    for (auto &gate : seq->gates) {
      if (gate->gate->is_quantum_gate() && !executed[gate.get()]) {
        bool ok = true;
        for (auto &output : gate->output_wires) {
          if (!executable[output->index]) {
            ok = false;
          }
        }
        if (!gate->gate->is_sparse()) {
          for (auto &output : gate->output_wires) {
            if (!local_qubit[output->index]) {
              ok = false;
            }
          }
        }
        if (ok) {
          // execute
          /*for (auto &output : gate->output_nodes) {
            std::cout << output->index << " ";
          }
          std::cout << "execute\n";*/
          executed[gate.get()] = true;
        } else {
          // not executable, block the qubits
          all_done = false;
          for (auto &output : gate->output_wires) {
            executable[output->index] = false;
          }
        }
      }
    }
    if (all_done) {
      break;
    }
    num_iterations++;
    // count global and local gates
    std::vector<bool> first_unexecuted_gate(num_qubits, false);
    std::vector<int> local_gates(num_qubits, 0);
    std::vector<int> global_gates(num_qubits, 0);
    bool first = true;
    for (auto &gate : seq->gates) {
      if (gate->gate->is_quantum_gate() && !executed[gate.get()]) {
        bool local = true;
        if (!gate->gate->is_sparse()) {
          for (auto &output : gate->output_wires) {
            if (!local_qubit[output->index]) {
              local = false;
            }
          }
        }
        for (auto &output : gate->output_wires) {
          if (local) {
            local_gates[output->index]++;
          } else {
            global_gates[output->index]++;
          }
          if (first) {
            first_unexecuted_gate[output->index] = true;
          }
        }
        first = false;
      }
    }
    auto cmp = [&](int a, int b) {
      if (first_unexecuted_gate[b])
        return false;
      if (first_unexecuted_gate[a])
        return true;
      if (global_gates[a] != global_gates[b]) {
        return global_gates[a] > global_gates[b];
      }
      if (local_gates[a] != local_gates[b]) {
        return local_gates[a] > local_gates[b];
      }
      // Use the qubit index as a final tiebreaker.
      return a < b;
    };
    std::vector<int> candidate_indices(num_qubits, 0);
    for (int i = 0; i < num_qubits; i++) {
      candidate_indices[i] = i;
      local_qubit[i] = false;
    }
    std::sort(candidate_indices.begin(), candidate_indices.end(), cmp);
    std::cout << "Iteration " << num_iterations << ": {";
    for (int i = 0; i < num_local_qubits; i++) {
      local_qubit[candidate_indices[i]] = true;
      std::cout << candidate_indices[i];
      if (i < num_local_qubits - 1) {
        std::cout << ", ";
      }
    }
    std::cout << "}" << std::endl;
    local_qubits.push_back(local_qubit);
  }
  std::cout << num_iterations << " iterations." << std::endl;
  return num_iterations;
}
} // namespace quartz

namespace sim {

/* public functions for qcircuit::Circuit */
template <typename DT>
qcircuit::Circuit<DT>::Circuit(std::vector<unsigned> const &perm,
                               unsigned nlocal) {
  num_qubits = perm.size();
  for (int i = 0; i < num_qubits; i++) {
    permutation[i] = perm[i];
  }
  n_local = nlocal;
  n_global = num_qubits - nlocal;
}

template <typename DT>
qcircuit::Circuit<DT>::Circuit(unsigned nqubits, unsigned nlocal) {
  num_qubits = nqubits;
  for (int i = 0; i < num_qubits; i++) {
    permutation[i] = i;
  }
  n_local = nlocal;
  n_global = num_qubits - nlocal;
}

template <typename DT>
bool qcircuit::Circuit<DT>::load_circuit_from_file(
    std::string const &filename) {
  std::fstream input(filename, std::ios::in);

  unsigned nqubits = 0;
  unsigned ngates = 0;
  // line[0]: nqubits, ngates
  input >> nqubits;
  assert(nqubits == num_qubits);

  input >> ngates;
  gates.reserve(ngates);

  std::string line;
  line.reserve(128);
  std::string gate_name;
  gate_name.reserve(16);

  for (unsigned i = 0; i < ngates; i++) {
    std::stringstream ss(line);
    // normal gate lines: gate_name, target_qubits, params
    // controled gates: c n_control control_qubits gate_name target_qubits
    // params shuffle gate: sf new_perm
    ss >> gate_name;
    if (!parse_gate(ss, gate_name)) {
      return false;
    }
  }

  return true;
}

// TODO: add mode to use heuristics or ILP
template <typename DT>
bool qcircuit::Circuit<DT>::compile(quartz::CircuitSeq *seq,
                                    quartz::Context *ctx,
                                    quartz::PythonInterpreter *interpreter,
                                    bool use_ilp) {
  // 1. ILP/heuristics
  std::vector<std::vector<bool>> local_qubits;
  if (!use_ilp)
    int result = num_iterations_by_heuristics(seq, n_local, local_qubits);
  else {
    local_qubits =
        compute_local_qubits_with_ilp(*seq, n_local, ctx, interpreter);
  }
  // fprintf(fout, " %d", result);

  // 2. DP, fuse gates and add shuffle gates
  int idx = 0;
  auto schedules = get_schedules(*seq, local_qubits, ctx);
  for (auto &schedule : schedules) {
    // add shuffle gate
    std::vector<int> target;
    std::vector<int> global;
    for (int i = 0; i < num_qubits; i++) {
      if (local_qubits[idx][i]) {
        target.push_back(i);
      } else {
        global.push_back(i);
      }
    }

    for (int i = 0; i < n_global; i++) {
      target.push_back(global[i]);
    }
    Gate<DT> gate{SHUFFLE, num_qubits, 0, target, {}, {}, {}};
    gates.push_back(gate);

    schedule.compute_kernel_schedule(
        {0, 10.4, 10.400001, 10.400002, 11, 40, 46, 66});
    std::cout << "cost = " << schedule.cost_ << std::endl;
    // schedule.print_kernel_schedule();
    int num_kernels = schedule.kernels.size();
    for (int i = 0; i < num_kernels; i++) {
      SimGateType g_type = FUSED;
      unsigned n_target = schedule.kernel_qubits[i].size();
      unsigned n_control = 0;
      std::vector<int> target;
      std::vector<int> control;
      // TODO: use template for matrix type, to be compatible with Legion-based
      // version std::vector<std::complex<DT>> mat;
      std::cout << "Fusing Kernel " << i << ": qubits [";
      for (int j = 0; j < (int)schedule.kernel_qubits[i].size(); j++) {
        std::cout << schedule.kernel_qubits[i][j];
        target.push_back(schedule.kernel_qubits[i][j]);
        if (j != (int)schedule.kernel_qubits[i].size() - 1) {
          std::cout << ", ";
        }
      }
      std::cout << "], gates ";
      std::cout << schedule.kernels[i].to_string() << std::endl;
      // mat = FuseGates<DT>(kernels);
      auto mat = FuseGates(schedule.kernels[i], schedule.kernel_qubits[i], ctx);

      // add fused gates kernels
      Gate<DT> gate{g_type, n_target, n_control, target, control, {}, mat};
      gates.push_back(gate);
    }

    idx++;
  }

  printf("Compilation Done! Start simulating...\n");

  return true;
}

template <typename DT>
void qcircuit::Circuit<DT>::simulate(int ndevices, bool use_mpi) {
  using Simulator = SimulatorCuQuantum<DT>;
  Simulator simulator(n_local, n_global, ndevices);
  std::vector<unsigned> init_perm;
  for (int i = 0; i < n_local + n_global; i++) {
    init_perm.push_back(i);
  }
  if (!use_mpi)
    simulator.InitStateSingle(init_perm);
  else
    simulator.InitStateMulti(init_perm);

  printf("Init State Vectors!\n");

  printf("Test SHM\n");
  std::vector<KernelGate> kernelgates;
  qComplex mat[2][2] = {make_qComplex(0, 0), make_qComplex(0, -1), make_qComplex(0, 1), make_qComplex(0, 0)};
  for (int i=0; i<50; i++){
    KernelGate kg(KernelGateType::Y, 5, 0, mat);
    kernelgates.push_back(kg);
  }
  // just for test
  qindex active_qubits_logical = 0;
  for (int i = 0; i < SHARED_MEM_SIZE; i++) {
    active_qubits_logical |= qindex(1) << i;
  }
  simulator.ApplyKernelGates(kernelgates, active_qubits_logical);
  simulator.Destroy();
  printf("Destroyed the simulator\n");
  return;

  int index = 0;
  while (index < gates.size()) {
    Gate<DT> gate = gates[index++];
    if (gate.gtype == SHUFFLE) {
      simulator.ApplyShuffle(gate);
    } else {
      for (int i = 0; i < ndevices; i++)
        simulator.ApplyGate(gate, i);
    }
  }
  printf("Finish Simulating!\n");
  simulator.Destroy();
  printf("Destroyed the simulator\n");
}

/* private functiona for qcircuit::Circuit */
// TODO: add more gates
template <typename DT>
bool qcircuit::Circuit<DT>::parse_gate(std::stringstream &ss,
                                       std::string const &gate_name) {
  int q0, q1;
  DT isqrt2 = 1 / std::sqrt(2);
  unsigned n_control;
  std::vector<int> control;
  SimGateType g_type = NORMAL;
  std::string tgate_name;

  if (gate_name == "c") {
    ss >> n_control;
    g_type = CONTROL;
    for (int i = 0; i < n_control; i++) {
      int c_qubit;
      ss >> c_qubit;
      control.push_back(c_qubit);
    }
    tgate_name.reserve(16);
    ss >> tgate_name;
  }

  std::string gname = n_control == 0 ? gate_name : tgate_name;

  if (gname == "h") {
    ss >> q0;
    Gate<DT> gate{g_type,
                  1,
                  n_control,
                  {q0},
                  control,
                  {},
                  {{isqrt2, 0}, {isqrt2, 0}, {isqrt2, 0}, {-isqrt2, 0}}};
    gates.push_back(gate);
  } else if (gname == "t") {
    ss >> q0;
    Gate<DT> gate{g_type,
                  1,
                  n_control,
                  {q0},
                  control,
                  {},
                  {{1, 0}, {0, 0}, {0, 0}, {isqrt2, isqrt2}}};
    gates.push_back(gate);
  } else if (gname == "x") {
    ss >> q0;
    Gate<DT> gate{g_type,
                  1,
                  n_control,
                  {q0},
                  control,
                  {},
                  {{0, 0}, {1, 0}, {1, 0}, {0, 0}}};
    gates.push_back(gate);
  } else if (gate_name == "y") {
    ss >> q0;
    Gate<DT> gate{g_type,
                  1,
                  n_control,
                  {q0},
                  control,
                  {},
                  {{0, 0}, {0, -1}, {0, 1}, {0, 0}}};
    gates.push_back(gate);
  } else if (gname == "z") {
    ss >> q0;
    Gate<DT> gate{g_type,
                  1,
                  n_control,
                  {q0},
                  control,
                  {},
                  {{1, 0}, {0, 0}, {0, 0}, {-1, 0}}};
    gates.push_back(gate);
  } else if (gname == "cz") {
    ss >> q0 >> q1;
    Gate<DT> gate{
        g_type, 2, n_control, {q0}, control, {}, {1, 0, 0, 0, 0, 0, 0,  0,
                                                  0, 0, 1, 0, 0, 0, 0,  0,
                                                  0, 0, 0, 0, 1, 0, 0,  0,
                                                  0, 0, 0, 0, 0, 0, -1, 0}};
    gates.push_back(gate);
  } else if (gname == "cnot" || gate_name == "cx") {
    ss >> q0 >> q1;
    Gate<DT> gate{
        g_type, 2, n_control, {q0, q1}, control, {}, {1, 0, 0, 0, 0, 0, 0, 0,
                                                      0, 0, 0, 0, 0, 0, 1, 0,
                                                      0, 0, 0, 0, 1, 0, 0, 0,
                                                      0, 0, 1, 0, 0, 0, 0, 0}};
    gates.push_back(gate);
  } else if (gname == "shuffle") {
    std::vector<int> target;
    for (int i = 0; i < num_qubits; i++) {
      int qubit;
      ss >> qubit;
      target.push_back(qubit);
    }
    Gate<DT> gate{SHUFFLE, num_qubits, 0, target, {}, {}, {}};
    gates.push_back(gate);
  } else {
    return false;
  }

  return true;
}

// an naive impl for fusing gates: can be more efficient
template <typename DT>
std::vector<std::complex<DT>>
qcircuit::Circuit<DT>::FuseGates(const quartz::CircuitSeq &seq,
                                 const std::vector<int> &kernel_qubits,
                                 quartz::Context *ctx) {
  // expand all gates to 2^n_target * 2^n_target matrix: identity X mat =>
  // matrix shuffle
  std::vector<int> qubits = kernel_qubits;
  unsigned ksize = unsigned(1) << qubits.size();
  unsigned vec_size = ksize * ksize;
  // std::vector<std::complex<DT>> res_mat(vec_size, std::complex<DT> (1, 0));
  std::vector<std::complex<DT>> res_mat;
  res_mat.resize(vec_size);
  for (unsigned i = 0; i < ksize; ++i) {
    res_mat[(ksize * i + i)] = std::complex<DT>(1, 0);
  }

  printf("Fused Kernel Size: %d\n", ksize);
  // reorder qubits to increasing order
  std::sort(qubits.begin(), qubits.end());

  for (int i = 0; i < seq.gates.size(); i++) {
    printf("Gate %d: [", i);
    std::vector<int> qubit_indices;
    std::vector<ParamType> params;
    for (const auto &input_wire : seq.gates[i]->input_wires) {
      if (input_wire->is_qubit()) {
        qubit_indices.push_back(input_wire->index);
        printf("%d, ", input_wire->index);
      } else {
        params.push_back(ctx->input_parameters[input_wire->index]);
      }
    }
    printf("]\n");

    // getting mask and qubit perm
    std::vector<int> qperm;
    qperm.resize(qubit_indices.size());
    std::vector<int> ordered_qubit = qubit_indices;
    std::sort(ordered_qubit.begin(), ordered_qubit.end());
    unsigned mask = 0;
    // get qubit mask for the gate to be fused
    for (int t = 0; t < qubit_indices.size(); t++) {
      int it = std::find(qubits.begin(), qubits.end(), qubit_indices[t]) -
               qubits.begin();
      mask |= (1 << it);
      int it2 = std::find(qubit_indices.begin(), qubit_indices.end(),
                          ordered_qubit[t]) -
                qubit_indices.begin();
      qperm[it2] = t;
    }

    printf("qperm:[\n");
    for (int e = 0; e < qperm.size(); e++) {
      printf("%d, ", qperm[e]);
    }
    printf("\n");

    printf("MatShuffle and MatMul\n");

    auto *m = seq.gates[i]->gate->get_matrix(params);
    std::vector<std::complex<double>> temp_d = m->flatten();
    std::vector<std::complex<DT>> temp(temp_d.begin(), temp_d.end());
    // matrix shuffle: to increasing order
    printf("matrix to be fused:\n");
    for (int i = 0; i < temp.size(); i++) {
      printf("(%f, %f)", temp[i].real(), temp[i].imag());
    }
    printf("\n");
    MatShuffle(temp, qubit_indices.size(), qperm);
    MatMul(mask, qubits.size(), res_mat, temp, qubit_indices.size());
  }

  return res_mat;
}

template <typename DT>
void qcircuit::Circuit<DT>::MatMul(unsigned mask, unsigned n_fused, M &res_mat,
                                   const M &m1, unsigned m_size) {
  // expand m1
  printf("Matrix Multiplication\n");
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
      std::complex<DT> re = std::complex<DT>(0, 0);
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

  for (int i = 0; i < res_mat.size(); i++) {
    printf("(%.1f, %.1f)", res_mat[i].real(), res_mat[i].imag());
  }
  printf("\n");
}

template <typename DT>
void qcircuit::Circuit<DT>::MatShuffle(M &res_mat, unsigned n_qubit,
                                       const std::vector<int> &perm) {

  std::vector<std::complex<DT>> temp_mat = res_mat;
  printf("Matrix Shuffle\n");

  unsigned n = unsigned{1} << n_qubit;

  for (unsigned i = 0; i < n; ++i) {
    unsigned i_ = i;
    unsigned row = 0;

    for (unsigned q = 0; q < n_qubit; ++q) {
      row |= ((i_ >> q) & 1) << perm[q];
    }

    for (unsigned j = 0; j < n; ++j) {
      unsigned j_ = j;
      unsigned col = 0;

      for (unsigned q = 0; q < n_qubit; ++q) {
        col |= ((j_ >> q) & 1) << perm[q];
      }

      res_mat[n * i + j] = temp_mat[n * row + col];
    }
  }

  for (int i = 0; i < res_mat.size(); i++) {
    printf("(%.1f, %.1f)", res_mat[i].real(), res_mat[i].imag());
  }
  printf("\n");
}

template class qcircuit::Circuit<double>;
template class qcircuit::Circuit<float>;

} // namespace sim
