#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include <assert.h>
#include <algorithm>
#include <unordered_map>

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
}

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
bool qcircuit::Circuit<DT>::compile(quartz::CircuitSeq *seq, quartz::Context *ctx) {
  // 1. ILP/heuristics
  std::vector<std::vector<bool>> local_qubits;
  int result =
      num_iterations_by_heuristics(seq, n_local, local_qubits);
  // fprintf(fout, " %d", result);

  // 2. DP, fuse gates and add shuffle gates
  int idx = 0;
  auto schedules = get_schedules(*seq, local_qubits, ctx);
  for (auto &schedule : schedules) {
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
      // TODO: use template for matrix type, to be compatible with Legion-based version
      // std::vector<std::complex<DT>> mat;
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
      auto mat = FuseGates<DT>(schedule.kernels[i], schedule.kernel_qubits[i], ctx);

      // add fused gates kernels 
      Gate<DT> gate{g_type,
                    n_target,
                    n_control,
                    target,
                    control,
                    {},
                    mat};
      gates.push_back(gate);
    }

    // add shuffle gate
    std::vector<int> target;
    std::vector<int> global;
    for(int i = 0; i < num_qubits; i++) {
      if (local_qubits[idx][i]){
        target.push_back(i);
      }
      else {
        global.push_back(i);
      }
    }

    for(int i = 0; i < n_global; i++) {
      target.push_back(global[i]);
    }
    Gate<DT> gate{SHUFFLE, num_qubits, 0, target, {}, {}, {}};
    gates.push_back(gate);

    idx++;
  }
}

template <typename DT>
void qcircuit::Circuit<DT>::simulate(int ndevices) {
  using Simulator = SimulatorCuQuantum<DT>;
  Simulator simulator(n_local, n_global, ndevices);
  std::vector<unsigned> init_perm;
  for (int i = 0; i < n_local + n_global; i++) {
    init_perm.push_back(i);
  }
  simulator.InitState(init_perm);
  printf("Init State Vectors!\n");
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
std::vector<std::complex<DT>> qcircuit::Circuit<DT>::FuseGates(const quartz::CircuitSeq& seq, const std::vector<int>& qubits, quartz::Context *ctx){
  // expand all gates to 2^n_target * 2^n_target matrix: identity X mat => matrix shuffle
  unsigned ksize = unsigned(1) << qubits.size();
  unsigned vec_size = ksize * ksize;
  std::vector<std::complex<DT>> res_mat(vec_size, std::complex<DT> (1, 0)); 
  printf("Fused Kernel Size: %d\n", ksize);
  for(int i = 0; i < seq.gates.size(); i++){
    std::vector<int> qubit_indices;
    std::vector<ParamType> params;
    for (const auto &input_wire : seq.gates[i]->input_wires) {
      if (input_wire->is_qubit()) {
        // TODO: check the order of the qubit??
        qubit_indices.push_back(input_wire->index);
      } else {
        params.push_back(ctx->input_parameters[input_wire->index]);
      }
    }
    // currently assume the matrix from quartz is for increasing order
    std::reverse(qubit_indices.begin(), qubit_indices.end());
    unsigned mask = 0;
    // get qubit mask for the gate to be fused
    for (int t = 0; t < qubit_indices.size(); t++){
      int it = std::find(qubits.begin(), qubits.end(), qubit_indices[t]) - qubits.begin();
      mask |= (1 << it);
    }

    if (seq.gates[i]->gate->is_parameter_gate()) {
      auto *m = seq.gates[i]->gate->get_matrix(params);
      std::vector<std::complex<DT>> temp = m->flatten();
      MatMut<DT>(mask, qubits.size(), res_mat, temp, qubit_indices.size());
    } else {
      // A quantum gate. Update the distribution.
      assert(gates[i]->gate->is_quantum_gate());
      auto *m = seq.gates[i]->gate->get_matrix();
      std::vector<std::complex<DT>> temp = m->flatten();
      MatMul<DT>(mask, qubits.size(), res_mat, temp, qubit_indices.size());
    }
  }

  return res_mat;
}

template class qcircuit::Circuit<double>;
template class qcircuit::Circuit<float>;

} // namespace sim