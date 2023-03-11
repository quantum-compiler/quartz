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
    permutation.push_back(perm[i]);
  }
  n_local = nlocal;
  n_global = num_qubits - nlocal;
}

template <typename DT>
qcircuit::Circuit<DT>::Circuit(unsigned nqubits, unsigned nlocal) {
  num_qubits = nqubits;
  for (int i = 0; i < num_qubits; i++) {
    permutation.push_back(i);
    pos.push_back(i);
  }
  n_local = nlocal;
  n_global = num_qubits - nlocal;
}

template <typename DT>
qcircuit::Circuit<DT>::Circuit(unsigned nqubits, unsigned nlocal, int myrank, int nranks) {
  num_qubits = nqubits;
  for (int i = 0; i < num_qubits; i++) {
    permutation.push_back(i);
    pos.push_back(i);
  }
  n_local = nlocal;
  n_global = num_qubits - nlocal;
  myRank = myrank;
  nRanks = nranks;
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
    // if (!parse_gate(ss, gate_name)) {
    //   return false;
    // }
  }

  return true;
}

// TODO: add mode to use heuristics or ILP
template <typename DT>
bool qcircuit::Circuit<DT>::compile(quartz::CircuitSeq *seq,
                                    quartz::Context *ctx,
                                    quartz::PythonInterpreter *interpreter,
                                    int ndevices,
                                    bool use_ilp) {
  // 1. ILP/heuristics
  std::vector<std::vector<int>> local_qubits;
  if (!use_ilp)
    // int result = num_iterations_by_heuristics(seq, n_local, local_qubits);
    assert(0);
  else {
    local_qubits =
        compute_local_qubits_with_ilp(*seq, n_local, ctx, interpreter);
  }
  // fprintf(fout, " %d", result);

  // 2. DP, fuse gates and add shuffle gates
  quartz::KernelCost kernel_cost(
      /*fusion_kernel_costs=*/{0, 10.4, 10.400001, 10.400002, 11, 40, 46, 66},
      /*shared_memory_init_cost=*/10,
      /*shared_memory_gate_cost=*/[](quartz::GateType type) { return 0.8; },
      /*shared_memory_total_qubits=*/10, /*shared_memory_cacheline_qubits=*/3);
  auto schedules = get_schedules(*seq, local_qubits, kernel_cost, ctx, /*absorb_single_qubit_gates=*/true);
  int idx = 0;
  int num_fuse = 0;
  int num_shm = 0;
  local_mask.assign(num_qubits, false);
  for (auto &schedule : schedules) {
    // add shuffle gate
    std::vector<int> target;
    for (int i = 0; i < n_local; i++) {
      target.push_back(local_qubits[idx][i]);
      local_mask[local_qubits[idx][i]] = true;
    }

    for (int i = 0; i < n_global; i++) {
      if(!local_mask[i])
        target.push_back(i);
    }
    Gate<DT> gate{SHUFFLE, num_qubits, 0, target, {}, {}, {}};
    gates.push_back(gate);
    update_layout(target);
    idx++;

    for (auto &kernel : schedule.kernels) {
      if (kernel.type == quartz::KernelType::fusion) {
        num_fuse++;
        FuseGates(kernel, ctx);
      }
      else if (kernel.type == quartz::KernelType::shared_memory) {
        num_shm++;
        SimGateType g_type = SHM;
        qindex active_qubits_logical = 0;
        printf("SHM Kernel Physical (%d): [ ", kernel.gates.gates.size());
        for (int i = 0; i < kernel.qubits.size(); i++) {
          active_qubits_logical |= qindex(1) << kernel.qubits[i];
          if (i != kernel.qubits.size() - 1)
            printf("%d,", pos[kernel.qubits[i]]);
          else
            printf("%d]\n", pos[kernel.qubits[i]]);
        }
        // if kernel.qubits.size() < SHARED_MEM_SIZE: fill it
        if (kernel.qubits.size() < SHARED_MEM_SIZE) {
          int cnt = kernel.qubits.size();
          for (int k = 0; k < SHARED_MEM_SIZE; k++) {
            if (!(active_qubits_logical & (1ll << permutation[k]))) {
              assert(k >= 3);
              cnt++;
              active_qubits_logical |= (1ll << permutation[k]);
              if (cnt == SHARED_MEM_SIZE)
                  break;
            }
          }
        }
        active_logical_qs.push_back(active_qubits_logical);
        std::cout << kernel.to_string() << std::endl;

        std::vector<KernelGate> kernelgates;
        
        for(auto &gate : kernel.gates.gates) {
          // get logical target qubit
          std::vector<int> qubit_indices;
          std::vector<ParamType> params;
          for (const auto &input_wire : gate->input_wires) {
            if (input_wire->is_qubit()) {
              qubit_indices.push_back(input_wire->index);
            } else {
              params.push_back(ctx->input_parameters[input_wire->index]);
            }
          }
          auto *m = gate->gate->get_matrix(params);
          std::vector<std::complex<qreal>> mat = m->flatten();
          // other way from quartz::GateType to KernelGateType?
          // target and control will be converted to related qubit when executing
          if (gate->gate->get_num_control_qubits() == 2) {
            // don't want to hardcode..
            qComplex mat_[2][2] = {(mat[3*8+3].real(),mat[3*8+3].imag()), (mat[3*8+7].real(), mat[3*8+7].imag()), (mat[7*8+3].real(), mat[7*8+3].imag()), (mat[7*8+7].real(), mat[7*8+7].imag())};
            qindex mask = active_qubits_logical;
            char isGlobalControl1 = (mask >> qubit_indices[0]) & 1;
            char isGlobalControl2 = (mask >> qubit_indices[1]) & 1;
            char isGlobalTarget = (mask >> qubit_indices[2]) & 1;
            KernelGate kg(toKernel(gate->gate->tp), qubit_indices[1], isGlobalControl2, qubit_indices[0], isGlobalControl1, qubit_indices[2], isGlobalTarget, mat_);
            kernelgates.push_back(kg);
          }
          else if (gate->gate->get_num_control_qubits() == 1) {
            // compress mat
            qComplex mat_[2][2] = {(mat[1*4+1].real(),mat[1*4+1].imag()), (mat[1*4+3].real(), mat[1*4+3].imag()), (mat[3*4+1].real(), mat[3*4+1].imag()), (mat[3*4+3].real(), mat[3*4+3].imag())};
            qindex mask = active_qubits_logical;
            char isGlobalControl1 = (mask >> qubit_indices[0]) & 1;
            char isGlobalTarget = (mask >> qubit_indices[1]) & 1;
            KernelGate kg(toKernel(gate->gate->tp), qubit_indices[0], isGlobalControl1, qubit_indices[1], isGlobalTarget, mat_);
            kernelgates.push_back(kg);
          }
          else if (gate->gate->get_num_control_qubits() == 0) {
            qComplex mat_[2][2] = {(mat[0].real(),mat[0].imag()), (mat[1].real(), mat[1].imag()), (mat[2].real(), mat[2].imag()), (mat[3].real(), mat[3].imag())};
            qindex mask = active_qubits_logical;
            char isGlobalTarget = (mask >> qubit_indices[0]) & 1;
            KernelGate kg(toKernel(gate->gate->tp), qubit_indices[0], isGlobalTarget, mat_);
            kernelgates.push_back(kg); 
          }            
        }
      
        task_map.push_back(SimGateType::SHM);
      }
    }
  }

  printf("Compilation Done! \n");
  printf("Num Shuffles: %d\n", schedules.size());
  printf("Num FUSION Kernel: %d\n", num_fuse);
  printf("Num SHM Kernel: %d\n", num_shm);
  printf("Start Simulating...\n");
  

  return true;
}

template <typename DT>
void qcircuit::Circuit<DT>::simulate(bool use_mpi) {
  using Simulator = SimulatorCuQuantum<DT>;
  Simulator simulator(n_local, n_global, n_devices, myRank, nRanks);
  std::vector<unsigned> init_perm;
  for (int i = 0; i < n_local + n_global; i++) {
    init_perm.push_back(i);
  }
  if (!use_mpi)
    simulator.InitStateSingle(init_perm);
  else
    simulator.InitStateMulti(init_perm);

  printf("Init State Vectors!\n");

  int normal_idx = 0;
  int shm_idx = 0;
  for (auto &task : task_map) {
    if (task == SHUFFLE) {
      simulator.ApplyShuffle(gates[normal_idx]);
      normal_idx++;
    }
    else if (task == FUSED) {
      for (int i = 0; i < n_devices; i++){
        simulator.ApplyGate(gates[normal_idx], i);
        normal_idx++;
      }
    }
    else if (task == SHM) {
      simulator.ApplyKernelGates(shm_gates[shm_idx], active_logical_qs[shm_idx]);
      shm_idx++;
    }
  }
  
  printf("Finish Simulating!\n");
  simulator.Destroy();
  printf("Destroyed the simulator\n");
}

/* private functiona for qcircuit::Circuit */

// an naive impl for fusing gates: can be more efficient
template <typename DT>
bool qcircuit::Circuit<DT>::FuseGates(const quartz::Kernel &kernel,
                                 quartz::Context *ctx) {
  SimGateType g_type = FUSED;
  unsigned n_target = kernel.qubits.size();
  unsigned n_control = 0;
  std::vector<int> target;
  std::vector<int> control;
  
  //reset qubit group map for current fusion kernel's qubits
  int fuse = 0;
  qubit_group_map_fusion.clear();
  for (int i = 0; i < n_local; i++) {
    if (std::find(kernel.qubits.begin(), kernel.qubits.end(), permutation[i]) != kernel.qubits.end()) {
        target.push_back(i); //physical target
        qubit_group_map_fusion[permutation[i]] = fuse++;
    }
  }

  printf( "Fusing Kernel (%d gates): qubits [", kernel.gates.gates.size());
  for (int j = 0; j < (int)kernel.qubits.size(); j++) {
    std::cout << kernel.qubits[j];
    target.push_back(kernel.qubits[j]);
    if (j != (int)kernel.qubits.size() - 1) {
      std::cout << ", ";
    }
  }
  std::cout << "], gates ";
  std::cout << kernel.to_string() << std::endl;


  std::vector<int> qubits = kernel.qubits;
  unsigned ksize = unsigned(1) << kernel.qubits.size();
  unsigned vec_size = ksize * ksize;
  std::vector<std::vector<std::complex<DT>>> res_mats;
  std::vector<std::complex<DT>> res_mat;
  res_mat.resize(vec_size);
  for (unsigned i = 0; i < ksize; ++i) {
    res_mat[(ksize * i + i)] = std::complex<DT>(1, 0);
  }
  for (int i = 0; i < n_devices; i++) {
    res_mats.push_back(res_mat);
  }

  // printf("Fused Kernel Size: %d\n", ksize);
  // // reorder qubits to increasing order
  // std::sort(qubits.begin(), qubits.end());

  for (int i = 0; i < kernel.gates.gates.size(); i++) {
    printf("Gate %d: [", i);
    std::vector<int> qubit_indices;
    std::vector<ParamType> params;
    for (const auto &input_wire : kernel.gates.gates[i]->input_wires) {
      if (input_wire->is_qubit()) {
        qubit_indices.push_back(input_wire->index);
        printf("%d, ", input_wire->index);
      } else {
        params.push_back(ctx->input_parameters[input_wire->index]);
      }
    }
    printf("]\n");

    for (int d = 0; d < n_devices; d++) {
      unsigned mask = 0;
      std::vector<int> qperm;
      std::vector<ComplexType> temp_mat_;
      if(getMat_per_device(ctx, myRank*n_devices+d, kernel.gates.gates[i]->gate, qubit_indices, params, temp_mat_, mask, qperm)) {
        //do MatMul
        M temp_mat(temp_mat_.begin(), temp_mat_.end());
        MatShuffle(temp_mat, qubit_indices.size(), qperm);
        MatMul(mask, qubits.size(), res_mats[i], temp_mat, qubit_indices.size());
      }
      //otherwise skip
    } 
  }

  // add fused gates kernels
  Gate<DT> gate{g_type, n_target, n_control, target, control, {}, res_mats};
  gates.push_back(gate);
  task_map.push_back(SimGateType::FUSED);

  return true;
}

template <typename DT>
void qcircuit::Circuit<DT>::MatMul(unsigned mask, unsigned n_fused, M &res_mat,
                                   const M &m1, unsigned m_size) {
  // expand m1
  // printf("Matrix Multiplication\n");
  unsigned n1 = unsigned{1} << m_size;
  unsigned n = unsigned{1} << n_fused;
  std::vector<std::complex<DT>> temp_mat = res_mat;

  for (unsigned i = 0; i < n; ++i) {
    unsigned i_ = i;
    unsigned row_m1 = 0;
    unsigned pos = 0;
    for (unsigned q = 0; q < n_fused; ++q) {
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
        for (unsigned q = 0; q < n_fused; ++q) {
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

  // for (int i = 0; i < res_mat.size(); i++) {
  //   printf("(%.1f, %.1f)", res_mat[i].real(), res_mat[i].imag());
  // }
  // printf("\n");
}

template <typename DT>
void qcircuit::Circuit<DT>::MatShuffle(M &res_mat, unsigned n_qubit,
                                       const std::vector<int> &perm) {

  std::vector<std::complex<DT>> temp_mat = res_mat;
  // printf("Matrix Shuffle\n");

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

  // for (int i = 0; i < res_mat.size(); i++) {
  //   printf("(%.1f, %.1f)", res_mat[i].real(), res_mat[i].imag());
  // }
  // printf("\n");
}

#define IS_HIGH_PART(part_id, logicIdx) ((part_id >> (pos[logicIdx] - n_local) & 1) > 0)
template <typename DT>
bool qcircuit::Circuit<DT>::getMat_per_device(quartz::Context *ctx, int part_id, quartz::Gate* gate, std::vector<int> qubit_indices, std::vector<ParamType>& params, std::vector<ComplexType>& res, unsigned &mask, std::vector<int>& perm) {
  if (gate->get_num_control_qubits() == 2) { //ccx
    int c2 = qubit_indices[0];
    int c1 = qubit_indices[1];
    int t = qubit_indices[2];
    if(local_mask[c2] && !local_mask[c1]) {
      int c = c1; c1 = c2; c2 = c;
    }
    if (local_mask[c1] && local_mask[c2]) { // CCU(c1, c2, t)
      auto *m = gate->get_matrix(params);
      res = m->flatten();
      std::vector<int> position;
      for (int t = 0; t < qubit_indices.size(); t++) {
        int v = qubit_group_map_fusion.at(qubit_indices[t]);
        position.push_back(v);
        mask |= 1 << v;
      }
      std::vector<int> sorted_position_;
      std::sort(sorted_position_.begin(), sorted_position_.end());
      assert(qubit_indices.size()==3);
      perm.resize(qubit_indices.size());
      for (int t = 0; t < qubit_indices.size(); t++) {
        int it2 = std::find(position.begin(), position.end(),
                            sorted_position_[t]) -
                  position.begin();
        perm[it2] = t;
      }
      return true;
    } else if (local_mask[c1] && !local_mask[c2]) {
        if (IS_HIGH_PART(part_id, c2)) { // CU(c1, t)
          //cx/cz gate mat
          if(gate->tp == quartz::GateType::ccx) {
            quartz::Gate* new_gate = ctx->get_gate(quartz::GateType::cx);
            auto *m = new_gate->get_matrix();
            res = m->flatten();
          }
          else if(gate->tp == quartz::GateType::ccz) {
            quartz::Gate* new_gate = ctx->get_gate(quartz::GateType::cz);
            auto *m = new_gate->get_matrix();
            res = m->flatten();
          }
          mask |= 1 << qubit_group_map_fusion.at(c1);
          mask |= 1 << qubit_group_map_fusion.at(t);
          perm.resize(2);
          perm[0] = qubit_group_map_fusion.at(c1) > qubit_group_map_fusion.at(t) ? 1 : 0;
          perm[1] = 1 - perm[0];
          return true;
          
        } else { // ID(t)
          return false;
        }
    } else { // !local_mask[c1] && !local_mask[c2]
        if (IS_HIGH_PART(part_id, c1) && IS_HIGH_PART(part_id, c2)) { // U(t)
          if(gate->tp == quartz::GateType::ccx) {
            quartz::Gate* new_gate = ctx->get_gate(quartz::GateType::x);
            auto *m = new_gate->get_matrix();
            res = m->flatten();
          }
          else if(gate->tp == quartz::GateType::ccz) {
            quartz::Gate* new_gate = ctx->get_gate(quartz::GateType::cz);
            auto *m = new_gate->get_matrix();
            res = m->flatten();
          }
          mask |= 1 << qubit_group_map_fusion.at(t);
          perm.resize(1);
          perm[0] = 0;
          return true;
        } else { // ID(t)
            return false;
        }
    }
  }
  else if (gate->get_num_control_qubits() == 1) {//cz, cp, cx
    int c = qubit_indices[0], t = qubit_indices[1];
    if (local_mask[c] && local_mask[t]) { // CU(c, t)
      auto *m = gate->get_matrix();
      res = m->flatten();
      mask |= 1 << qubit_group_map_fusion.at(c);
      mask |= 1 << qubit_group_map_fusion.at(t);
      perm.resize(2);
      perm[0] = qubit_group_map_fusion.at(c) > qubit_group_map_fusion.at(t) ? 1 : 0;
      perm[1] = 1 - perm[0];
      return true;    
    } else if (local_mask[c] && !local_mask[t]) { // U(c)
        if (IS_HIGH_PART(part_id, t)) { // U(t)
          // for this case, control qubit will become target, can be proved
          if(gate->tp == quartz::GateType::cz) {
            quartz::Gate* new_gate = ctx->get_gate(quartz::GateType::z);
            auto *m = new_gate->get_matrix();
            res = m->flatten();
          }
          else if(gate->tp == quartz::GateType::cp) {
            quartz::Gate* new_gate = ctx->get_gate(quartz::GateType::p);
            auto *m = new_gate->get_matrix(params);
            res = m->flatten();
          }
          mask |= 1 << qubit_group_map_fusion.at(c);
          perm.resize(1);
          perm[0] = 0;
          return true;
        }
        else {
          return false;
        }
    } else if (!local_mask[c] && local_mask[t]) {
        if (IS_HIGH_PART(part_id, c)) { // U(t)
          if(gate->tp == quartz::GateType::cx) {
            quartz::Gate* new_gate = ctx->get_gate(quartz::GateType::x);
            auto *m = new_gate->get_matrix();
            res = m->flatten();
          }
          else if(gate->tp == quartz::GateType::cz) {
            quartz::Gate* new_gate = ctx->get_gate(quartz::GateType::z);
            auto *m = new_gate->get_matrix();
            res = m->flatten();
          }
          else if(gate->tp == quartz::GateType::cp) {
            quartz::Gate* new_gate = ctx->get_gate(quartz::GateType::p);
            auto *m = new_gate->get_matrix(params);
            res = m->flatten();
          }
          mask |= 1 << qubit_group_map_fusion.at(t);
          perm.resize(1);
          perm[0] = 0;
          return true;
        } else {
            return false;
        }
    } else { // !local_mask[c] && !local_mask[t]
        assert(gate->tp == quartz::GateType::cz || gate->tp == quartz::GateType::cp);
        if (IS_HIGH_PART(part_id, c)) {
            switch (gate->tp) {
                case quartz::GateType::cz: {
                    if (IS_HIGH_PART(part_id, t)) {
                        quartz::Gate* new_gate = ctx->get_gate(quartz::GateType::z);
                        auto *m = new_gate->get_matrix();
                        res = m->flatten();
                        res[0] = res[3];
                        mask |= 1 << 0;
                        perm.resize(1);
                        perm[0] = 0;
                        return true;  
                    }
                }
                case quartz::GateType::cp: {
                    if (IS_HIGH_PART(part_id, t)) {
                        quartz::Gate* new_gate = ctx->get_gate(quartz::GateType::p);
                        auto *m = new_gate->get_matrix(params);
                        res = m->flatten();
                        res[0] = res[3];
                        mask |= 1 << 0;
                        perm.resize(1);
                        perm[0] = 0;
                        return true;
                    }
                }
            }
        } else {
            return false;
        }
    }
  }
  else if (gate->get_num_control_qubits() == 0) {
    int t = qubit_indices[0];
      if (!local_mask[t]) { // GCC(t)
        switch (gate->tp) {
            case quartz::GateType::p: {
                if (IS_HIGH_PART(part_id, t)) {
                    quartz::Gate* new_gate = ctx->get_gate(quartz::GateType::p);
                    auto *m = new_gate->get_matrix(params);
                    res = m->flatten();
                    res[0] = res[3];
                    mask |= 1 << 0;
                    perm.resize(1);
                    perm[0] = 0;
                    return true;
                } else {
                    return false;
                }
            }
            case quartz::GateType::z: {
                if (IS_HIGH_PART(part_id, t)) {
                    quartz::Gate* new_gate = ctx->get_gate(quartz::GateType::z);
                    auto *m = new_gate->get_matrix();
                    res = m->flatten();
                    res[0] = res[3];
                    mask |= 1 << 0;
                    perm.resize(1);
                    perm[0] = 0;
                    return true;  
                } else {
                    return false;
                }
            }
            // case quartz::GateType::x: { //TODO
            //     if (IS_HIGH_PART(part_id, t)) {
            //         //change device logic <-> physic mapping
            //         return false;
            //     } else {
            //         return false;
            //     }
            // }
        }
      } else { // local_mask[t] -> U(t)
          auto *m = gate->get_matrix();
          res = m->flatten();
          mask |= 1 << qubit_group_map_fusion.at(t);
          perm.resize(1);
          perm[0] = 0;
          return true;
      }
  }
}

template <typename DT>
void qcircuit::Circuit<DT>::update_layout(std::vector<int> targets) {
  
  std::vector<int> new_global_pos;
  int nGlobalSwaps = n_global;
  int nLocalSwaps = 0;
  int num_swaps = 0;
  for (int i = 0; i < n_global; i++) {
    new_global_pos.push_back(pos[targets[i + n_local]]);
    if(pos[targets[i + n_local]] < n_local) num_swaps++;
  }
  std::sort(new_global_pos.begin(), new_global_pos.end());
  
  unsigned local_mask = 0;
  unsigned global_mask = 0;
  int j1 = 0;
  for (int i = n_global - 1; i >= 0; i--) {
    if(new_global_pos[i] >= n_local) {
      global_mask |= 1 << (new_global_pos[i] - n_local);
      nGlobalSwaps--;
    }
    else {
      // for nccl-based comm, local transpose
      if(new_global_pos[i] >= (n_local - num_swaps)) {
        local_mask |= 1 << (new_global_pos[i] - n_local + num_swaps);
      }
      else {
        nLocalSwaps++;
        for (int j = num_swaps - 1; j >= 0; j--) {
          if(!(local_mask >> j & 1)) {
            std::swap(pos[permutation[new_global_pos[i]]], pos[permutation[n_local - num_swaps + j]]);
            std::swap(permutation[new_global_pos[i]], permutation[n_local - num_swaps + j]);
            local_mask |= 1 << j;
            break;
          }
        }
      }
    }
  }

  for (int i = 0; i < n_global; i++) {
    if((~global_mask) >> i & 1) {
      std::swap(pos[permutation[n_local-nGlobalSwaps]], pos[permutation[i+n_local]]);
      std::swap(permutation[n_local-nGlobalSwaps], permutation[i+n_local]);
      nGlobalSwaps--;
    }     
  }
}

template <typename DT>
KernelGateType qcircuit::Circuit<DT>::toKernel(quartz::GateType type) {
    switch (type) {
      case quartz::GateType::ccx:
        return KernelGateType::CCX;
      case quartz::GateType::cx:
        return KernelGateType::CNOT;
      case quartz::GateType::cz:
        return KernelGateType::CZ;
      case quartz::GateType::cp:
        return KernelGateType::CU1;
      case quartz::GateType::u1:
        return KernelGateType::U1;
      case quartz::GateType::u2:
        return KernelGateType::U2;
      case quartz::GateType::u3:
        return KernelGateType::U3;
      case quartz::GateType::h:
        return KernelGateType::H;
      case quartz::GateType::x:
        return KernelGateType::X;
      case quartz::GateType::y:
        return KernelGateType::Y;
      case quartz::GateType::z:
        return KernelGateType::Z;
      case quartz::GateType::s:
        return KernelGateType::S;
      case quartz::GateType::sdg:
        return KernelGateType::SDG;
      case quartz::GateType::rx:
        return KernelGateType::RX;
      case quartz::GateType::ry:
        return KernelGateType::RY;
      case quartz::GateType::rz:
        return KernelGateType::RZ;
      case quartz::GateType::t:
        return KernelGateType::T;
      case quartz::GateType::tdg:
        return KernelGateType::TDG;
      default:
          assert(false);
    }
  }

template class qcircuit::Circuit<double>;
template class qcircuit::Circuit<float>;

} // namespace sim
