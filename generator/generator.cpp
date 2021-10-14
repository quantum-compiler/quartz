#include "generator.h"

#include <cassert>

void Generator::generate_dfs(int num_qubits,
                             int max_num_input_parameters,
                             int max_num_gates,
                             Dataset &dataset) {
  DAG *dag = new DAG(num_qubits, max_num_input_parameters);
  // We need a large vector for both input and internal parameters.
  std::vector<int> used_parameters(max_num_input_parameters + max_num_gates, 0);
  dfs(0, max_num_gates, dag, used_parameters, dataset);
  /*for (const auto& it : dataset) {
    for (const auto& dag : it.second) {
      printf("key = %llu \n", dag->hash(context));
      dag->print(context);
    }
  }*/
  delete dag;
}

void Generator::generate(int num_qubits,
                         int num_input_parameters,
                         int max_num_gates,
                         Dataset &dataset) {
  auto empty_dag = std::make_unique<DAG>(num_qubits, num_input_parameters);
  std::vector<DAG *> dags_to_search(1, empty_dag.get());
  std::vector<std::vector<DAG *>> dags(1, dags_to_search);
  for (int num_gates = 1; num_gates <= max_num_gates; num_gates++) {
    dags_to_search.clear();
    bfs(dags, dataset, &dags_to_search);
    dags.push_back(dags_to_search);
  }
}

void Generator::dfs(int gate_idx,
                    int max_num_gates,
                    DAG *dag,
                    std::vector<int> &used_parameters,
                    Dataset &dataset) {
  bool pass_checks = true;
  // check that qubits are used in an increasing order
  for (int i = 1; i < dag->get_num_qubits(); i++)
    if (dag->outputs[i] != dag->nodes[i].get()
        && dag->outputs[i - 1] == dag->nodes[i - 1].get())
      pass_checks = false;
  // check that input parameters are used in an increasing order
  for (int i = 1; i < dag->get_num_input_parameters(); i++)
    if (used_parameters[i] > 0 && used_parameters[i - 1] == 0)
      pass_checks = false;
  // Note that we do not check that internal parameters are used in an
  // increasing order here.
  // Return if we fail any checks
  if (!pass_checks)
    return;

  int num_unused_internal_parameter = 0;
  for (int i = dag->get_num_input_parameters();
       i < dag->get_num_total_parameters(); i++) {
    if (used_parameters[i] == 0)
      num_unused_internal_parameter++;
  }

  bool save_into_dataset = (num_unused_internal_parameter == 0);
  if (save_into_dataset) {
    // save a clone of dag to dataset
    dataset.insert(context, dag->clone_and_shrink_unused_input_parameters());
  }

  // check that this circuit is different with any other circuits in the dataset
  for (auto &other_dag : dataset[dag->hash(context)]) {
    // we could use BFS to avoid searching DAGs with more gates at first
    if (dag->get_num_gates() >= other_dag->get_num_gates()
        && verifier_.equivalent_on_the_fly(context, dag, other_dag.get())) {
      return;
    }
  }

  if (gate_idx >= max_num_gates)
    return;
  std::vector<int> qubit_indices;
  std::vector<int> parameter_indices;
  for (const auto &idx : context->get_supported_gates()) {
    Gate *gate = context->get_gate(idx);
    if (gate->get_num_qubits() == 0) {
      if (gate->get_num_parameters() == 1) {
        assert(false && "Unsupported gate type");
      } else if (gate->get_num_parameters() == 2) {
        // Case: 0-qubit operators with 2 parameters
        bool new_input_parameter_searched = false;
        for (int p1 = 0; p1 < dag->get_num_total_parameters(); p1++) {
          if (p1 < dag->get_num_input_parameters() && !used_parameters[p1]) {
            // We should use the new (unused) input parameter with smallest
            // index as the first input if there are more than one of them.
            if (new_input_parameter_searched) {
              continue;
            } else {
              new_input_parameter_searched = true;
            }
          }
          parameter_indices.push_back(p1);
          used_parameters[p1] += 1;
          for (int p2 = 0; p2 < dag->get_num_total_parameters(); p2++) {
            if (gate->is_commutative() && p1 > p2) {
              // For commutative gates, enforce p1 <= p2
              continue;
            }
            parameter_indices.push_back(p2);
            used_parameters[p2] += 1;
            int output_param_index;
            bool ret = dag->add_gate(qubit_indices,
                                     parameter_indices,
                                     gate,
                                     &output_param_index);
            assert(ret);
            dfs(gate_idx + 1, max_num_gates, dag, used_parameters, dataset);
            ret = dag->remove_last_gate();
            assert(ret);
            used_parameters[p2] -= 1;
            parameter_indices.pop_back();
          }
          used_parameters[p1] -= 1;
          parameter_indices.pop_back();
        }
      } else {
        assert(false && "Unsupported gate type");
      }
    } else if (gate->get_num_qubits() == 1) {
      if (gate->get_num_parameters() == 0) {
        // Case: 1-qubit operators without parameters
        for (int i = 0; i < dag->get_num_qubits(); i++) {
          qubit_indices.push_back(i);
          bool
              ret =
              dag->add_gate(qubit_indices, parameter_indices, gate, nullptr);
          assert(ret);
          dfs(gate_idx + 1, max_num_gates, dag, used_parameters, dataset);
          ret = dag->remove_last_gate();
          assert(ret);
          qubit_indices.pop_back();
        }
      } else if (gate->get_num_parameters() == 1) {
        // Case: 1-qubit operators with 1 parameter
        for (int q1 = 0; q1 < dag->get_num_qubits(); q1++) {
          qubit_indices.push_back(q1);
          for (int p1 = 0; p1 < dag->get_num_total_parameters(); p1++) {
            parameter_indices.push_back(p1);
            bool ret =
                dag->add_gate(qubit_indices, parameter_indices, gate, nullptr);
            assert(ret);
            used_parameters[p1] += 1;
            dfs(gate_idx + 1, max_num_gates, dag, used_parameters, dataset);
            used_parameters[p1] -= 1;
            ret = dag->remove_last_gate();
            assert(ret);
            parameter_indices.pop_back();
          }
          qubit_indices.pop_back();
        }
      } else {
        assert(false && "To be implemented...");
      }
    } else if (gate->get_num_qubits() == 2) {
      if (gate->get_num_parameters() == 0) {
        // Case: 2-qubit operators without parameters
        bool new_qubit_searched = false;
        for (int q1 = 0; q1 < dag->get_num_qubits(); q1++) {
          if (q1 < dag->get_num_input_parameters() && !dag->qubit_used(q1)) {
            // We should use the new (unused) qubit with smallest index
            // as the first input if there are more than one of them.
            if (new_qubit_searched) {
              continue;
            } else {
              new_qubit_searched = true;
            }
          }
          qubit_indices.push_back(q1);
          for (int q2 = 0; q2 < dag->get_num_qubits(); q2++) {
            if (q1 == q2) continue;
            qubit_indices.push_back(q2);
            bool ret =
                dag->add_gate(qubit_indices, parameter_indices, gate, nullptr);
            assert(ret);
            dfs(gate_idx + 1, max_num_gates, dag, used_parameters, dataset);
            ret = dag->remove_last_gate();
            assert(ret);
            qubit_indices.pop_back();
          }
          qubit_indices.pop_back();
        }
      } else {
        assert(false && "To be implemented...");
      }
    } else {
      // Current only support 1- and 2-qubit gates
      assert(false && "Unsupported gate");
    }
  }
}

void Generator::bfs(const std::vector<std::vector<DAG *>> &dags, Dataset &dataset, std::vector<DAG *> *new_representatives) {
  auto try_to_add_to_result = [&](DAG *new_dag) {
    // A new DAG with |current_max_num_gates| + 1 gates.
    if (!verifier_.redundant(context, new_dag)) {
      bool ret = dataset.insert(context, std::make_unique<DAG>(*new_dag));
      if (ret) {
        // The DAG's hash value is new to the DAG.
        // XXX: presuming different hash values imply different DAGs.
        // Note: this is the second instance of DAG we create in this function.
        auto rep = std::make_unique<DAG>(*new_dag);
        auto rep_ptr = rep.get();
        context->set_representative(std::move(rep));
        if (new_representatives) {
          new_representatives->push_back(rep_ptr);
        }
      }
    }
  };
  int current_max_num_gates = (int) dags.size() - 1;
  std::vector<int> params_used_times;
  for (int num_gates = 0; num_gates <= current_max_num_gates; num_gates++) {
    for (auto &dag : dags[num_gates]) {
      // Add (current_max_num_gates - num_gates) parameter gates.
      std::vector<std::unique_ptr<DAG>> dags_to_search;
      // Assume all parameters are used in the current dag now.
      params_used_times.assign(dag->get_num_total_parameters(), 1);
      dfs_parameter_gates(std::make_unique<DAG>(*dag),
                          current_max_num_gates
                              - num_gates, /*max_unused_params=*/
                          kMaxParamInputPerQuantumGate, /*current_unused_params=*/
                          0,
                          params_used_times,
                          dags_to_search);

      std::vector<int> qubit_indices, parameter_indices;
      // Add 1 quantum gate.
      // So the last gate must be a quantum gate.
      for (const auto &idx : context->get_supported_quantum_gates()) {
        Gate *gate = context->get_gate(idx);
        if (gate->get_num_qubits() == 1) {
          if (gate->get_num_parameters() == 0) {
            // Case: 1-qubit operators without parameters
            if (num_gates != current_max_num_gates) {
              // We could only have added 0 new parameter gates to get here.
              continue;
            }
            for (auto &dag_to_search : dags_to_search) {
              for (int i = 0; i < dag_to_search->get_num_qubits(); i++) {
                qubit_indices.push_back(i);
                bool
                    ret =
                    dag_to_search->add_gate(qubit_indices,
                                            parameter_indices,
                                            gate,
                                            nullptr);
                assert(ret);
                try_to_add_to_result(dag_to_search.get());
                ret = dag_to_search->remove_last_gate();
                assert(ret);
                qubit_indices.pop_back();
              }
            }
          } else if (gate->get_num_parameters() == 1) {
            // Case: 1-qubit operators with 1 parameter
            for (auto &dag_to_search : dags_to_search) {
              for (int q1 = 0; q1 < dag_to_search->get_num_qubits(); q1++) {
                qubit_indices.push_back(q1);
                // We must use the new parameter when |num_gates| <
                // |current_max_num_gates|.
                for (int p1 = (num_gates == current_max_num_gates ? 0 :
                               dag_to_search->get_num_total_parameters() - 1);
                     p1 < dag_to_search->get_num_total_parameters(); p1++) {
                  parameter_indices.push_back(p1);
                  bool ret =
                      dag_to_search->add_gate(qubit_indices,
                                              parameter_indices,
                                              gate,
                                              nullptr);
                  assert(ret);
                  try_to_add_to_result(dag_to_search.get());
                  ret = dag_to_search->remove_last_gate();
                  assert(ret);
                  parameter_indices.pop_back();
                }
                qubit_indices.pop_back();
              }
            }
          } else {
            assert(false && "To be implemented...");
          }
        } else if (gate->get_num_qubits() == 2) {
          if (gate->get_num_parameters() == 0) {
            // Case: 2-qubit operators without parameters
            if (num_gates != current_max_num_gates) {
              // We could only have added 0 new parameter gates to get here.
              continue;
            }
            for (auto &dag_to_search : dags_to_search) {
              for (int q1 = 0; q1 < dag_to_search->get_num_qubits(); q1++) {
                qubit_indices.push_back(q1);
                for (int q2 = 0; q2 < dag_to_search->get_num_qubits(); q2++) {
                  if (q1 == q2)
                    continue;
                  qubit_indices.push_back(q2);
                  bool ret =
                      dag_to_search->add_gate(qubit_indices,
                                              parameter_indices,
                                              gate,
                                              nullptr);
                  assert(ret);
                  try_to_add_to_result(dag_to_search.get());
                  ret = dag_to_search->remove_last_gate();
                  assert(ret);
                  qubit_indices.pop_back();
                }
                qubit_indices.pop_back();
              }
            }
          } else {
            assert(false && "To be implemented...");
          }
        } else {
          // Current only support 1- and 2-qubit gates
          assert(false && "Unsupported gate");
        }
      }
    }
  }
}

void Generator::dfs_parameter_gates(std::unique_ptr<DAG> dag,
                                    int remaining_gates,
                                    int max_unused_params,
                                    int current_unused_params,
                                    std::vector<int> &params_used_times,
                                    std::vector<std::unique_ptr<DAG>> &result) {
  if (remaining_gates == 0) {
    result.push_back(std::move(dag));
    return;
  }
  for (const auto &idx : context->get_supported_parameter_gates()) {
    Gate *gate = context->get_gate(idx);
    if (gate->get_num_parameters() == 1) {
      assert(false && "Unsupported gate type");
    } else if (gate->get_num_parameters() == 2) {
      // Case: 0-qubit operators with 2 parameters
      std::vector<int> param_indices(2);
      for (param_indices[0] = 0;
           param_indices[0] < dag->get_num_total_parameters();
           param_indices[0]++) {
        if (param_indices[0] >= dag->get_num_input_parameters()) {
          // Only enforce all internal parameters are used
          if (!params_used_times[param_indices[0]])
            current_unused_params--;
          params_used_times[param_indices[0]]++;
        }
        for (param_indices[1] = 0;
             param_indices[1] < dag->get_num_total_parameters();
             param_indices[1]++) {
          if (gate->is_commutative() && param_indices[0] > param_indices[1]) {
            // For commutative gates, enforce param_indices[0] <= param_indices[1]
            continue;
          }
          if (param_indices[1] >= dag->get_num_input_parameters()) {
            if (!params_used_times[param_indices[1]])
              current_unused_params--;
            params_used_times[param_indices[1]]++;
          }
          if (current_unused_params + 1/*new parameter*/
              - (remaining_gates - 1) * (kMaxParamInputPerParamGate - 1)
              > max_unused_params) {
            // Too many unused parameters, prune it
            // Restore |params_used_times[param_indices[1]]|
            if (param_indices[1] >= dag->get_num_input_parameters()) {
              params_used_times[param_indices[1]]--;
              if (!params_used_times[param_indices[0]])
                current_unused_params++;
            }
            continue;
          }
          int output_param_index;
          auto new_dag = std::make_unique<DAG>(*dag);
          bool ret = new_dag->add_gate({},
                                       param_indices,
                                       gate,
                                       &output_param_index);
          assert(ret);
          if (output_param_index >= params_used_times.size()) {
            params_used_times.resize(output_param_index + 1);
          }
          params_used_times[output_param_index] = 0;
          dfs_parameter_gates(std::move(new_dag),
                              remaining_gates - 1,
                              max_unused_params,
                              current_unused_params + 1/*new parameter*/,
                              params_used_times,
                              result);
          assert(ret);
          if (param_indices[1] >= dag->get_num_input_parameters()) {
            params_used_times[param_indices[1]]--;
            if (!params_used_times[param_indices[1]])
              current_unused_params++;
          }
        }
        if (param_indices[0] >= dag->get_num_input_parameters()) {
          params_used_times[param_indices[0]]--;
          if (!params_used_times[param_indices[0]])
            current_unused_params++;
        }
      }
    } else {
      assert(false && "Unsupported gate type");
    }
  }
}
