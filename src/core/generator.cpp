#include "generator.h"
#include "../dataset/equivalence_set.h"

#include <cassert>

void Generator::generate_dfs(int num_qubits,
                             int max_num_input_parameters,
                             int max_num_quantum_gates,
                             int max_num_param_gates,
                             Dataset &dataset,
                             bool restrict_search_space) {
  DAG *dag = new DAG(num_qubits, max_num_input_parameters);
  // Generate all possible parameter gates at the beginning.
  assert(max_num_param_gates == 1);
  dag->generate_parameter_gates(context);
  // We need a large vector for both input and internal parameters.
  // std::vector<int> used_parameters(max_num_input_parameters + max_num_param_gates, 0);
  std::vector<int> used_parameters(dag->get_num_total_parameters(), 0);
  dfs(0,
      max_num_quantum_gates,
      max_num_param_gates,
      dag,
      used_parameters,
      dataset,
      restrict_search_space);
  delete dag;
}

void Generator::generate(int num_qubits,
                         int num_input_parameters,
                         int max_num_quantum_gates,
                         int max_num_param_gates,
                         Dataset *dataset,
                         bool verify_equivalences,
                         EquivalenceSet *equiv_set,
                         bool verbose) {
  auto empty_dag = std::make_unique<DAG>(num_qubits, num_input_parameters);
  // Generate all possible parameter gates at the beginning.
  assert(max_num_param_gates == 1);
  empty_dag->generate_parameter_gates(context);
  empty_dag->hash(context);  // generate other hash values
  std::vector<DAG *> dags_to_search(1, empty_dag.get());
  if (verify_equivalences) {
    assert(equiv_set);
    auto equiv_class = std::make_unique<EquivalenceClass>();
    equiv_class->insert(std::make_unique<DAG>(*empty_dag));
    equiv_set->insert_class(context, std::move(equiv_class));
  } else {
    context->set_representative(std::make_unique<DAG>(*empty_dag));
  }
  dataset->insert(context, std::move(empty_dag));
  std::vector<std::vector<DAG *>> dags(1, dags_to_search);

  // To avoid EquivalenceSet deleting the DAGs in |dags| when calling clear().
  std::vector<std::unique_ptr<DAG>> dag_holder;

  for (int num_gates = 1; num_gates <= max_num_quantum_gates; num_gates++) {
    if (verbose) {
      std::cout << "BFS: " << dags_to_search.size()
                << " representative DAGs to search with "
                << num_gates - 1 << " gates." << std::endl;
    }
    if (!verify_equivalences) {
      assert(dataset);
      dags_to_search.clear();
      bfs(dags,
          max_num_param_gates,
          *dataset,
          &dags_to_search,
          verify_equivalences,
          nullptr);
      dags.push_back(dags_to_search);
    } else {
      assert(dataset);
      assert(equiv_set);
      bfs(dags,
          max_num_param_gates,
          *dataset,
          nullptr,
          verify_equivalences,
          equiv_set);
      // Do not verify when |num_gates == max_num_quantum_gates|.
      // This is to make the behavior the same when |verify_equivalences| is
      // true or false.
      if (num_gates == max_num_quantum_gates) {
        break;
      }
      bool ret = dataset->save_json(context, "tmp_before_verify.json");
      assert(ret);

      // Assume working directory is cmake-build-debug/ here.
      system(
          "python ../python/verify_equivalences.py tmp_before_verify.json tmp_after_verify.json");

      dags_to_search.clear();
      ret = equiv_set->load_json(context,
                                 "tmp_after_verify.json",
                                 &dags_to_search);
      assert(ret);
      for (auto &dag : dags_to_search) {
        auto new_dag = std::make_unique<DAG>(*dag);
        dag = new_dag.get();
        dag_holder.push_back(std::move(new_dag));
      }

      dags.push_back(dags_to_search);
      /* Seems problematic
      equiv_set->remove_common_first_or_last_gates(context);
      std::vector<DAG *> simplified_dags_to_search;
      simplified_dags_to_search.reserve(dags_to_search.size());
      for (auto &dag : dags_to_search) {
        if (equiv_set->contains(context, dag)) {
          simplified_dags_to_search.push_back(dag);
        }
      }
      dags.push_back(simplified_dags_to_search);
      */
    }
  }
}

void Generator::dfs(int gate_idx,
                    int max_num_gates,
                    int max_remaining_param_gates,
                    DAG *dag,
                    std::vector<int> &used_parameters,
                    Dataset &dataset,
                    bool restrict_search_space) {
  if (restrict_search_space) {
    // An optimization to restrict the search space, but may also cause
    // the equivalences found to be incomplete.
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
  }

  static int tmp = 0;
  tmp++;
  if (tmp == (tmp & (-tmp))) {
    std::cout << "DFS " << tmp << " " << dataset.num_hash_values() << std::endl;
  }

  /*int num_unused_internal_parameter = 0;
  for (int i = dag->get_num_input_parameters();
       i < dag->get_num_total_parameters(); i++) {
    if (used_parameters[i] == 0)
      num_unused_internal_parameter++;
  }

  bool save_into_dataset = (num_unused_internal_parameter == 0);
  if (save_into_dataset) {
    // save a clone of dag to |dataset|
    dataset.insert(context, dag->clone_and_shrink_unused_input_parameters());
  }*/
  dataset.insert(context, dag->clone());

  // Check that this circuit is different with any other circuits in the
  // |dataset|.
  // Optimization disabled.
  /*for (auto &other_dag : dataset[dag->hash(context)]) {
    // we could use BFS to avoid searching DAGs with more gates at first
    if (dag->get_num_gates() >= other_dag->get_num_gates()
        && verifier_.equivalent_on_the_fly(context, dag, other_dag.get())) {
      return;
    }
  }*/

  if (gate_idx >= max_num_gates)
    return;
  std::vector<int> qubit_indices;
  std::vector<int> parameter_indices;
  for (const auto &idx : context->get_supported_quantum_gates()) {
    Gate *gate = context->get_gate(idx);
    if (gate->get_num_qubits() == 0) {
      assert(false);  // We only search for quantum gates here.
      if (!max_remaining_param_gates) {
        // We can't add more parameter gates.
        continue;
      }
      if (gate->get_num_parameters() == 1) {
        for (int p = 0; p < dag->get_num_total_parameters(); p++) {
          parameter_indices.push_back(p);
          used_parameters[p] += 1;
          int output_param_index;
          bool ret = dag->add_gate(qubit_indices,
                                   parameter_indices,
                                   gate,
                                   &output_param_index);
          assert(ret);
          dfs(gate_idx + 1,
              max_num_gates,
              max_remaining_param_gates - 1,
              dag,
              used_parameters,
              dataset,
              restrict_search_space);
          ret = dag->remove_last_gate();
          assert(ret);
          used_parameters[p] -= 1;
          parameter_indices.pop_back();
        }
      } else if (gate->get_num_parameters() == 2) {
        // Case: 0-qubit operators with 2 parameters
        bool new_input_parameter_searched = false;
        for (int p1 = 0; p1 < dag->get_num_total_parameters(); p1++) {
          if (restrict_search_space) {
            if (p1 < dag->get_num_input_parameters() && !used_parameters[p1]) {
              // We should use the new (unused) input parameter with smallest
              // index as the first input if there are more than one of them.
              if (new_input_parameter_searched) {
                continue;
              } else {
                new_input_parameter_searched = true;
              }
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
            dfs(gate_idx + 1,
                max_num_gates,
                max_remaining_param_gates - 1,
                dag,
                used_parameters,
                dataset,
                restrict_search_space);
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
      for (int i = 0; i < dag->get_num_qubits(); i++) {
        qubit_indices.push_back(i);
        auto search_parameters = [&](int num_remaining_parameters,
                                     auto &search_parameters_ref/*feed in the lambda implementation to itself as a parameter*/) {
          if (num_remaining_parameters == 0) {
            bool
                ret =
                dag->add_gate(qubit_indices,
                              parameter_indices,
                              gate,
                              nullptr);
            assert(ret);
            dfs(gate_idx + 1,
                max_num_gates,
                max_remaining_param_gates,
                dag,
                used_parameters,
                dataset,
                restrict_search_space);
            ret = dag->remove_last_gate();
            assert(ret);
            return;
          }

          for (int p1 = 0;
               p1 < dag->get_num_total_parameters(); p1++) {
            parameter_indices.push_back(p1);
            used_parameters[p1] += 1;
            search_parameters_ref(num_remaining_parameters - 1,
                                  search_parameters_ref);
            used_parameters[p1] -= 1;
            parameter_indices.pop_back();
          }
        };
        search_parameters(gate->get_num_parameters(), search_parameters);

        qubit_indices.pop_back();
      }
    } else if (gate->get_num_qubits() == 2) {
      if (gate->get_num_parameters() == 0) {
        // Case: 2-qubit operators without parameters
        bool new_qubit_searched = false;
        for (int q1 = 0; q1 < dag->get_num_qubits(); q1++) {
          if (restrict_search_space) {
            if (q1 < dag->get_num_input_parameters() && !dag->qubit_used(q1)) {
              // We should use the new (unused) qubit with smallest index
              // as the first input if there are more than one of them.
              if (new_qubit_searched) {
                continue;
              } else {
                new_qubit_searched = true;
              }
            }
          }
          qubit_indices.push_back(q1);
          for (int q2 = 0; q2 < dag->get_num_qubits(); q2++) {
            if (q1 == q2) continue;
            qubit_indices.push_back(q2);
            bool ret =
                dag->add_gate(qubit_indices, parameter_indices, gate, nullptr);
            assert(ret);
            dfs(gate_idx + 1,
                max_num_gates,
                max_remaining_param_gates,
                dag,
                used_parameters,
                dataset,
                restrict_search_space);
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

void Generator::bfs(const std::vector<std::vector<DAG *>> &dags,
                    int max_num_param_gates,
                    Dataset &dataset,
                    std::vector<DAG *> *new_representatives,
                    bool verify_equivalences,
                    const EquivalenceSet *equiv_set) {
  auto try_to_add_to_result = [&](DAG *new_dag) {
    // A new DAG with |current_max_num_gates| + 1 gates.
    if (verify_equivalences) {
      assert(equiv_set);
      if (!verifier_.redundant(context, equiv_set, new_dag)) {
        auto new_new_dag = std::make_unique<DAG>(*new_dag);
        auto new_new_dag_ptr = new_new_dag.get();
        dataset.insert(context, std::move(new_new_dag));
        if (new_representatives) {
          // Warning: this is not the new representatives -- only the new DAGs.
          new_representatives->push_back(new_new_dag_ptr);
        }
      }
    } else {
      // If we will not verify the equivalence later, we should update
      // the representatives in the context now.
      if (verifier_.redundant(context, new_dag)) {
        return;
      }
      bool ret = dataset.insert(context, std::make_unique<DAG>(*new_dag));
      // Presuming different hash values imply different DAGs.
      if (ret) {
        // The DAG's hash value is new to the dataset.
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
  for (auto &dag : dags.back()) {
    std::vector<int> qubit_indices, parameter_indices;
    // Add 1 quantum gate.
    for (const auto &idx : context->get_supported_quantum_gates()) {
      Gate *gate = context->get_gate(idx);
      if (gate->get_num_qubits() == 1) {
        // Case: 1-qubit operators
        for (int i = 0; i < dag->get_num_qubits(); i++) {
          qubit_indices.push_back(i);
          auto search_parameters = [&](int num_remaining_parameters,
                                       auto &search_parameters_ref/*feed in the lambda implementation to itself as a parameter*/) {
            if (num_remaining_parameters == 0) {
              bool
                  ret =
                  dag->add_gate(qubit_indices,
                                parameter_indices,
                                gate,
                                nullptr);
              assert(ret);
              try_to_add_to_result(dag);
              ret = dag->remove_last_gate();
              assert(ret);
              return;
            }

            for (int p1 = 0;
                 p1 < dag->get_num_total_parameters(); p1++) {
              parameter_indices.push_back(p1);
              search_parameters_ref(num_remaining_parameters - 1,
                                    search_parameters_ref);
              parameter_indices.pop_back();
            }
          };
          search_parameters(gate->get_num_parameters(), search_parameters);

          qubit_indices.pop_back();
        }
      } else if (gate->get_num_qubits() == 2) {
        if (gate->get_num_parameters() == 0) {
          // Case: 2-qubit operators without parameters
          for (int q1 = 0; q1 < dag->get_num_qubits(); q1++) {
            qubit_indices.push_back(q1);
            for (int q2 = 0; q2 < dag->get_num_qubits(); q2++) {
              if (q1 == q2)
                continue;
              qubit_indices.push_back(q2);
              bool ret =
                  dag->add_gate(qubit_indices,
                                parameter_indices,
                                gate,
                                nullptr);
              assert(ret);
              try_to_add_to_result(dag);
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
    dag->hash(context);  // restore hash value
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
      std::vector<int> param_indices(1);
      for (param_indices[0] = 0;
           param_indices[0] < dag->get_num_total_parameters();
           param_indices[0]++) {
        if (param_indices[0] >= dag->get_num_input_parameters()) {
          if (!params_used_times[param_indices[0]])
            current_unused_params--;
          params_used_times[param_indices[0]]++;
        }
        if (current_unused_params + 1/*new parameter*/
            - (remaining_gates - 1) * (kMaxParamInputPerParamGate - 1)
            > max_unused_params) {
          // Too many unused parameters, prune it
          // Restore |params_used_times[param_indices[1]]|
          if (param_indices[0] >= dag->get_num_input_parameters()) {
            params_used_times[param_indices[0]]--;
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
        if (param_indices[0] >= dag->get_num_input_parameters()) {
          params_used_times[param_indices[0]]--;
          if (!params_used_times[param_indices[0]])
            current_unused_params++;
        }
      }
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
              if (!params_used_times[param_indices[1]])
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
