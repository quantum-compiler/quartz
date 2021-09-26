#include "generator.h"

#include <cassert>

void Generator::generate(int num_qubits,
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
    dataset[dag->hash(context)].insert(new DAG(*dag));
  }

  // check that this circuit is different with any other circuits in the dataset
  for (auto &other_dag : dataset[dag->hash(context)]) {
    // we could use BFS to avoid searching DAGs with more gates at first
    if (dag->get_num_gates() >= other_dag->get_num_gates()
        && verifier_.equivalent_on_the_fly(context, dag, other_dag)) {
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
