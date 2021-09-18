#include "generator.h"

#include <cassert>

void Generator::generate(Context *ctx,
                         int num_qubits,
                         int max_num_parameters,
                         int max_num_gates)
{
  DAG* dag = new DAG(num_qubits, max_num_parameters);
  std::unordered_map<size_t, std::unordered_set<DAG*> > dataset;
  std::vector<bool> used_parameters(max_num_parameters, false);
  dfs(ctx, 0, max_num_gates, 0, 0, dag, used_parameters, dataset);
}

void Generator::dfs(Context *ctx,
                    int gate_idx,
                    int max_num_gates,
                    int next_unused_qubit_id,
                    DAG *dag,
                    std::vector<bool> &used_parameters,
                    std::unordered_map<size_t,
                                       std::unordered_set<DAG *> > &dataset)
{
  bool pass_checks = true;
  // check that qubits are used in an increasing order
  for (int i = 1; i < dag->get_num_qubits(); i++)
    if (outputs[i] == nodes[i].get() && outputs[i-1] != nodes[i-1].get())
      pass_checks = false;
  // check that parameters are used in an increasing order
  for (int i = 1; i < dag->get_num_parameters(); i++)
    if (used_parameters[i] && !used_parameters[i-1])
      pass_checks = false;
  // Return if we fail any checks
  if (!pass_checks)
    return;

  // save a clone of dag to dataset
  dataset[dag->hash(ctx)].insert(new DAG(*dag));

  if (gate_idx > max_num_gates)
    return;
  std::vector<int> qubit_indices;
  std::vector<int> parameter_indices;
  for (const auto& idx : ctx->get_supported_gates()) {
    Gate* gate = ctx->get_gate(idx);
    if (gate->get_num_qubits() == 1) {
      if (gate->get_num_parameters() == 0) {
        // Case: 1-qubit operators without parameters
        for (int i = 0; i < dag->get_num_qubits(); i++) {
          qubit_indices.push_back(i);
          dag->add_gate(qubit_indices, parameter_indices, gate, NULL);
          dfs(ctx, gate_idx+1, max_num_gates, dag, used_parameters, dataset);
          dag->remove_last_gate();
          qubit_indices.pop_back();
        }
      } else if (gate->get_num_parameters() == 1) {
        // Case: 1-qubit operators with 1 parameter
        for (int q1 = 0; q1 < dag->get_num_qubits(); q1++) {
          qubit_indices.push_back(q1);
          for (int p1 = 0; p1 < dag->get_num_total_parameters(); p1++) {
            parameter_indices.push_back(p1);
            dag->add_gate(qubit_indices, parameter_indices, gate, NULL);
            bool old_used_p1 = false;
            if (p1 < dag->get_num_input_parameters()) {
              old_used_p1 = used_parameters[p1];
              used_parameters[p1] = true;
            }
            dfs(ctx, gate_idx+1, max_num_gates, dag, used_parameters, dataset);
            if (p1 < dag->get_num_input_parameters()) {
              used_parameters[p1] = old_used_p1;
            }
            dag->remove_last_gate();
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
        for (int q1 = 0; q1 < dag->get_num_qubits(); q1++) {
          qubit_indices.push_back(q1);
          for (int q2 = 0; q2 < dag->get_num_qubits(); q2++) {
            if (q1 == q2) continue;
            qubit_indices.push_back(q2);
            dag->add_gate(qubit_indices, parameter_indices, gate, NULL);
            dfs(ctx, gate_idx+1, max_num_gates, dag, used_parameters, dataset);
            dag->remove_last_gate();
            qubit_indices.pop();
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

