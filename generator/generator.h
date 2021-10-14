#pragma once

#include "../context/context.h"
#include "../dag/dag.h"
#include "../dataset/dataset.h"
#include "../verifier/verifier.h"

#include <unordered_set>

class Generator {
 public:
  explicit Generator(Context *ctx) : context(ctx) {}
  void generate_dfs(int num_qubits,
                    int max_num_input_parameters,
                    int max_num_gates,
                    Dataset &dataset);

  // Use BFS to generate all equivalent DAGs with |num_qubits| qubits,
  // |num_input_parameters| input parameters (probably with some unused),
  // and <= |max_num_gates| gates.
  void generate(int num_qubits,
                int num_input_parameters,
                int max_num_gates,
                Dataset &dataset);

 private:
  void dfs(int gate_idx,
           int max_num_gates,
           DAG *dag,
           std::vector<int> &used_parameters,
           Dataset &dataset);

  // |dags[i]| is the DAGs with |i| gates.
  void bfs(const std::vector<std::vector<DAG *>> &dags,
           Dataset &dataset,
           std::vector<DAG *> *new_representatives);

  void dfs_parameter_gates(std::unique_ptr<DAG> dag,
                           int remaining_gates,
                           int max_unused_params,
                           int current_unused_params,
                           std::vector<int> &params_used_times,
                           std::vector<std::unique_ptr<DAG>> &result);

  Context *context;
  Verifier verifier_;
};
