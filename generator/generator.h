#pragma once

#include "../context/context.h"
#include "../dag/dag.h"
#include "../dataset/dataset.h"
#include "../verifier/verifier.h"

#include <unordered_set>

class Generator {
 public:
  explicit Generator(Context *ctx) : context(ctx) {}
  void generate(int num_qubits,
                int max_num_input_parameters,
                int max_num_gates,
                Dataset &dataset);

 private:
  void dfs(int gate_idx,
           int max_num_gates,
           DAG *dag,
           std::vector<int> &used_parameters,
           Dataset &dataset);

  void bfs(std::vector<DAG *> dags_to_search, Dataset &dataset);

  Context *context;
  Verifier verifier_;
};
