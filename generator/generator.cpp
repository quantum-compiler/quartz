#include "generator.h"

void Generator::generate(Context *ctx,
                         int num_qubits,
                         int max_num_parameters,
                         int max_num_gates)
{
  DAG* dag = new DAG(num_qubits, max_num_parameters);
  std::unordered_map<size_t, std::unordered_set<DAG*> > dataset;
  dfs(0, max_num_gates, dag, dataset);
}

void dfs(Context *ctx,
         int gid,
         int max_num_gates,
         int next_qid,
         int next_pid,
         DAG* dag,
         std::unordered_map<size_t, std::unordered_set<DAG*> >& dataset)
{
  if (gid >= max_num_gates)
    return;
  // save our hash value
  dataset[dag->hash(ctx)].insert(dag);
  for (const auto& idx : ctx->get_supported_gates()) {
    Gate* gate = ctx->get_gate(idx);
    
  }
}
