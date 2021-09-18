#include "../context/context.h"
#include "../dag/dag.h"

#include <unordered_set>

class Generator {
 public:
  void generate(Context *ctx,
                int num_qubits,
                int max_num_parameters,
                int max_num_gates,
                std::unordered_map<DAGHashType, std::unordered_set<DAG*> > &dataset);

 private:
  void dfs(Context *ctx,
           int gate_idx,
           int max_num_gates,
           DAG *dag,
           std::vector<bool> &used_parameters,
           std::unordered_map<DAGHashType, std::unordered_set<DAG *> > &dataset);
};
