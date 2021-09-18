#include "../context/context.h"
#include "../dag/dag.h"

#include <unordered_set>

class Generator {
public:
  Generator(Context* ctx) : context(ctx) {}
  void generate(int num_qubits,
                int max_num_parameters,
                int max_num_gates);

 private:
  void dfs(int gate_idx,
           int max_num_gates,
           DAG *dag,
           std::vector<bool> &used_parameters,
           std::unordered_map<size_t, std::unordered_set<DAG *> > &dataset);
private:
  Context* context;
};
