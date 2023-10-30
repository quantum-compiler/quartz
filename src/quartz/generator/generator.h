#pragma once

#include "../context/context.h"
#include "../dataset/dataset.h"
#include "../dataset/equivalence_set.h"
#include "../verifier/verifier.h"
#include "quartz/circuitseq/circuitseq.h"

#include <chrono>
#include <unordered_set>

namespace quartz {
class Generator {
 public:
  explicit Generator(Context *ctx) : context(ctx) {}

  // Use DFS to generate all equivalent DAGs with |num_qubits| qubits,
  // <= |max_num_input_parameters| input parameters,
  // and <= |max_num_quantum_gates| gates.
  // If |restrict_search_space| is false, we search for all possible DAGs
  // with no unused internal parameters.
  // If |restrict_search_space| is true, we only search for DAGs which:
  //   - Use qubits in an increasing order;
  //   - Use input parameters in an increasing order;
  //   - When a gate uses more than one fresh new qubits or fresh new
  //   input
  //     parameters, restrict the order (for example, if a CX gate uses
  //     two fresh new qubits, the control qubit must have the smaller
  //     index).
  // If |unique_parameters| is true, we only search for DAGs that use
  // each input parameters only once (note: use a doubled parameter, i.e.,
  // Rx(2theta) is considered using the parameter theta once).
  void generate_dfs(int num_qubits, int max_num_input_parameters,
                    int max_num_quantum_gates, int max_num_param_gates,
                    Dataset &dataset, bool restrict_search_space,
                    bool unique_parameters);

  /**
   * Use BFS to generate all equivalent DAGs with |num_qubits| qubits,
   * |num_input_parameters| input parameters (probably with some unused),
   * and <= |max_num_quantum_gates| gates.
   *
   * @param num_qubits number of qubits in the circuits generated.
   * @param num_input_parameters number of input parameters in the circuits
   * generated.
   * @param max_num_quantum_gates max number of quantum gates in the circuits
   * generated.
   * @param max_num_param_gates currently unused.
   * @param dataset the |Dataset| object to store the result.
   * @param invoke_python_verifier if true, invoke Z3 verifier in Python to
   * verify that the equivalences we found are indeed equivalent. Otherwise,
   * we will simply trust the one-time random testing result, which may
   * treat hash collision as equivalent. XXX: when this is false, we will
   * treat any CircuitSeq with hash values differ no more than 1 with any
   * representative as equivalent.
   * @param equiv_set should be an empty |EquivalenceSet| object at the
   * beginning, and will store the intermediate ECC sets during generation.
   * @param unique_parameters if true, we only search for DAGs that use
   * each input parameters only once (note: use a doubled parameter, i.e.,
   * Rx(2theta) is considered using the parameter theta once).
   * @param verbose print debug message or not.
   * @param record_verification_time use |std::chrono::steady_clock| to
   * record the verification time or not.
   */
  void generate(
      int num_qubits, int num_input_parameters, int max_num_quantum_gates,
      int max_num_param_gates, Dataset *dataset, bool invoke_python_verifier,
      EquivalenceSet *equiv_set, bool unique_parameters, bool verbose = false,
      decltype(std::chrono::steady_clock::now() -
               std::chrono::steady_clock::now()) *record_verification_time =
          nullptr);

 private:
  void initialize_supported_quantum_gates();

  void dfs(int gate_idx, int max_num_gates, int max_remaining_param_gates,
           CircuitSeq *dag, std::vector<int> &used_parameters, Dataset &dataset,
           bool restrict_search_space, bool unique_parameters);

  // Requires initialize_supported_quantum_gates() to be called first.
  // |dags[i]| is the DAGs with |i| gates.
  void bfs(const std::vector<std::vector<CircuitSeq *>> &dags,
           int max_num_param_gates, Dataset &dataset,
           std::vector<CircuitSeq *> *new_representatives,
           bool invoke_python_verifier, const EquivalenceSet *equiv_set,
           bool unique_parameters);

  void dfs_parameter_gates(std::unique_ptr<CircuitSeq> dag, int remaining_gates,
                           int max_unused_params, int current_unused_params,
                           std::vector<int> &params_used_times,
                           std::vector<std::unique_ptr<CircuitSeq>> &result);

  Context *context;
  // |supported_quantum_gates_[i]|: supported quantum gates with |i| qubits.
  std::vector<std::vector<GateType>> supported_quantum_gates_;
  Verifier verifier_;
};

} // namespace quartz
