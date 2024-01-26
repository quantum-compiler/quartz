#pragma once

#include "quartz/circuitseq/circuitseq.h"
#include "quartz/context/context.h"
#include "quartz/dataset/dataset.h"
#include "quartz/dataset/equivalence_set.h"
#include "quartz/verifier/verifier.h"

#include <chrono>
#include <unordered_set>

namespace quartz {
class Generator {
 public:
  /**
   * Create a generator for a context. The context should have the gate set
   * ready, all parameter expressions generated, and all random testing
   * distributions generated. This can be done by calling
   * Context::Context(const std::vector<GateType> &supported_gates,
   *                  int num_qubits,
   *                  int num_input_symbolic_params).
   * Please create different generator objects with different contexts if
   * you need different parameter expressions, different numbers of parameters,
   * or different gate sets. It is OK to use the same generator for different
   * numbers of qubits or different numbers of gates, and the context should
   * be created with the maximum number of qubits.
   * @param ctx A context satisfying the above conditions.
   */
  explicit Generator(Context *ctx) : ctx_(ctx) {}

  /**
   * Use BFS to generate all equivalent circuits with |num_qubits| qubits
   * and <= |max_num_quantum_gates| gates,
   * using the gate set and the input parameters (expressions) in the context.
   *
   * @param num_qubits number of qubits in the circuits generated.
   * @param max_num_quantum_gates max number of quantum gates in the circuits
   * generated.
   * @param dataset the |Dataset| object to store the result.
   * @param invoke_python_verifier if true, invoke Z3 verifier in Python to
   * verify that the equivalences we found are indeed equivalent. Otherwise,
   * we will simply trust the one-time random testing result, which may
   * treat hash collision as equivalent. Note: when this is false, we will
   * treat any CircuitSeq with hash values differ no more than 1 with any
   * representative as equivalent.
   * @param equiv_set should be an empty |EquivalenceSet| object at the
   * beginning, and will store the intermediate ECC sets during generation.
   * @param unique_parameters if true, we only search for DAGs that use
   * each input parameter (expression) only once (note: using a doubled
   * parameter, i.e., Rx(2theta) is considered using the parameter theta once).
   * @param verbose print debug message or not.
   * @param record_verification_time use |std::chrono::steady_clock| to
   * record the verification time or not.
   * @return True if the generation is successful.
   */
  bool generate(
      int num_qubits, int max_num_quantum_gates, Dataset *dataset,
      bool invoke_python_verifier, EquivalenceSet *equiv_set,
      bool unique_parameters, bool verbose = false,
      std::chrono::steady_clock::duration *record_verification_time = nullptr);

 private:
  void initialize_supported_quantum_gates();

  // Requires initialize_supported_quantum_gates() to be called first.
  // |dags[i]| is the DAGs with |i| gates.
  void bfs(const std::vector<std::vector<CircuitSeq *>> &dags, Dataset &dataset,
           std::vector<CircuitSeq *> *new_representatives,
           bool invoke_python_verifier, const EquivalenceSet *equiv_set,
           bool unique_parameters);

  Context *ctx_;
  // |supported_quantum_gates_[i]|: supported quantum gates with |i| qubits.
  std::vector<std::vector<GateType>> supported_quantum_gates_;
  std::vector<InputParamMaskType> input_param_masks_;
  Verifier verifier_;
};

}  // namespace quartz
