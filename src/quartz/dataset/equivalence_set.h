#pragma once

#include "quartz/circuitseq/circuitseq.h"
#include "quartz/context/context.h"
#include "representative_set.h"

#include <functional>
#include <list>
#include <map>
#include <set>
#include <unordered_map>
#include <unordered_set>

namespace quartz {
class EquivalenceSet;

class EquivalenceClass {
public:
  // Returns all DAGs in this equivalence class.
  [[nodiscard]] std::vector<CircuitSeq *> get_all_dags() const;

  void insert(std::unique_ptr<CircuitSeq> dag);

  [[nodiscard]] int size() const;
  void reserve(std::size_t new_cap);

  // Extract all DAGs in this equivalence class, and make this class
  // empty.
  std::vector<std::unique_ptr<CircuitSeq>> extract();

  // Replace |dags_| with |dags|.
  void set_dags(std::vector<std::unique_ptr<CircuitSeq>> dags);

  // The first CircuitSeq is the representative.
  CircuitSeq *get_representative();

  // Returns whether this equivalence class contains (a CircuitSeq fully
  // equivalent to) |circuitseq|.
  [[nodiscard]] bool contains(const CircuitSeq &dag) const;

  // If this equivalence class contains |circuitseq|, set |circuitseq| as the
  // representative of the class and return true. Otherwise, return false.
  [[nodiscard]] bool set_as_representative(const CircuitSeq &dag);

  // For each pair of circuits in this class, if they share
  // a common "first" gate or a common "last" gate, remove the latter one.
  // Here "first" means a quantum gate which does not topologically depend
  // on any other quantum gates, and "last" means a quantum gate which can
  // appear at last in some topological order of the CircuitSeq.
  // Return the number of circuits removed.
  int remove_common_first_or_last_gates(
      Context *ctx,
      std::unordered_set<CircuitSeqHashType> &hash_values_to_remove);

  // Return the number of circuits modified.
  int remove_unused_internal_parameters(Context *ctx);

  // Return the hash of the first CircuitSeq.
  CircuitSeqHashType hash(Context *ctx);

  // Sort the circuits in this equivalence class by CircuitSeq::less_than().
  void sort();

  // Compare two ECCs in a deterministic order.
  // ECCs with fewer DAGs are considered smaller.
  // For ECCs with the same number of DAGs, an alphabetical order of
  // DAGs is used.
  static bool less_than(const EquivalenceClass &ecc1,
                        const EquivalenceClass &ecc2);

private:
  std::vector<std::unique_ptr<CircuitSeq>> dags_;
};

class UniquePtrEquivalenceClassComparator {
public:
  bool operator()(const std::unique_ptr<EquivalenceClass> &ecc1,
                  const std::unique_ptr<EquivalenceClass> &ecc2) const {
    if (!ecc1 || !ecc2) {
      // nullptr
      return !ecc2;
    }
    return EquivalenceClass::less_than(*ecc1, *ecc2);
  }
};

// This class stores all equivalence classes.
class EquivalenceSet {
public:
  // |new_representatives| is for Generator::generate().
  // It will be pushed back all representatives previously not in
  // the equivalence set.
  bool load_json(Context *ctx, const std::string &file_name,
                 std::vector<CircuitSeq *> *new_representatives = nullptr);

  bool save_json(const std::string &file_name) const;

  void clear();

  std::unique_ptr<RepresentativeSet> get_representative_set() const;

  // A final pass of simplification before feeding the equivalences
  // to the optimizer.
  // Returns if the pass does some simplification or not.
  bool simplify(Context *ctx,
                bool normalize_to_minimal_circuit_representation = true,
                bool common_subcircuit_pruning = true,
                bool other_simplification = true, bool verbose = false);

  // Sort the circuits in each equivalence class by CircuitSeq::less_than().
  void sort();

  // Remove equivalence classes with only one CircuitSeq.
  // Return the number of equivalent classes removed.
  int remove_singletons(Context *ctx, bool verbose = false);

  // Normalize each CircuitSeq to have the canonical representation.
  // Return the number of equivalent classes modified.
  int normalize_to_canonical_representations(Context *ctx,
                                             bool verbose = false);

  // Remove unused internal parameters.
  // Return the number of equivalent classes modified.
  int remove_unused_internal_params(Context *ctx, bool verbose = false);

  // Remove unused qubits and input parameters if they are unused in
  // each CircuitSeq of an equivalent class.
  // Return the number of equivalent classes removed
  // (and possibly inserted again).
  int remove_unused_qubits_and_input_params(Context *ctx, bool verbose = false);

  // For each pair of circuits in one equivalence class, if they share
  // a common "first" gate or a common "last" gate, remove the latter one.
  // Here "first" means a quantum gate which does not topologically depend
  // on any other quantum gates, and "last" means a quantum gate which can
  // appear at last in some topological order of the CircuitSeq.
  // Return the number of equivalent classes modified.
  int remove_common_first_or_last_gates(Context *ctx, bool verbose = false);

  // If there are two equivalence classes that are equivalent under
  // permutation of parameters, remove one of them.
  // Return the number of equivalent classes removed.
  int remove_parameter_permutations(Context *ctx, bool verbose = false);

  // This function runs in O(1).
  [[nodiscard]] int num_equivalence_classes() const;

  // This function runs in O(|classes_|.size()).
  [[nodiscard]] int num_total_dags() const;

  // Returns the position in |classes_|, or -1 if not found.
  [[nodiscard]] int first_class_with_common_first_or_last_gates() const;

  [[nodiscard]] std::string get_class_id(int num_class) const;

  [[nodiscard]] std::vector<std::vector<CircuitSeq *>>
  get_all_equivalence_sets() const;

  [[nodiscard]] std::vector<EquivalenceClass *>
  get_possible_classes(const CircuitSeqHashType &hash_value) const;

  // A hacky function to insert a single class to the equivalence set.
  // There's no guarantee that the class inserted is different with any
  // other classes already in the set.
  void insert_class(Context *ctx,
                    std::unique_ptr<EquivalenceClass> equiv_class);

  // Calls |equiv_class->insert(circuitseq)| and updates |possible_classes_|.
  void insert(Context *ctx, EquivalenceClass *equiv_class,
              std::unique_ptr<CircuitSeq> dag);

  // If the whole equivalence set contains a CircuitSeq fully equivalent to
  // |circuitseq|, return the equivalence class(es) containing it. Otherwise,
  // return an empty vector.
  [[nodiscard]] std::vector<EquivalenceClass *>
  get_containing_class(Context *ctx, CircuitSeq *dag) const;

private:
  void set_possible_class(const CircuitSeqHashType &hash_value,
                          EquivalenceClass *equiv_class);
  void remove_possible_class(const CircuitSeqHashType &hash_value,
                             EquivalenceClass *equiv_class);

  std::vector<std::unique_ptr<EquivalenceClass>> classes_;

  // A map from the hash value to all equivalence classes with at least
  // one CircuitSeq of the hash value.
  std::unordered_map<CircuitSeqHashType, std::set<EquivalenceClass *>>
      possible_classes_;
};

} // namespace quartz
