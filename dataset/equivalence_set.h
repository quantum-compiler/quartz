#pragma once

#include "../context/context.h"
#include "../dag/dag.h"

#include <list>
#include <set>
#include <unordered_set>
#include <unordered_map>

class EquivalenceSet {
 public:
  bool load_json(Context *ctx, const std::string &file_name);

  bool save_json(const std::string &file_name) const;

  // Normalize each clause of equivalent DAGs to have the minimum
  // (according to DAG::less_than) minimal representation.
  // Warning: see comments in DAG::minimal_representation().
  void normalize_to_minimal_representations(Context *ctx);

  void clear();

  // Remove unused qubits and input parameters if they are unused in
  // each DAG of an equivalent class.
  // Return the number of equivalent classes (modified and then) inserted.
  // This function potentially changes the ordering of equivalent classes
  // in the dataset.
  int remove_unused_qubits_and_input_params(Context *ctx);

  // This function runs in O(|dataset|.size()).
  [[nodiscard]] int num_equivalence_classes() const;

  // This function runs in O(num_equivalence_classes()).
  [[nodiscard]] int num_total_dags() const;

  void set_representatives(Context *ctx,
                           std::vector<DAG *> *new_representatives) const;

  [[nodiscard]] DAGHashType has_common_first_or_last_gates() const;

  // We cannot use std::vector here because that would need
  // std::unordered_set<std::unique_ptr<DAG>> to be copy-constructible.
  //
  // Each std::unordered_set represents a clause of equivalent DAGs.
  std::unordered_map<DAGHashType,
                     std::list<std::set<std::unique_ptr<DAG>, /*Compare=*/
                                        UniquePtrDAGComparator>>>
      dataset;
};
