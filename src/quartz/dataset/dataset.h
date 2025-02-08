#pragma once

#include "quartz/circuitseq/circuitseq.h"
#include "quartz/context/context.h"

#include <unordered_set>

namespace quartz {

class Dataset {
 public:
  /**
   * Dump a Json file for verifier.py.
   * @param ctx The context.
   * @param file_name The file name to write to.
   * @return True iff the file is saved successfully.
   */
  bool save_json(Context *ctx, const std::string &file_name) const;

  // Return the number of DAGs removed.
  int remove_singletons(Context *ctx);

  // Normalize each CircuitSeq to have the canonical representation.
  // Return the number of DAGs removed.
  int normalize_to_canonical_representations(Context *ctx);

  /**
   * Sort the circuits with the same hash value by CircuitSeq::less_than().
   */
  void sort();

  // This function runs in O(1).
  [[nodiscard]] int num_hash_values() const;

  // This function runs in O(num_hash_values()).
  [[nodiscard]] int num_total_dags() const;

  auto &operator[](const CircuitSeqHashType &val) { return dataset[val]; }

  // Returns true iff the hash value is new to the |dataset|.
  bool insert(Context *ctx, std::unique_ptr<CircuitSeq> dag);

  /**
   * Insert a CircuitSeq with a pre-defined hash value. This function is used
   * when the hash value cannot be computed, and should NOT be used in Quartz's
   * generator.
   * @param hash_value The hash value to be inserted.
   * @param dag The CircuitSeq to be inserted.
   * @return True iff the hash value is new to the |dataset|.
   */
  bool insert_with_hash(const CircuitSeqHashType &hash_value,
                        std::unique_ptr<CircuitSeq> dag);

  // Inserts the CircuitSeq to an existing set if the hash value plus or minus 1
  // is found. Returns true iff there is no such existing set.
  bool insert_to_nearby_set_if_exists(Context *ctx,
                                      std::unique_ptr<CircuitSeq> dag);

  // Make this Dataset a brand new one.
  void clear();

  std::unordered_map<CircuitSeqHashType,
                     std::vector<std::unique_ptr<CircuitSeq>>>
      dataset;
};
}  // namespace quartz
