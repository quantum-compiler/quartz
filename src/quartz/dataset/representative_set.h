#pragma once

#include "quartz/circuitseq/circuitseq.h"
#include "quartz/context/context.h"

namespace quartz {

class RepresentativeSet {
 public:
  bool load_json(Context *ctx, const std::string &file_name);

  bool save_json(const std::string &file_name) const;

  void clear();

  // Returns all DAGs in this representative set.
  [[nodiscard]] std::vector<CircuitSeq *> get_all_dags() const;

  void insert(std::unique_ptr<CircuitSeq> dag);

  [[nodiscard]] int size() const;
  void reserve(std::size_t new_cap);

  // Extract all DAGs in this equivalence class, and make this class
  // empty.
  std::vector<std::unique_ptr<CircuitSeq>> extract();

  // Replace |dags_| with |dags|.
  void set_dags(std::vector<std::unique_ptr<CircuitSeq>> dags);

  // Sort the circuits in this equivalence class by CircuitSeq::less_than().
  void sort();

 private:
  std::vector<std::unique_ptr<CircuitSeq>> dags_;
};

}  // namespace quartz
