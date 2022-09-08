#pragma once

#include "../context/context.h"
#include "../dag/dag.h"

namespace quartz {

class RepresentativeSet {
public:
  bool load_json(Context *ctx, const std::string &file_name);

  bool save_json(const std::string &file_name) const;

  void clear();

  // Returns all DAGs in this representative set.
  [[nodiscard]] std::vector<DAG *> get_all_dags() const;

  void insert(std::unique_ptr<DAG> dag);

  [[nodiscard]] int size() const;
  void reserve(std::size_t new_cap);

  // Extract all DAGs in this equivalence class, and make this class
  // empty.
  std::vector<std::unique_ptr<DAG>> extract();

  // Replace |dags_| with |dags|.
  void set_dags(std::vector<std::unique_ptr<DAG>> dags);

  // Sort the circuits in this equivalence class by DAG::less_than().
  void sort();

private:
  std::vector<std::unique_ptr<DAG>> dags_;
};

} // namespace quartz
