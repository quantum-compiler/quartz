#include "dataset.h"

#include <cassert>
#include <fstream>
#include <iomanip>

namespace quartz {
int Dataset::num_hash_values() const { return (int)dataset.size(); }

int Dataset::num_total_dags() const {
  int ret = 0;
  for (const auto &it : dataset) {
    ret += (int)it.second.size();
  }
  return ret;
}

bool Dataset::save_json(Context *ctx, const std::string &file_name) const {
  std::ofstream fout;
  fout.open(file_name, std::ofstream::out);
  if (!fout.is_open()) {
    return false;
  }

  fout << "[" << std::endl;

  fout << ctx->param_info_to_json() << "," << std::endl;

  fout << "{" << std::endl;
  bool start0 = true;
  for (const auto &it : dataset) {
    if (it.second.empty()) {
      // Empty CircuitSeq set
      continue;
    }
    if (start0) {
      start0 = false;
    } else {
      fout << ",";
    }
    fout << "\"" << std::hex << it.first << "\": [" << std::endl;
    bool start = true;
    for (const auto &dag : it.second) {
      if (start) {
        start = false;
      } else {
        fout << ",";
      }
      fout << dag->to_json();
    }
    fout << "]" << std::endl;
  }
  fout << "}" << std::endl;

  fout << "]" << std::endl;
  return true;
}

int Dataset::remove_singletons(Context *ctx) {
  int num_removed = 0;
  for (auto it = dataset.begin(); it != dataset.end();) {
    if (it->second.size() != 1) {
      it++;
      continue;
    }
    auto &dag = it->second[0];
    auto it_hash_value = dag->hash(ctx);
    bool found_possible_equivalence = false;
    for (auto &hash_value : dag->other_hash_values()) {
      auto find_other = dataset.find(hash_value);
      if (find_other != dataset.end() && !find_other->second.empty()) {
        found_possible_equivalence = true;
        break;
      }
      assert(hash_value == it_hash_value + 1);  // Only deal with this case...
    }
    // ...so that we know for sure that only DAGs with hash value equal
    // to |it_hash_value - 1| can have other_hash_values() containing
    // |it_hash_value|.
    auto find_other = dataset.find(it_hash_value - 1);
    if (find_other != dataset.end() && !find_other->second.empty()) {
      found_possible_equivalence = true;
    }
    if (found_possible_equivalence) {
      it++;
      continue;
    }
    // Remove |it|.
    auto remove_it = it;
    it++;
    dataset.erase(remove_it);
    num_removed++;
  }
  return num_removed;
}

int Dataset::normalize_to_canonical_representations(Context *ctx) {
  int num_removed = 0;
  std::vector<std::unique_ptr<CircuitSeq>> dags_to_insert_afterwards;
  auto dag_already_exists =
      [](const CircuitSeq &dag,
         const std::vector<std::unique_ptr<CircuitSeq>> &new_dags) {
        for (auto &other_dag : new_dags) {
          if (dag.fully_equivalent(*other_dag)) {
            return true;
          }
        }
        return false;
      };

  for (auto &item : dataset) {
    auto &current_hash_value = item.first;
    auto &dags = item.second;
    auto size_before = dags.size();
    std::vector<std::unique_ptr<CircuitSeq>> new_dags;
    std::unique_ptr<CircuitSeq> new_dag;

    for (auto &dag : dags) {
      bool is_canonical = dag->canonical_representation(&new_dag, ctx);
      if (!is_canonical) {
        if (!dag_already_exists(*new_dag, new_dags)) {
          new_dags.push_back(std::move(new_dag));
        }
        dag = nullptr;  // delete the original CircuitSeq
      }
    }
    if (!new_dags.empty()) {
      // |item| is modified.
      for (auto &dag : dags) {
        // Put all dags into |new_dags|.
        if (dag != nullptr) {
          if (!dag_already_exists(*dag, new_dags)) {
            new_dags.push_back(std::move(dag));
          }
        }
      }
      // Update |dags|.
      dags.clear();
      for (auto &dag : new_dags) {
        const auto hash_value = dag->hash(ctx);
        if (hash_value == current_hash_value) {
          dags.push_back(std::move(dag));
        } else {
          // The hash value changed due to floating-point errors.
          // Insert |circuitseq| later to avoid corrupting the iterator
          // of |dataset|.
          dags_to_insert_afterwards.push_back(std::move(dag));
        }
      }
      auto size_after = dags.size();
      num_removed += (int)(size_before - size_after);
    }
  }
  for (auto &dag : dags_to_insert_afterwards) {
    const auto hash_value = dag->hash(ctx);
    if (!dag_already_exists(*dag, dataset[hash_value])) {
      num_removed--;  // Insert |circuitseq| back.
      dataset[hash_value].push_back(std::move(dag));
    }
  }
  return num_removed;
}

void Dataset::sort() {
  for (auto &it : dataset) {
    std::sort(it.second.begin(), it.second.end(),
              UniquePtrCircuitSeqComparator());
  }
}

bool Dataset::insert(Context *ctx, std::unique_ptr<CircuitSeq> dag) {
  const auto hash_value = dag->hash(ctx);
  bool ret = dataset.count(hash_value) == 0;
  dataset[hash_value].push_back(std::move(dag));
  return ret;
}

bool Dataset::insert_to_nearby_set_if_exists(Context *ctx,
                                             std::unique_ptr<CircuitSeq> dag) {
  const auto hash_value = dag->hash(ctx);
  for (const auto &hash_value_offset : {0, 1, -1}) {
    auto it = dataset.find(hash_value + hash_value_offset);
    if (it != dataset.end()) {
      // Found a nearby set, insert the circuitseq.
      it->second.push_back(std::move(dag));
      return false;
    }
  }
  // The hash value is new.
  dataset[hash_value].push_back(std::move(dag));
  return true;
}

void Dataset::clear() {
  // Caveat here: if only dataset.clear() is called, the behavior will be
  // different with a brand new Dataset.
  dataset = std::unordered_map<CircuitSeqHashType,
                               std::vector<std::unique_ptr<CircuitSeq>>>();
}

}  // namespace quartz
