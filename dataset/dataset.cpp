#include "dataset.h"

#include <fstream>
#include <iomanip>

int Dataset::num_hash_values() const {
  return (int) dataset.size();
}

int Dataset::num_total_dags() const {
  int ret = 0;
  for (const auto &it : dataset) {
    ret += (int) it.second.size();
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

  // The generated parameters for random testing.
  auto all_parameters = ctx->get_all_generated_parameters();
  fout << "[";
  bool start0 = true;
  for (auto &param : all_parameters) {
    if (start0) {
      start0 = false;
    } else {
      fout << ", ";
    }
    fout << std::scientific << std::setprecision(17) << param;
  }
  fout << "]," << std::endl;

  fout << "{" << std::endl;
  start0 = true;
  for (const auto &it : dataset) {
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
  for (auto it = dataset.begin(); it != dataset.end(); ) {
    if (it->second.size() != 1) {
      it++;
      continue;
    }
    auto &dag = it->second[0];
    dag->hash(ctx);
    bool found_possible_equivalence = false;
    for (auto &hash_value : dag->other_hash_values()) {
      if (dataset.count(hash_value) > 0) {
        found_possible_equivalence = true;
        break;
      }
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

int Dataset::normalize_to_minimal_circuit_representations(Context *ctx) {
  int num_removed = 0;
  for (auto &item : dataset) {
    auto &dags = item.second;
    auto size_before = dags.size();
    std::vector<std::unique_ptr<DAG>> new_dags;
    std::unique_ptr<DAG> new_dag;
    auto dag_already_exists_in_new_dags = [&] (const DAG &dag) {
      for (auto &other_dag : new_dags) {
        if (dag.fully_equivalent(*other_dag)) {
          return true;
        }
      }
      return false;
    };

    for (auto &dag : dags) {
      bool is_minimal = dag->minimal_circuit_representation(&new_dag);
      if (!is_minimal) {
        if (!dag_already_exists_in_new_dags(*new_dag)) {
          new_dags.push_back(std::move(new_dag));
        }
        dag = nullptr;  // delete the original DAG
      }
    }
    if (!new_dags.empty()) {
      // |item| is modified.
      for (auto &dag : dags) {
        // Put all dags into |new_dags|.
        if (dag != nullptr) {
          if (!dag_already_exists_in_new_dags(*dag)) {
            new_dags.push_back(std::move(dag));
          }
        }
      }
      dags = std::move(new_dags);  // update |dags|.
      auto size_after = dags.size();
      num_removed += (int) (size_before - size_after);
    }
  }
  return num_removed;
}

bool Dataset::insert(Context *ctx, std::unique_ptr<DAG> dag) {
  const auto hash_value = dag->hash(ctx);
  bool ret = dataset.count(hash_value) == 0;
  dataset[hash_value].push_back(std::move(dag));
  return ret;
}

void Dataset::clear() {
  dataset.clear();
}
