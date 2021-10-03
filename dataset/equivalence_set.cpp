#include "equivalence_set.h"

#include <cassert>
#include <fstream>
#include <limits>

bool EquivalenceSet::load_json(Context *ctx, const std::string &file_name) {
  std::ifstream fin;
  fin.open(file_name, std::ifstream::in);
  if (!fin.is_open()) {
    return false;
  }
  fin.ignore(std::numeric_limits<std::streamsize>::max(), '{');
  while (true) {
    char ch;
    fin.get(ch);
    while (ch != '\"' && ch != '}') {
      fin.get(ch);
    }
    if (ch == '}') {
      break;
    }

    // New equivalence item

    // the tag
    DAGHashType hash_value;
    fin >> std::hex >> hash_value;
    fin.ignore(); // '_'
    int id;
    fin >> std::dec >> id;
    bool insert_at_end_of_list = false;
    if (dataset[hash_value].size() <= id) {
      dataset[hash_value].resize(id + 1);
      insert_at_end_of_list = true;
    }
    auto insert_pos = dataset[hash_value].begin();
    if (!insert_at_end_of_list) {
      for (int i = 0; i < id; i++) {
        insert_pos++;
      }
    }
    auto &dag_set =
        (insert_at_end_of_list ? dataset[hash_value].back() : *insert_pos);
    assert (dag_set.empty());

    // the DAGs
    fin.ignore(std::numeric_limits<std::streamsize>::max(), '[');
    while (true) {
      fin.get(ch);
      while (ch != '[' && ch != ']') {
        fin.get(ch);
      }
      if (ch == ']') {
        break;
      }

      // New DAG
      fin.unget();  // '['
      dag_set.insert(DAG::read_json(ctx, fin));
    }
  }
  return true;
}

bool EquivalenceSet::save_json(const std::string &save_file_name) const {
  std::ofstream fout;
  fout.open(save_file_name, std::ofstream::out);
  if (!fout.is_open()) {
    return false;
  }
  fout << "{" << std::endl;
  bool start0 = true;
  for (const auto &item : dataset) {
    int id = 0;
    for (const auto &equiv_set : item.second) {
      if (start0) {
        start0 = false;
      } else {
        fout << ",";
      }
      fout << "\"" << std::hex << item.first << "_" << std::dec << id++
           << "\": [" << std::endl;
      bool start = true;
      for (const auto &dag : equiv_set) {
        if (start) {
          start = false;
        } else {
          fout << ",";
        }
        fout << dag->to_json();
      }
      fout << "]" << std::endl;
    }
  }
  fout << "}" << std::endl;
  return true;
}

void EquivalenceSet::normalize_to_minimal_representations(Context *ctx) {
  // TODO: why is the result different for each run?
  auto old_dataset = std::move(dataset);
  dataset = std::unordered_map<DAGHashType,
                               std::list<std::set<std::unique_ptr<DAG>,
                                                  UniquePtrDAGComparator>>>();
  for (auto &item : old_dataset) {
    const auto &hash_tag = item.first;
    auto &equivalence_list = item.second;
    for (auto &equiv_set : equivalence_list) {
      // Compute the minimal minimal-representation in the set.
      DAG *set_minrep_pos = nullptr;
      std::unique_ptr<DAG> set_minrep = nullptr;
      std::vector<int> set_qubit_perm, set_param_perm;
      std::unique_ptr<DAG> dag_minrep = nullptr;  // temporary variables
      std::vector<int> dag_qubit_perm, dag_param_perm;
      bool trivial_equivalence = true;
      for (auto &dag : equiv_set) {
        dag->minimal_representation(&dag_minrep,
                                    &dag_qubit_perm,
                                    &dag_param_perm);
        if (!set_minrep || dag_minrep->less_than(*set_minrep)) {
          if (set_minrep) {
            // We found two DAGs with different minimal-representations in the
            // set.
            trivial_equivalence = false;
          }
          // destroying the previous content of |set_minrep|
          set_minrep = std::move(dag_minrep);
          set_qubit_perm = std::move(dag_qubit_perm);
          set_param_perm = std::move(dag_param_perm);
          set_minrep_pos = dag.get();
        } else if (!dag_minrep->fully_equivalent(*set_minrep)) {
          // We found two DAGs with different minimal-representations in the
          // set.
          trivial_equivalence = false;
        }
      }
      if (trivial_equivalence) {
        continue;
      }
      assert (set_minrep);
      const auto new_hash_tag = set_minrep->hash(ctx);

      // Find this equivalence in the new equivalence set.
      bool equiv_found = false;
      std::set<std::unique_ptr<DAG>, UniquePtrDAGComparator>
          *new_equiv_set_pos = nullptr;
      for (auto &new_equiv_set : dataset[new_hash_tag]) {
        // Compare by the content of the DAG.
        if (new_equiv_set.count(set_minrep) > 0) {
          equiv_found = true;
          new_equiv_set_pos = &new_equiv_set;
          break;
        }
      }

      if (!equiv_found) {
        // If not found, insert a new equivalence set.
        dataset[new_hash_tag].emplace_back();
        new_equiv_set_pos = &dataset[new_hash_tag].back();
      }

      // Insert the permuted DAGs into the new equivalence set.
      auto &new_equiv_set = *new_equiv_set_pos;
      for (auto &dag : equiv_set) {
        if (dag.get() == set_minrep_pos) {
          // Optimization: if |dag| is the DAG with the minimal
          // minimal-representation in the set, do not re-compute the
          // permuted DAG.
          if (equiv_found) {
            // Already in |new_equiv_set|.
            continue;
          }
          new_equiv_set.insert(std::move(set_minrep));
          // Note that |set_minrep| is not usable anymore after std::move.
        } else {
          // Optimization: if |dag|'s minimal-representation is already in the
          // set, do not insert it again.
          dag->minimal_representation(&dag_minrep);
          if (new_equiv_set.count(dag_minrep) > 0) {
            continue;
          }
          new_equiv_set.insert(dag->get_permuted_dag(set_qubit_perm,
                                                     set_param_perm));
        }
      }
      // TODO: why does this increase the number of equivalence classes?
      /*for (auto it = new_equiv_set.begin(); it != new_equiv_set.end(); ) {
        // Optimization: if |dag|'s minimal-representation is already in the
        // set, erase |dag|.
        auto &dag = *it;
        bool is_minimal = dag->minimal_representation(&dag_minrep);
        if (!is_minimal && new_equiv_set.count(dag_minrep) > 0) {
          new_equiv_set.erase(it++);
        } else {
          it++;
        }
      }*/
    }
  }
}

int EquivalenceSet::num_equivalence_classes() const {
  int result = 0;
  for (const auto &item : dataset) {
    result += (int) item.second.size();
  }
  return result;
}

int EquivalenceSet::num_total_dags() const {
  int result = 0;
  for (const auto &item : dataset) {
    for (const auto &dag_set : item.second) {
      result += (int) dag_set.size();
    }
  }
  return result;
}
