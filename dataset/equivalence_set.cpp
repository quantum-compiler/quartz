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

void EquivalenceSet::normalize_to_minimal_representations(Context *ctx) {
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
      std::unique_ptr<DAG> dag_minrep = nullptr;  // temporary variables
      for (auto &dag : equiv_set) {
        dag->minimal_representation(&dag_minrep);
        if (!set_minrep || dag_minrep->less_than(*set_minrep)) {
          // destroying the previous content of |set_minrep|
          set_minrep = std::move(dag_minrep);
          set_minrep_pos = dag.get();
        }
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

      // TODO: implement
    }
  }
}
