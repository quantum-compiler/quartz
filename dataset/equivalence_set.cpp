#include "equivalence_set.h"

#include <cassert>
#include <fstream>
#include <limits>
#include <queue>

std::vector<DAG *> EquivalenceClass::get_all_dags() const {
  std::vector<DAG *> result;
  result.reserve(dags_.size());
  for (const auto &dag : dags_) {
    result.push_back(dag.get());
  }
  return result;
}

void EquivalenceClass::insert(std::unique_ptr<DAG> dag) {
  dags_.push_back(std::move(dag));
}

int EquivalenceClass::size() const {
  return (int) dags_.size();
}

bool EquivalenceSet::load_json(Context *ctx, const std::string &file_name) {
  std::ifstream fin;
  fin.open(file_name, std::ifstream::in);
  if (!fin.is_open()) {
    return false;
  }

  // Equivalences between equivalence classes with different hash values.
  using EquivClassTag = std::pair<DAGHashType, int>;
  // This vector stores edges in an undirected graph with nodes being
  // equivalence classes.
  std::unordered_map<EquivClassTag, std::vector<EquivClassTag>, PairHash>
      equiv_edges;
  fin.ignore(std::numeric_limits<std::streamsize>::max(), '[');
  fin.ignore(std::numeric_limits<std::streamsize>::max(), '[');
  while (true) {
    char ch;
    fin.get(ch);
    while (ch != '[' && ch != ']') {
      fin.get(ch);
    }
    if (ch == ']') {
      break;
    }

    // New equivalence between a pair of equivalence class

    DAGHashType hash_value;
    int id;

    // the tags
    fin.ignore(std::numeric_limits<std::streamsize>::max(), '\"');
    fin >> std::hex >> hash_value;
    fin.ignore(); // '_'
    fin >> std::dec >> id;
    fin.ignore(std::numeric_limits<std::streamsize>::max(), '\"');
    EquivClassTag class1 = std::make_pair(hash_value, id);

    fin.ignore(std::numeric_limits<std::streamsize>::max(), '\"');
    fin >> std::hex >> hash_value;
    fin.ignore(); // '_'
    fin >> std::dec >> id;
    fin.ignore(std::numeric_limits<std::streamsize>::max(), '\"');
    EquivClassTag class2 = std::make_pair(hash_value, id);

    equiv_edges[class1].push_back(class2);
    equiv_edges[class2].push_back(class1);
    fin.ignore(std::numeric_limits<std::streamsize>::max(), ']');
  }

  // BFS to merge the equivalence classes.
  std::unordered_map<EquivClassTag, int, PairHash> merged_equiv_class_id;
  int num_merged_equiv_classes = 0;
  for (const auto &start_pair : equiv_edges) {
    const auto &start_node = start_pair.first;
    if (merged_equiv_class_id.count(start_node) > 0) {
      // Already searched.
      continue;
    }
    std::queue<EquivClassTag> to_visit;
    // Create a new equivalence class.
    merged_equiv_class_id[start_node] = num_merged_equiv_classes;
    to_visit.push(start_node);
    while (!to_visit.empty()) {
      auto node = to_visit.front();
      to_visit.pop();
      for (const auto &next_node : equiv_edges[node]) {
        if (merged_equiv_class_id.count(next_node) == 0) {
          // Not searched yet
          merged_equiv_class_id[next_node] = num_merged_equiv_classes;
          to_visit.push(next_node);
        }
      }
    }
    num_merged_equiv_classes++;
  }

  // The |num_merged_equiv_classes| classes are not yet created.
  std::vector<EquivalenceClass *>
      merged_equiv_class(num_merged_equiv_classes, nullptr);

  // Input the equivalence classes.
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

    // New equivalence class

    // the tag
    DAGHashType hash_value;
    fin >> std::hex >> hash_value;
    fin.ignore(); // '_'
    int id;
    fin >> std::dec >> id;
    EquivClassTag class_tag = std::make_pair(hash_value, id);
    bool merged = merged_equiv_class_id.count(class_tag) > 0;
    EquivalenceClass *equiv_class;
    if (merged) {
      if (!merged_equiv_class[merged_equiv_class_id[class_tag]]) {
        classes_.push_back(std::make_unique<EquivalenceClass>());
        merged_equiv_class[merged_equiv_class_id[class_tag]] =
            classes_.back().get();
      }
      equiv_class = merged_equiv_class[merged_equiv_class_id[class_tag]];
    } else {
      classes_.push_back(std::make_unique<EquivalenceClass>());
      equiv_class = classes_.back().get();
    }
    assert(equiv_class);

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
      auto dag = DAG::read_json(ctx, fin);
      auto dag_hash_value = dag->hash(ctx);
      // Due to floating point errors and for compatibility of different
      // platforms, |dag_hash_value| can be different from |hash_value|.
      // So we have recalculated it here.
      set_possible_class(dag_hash_value, equiv_class);
      equiv_class->insert(std::move(dag));
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
  for (const auto &item : dataset_prev) {
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
  assert(false);  // TODO
  // TODO: why is the result different for each run?
  auto old_dataset = std::move(dataset_prev);
  dataset_prev = std::unordered_map<DAGHashType,
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
      for (auto &new_equiv_set : dataset_prev[new_hash_tag]) {
        // Compare by the content of the DAG.
        if (new_equiv_set.count(set_minrep) > 0) {
          equiv_found = true;
          new_equiv_set_pos = &new_equiv_set;
          break;
        }
      }

      if (!equiv_found) {
        // If not found, insert a new equivalence set.
        dataset_prev[new_hash_tag].emplace_back();
        new_equiv_set_pos = &dataset_prev[new_hash_tag].back();
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

void EquivalenceSet::clear() {
  possible_classes_.clear();
  classes_.clear();
  // dataset_prev.clear();
}

int EquivalenceSet::remove_unused_qubits_and_input_params(Context *ctx) {
  assert(false);  // TODO
  // We cannot use vector here (otherwise there will be build errors).
  std::list<std::set<std::unique_ptr<DAG>, UniquePtrDAGComparator>>
      new_dag_sets;
  for (auto it0 = dataset_prev.begin(); it0 != dataset_prev.end();) {
    auto &item = *it0;
    for (auto it1 = item.second.begin(); it1 != item.second.end();) {
      auto &dag_set = *it1;
      assert(!dag_set.empty());
      auto &rep = *dag_set.begin();
      std::vector<bool> qubit_used(rep->get_num_qubits(), false);
      std::vector<bool>
          input_param_used(rep->get_num_input_parameters(), false);
      for (const auto &dag : dag_set) {
        assert(qubit_used.size() == dag->get_num_qubits());
        for (int i = 0; i < (int) qubit_used.size(); i++) {
          if (!qubit_used[i]) {
            if (dag->qubit_used(i)) {
              qubit_used[i] = true;
            }
          }
        }

        if (dag->get_num_input_parameters() > (int) input_param_used.size()) {
          input_param_used.resize(dag->get_num_input_parameters(), false);
        }
        for (int i = 0; i < dag->get_num_input_parameters(); i++) {
          if (!input_param_used[i]) {
            if (dag->input_param_used(i)) {
              input_param_used[i] = true;
            }
          }
        }
      }
      std::vector<int> unused_qubits, unused_input_params;
      for (int i = 0; i < (int) qubit_used.size(); i++) {
        if (!qubit_used[i]) {
          unused_qubits.push_back(i);
        }
      }
      for (int i = 0; i < (int) input_param_used.size(); i++) {
        if (!input_param_used[i]) {
          unused_input_params.push_back(i);
        }
      }
      if (unused_qubits.empty() && unused_input_params.empty()) {
        // No unused ones
        it1++;
        continue;
      }

      // Only keep the ones with a (possibly empty) suffix of input parameters
      // removed, because others must be redundant
      // Warning: this optimization presumes that initially all circuits
      // share the same number of input parameters.
      bool keep_dag_set = true;
      if (!unused_input_params.empty()) {
        for (int i = 0; i < (int) unused_input_params.size(); i++) {
          if (unused_input_params[i]
              != input_param_used.size() - unused_input_params.size() + i) {
            keep_dag_set = false;
            break;
          }
        }
      }

      if (keep_dag_set) {
        // Construct a new DAG set
        new_dag_sets.emplace_back();
        auto &new_dag_set = new_dag_sets.back();
        bool has_hash_value = false;
        DAGHashType hash_value;
        for (auto &dag : dag_set) {
          auto new_dag = std::make_unique<DAG>(*dag);  // clone the DAG here
          bool ret = new_dag->remove_unused_qubits(unused_qubits);
          assert(ret);
          ret = new_dag->remove_unused_input_params(unused_input_params);
          assert(ret);
          auto new_dag_hash = new_dag->hash(ctx);
          if (has_hash_value) {
            assert(new_dag_hash == hash_value);
          } else {
            hash_value = new_dag_hash;
            has_hash_value = true;
          }
          new_dag_set.insert(std::move(new_dag));
        }
        // Do not insert the new DAG set to the original dataset_prev
        // to avoid invalidating iterators
      }

      // Erase the original one
      auto it1_to_erase = it1;
      it1++;
      item.second.erase(it1_to_erase);
    }
    if (item.second.empty()) {
      // Erase the whole hash value
      auto it0_to_erase = it0;
      it0++;
      dataset_prev.erase(it0_to_erase);
    } else {
      it0++;
    }
  }
  const int num_dag_set_inserted = (int) new_dag_sets.size();
  for (auto &new_dag_set : new_dag_sets) {
    auto hash_value = (*new_dag_set.begin())->hash(ctx);
    // Note: we don't check if the new equivalence set is already equivalent
    // to an existing one here.
    dataset_prev[hash_value].push_back(std::move(new_dag_set));
  }
  return num_dag_set_inserted;
}

int EquivalenceSet::num_equivalence_classes() const {
  return (int) classes_.size();
}

int EquivalenceSet::num_total_dags() const {
  int result = 0;
  for (const auto &item : classes_) {
    result += item->size();
  }
  return result;
}

void EquivalenceSet::set_representatives(Context *ctx,
                                         std::vector<DAG *> *new_representatives) const {
  for (const auto &item : dataset_prev) {
    int id = 0;
    for (const auto &dag_set : item.second) {
      assert(!dag_set.empty());
      auto &rep = *dag_set.begin();
      // Warning: we need the Dataset class to preserve the order of DAGs with
      // the same hash value here.
      if (!ctx->has_representative(rep->hash(ctx), id)) {
        // Set rep as representative
        auto new_rep = std::make_unique<DAG>(*rep);
        if (new_representatives) {
          new_representatives->push_back(new_rep.get());
        }
        ctx->set_representative(std::move(new_rep), id);
      }
      id++;
    }
  }
}

DAGHashType EquivalenceSet::has_common_first_or_last_gates() const {
  for (const auto &item : dataset_prev) {
    for (const auto &dag_set : item.second) {
      // brute force here
      for (const auto &dag1 : dag_set) {
        if (dag1->get_num_gates() == 0) {
          continue;
        }
        for (const auto &dag2 : dag_set) {
          if (dag1 == dag2) {
            continue;
          }
          if (dag2->get_num_gates() == 0) {
            continue;
          }
          if (DAG::same_gate(*dag1, 0, *dag2, 0)) {
            int id = 0;
            bool same = true;
            while (dag1->edges[id]->gate->is_parameter_gate()) {
              // A prefix of only parameter gates doesn't count.
              id++;
              if (id >= dag1->get_num_gates() || id >= dag2->get_num_gates()) {
                same = false;
                break;
              }
              same = DAG::same_gate(*dag1, id, *dag2, id);
            }
            if (same) {
              return item.first;
            }
          }
          if (DAG::same_gate(*dag1,
                             dag1->get_num_gates() - 1,
                             *dag2,
                             dag2->get_num_gates() - 1)) {
            assert(dag1->edges[dag1->get_num_gates()
                - 1]->gate->is_quantum_gate());
            return item.first;
          }
        }
      }
    }
  }
  return 0;  // no common first or last gates found
}

std::vector<std::vector<DAG *>> EquivalenceSet::get_all_equivalence_sets() const {
  std::vector<std::vector<DAG *>> result;
  result.reserve(num_equivalence_classes());
  for (const auto &item : classes_) {
    result.push_back(item->get_all_dags());
  }
  return result;
}

void EquivalenceSet::set_possible_class(const DAGHashType &hash_value,
                                        EquivalenceClass *equiv_class) {
  auto &possible_classes = possible_classes_[hash_value];
  if (std::find(possible_classes.begin(), possible_classes.end(), equiv_class)
      == possible_classes.end()) {
    possible_classes.push_back(equiv_class);
  }
}
