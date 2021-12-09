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

void EquivalenceClass::reserve(std::size_t new_cap) {
  dags_.reserve(new_cap);
}

std::vector<std::unique_ptr<DAG>> EquivalenceClass::extract() {
  return std::move(dags_);
}

void EquivalenceClass::set_dags(std::vector<std::unique_ptr<DAG>> dags) {
  dags_ = std::move(dags);
}

DAG *EquivalenceClass::get_representative() {
  assert(!dags_.empty());
  return dags_[0].get();
}

bool EquivalenceClass::contains(const DAG &dag) const {
  for (const auto &dag_in_class : dags_) {
    if (dag.fully_equivalent(*dag_in_class)) {
      return true;
    }
  }
  return false;
}

bool EquivalenceClass::set_as_representative(const DAG &dag) {
  if (dag.fully_equivalent(*dags_[0])) {
    // |dag| is already the representative.
    return true;
  }
  for (int i = 1; i < (int) dags_.size(); i++) {
    if (dag.fully_equivalent(*dags_[i])) {
      std::swap(dags_[0], dags_[i]);
      return true;
    }
  }
  return false;
}

int EquivalenceClass::remove_common_first_or_last_gates(Context *ctx,
                                                        std::unordered_set<
                                                            DAGHashType> &hash_values_to_remove) {
  assert(hash_values_to_remove.empty());
  std::vector<DAGHyperEdge *> all_first_gates, all_last_gates;
  std::vector<int> removing_ids;
  for (int i = 0; i < (int) dags_.size(); i++) {
    auto first_gates = dags_[i]->first_quantum_gates();
    auto last_gates = dags_[i]->last_quantum_gates();
    bool remove = false;
    for (auto &first_gate : first_gates) {
      if (remove) {
        break;
      }
      for (auto &other_first_gate : all_first_gates) {
        if (DAG::same_gate(first_gate, other_first_gate)) {
          remove = true;
          break;
        }
      }
    }
    for (auto &last_gate : last_gates) {
      if (remove) {
        break;
      }
      for (auto &other_last_gate : all_last_gates) {
        if (DAG::same_gate(last_gate, other_last_gate)) {
          remove = true;
          break;
        }
      }
    }
    if (remove) {
      removing_ids.push_back(i);
      hash_values_to_remove.insert(dags_[i]->hash(ctx));
      for (const auto &other_hash : dags_[i]->other_hash_values()) {
        hash_values_to_remove.insert(other_hash);
      }
    } else {
      all_first_gates.insert(all_first_gates.end(),
                             first_gates.begin(),
                             first_gates.end());
      all_last_gates.insert(all_last_gates.end(), last_gates.begin(),
                            last_gates.end());
    }
  }
  if (removing_ids.empty()) {
    return 0;
  }

  // Update the pointers to this equivalence class.
  auto removing_it = removing_ids.begin();
  for (int i = 0; i < (int) dags_.size(); i++) {
    if (removing_it != removing_ids.end() && *removing_it == i) {
      removing_it++;
    } else {
      // Not removed, keep the hash values.
      hash_values_to_remove.erase(dags_[i]->hash(ctx));
      for (const auto &other_hash : dags_[i]->other_hash_values()) {
        hash_values_to_remove.erase(other_hash);
      }
      if (hash_values_to_remove.empty()) {
        break;
      }
    }
  }

  std::vector<std::unique_ptr<DAG>> previous_dags;
  std::swap(dags_, previous_dags);
  // |dags_| is empty now.
  assert(previous_dags.size() >= removing_ids.size());
  dags_.reserve(previous_dags.size() - removing_ids.size());
  removing_it = removing_ids.begin();
  for (int i = 0; i < (int) previous_dags.size(); i++) {
    if (removing_it != removing_ids.end() && *removing_it == i) {
      removing_it++;
    } else {
      // not removed
      dags_.push_back(std::move(previous_dags[i]));
    }
  }
  return (int) removing_ids.size();
}

int EquivalenceClass::remove_unused_internal_parameters(Context *ctx) {
  int num_dag_modified = 0;
  for (auto &dag : dags_) {
    if (dag->remove_unused_internal_parameters()) {
      num_dag_modified++;
      // Restore the hash value.
      // (probably |dag->hash_value_valid_ = true;| also works)
      dag->hash(ctx);
    }
  }
  return num_dag_modified;
}

bool EquivalenceSet::load_json(Context *ctx,
                               const std::string &file_name,
                               std::vector<DAG *> *new_representatives) {
  std::ifstream fin;
  fin.open(file_name, std::ifstream::in);
  if (!fin.is_open()) {
    return false;
  }

  // If the current equivalence set is not empty, keep the representatives.
  std::vector<std::unique_ptr<DAG>> representatives;
  representatives.reserve(classes_.size());
  for (auto &item : classes_) {
    auto dags = item->extract();
    if (!dags.empty()) {
      representatives.push_back(std::move(dags[0]));
    }
  }
  clear();

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
      for (const auto &other_hash_value : dag->other_hash_values()) {
        set_possible_class(other_hash_value, equiv_class);
      }
      equiv_class->insert(std::move(dag));
    }
  }

  // Move all previous representatives to the beginning of the corresponding
  // equivalence class, and find new representatives.
  std::unordered_set<EquivalenceClass *> existing_classes;
  for (auto &rep : representatives) {
    EquivalenceClass *found_equiv_class = nullptr;
    for (auto &equiv_class : get_possible_classes(rep->hash(ctx))) {
      if (equiv_class->set_as_representative(*rep)) {
        found_equiv_class = equiv_class;
        break;
      }
    }
    if (!found_equiv_class) {
      for (const auto &other_hash_value : rep->other_hash_values()) {
        for (auto &equiv_class : get_possible_classes(other_hash_value)) {
          if (equiv_class->set_as_representative(*rep)) {
            found_equiv_class = equiv_class;
            break;
          }
        }
        if (found_equiv_class) {
          break;
        }
      }
    }
    if (new_representatives) {
      existing_classes.insert(found_equiv_class);
    }
  }
  if (new_representatives) {
    for (auto &item : classes_) {
      if (existing_classes.count(item.get()) == 0) {
        // A new equivalence class.
        new_representatives->push_back(item->get_representative());
      }
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

  // To adapt the format
  fout << "[[]," << std::endl;

  fout << "{" << std::endl;
  bool start0 = true;
  int id = 0;
  for (const auto &item : classes_) {
    if (start0) {
      start0 = false;
    } else {
      fout << ",";
    }
    fout << "\"" << get_class_id(id++) << "\": [" << std::endl;
    bool start = true;
    for (const auto &dag : item->get_all_dags()) {
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

  // To adapt the format
  fout << "]" << std::endl;

  return true;
}

void EquivalenceSet::normalize_to_minimal_representations(Context *ctx) {
  assert(false);  // TODO
  /*
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
//      for (auto it = new_equiv_set.begin(); it != new_equiv_set.end(); ) {
//        // Optimization: if |dag|'s minimal-representation is already in the
//        // set, erase |dag|.
//        auto &dag = *it;
//        bool is_minimal = dag->minimal_circuit_representation(&dag_minrep);
//        if (!is_minimal && new_equiv_set.count(dag_minrep) > 0) {
//          new_equiv_set.erase(it++);
//        } else {
//          it++;
//        }
//      }
    }
  }
  */
}

void EquivalenceSet::clear() {
  possible_classes_.clear();
  classes_.clear();
}

bool EquivalenceSet::simplify(Context *ctx,
                              bool common_subcircuit_pruning,
                              bool other_simplification) {
  bool ever_simplified = false;
  // If there are 2 continuous optimizations with no effect, break.
  constexpr int kNumOptimizationsToPerform = 4;
  // Initially we want to run all optimizations once.
  int remaining_optimizations = kNumOptimizationsToPerform + 1;
  while (true) {
    if (other_simplification && remove_singletons(ctx)) {
      remaining_optimizations = kNumOptimizationsToPerform;
      ever_simplified = true;
    } else if (!--remaining_optimizations) {
      break;
    }
    if (other_simplification && remove_unused_internal_params(ctx)) {
      remaining_optimizations = kNumOptimizationsToPerform;
      ever_simplified = true;
    } else if (!--remaining_optimizations) {
      break;
    }
    if (other_simplification && remove_unused_qubits_and_input_params(ctx)) {
      remaining_optimizations = kNumOptimizationsToPerform;
      ever_simplified = true;
    } else if (!--remaining_optimizations) {
      break;
    }
    if (other_simplification && remove_parameter_permutations(ctx)) {
      remaining_optimizations = kNumOptimizationsToPerform;
      ever_simplified = true;
    } else if (!--remaining_optimizations) {
      break;
    }
    if (common_subcircuit_pruning && remove_common_first_or_last_gates(ctx)) {
      remaining_optimizations = kNumOptimizationsToPerform;
      ever_simplified = true;
    } else if (!--remaining_optimizations) {
      break;
    }
  }
  return ever_simplified;
}

int EquivalenceSet::remove_singletons(Context *ctx) {
  bool have_singletons_to_remove = false;
  for (auto &item : classes_) {
    if (item->size() <= 1) {
      have_singletons_to_remove = true;
      break;
    }
  }
  if (!have_singletons_to_remove) {
    return 0;
  }

  int num_removed = 0;
  std::vector<std::unique_ptr<EquivalenceClass>> prev_classes;
  std::swap(prev_classes, classes_);
  // Now |classes_| is empty.
  classes_.reserve(prev_classes.size());
  for (auto &item : prev_classes) {
    if (item->size() > 1) {
      classes_.push_back(std::move(item));
    } else {
      num_removed++;
      // Remove all pointers to the equivalence class.
      if (item->size() > 0) {
        for (auto &dag : item->get_all_dags()) {
          remove_possible_class(dag->hash(ctx), item.get());
          for (const auto &other_hash : dag->other_hash_values()) {
            remove_possible_class(other_hash, item.get());
          }
        }
      }
    }
  }
  assert(num_removed > 0);
  return num_removed;
}

int EquivalenceSet::remove_unused_internal_params(Context *ctx) {
  int num_class_modified = 0;
  for (auto &item : classes_) {
    if (item->remove_unused_internal_parameters(ctx)) {
      num_class_modified++;
    }
  }
  return num_class_modified;
}

int EquivalenceSet::remove_unused_qubits_and_input_params(Context *ctx) {
  std::vector<EquivalenceClass *> classes_to_remove;
  std::vector<std::unique_ptr<EquivalenceClass>> classes_to_insert;
  for (auto &item : classes_) {
    auto dags = item->get_all_dags();
    if (dags.empty()) {
      classes_to_remove.emplace_back(item.get());
      continue;
    }
    auto &rep = dags.front();
    std::vector<bool> qubit_used(rep->get_num_qubits(), false);
    std::vector<bool>
        input_param_used(rep->get_num_input_parameters(), false);
    for (const auto &dag : dags) {
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
      continue;
    }

    // Lazily remove the original DAG class.
    classes_to_remove.emplace_back(item.get());
    // Remove all pointers to the current class.
    for (auto &dag : dags) {
      remove_possible_class(dag->hash(ctx), item.get());
      for (const auto &other_hash : dag->other_hash_values()) {
        remove_possible_class(other_hash, item.get());
      }
    }

    // Only keep the ones with a (possibly empty) suffix of input parameters
    // removed, because others must be redundant
    // Warning: this optimization presumes that initially all circuits
    // share the same number of input parameters.
    bool keep_dag_class = true;
    if (!unused_input_params.empty()) {
      for (int i = 0; i < (int) unused_input_params.size(); i++) {
        if (unused_input_params[i]
            != input_param_used.size() - unused_input_params.size() + i) {
          keep_dag_class = false;
          break;
        }
      }
    }

    if (keep_dag_class) {
      // Construct a new DAG class
      classes_to_insert.push_back(std::make_unique<EquivalenceClass>());
      auto &new_dag_class = classes_to_insert.back();
      new_dag_class->reserve(item->size());
      auto dags_unique_ptr = item->extract();
      bool already_exist = false;
      // We only need to check the first DAG to see if the class already exists.
      bool first_dag = true;
      for (auto &dag : dags_unique_ptr) {
        bool ret = dag->remove_unused_qubits(unused_qubits);
        assert(ret);
        ret = dag->remove_unused_input_params(unused_input_params);
        assert(ret);
        auto check_hash_value = [&](const DAGHashType &hash_value) {
          if (already_exist) {
            return;
          }
          for (auto &possible_class : get_possible_classes(hash_value)) {
            for (auto &other_dag : possible_class->get_all_dags()) {
              if (dag->fully_equivalent(*other_dag)) {
                already_exist = true;
                break;
              }
            }
            if (already_exist) {
              break;
            }
          }
        };
        if (first_dag) {
          auto hash_value = dag->hash(ctx);
          check_hash_value(hash_value);
          for (const auto &other_hash : dag->other_hash_values()) {
            check_hash_value(other_hash);
          }
          if (already_exist) {
            break;
          }
          first_dag = false;
        }
        new_dag_class->insert(std::move(dag));
      }
      if (already_exist) {
        // Remove the new class.
        classes_to_insert.pop_back();
        keep_dag_class = false;  // unused
      } else {
        // Add pointers to the new class.
        for (auto &dag : new_dag_class->get_all_dags()) {
          assert(dag);
          set_possible_class(dag->hash(ctx), new_dag_class.get());
          for (const auto &other_hash : dag->other_hash_values()) {
            set_possible_class(other_hash, new_dag_class.get());
          }
        }
      }
    }
  }

  if (classes_to_remove.empty()) {
    assert(classes_to_insert.empty());
    return 0;
  }

  std::vector<std::unique_ptr<EquivalenceClass>> prev_classes;
  std::swap(prev_classes, classes_);
  // Now |classes_| is empty.
  assert(prev_classes.size() + classes_to_insert.size()
             >= classes_to_remove.size());
  classes_.reserve(prev_classes.size() + classes_to_insert.size()
                       - classes_to_remove.size());
  auto remove_it = classes_to_remove.begin();
  for (auto &item : prev_classes) {
    if (remove_it != classes_to_remove.end() && item.get() == *remove_it) {
      // Remove the equivalence class.
      remove_it++;
    } else {
      assert(item->size() > 0);
      classes_.push_back(std::move(item));
    }
  }

  for (auto &item : classes_to_insert) {
    classes_.push_back(std::move(item));
  }

  return (int) classes_to_remove.size();
}

int EquivalenceSet::remove_common_first_or_last_gates(Context *ctx) {
  int num_classes_modified = 0;
  for (auto &item : classes_) {
    std::unordered_set<DAGHashType> hash_values_to_remove;
    if (item->remove_common_first_or_last_gates(ctx, hash_values_to_remove)) {
      num_classes_modified++;
      for (const auto &hash_value : hash_values_to_remove) {
        remove_possible_class(hash_value, item.get());
      }
    }
  }
  return num_classes_modified;
}

int EquivalenceSet::remove_parameter_permutations(Context *ctx) {
  std::vector<EquivalenceClass *> classes_to_remove;
  for (auto &item : classes_) {
    if (item->size() == 0) {
      continue;
    }
    const auto &dags = item->get_all_dags();
    int min_num_input_param = dags[0]->get_num_input_parameters();
    for (auto &dag : dags) {
      min_num_input_param =
          std::min(min_num_input_param, dag->get_num_input_parameters());
      if (min_num_input_param <= 1) {
        break;
      }
    }
    if (min_num_input_param <= 1) {
      // No way to permute the parameters.
      continue;
    }
    // |qubit_permutation| is always the identity.
    std::vector<int> qubit_permutation(dags[0]->get_num_qubits());
    for (int i = 0; i < (int) qubit_permutation.size(); i++) {
      qubit_permutation[i] = i;
    }
    std::vector<int> param_permutation(min_num_input_param);
    for (int i = 0; i < min_num_input_param; i++) {
      param_permutation[i] = i;
    }
    bool found_permuted_equivalence = false;
    while (std::next_permutation(param_permutation.begin(),
                                 param_permutation.end())) {
      // Check all permutations except for the identity.
      EquivalenceClass *permuted_class = nullptr;
      std::vector<std::unique_ptr<DAG>> permuted_dags;
      permuted_dags.reserve(dags.size());
      for (auto &dag : dags) {
        permuted_dags.emplace_back(dag->get_permuted_dag(qubit_permutation,
                                                         param_permutation));
      }
      for (auto &permuted_dag : permuted_dags) {
        permuted_class = get_containing_class(ctx, permuted_dag.get());
        if (permuted_class) {
          // Found the permuted class.
          break;
        }
      }
      if (permuted_class && permuted_class != item.get()) {
        found_permuted_equivalence = true;
        // Update the permuted class using this class.
        for (auto &permuted_dag : permuted_dags) {
          if (!permuted_class->contains(*permuted_dag)) {
            insert(ctx, permuted_class, std::move(permuted_dag));
          }
        }
        break;
      }
    }
    if (found_permuted_equivalence) {
      // Remove this equivalence class.
      classes_to_remove.push_back(item.get());
      for (auto &dag : dags) {
        remove_possible_class(dag->hash(ctx), item.get());
        for (const auto &other_hash : dag->other_hash_values()) {
          remove_possible_class(other_hash, item.get());
        }
      }
    }
  }

  if (classes_to_remove.empty()) {
    return 0;
  }

  std::vector<std::unique_ptr<EquivalenceClass>> prev_classes;
  std::swap(prev_classes, classes_);
  // Now |classes_| is empty.
  assert(prev_classes.size() >= classes_to_remove.size());
  classes_.reserve(prev_classes.size() - classes_to_remove.size());
  auto remove_it = classes_to_remove.begin();
  for (auto &item : prev_classes) {
    if (remove_it != classes_to_remove.end() && item.get() == *remove_it) {
      // Remove the equivalence class.
      remove_it++;
    } else {
      assert(item->size() > 0);
      classes_.push_back(std::move(item));
    }
  }

  return (int) classes_to_remove.size();
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

int EquivalenceSet::first_class_with_common_first_or_last_gates() const {
  int class_id = 0;
  for (const auto &item : classes_) {
    const auto &dags = item->get_all_dags();
    // brute force here
    for (const auto &dag1 : dags) {
      if (dag1->get_num_gates() == 0) {
        continue;
      }
      for (const auto &dag2 : dags) {
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
            return class_id;
          }
        }
        if (DAG::same_gate(*dag1,
                           dag1->get_num_gates() - 1,
                           *dag2,
                           dag2->get_num_gates() - 1)) {
          assert(dag1->edges[dag1->get_num_gates()
              - 1]->gate->is_quantum_gate());
          return class_id;
        }
      }
    }
    class_id++;
  }
  return -1;  // no common first or last gates found
}

std::string EquivalenceSet::get_class_id(int num_class) const {
  return std::to_string(num_class) + "_"
      + std::to_string(classes_[num_class]->size());
}

std::vector<std::vector<DAG *>> EquivalenceSet::get_all_equivalence_sets() const {
  std::vector<std::vector<DAG *>> result;
  result.reserve(num_equivalence_classes());
  for (const auto &item : classes_) {
    result.push_back(item->get_all_dags());
  }
  return result;
}

std::vector<EquivalenceClass *> EquivalenceSet::get_possible_classes(const DAGHashType &hash_value) const {
  auto it = possible_classes_.find(hash_value);
  if (it == possible_classes_.end()) {
    return std::vector<EquivalenceClass *>();
  }
  return std::vector<EquivalenceClass *>(it->second.begin(),
                                         it->second.end());
}

void EquivalenceSet::insert_class(Context *ctx,
                                  std::unique_ptr<EquivalenceClass> equiv_class) {
  // Add pointers to the new class.
  for (auto &dag : equiv_class->get_all_dags()) {
    assert(dag);
    set_possible_class(dag->hash(ctx), equiv_class.get());
    for (const auto &other_hash : dag->other_hash_values()) {
      set_possible_class(other_hash, equiv_class.get());
    }
  }

  classes_.push_back(std::move(equiv_class));
}

void EquivalenceSet::insert(Context *ctx,
                            EquivalenceClass *equiv_class,
                            std::unique_ptr<DAG> dag) {
  DAG *dag_backup = dag.get();
  equiv_class->insert(std::move(dag));
  set_possible_class(dag_backup->hash(ctx), equiv_class);
  for (const auto &other_hash : dag_backup->other_hash_values()) {
    set_possible_class(other_hash, equiv_class);
  }
}

EquivalenceClass *EquivalenceSet::get_containing_class(Context *ctx,
                                                       DAG *dag) const {
  auto possible_classes = get_possible_classes(dag->hash(ctx));
  for (auto &equiv_class : possible_classes) {
    if (equiv_class->contains(*dag)) {
      return equiv_class;
    }
  }
  auto possible_class_set = std::unordered_set<EquivalenceClass *>(
      possible_classes.begin(),
      possible_classes.end());
  for (const auto &other_hash : dag->other_hash_values()) {
    possible_classes = get_possible_classes(other_hash);
    for (auto &equiv_class : possible_classes) {
      if (possible_class_set.count(equiv_class) == 0) {
        // not cached
        possible_class_set.insert(equiv_class);
        if (equiv_class->contains(*dag)) {
          return equiv_class;
        }
      }
    }
  }
  return nullptr;
}

void EquivalenceSet::set_possible_class(const DAGHashType &hash_value,
                                        EquivalenceClass *equiv_class) {
  auto &possible_classes = possible_classes_[hash_value];
  possible_classes.insert(equiv_class);
}

void EquivalenceSet::remove_possible_class(const DAGHashType &hash_value,
                                           EquivalenceClass *equiv_class) {
  auto &possible_classes = possible_classes_[hash_value];
  possible_classes.erase(equiv_class);
}
