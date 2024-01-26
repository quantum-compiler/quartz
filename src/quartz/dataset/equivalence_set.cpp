#include "equivalence_set.h"

#include <cassert>
#include <fstream>
#include <limits>
#include <queue>

namespace quartz {
std::vector<CircuitSeq *> EquivalenceClass::get_all_dags() const {
  std::vector<CircuitSeq *> result;
  result.reserve(dags_.size());
  for (const auto &dag : dags_) {
    result.push_back(dag.get());
  }
  return result;
}

void EquivalenceClass::insert(std::unique_ptr<CircuitSeq> dag) {
  dags_.push_back(std::move(dag));
}

int EquivalenceClass::size() const { return (int)dags_.size(); }

void EquivalenceClass::reserve(std::size_t new_cap) { dags_.reserve(new_cap); }

std::vector<std::unique_ptr<CircuitSeq>> EquivalenceClass::extract() {
  return std::move(dags_);
}

void EquivalenceClass::set_dags(std::vector<std::unique_ptr<CircuitSeq>> dags) {
  dags_ = std::move(dags);
}

CircuitSeq *EquivalenceClass::get_representative() {
  assert(!dags_.empty());
  return dags_[0].get();
}

bool EquivalenceClass::contains(const CircuitSeq &dag) const {
  for (const auto &dag_in_class : dags_) {
    if (dag.fully_equivalent(*dag_in_class)) {
      return true;
    }
  }
  return false;
}

bool EquivalenceClass::set_as_representative(const CircuitSeq &dag) {
  if (dag.fully_equivalent(*dags_[0])) {
    // |circuitseq| is already the representative.
    return true;
  }
  for (int i = 1; i < (int)dags_.size(); i++) {
    if (dag.fully_equivalent(*dags_[i])) {
      std::swap(dags_[0], dags_[i]);
      return true;
    }
  }
  return false;
}

int EquivalenceClass::remove_common_first_or_last_gates(
    Context *ctx,
    std::unordered_set<CircuitSeqHashType> &hash_values_to_remove) {
  assert(hash_values_to_remove.empty());
  std::vector<CircuitGate *> all_first_gates, all_last_gates;
  std::vector<int> removing_ids;
  for (int i = 0; i < (int)dags_.size(); i++) {
    auto first_gates = dags_[i]->first_quantum_gates();
    auto last_gates = dags_[i]->last_quantum_gates();
    bool remove = false;
    for (auto &first_gate : first_gates) {
      if (remove) {
        break;
      }
      for (auto &other_first_gate : all_first_gates) {
        if (CircuitSeq::same_gate(first_gate, other_first_gate)) {
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
        if (CircuitSeq::same_gate(last_gate, other_last_gate)) {
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
      all_first_gates.insert(all_first_gates.end(), first_gates.begin(),
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
  for (int i = 0; i < (int)dags_.size(); i++) {
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

  std::vector<std::unique_ptr<CircuitSeq>> previous_dags;
  std::swap(dags_, previous_dags);
  // |dags_| is empty now.
  assert(previous_dags.size() >= removing_ids.size());
  dags_.reserve(previous_dags.size() - removing_ids.size());
  removing_it = removing_ids.begin();
  for (int i = 0; i < (int)previous_dags.size(); i++) {
    if (removing_it != removing_ids.end() && *removing_it == i) {
      removing_it++;
    } else {
      // not removed
      dags_.push_back(std::move(previous_dags[i]));
    }
  }
  return (int)removing_ids.size();
}

CircuitSeqHashType EquivalenceClass::hash(Context *ctx) {
  for (auto &dag : dags_) {
    if (dag) {
      // Not nullptr
      return dag->hash(ctx);
    }
  }
  return 0;  // empty class
}

void EquivalenceClass::sort() {
  std::sort(dags_.begin(), dags_.end(), UniquePtrCircuitSeqComparator());
}

bool EquivalenceClass::less_than(const EquivalenceClass &ecc1,
                                 const EquivalenceClass &ecc2) {
  if (ecc1.size() != ecc2.size()) {
    return ecc1.size() < ecc2.size();
  }
  for (int i = 0; i < ecc1.size(); i++) {
    // deal with nullptrs
    if (!ecc1.dags_[i]) {
      if (!ecc2.dags_[i]) {
        continue;
      }
      return false;
    }
    if (!ecc2.dags_[i]) {
      return true;
    }
    if (!ecc1.dags_[i]->fully_equivalent(*ecc2.dags_[i])) {
      return ecc1.dags_[i]->less_than(*ecc2.dags_[i]);
    }
  }
  return false;
}

bool EquivalenceSet::load_json(Context *ctx, const std::string &file_name,
                               bool from_verifier,
                               std::vector<CircuitSeq *> *new_representatives) {
  std::ifstream fin;
  fin.open(file_name, std::ifstream::in);
  if (!fin.is_open()) {
    std::cerr << "EquivalenceSet fails to open " << file_name << std::endl;
    return false;
  }

  // If the current equivalence set is not empty, keep the
  // representatives.
  std::vector<std::unique_ptr<CircuitSeq>> representatives;
  representatives.reserve(classes_.size());
  for (auto &item : classes_) {
    auto dags = item->extract();
    if (!dags.empty()) {
      representatives.push_back(std::move(dags[0]));
    }
  }
  clear();

  // Equivalences between equivalence classes with different hash values.
  using EquivClassTag = std::pair<CircuitSeqHashType, int>;
  // This vector stores gates in an undirected graph with wires being
  // equivalence classes.
  std::unordered_map<EquivClassTag, std::vector<EquivClassTag>, PairHash>
      equiv_edges;
  fin.ignore(std::numeric_limits<std::streamsize>::max(), '[');
  if (!from_verifier) {
    ctx->load_param_info_from_json(fin);
  } else {
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

      CircuitSeqHashType hash_value;
      int id;

      // the tags
      fin.ignore(std::numeric_limits<std::streamsize>::max(), '\"');
      fin >> std::hex >> hash_value;
      fin.ignore();  // '_'
      fin >> std::dec >> id;
      fin.ignore(std::numeric_limits<std::streamsize>::max(), '\"');
      EquivClassTag class1 = std::make_pair(hash_value, id);

      fin.ignore(std::numeric_limits<std::streamsize>::max(), '\"');
      fin >> std::hex >> hash_value;
      fin.ignore();  // '_'
      fin >> std::dec >> id;
      fin.ignore(std::numeric_limits<std::streamsize>::max(), '\"');
      EquivClassTag class2 = std::make_pair(hash_value, id);

      equiv_edges[class1].push_back(class2);
      equiv_edges[class2].push_back(class1);
      fin.ignore(std::numeric_limits<std::streamsize>::max(), ']');
    }
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
  std::vector<EquivalenceClass *> merged_equiv_class(num_merged_equiv_classes,
                                                     nullptr);

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
    CircuitSeqHashType hash_value;
    fin >> std::hex >> hash_value;
    fin.ignore();  // '_'
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

      // New CircuitSeq
      fin.unget();  // '['
      auto dag = CircuitSeq::read_json(ctx, fin);
      auto dag_hash_value = dag->hash(ctx);
      // Due to floating point errors and for compatibility of
      // different platforms, |dag_hash_value| can be different from
      // |hash_value|. So we have recalculated it here.
      set_possible_class(dag_hash_value, equiv_class);
      for (const auto &other_hash_value : dag->other_hash_values()) {
        set_possible_class(other_hash_value, equiv_class);
      }
      equiv_class->insert(std::move(dag));
    }

    // If the equivalence class is merged with some others,
    // we need to select the new representative as the smaller one.
    // Select the new representative can be done in O(1), but doing in
    // O(nlogn) where n is the number of DAGs is fine here.
    // For some reason, even if the equivalence class is not merged with any
    // others here, we still need to sort the equivalence class.
    equiv_class->sort();
  }

  // Move all previous representatives to the beginning of the
  // corresponding equivalence class, and find new representatives.
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

bool EquivalenceSet::save_json(Context *ctx,
                               const std::string &save_file_name) const {
  std::ofstream fout;
  fout.open(save_file_name, std::ofstream::out);
  if (!fout.is_open()) {
    return false;
  }

  fout << "[" << std::endl;

  fout << ctx->param_info_to_json() << "," << std::endl;

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
      fout << dag->to_json(/*keep_hash_value=*/false);
    }
    fout << "]" << std::endl;
  }
  fout << "}" << std::endl;

  // To adapt the format
  fout << "]" << std::endl;

  return true;
}

void EquivalenceSet::clear() {
  possible_classes_.clear();
  classes_.clear();
}

std::unique_ptr<RepresentativeSet>
EquivalenceSet::get_representative_set() const {
  auto rep_set = std::make_unique<RepresentativeSet>();
  rep_set->reserve(num_equivalence_classes());
  for (auto &item : classes_) {
    rep_set->insert(item->get_representative()->clone());
  }
  return rep_set;
}

bool EquivalenceSet::simplify(Context *ctx,
                              bool normalize_to_minimal_circuit_representation,
                              bool common_subcircuit_pruning,
                              bool other_simplification, bool verbose) {
  bool ever_simplified = false;
  // If there are 2 continuous optimizations with no effect, break.
  // This number should be the total number of optimizations minus one.
  constexpr int kNumOptimizationsToPerform = 7;
  // Initially we want to run all optimizations once.
  int remaining_optimizations = kNumOptimizationsToPerform + 1;
  while (true) {
    if (other_simplification && remove_singletons(ctx, verbose)) {
      remaining_optimizations = kNumOptimizationsToPerform;
      ever_simplified = true;
    } else if (!--remaining_optimizations) {
      break;
    }
    if (normalize_to_minimal_circuit_representation &&
        normalize_to_canonical_representations(ctx, verbose)) {
      remaining_optimizations = kNumOptimizationsToPerform;
      ever_simplified = true;
    } else if (!--remaining_optimizations) {
      break;
    }
    if (other_simplification && remove_unused_qubits(ctx, verbose)) {
      remaining_optimizations = kNumOptimizationsToPerform;
      ever_simplified = true;
    } else if (!--remaining_optimizations) {
      break;
    }
    if (other_simplification && remove_parameter_non_prefix(ctx, verbose)) {
      remaining_optimizations = kNumOptimizationsToPerform;
      ever_simplified = true;
    } else if (!--remaining_optimizations) {
      break;
    }
    if (other_simplification &&
        remove_parameter_expression_substitutions(ctx, verbose)) {
      remaining_optimizations = kNumOptimizationsToPerform;
      ever_simplified = true;
    } else if (!--remaining_optimizations) {
      break;
    }
    if (other_simplification && remove_parameter_permutations(ctx, verbose)) {
      remaining_optimizations = kNumOptimizationsToPerform;
      ever_simplified = true;
    } else if (!--remaining_optimizations) {
      break;
    }
    if (other_simplification && remove_qubit_permutations(ctx, verbose)) {
      remaining_optimizations = kNumOptimizationsToPerform;
      ever_simplified = true;
    } else if (!--remaining_optimizations) {
      break;
    }
    if (common_subcircuit_pruning &&
        remove_common_first_or_last_gates(ctx, verbose)) {
      remaining_optimizations = kNumOptimizationsToPerform;
      ever_simplified = true;
    } else if (!--remaining_optimizations) {
      break;
    }
  }
  return ever_simplified;
}

void EquivalenceSet::sort() {
  for (auto &item : classes_) {
    item->sort();
  }
}

int EquivalenceSet::remove_singletons(Context *ctx, bool verbose) {
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
        if (verbose) {
          std::cout << "Remove singleton: "
                    << item->get_all_dags()[0]->hash(ctx) << std::endl;
        }
        for (auto &dag : item->get_all_dags()) {
          remove_possible_class(dag->hash(ctx), item.get());
          for (const auto &other_hash : dag->other_hash_values()) {
            remove_possible_class(other_hash, item.get());
          }
        }
      } else {
        if (verbose) {
          std::cout << "Remove empty ECC" << std::endl;
        }
      }
    }
  }
  assert(num_removed > 0);
  return num_removed;
}

int EquivalenceSet::normalize_to_canonical_representations(Context *ctx,
                                                           bool verbose) {
  int num_class_modified = 0;
  for (auto &item : classes_) {
    auto dags = item->extract();
    std::vector<std::unique_ptr<CircuitSeq>> new_dags;
    std::unique_ptr<CircuitSeq> new_dag;
    std::unordered_set<CircuitSeqHashType> hash_values_to_remove;
    int class_modified = 0;
    for (auto &dag : dags) {
      bool is_minimal = dag->canonical_representation(&new_dag, ctx);
      if (!is_minimal) {
        class_modified++;
        new_dags.push_back(std::move(new_dag));
        hash_values_to_remove.insert(dag->hash(ctx));
        for (const auto &other_hash : dag->other_hash_values()) {
          hash_values_to_remove.insert(other_hash);
        }
        dag = nullptr;  // delete the CircuitSeq
      }
    }
    if (!class_modified) {
      item->set_dags(std::move(dags));
      continue;
    }
    if (verbose) {
      std::cout << "Normalize to minimal circuit representations: "
                << new_dags[0]->hash(ctx) << ": " << class_modified
                << " DAGs modified." << std::endl;
    }
    std::unordered_set<CircuitSeqHashType> existing_hash_values;
    std::unordered_set<CircuitSeqHashType> hash_values_to_insert;
    num_class_modified++;
    item->set_dags({});  // insert the DAGs one by one
    for (auto &dag : dags) {
      if (dag) {
        existing_hash_values.insert(dag->hash(ctx));
        for (const auto &other_hash : dag->other_hash_values()) {
          existing_hash_values.insert(other_hash);
        }
        item->insert(std::move(dag));
      }
    }
    for (auto &hash_value : existing_hash_values) {
      hash_values_to_remove.erase(hash_value);
    }
    for (auto &dag : new_dags) {
      // New DAGs can be identical to some DAGs in the ECC.
      if (!item->contains(*dag)) {
        hash_values_to_insert.insert(dag->hash(ctx));
        for (const auto &other_hash : dag->other_hash_values()) {
          hash_values_to_insert.insert(other_hash);
        }
        item->insert(std::move(dag));
      }
    }
    // Update the hash values.
    for (auto &hash_value : hash_values_to_remove) {
      if (hash_values_to_insert.count(hash_value) > 0) {
        hash_values_to_insert.erase(hash_value);
      } else {
        // Remove the hash value.
        remove_possible_class(hash_value, item.get());
      }
    }
    for (auto &hash_value : hash_values_to_insert) {
      if (existing_hash_values.count(hash_value) == 0) {
        // Add the hash value.
        set_possible_class(hash_value, item.get());
      }
    }
  }
  return num_class_modified;
}

int EquivalenceSet::remove_unused_qubits(Context *ctx, bool verbose) {
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
    for (const auto &dag : dags) {
      assert(qubit_used.size() == dag->get_num_qubits());
      for (int i = 0; i < (int)qubit_used.size(); i++) {
        if (!qubit_used[i]) {
          if (dag->qubit_used(i)) {
            qubit_used[i] = true;
          }
        }
      }
    }
    std::vector<int> unused_qubits;
    for (int i = 0; i < (int)qubit_used.size(); i++) {
      if (!qubit_used[i]) {
        unused_qubits.push_back(i);
      }
    }
    if (unused_qubits.empty()) {
      // No unused ones
      continue;
    }

    // Lazily remove the original CircuitSeq class.
    classes_to_remove.emplace_back(item.get());
    // Remove all pointers to the current class.
    for (auto &dag : dags) {
      remove_possible_class(dag->hash(ctx), item.get());
      for (const auto &other_hash : dag->other_hash_values()) {
        remove_possible_class(other_hash, item.get());
      }
    }

    if (verbose) {
      std::cout << "Remove unused qubits and input params: " << item->hash(ctx)
                << std::endl;
    }

    // Construct a new CircuitSeq class
    classes_to_insert.push_back(std::make_unique<EquivalenceClass>());
    auto &new_dag_class = classes_to_insert.back();
    new_dag_class->reserve(item->size());
    auto dags_unique_ptr = item->extract();
    bool already_exist = false;
    // We only need to check the first CircuitSeq to see if the class
    // already exists.
    bool first_dag = true;
    for (auto &dag : dags_unique_ptr) {
      bool ret = dag->remove_unused_qubits(unused_qubits);
      assert(ret);
      auto check_hash_value = [&](const CircuitSeqHashType &hash_value) {
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
    } else {
      // Add pointers to the new class.
      for (auto &dag : new_dag_class->get_all_dags()) {
        assert(dag);
        set_possible_class(dag->hash(ctx), new_dag_class.get());
        for (const auto &other_hash : dag->other_hash_values()) {
          set_possible_class(other_hash, new_dag_class.get());
        }
      }
      if (verbose) {
        std::cout << "ECC " << item->hash(ctx) << " -> new ECC "
                  << new_dag_class->hash(ctx) << std::endl;
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
  assert(prev_classes.size() + classes_to_insert.size() >=
         classes_to_remove.size());
  classes_.reserve(prev_classes.size() + classes_to_insert.size() -
                   classes_to_remove.size());
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

  return (int)classes_to_remove.size();
}

int EquivalenceSet::remove_common_first_or_last_gates(Context *ctx,
                                                      bool verbose) {
  int num_classes_modified = 0;
  for (auto &item : classes_) {
    std::unordered_set<CircuitSeqHashType> hash_values_to_remove;
    if (item->remove_common_first_or_last_gates(ctx, hash_values_to_remove)) {
      num_classes_modified++;
      if (verbose) {
        std::cout << "Remove common first of last gates: ECC "
                  << item->hash(ctx) << " modified." << std::endl;
      }
      for (const auto &hash_value : hash_values_to_remove) {
        remove_possible_class(hash_value, item.get());
      }
    }
  }
  return num_classes_modified;
}

int EquivalenceSet::remove_parameter_non_prefix(Context *ctx, bool verbose) {
  std::vector<EquivalenceClass *> classes_to_remove;
  auto param_masks = ctx->get_param_masks();
  const int num_input_param = ctx->get_num_input_symbolic_parameters();
  for (auto &item : classes_) {
    if (item->size() == 0) {
      continue;
    }
    InputParamMaskType param_mask{0};
    const auto &dags = item->get_all_dags();
    // Assume input symbolic parameters have the lowest indices.
    assert(num_input_param <= 63);
    for (auto &dag : dags) {
      param_mask = param_mask | dag->get_input_param_usage_mask(param_masks);
    }
    param_mask += 1;  // 0..001..1 -> 0..010..0
    if ((param_mask & (-param_mask)) != param_mask) {
      // Not a power of 2 now -- not 0..001..1 or 0 before +1.
      // A gap in input symbolic parameter mask detected.
      // Remove this class.
      lazily_remove_class(item.get(), dags, ctx, classes_to_remove, verbose,
                          "Remove parameter non-prefixes");
    }
  }
  return do_remove_classes(classes_to_remove);
}

int EquivalenceSet::remove_parameter_expression_substitutions(Context *ctx,
                                                              bool verbose) {
  std::vector<EquivalenceClass *> classes_to_remove;
  auto param_masks = ctx->get_param_masks();
  const int num_input_param = ctx->get_num_input_symbolic_parameters();
  for (auto &item : classes_) {
    if (item->size() == 0) {
      continue;
    }
    const auto &dags = item->get_all_dags();
    // Assume input symbolic parameters have the lowest indices.
    assert(num_input_param <= 63);
    InputParamMaskType non_input_used_once_in_all_circuits =
        (((InputParamMaskType)1) << num_input_param) - 1;
    for (auto &dag : dags) {
      InputParamMaskType current_usage_mask{0};
      InputParamMaskType current_usage_mask_used_twice{0};
      InputParamMaskType current_usage_mask_input_param_only{0};
      for (auto &circuit_gate : dag->gates) {
        // A quantum gate using parameters.
        if (circuit_gate->gate->is_parametrized_gate()) {
          for (auto &input_wire : circuit_gate->input_wires) {
            if (input_wire->is_parameter()) {
              if (input_wire->type == CircuitWire::input_param) {
                current_usage_mask_input_param_only |=
                    param_masks[input_wire->index];
              }
              current_usage_mask_used_twice |=
                  current_usage_mask & param_masks[input_wire->index];
              current_usage_mask |= param_masks[input_wire->index];
            }
          }
        }
      }
      auto non_input_used_once =
          current_usage_mask & (~(current_usage_mask_used_twice |
                                  current_usage_mask_input_param_only));
      non_input_used_once_in_all_circuits =
          non_input_used_once_in_all_circuits & non_input_used_once;
      if (!non_input_used_once_in_all_circuits) {
        break;
      }
    }
    if (non_input_used_once_in_all_circuits) {
      // If there is any input symbolic parameter satisfying the condition,
      // remove this equivalence class.
      lazily_remove_class(item.get(), dags, ctx, classes_to_remove, verbose,
                          "Remove parameter expression substitutions");
    }
  }
  return do_remove_classes(classes_to_remove);
}

int EquivalenceSet::remove_parameter_permutations(Context *ctx, bool verbose) {
  // This function needs a deterministic order of |classes_| in order for
  // the result to be reproducible.
  // Therefore, we sort the ECCs here.
  std::sort(classes_.begin(), classes_.end(),
            UniquePtrEquivalenceClassComparator());
  std::vector<EquivalenceClass *> classes_to_remove;
  const int num_input_param = ctx->get_num_input_symbolic_parameters();
  auto param_masks = ctx->get_param_masks();
  for (auto &item : classes_) {
    if (item->size() == 0) {
      continue;
    }
    const auto &dags = item->get_all_dags();
    // Compute which input parameters are used in all circuits in this ECC.
    auto param_mask = dags[0]->get_input_param_usage_mask(param_masks);
    for (auto &dag : dags) {
      param_mask = param_mask & dag->get_input_param_usage_mask(param_masks);
    }
    if (param_mask == 0 || param_mask == (param_mask & (-param_mask))) {
      // At most 1 common parameter.
      // No way to permute the parameters.
      continue;
    }
    // |qubit_permutation| is always the identity.
    std::vector<int> qubit_permutation(dags[0]->get_num_qubits());
    for (int i = 0; i < (int)qubit_permutation.size(); i++) {
      qubit_permutation[i] = i;
    }
    std::vector<int> masked_param_location;
    for (int i = 0; i < num_input_param; i++) {
      if (param_mask & (((InputParamMaskType)1) << i)) {
        masked_param_location.push_back(i);
      }
    }
    std::vector<int> masked_param_permutation(masked_param_location.size());
    for (int i = 0; i < (int)masked_param_location.size(); i++) {
      masked_param_permutation[i] = i;
    }
    std::vector<int> input_param_permutation(num_input_param);
    for (int i = 0; i < num_input_param; i++) {
      input_param_permutation[i] = i;
    }
    bool found_permuted_equivalence = false;
    do {
      // Get the permutation for all input parameters from the masked
      // permutation.
      for (int i = 0; i < (int)masked_param_location.size(); i++) {
        input_param_permutation[masked_param_location[i]] =
            masked_param_location[masked_param_permutation[i]];
      }
      // Check all permutations including the identity (because
      // we want to merge ECCs with the same CircuitSeq).
      std::set<EquivalenceClass *> permuted_classes;
      std::vector<std::unique_ptr<CircuitSeq>> permuted_dags;
      permuted_dags.reserve(dags.size());
      for (auto &dag : dags) {
        permuted_dags.emplace_back(dag->get_permuted_seq(
            qubit_permutation, input_param_permutation, ctx));
      }
      for (auto &permuted_dag : permuted_dags) {
        for (const auto &permuted_class :
             get_containing_class(ctx, permuted_dag.get())) {
          permuted_classes.insert(permuted_class);
        }
      }
      EquivalenceClass *permuted_class = nullptr;
      for (auto &c : permuted_classes) {
        if (c != item.get() && (!permuted_class || EquivalenceClass::less_than(
                                                       *c, *permuted_class))) {
          permuted_class = c;
        }
      }
      if (permuted_class) {
        found_permuted_equivalence = true;
        if (verbose) {
          std::cout << "Remove parameter permutations: found ECC "
                    << permuted_class->hash(ctx) << std::endl;
          for (auto &dag : permuted_class->get_all_dags()) {
            std::cout << "  " << dag->to_json() << std::endl;
          }
        }
        // Update the permuted class using this class.
        for (auto &permuted_dag : permuted_dags) {
          if (!permuted_class->contains(*permuted_dag)) {
            if (verbose) {
              std::cout << "Remove parameter permutations: insert "
                        << permuted_dag->to_json() << std::endl;
            }
            insert(ctx, permuted_class, std::move(permuted_dag));
          }
        }
        break;
      }
    } while (std::next_permutation(masked_param_permutation.begin(),
                                   masked_param_permutation.end()));
    if (found_permuted_equivalence) {
      // Remove this equivalence class.
      lazily_remove_class(item.get(), dags, ctx, classes_to_remove, verbose,
                          "Remove parameter permutations");
    }
  }

  return do_remove_classes(classes_to_remove);
}

int EquivalenceSet::remove_qubit_permutations(Context *ctx, bool verbose) {
  // This function needs a deterministic order of |classes_| in order for
  // the result to be reproducible.
  // Therefore, we sort the ECCs here.
  std::sort(classes_.begin(), classes_.end(),
            UniquePtrEquivalenceClassComparator());
  std::vector<EquivalenceClass *> classes_to_remove;
  for (auto &item : classes_) {
    if (item->size() == 0) {
      continue;
    }
    const auto &dags = item->get_all_dags();
    int num_qubits = dags[0]->get_num_qubits();
    if (num_qubits <= 1) {
      // No way to permute the qubit.
      continue;
    }
    std::vector<int> qubit_permutation(num_qubits);
    for (int i = 0; i < num_qubits; i++) {
      qubit_permutation[i] = i;
    }
    bool found_permuted_equivalence = false;
    do {
      // Check all permutations including the identity (because
      // we want to merge ECCs with the same CircuitSeq).
      std::set<EquivalenceClass *> permuted_classes;
      std::vector<std::unique_ptr<CircuitSeq>> permuted_dags;
      permuted_dags.reserve(dags.size());
      for (auto &dag : dags) {
        permuted_dags.emplace_back(dag->get_permuted_seq(
            qubit_permutation, /*no parameter permutation*/ {}, ctx));
      }
      for (auto &permuted_dag : permuted_dags) {
        for (const auto &permuted_class :
             get_containing_class(ctx, permuted_dag.get())) {
          permuted_classes.insert(permuted_class);
        }
      }
      EquivalenceClass *permuted_class = nullptr;
      for (auto &c : permuted_classes) {
        if (c != item.get() && (!permuted_class || EquivalenceClass::less_than(
                                                       *c, *permuted_class))) {
          permuted_class = c;
        }
      }
      if (permuted_class) {
        found_permuted_equivalence = true;
        if (verbose) {
          std::cout << "Remove qubit permutations: found ECC "
                    << permuted_class->hash(ctx) << std::endl;
          for (auto &dag : permuted_class->get_all_dags()) {
            std::cout << "  " << dag->to_json() << std::endl;
          }
        }
        // Update the permuted class using this class.
        for (auto &permuted_dag : permuted_dags) {
          if (!permuted_class->contains(*permuted_dag)) {
            if (verbose) {
              std::cout << "Remove qubit permutations: insert "
                        << permuted_dag->to_json() << std::endl;
            }
            insert(ctx, permuted_class, std::move(permuted_dag));
          }
        }
        break;
      }
    } while (std::next_permutation(qubit_permutation.begin(),
                                   qubit_permutation.end()));
    if (found_permuted_equivalence) {
      // Remove this equivalence class.
      lazily_remove_class(item.get(), dags, ctx, classes_to_remove, verbose,
                          "Remove qubit permutations");
    }
  }
  return do_remove_classes(classes_to_remove);
}

int EquivalenceSet::num_equivalence_classes() const {
  return (int)classes_.size();
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
        if (CircuitSeq::same_gate(*dag1, 0, *dag2, 0)) {
          int id = 0;
          bool same = true;
          while (dag1->gates[id]->gate->is_parameter_gate()) {
            // A prefix of only parameter gates doesn't count.
            id++;
            if (id >= dag1->get_num_gates() || id >= dag2->get_num_gates()) {
              same = false;
              break;
            }
            same = CircuitSeq::same_gate(*dag1, id, *dag2, id);
          }
          if (same) {
            return class_id;
          }
        }
        if (CircuitSeq::same_gate(*dag1, dag1->get_num_gates() - 1, *dag2,
                                  dag2->get_num_gates() - 1)) {
          assert(
              dag1->gates[dag1->get_num_gates() - 1]->gate->is_quantum_gate());
          return class_id;
        }
      }
    }
    class_id++;
  }
  return -1;  // no common first or last gates found
}

std::string EquivalenceSet::get_class_id(int num_class) const {
  return std::to_string(num_class) + "_" +
         std::to_string(classes_[num_class]->size());
}

std::vector<std::vector<CircuitSeq *>>
EquivalenceSet::get_all_equivalence_sets() const {
  std::vector<std::vector<CircuitSeq *>> result;
  result.reserve(num_equivalence_classes());
  for (const auto &item : classes_) {
    result.push_back(item->get_all_dags());
  }
  return result;
}

std::vector<EquivalenceClass *> EquivalenceSet::get_possible_classes(
    const CircuitSeqHashType &hash_value) const {
  auto it = possible_classes_.find(hash_value);
  if (it == possible_classes_.end()) {
    return std::vector<EquivalenceClass *>();
  }
  return std::vector<EquivalenceClass *>(it->second.begin(), it->second.end());
}

void EquivalenceSet::insert_class(
    Context *ctx, std::unique_ptr<EquivalenceClass> equiv_class) {
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

void EquivalenceSet::insert(Context *ctx, EquivalenceClass *equiv_class,
                            std::unique_ptr<CircuitSeq> dag) {
  CircuitSeq *dag_backup = dag.get();
  equiv_class->insert(std::move(dag));
  set_possible_class(dag_backup->hash(ctx), equiv_class);
  for (const auto &other_hash : dag_backup->other_hash_values()) {
    set_possible_class(other_hash, equiv_class);
  }
}

std::vector<EquivalenceClass *>
EquivalenceSet::get_containing_class(Context *ctx, CircuitSeq *dag) const {
  std::set<EquivalenceClass *> result;
  auto possible_classes = get_possible_classes(dag->hash(ctx));
  for (auto &equiv_class : possible_classes) {
    if (equiv_class->contains(*dag)) {
      result.insert(equiv_class);
    }
  }
  auto possible_class_set = std::unordered_set<EquivalenceClass *>(
      possible_classes.begin(), possible_classes.end());
  for (const auto &other_hash : dag->other_hash_values()) {
    possible_classes = get_possible_classes(other_hash);
    for (auto &equiv_class : possible_classes) {
      if (possible_class_set.count(equiv_class) == 0) {
        // not cached
        possible_class_set.insert(equiv_class);
        if (equiv_class->contains(*dag)) {
          result.insert(equiv_class);
        }
      }
    }
  }
  return std::vector<EquivalenceClass *>(result.begin(), result.end());
}

void EquivalenceSet::lazily_remove_class(
    EquivalenceClass *equiv_class, const std::vector<CircuitSeq *> &dags,
    Context *ctx, std::vector<EquivalenceClass *> &classes_to_remove,
    bool verbose, const std::string &pass_name) {
  classes_to_remove.push_back(equiv_class);
  if (verbose) {
    std::cout << pass_name << ": remove " << equiv_class->hash(ctx)
              << std::endl;
    for (auto &dag : dags) {
      std::cout << "  " << dag->to_json() << std::endl;
    }
  }
  for (auto &dag : dags) {
    remove_possible_class(dag->hash(ctx), equiv_class);
    for (const auto &other_hash : dag->other_hash_values()) {
      remove_possible_class(other_hash, equiv_class);
    }
  }
}

int EquivalenceSet::do_remove_classes(
    const std::vector<EquivalenceClass *> &classes_to_remove) {
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

  return (int)classes_to_remove.size();
}

void EquivalenceSet::set_possible_class(const CircuitSeqHashType &hash_value,
                                        EquivalenceClass *equiv_class) {
  auto &possible_classes = possible_classes_[hash_value];
  possible_classes.insert(equiv_class);
}

void EquivalenceSet::remove_possible_class(const CircuitSeqHashType &hash_value,
                                           EquivalenceClass *equiv_class) {
  auto &possible_classes = possible_classes_[hash_value];
  possible_classes.erase(equiv_class);
}

}  // namespace quartz
