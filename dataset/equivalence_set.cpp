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
