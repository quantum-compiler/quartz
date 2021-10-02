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
    unsigned long long hash_value;
    fin >> std::hex >> hash_value;
    fin.ignore(); // '_'
    int id;
    fin >> std::dec >> id;
    EquivalenceHashType tag = std::make_pair(hash_value, id);
    assert (dataset.count(tag) == 0);
    auto &dag_set = dataset[tag];

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
