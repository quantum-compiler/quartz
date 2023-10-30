#include "representative_set.h"

#include <algorithm>
#include <fstream>
#include <limits>

namespace quartz {

bool RepresentativeSet::load_json(Context *ctx, const std::string &file_name) {
  std::ifstream fin;
  fin.open(file_name, std::ifstream::in);
  if (!fin.is_open()) {
    std::cerr << "RepresentativeSet fails to open " << file_name << std::endl;
    return false;
  }

  // the DAGs
  fin.ignore(std::numeric_limits<std::streamsize>::max(), '[');
  char ch;
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
    dags_.emplace_back(std::move(dag));
  }
  return true;
}

bool RepresentativeSet::save_json(const std::string &save_file_name) const {
  std::ofstream fout;
  fout.open(save_file_name, std::ofstream::out);
  if (!fout.is_open()) {
    return false;
  }

  fout << "[" << std::endl;
  bool start = true;
  for (const auto &dag : dags_) {
    if (start) {
      start = false;
    } else {
      fout << "," << std::endl;
    }
    auto dag_with_newline = dag->to_json();
    // remove newline
    fout << dag_with_newline.substr(0, dag_with_newline.size() - 1);
  }
  fout << "\n]" << std::endl;

  return true;
}

void RepresentativeSet::clear() { dags_.clear(); }

std::vector<CircuitSeq *> RepresentativeSet::get_all_dags() const {
  std::vector<CircuitSeq *> result;
  result.reserve(dags_.size());
  for (const auto &dag : dags_) {
    result.push_back(dag.get());
  }
  return result;
}

void RepresentativeSet::insert(std::unique_ptr<CircuitSeq> dag) {
  dags_.push_back(std::move(dag));
}

int RepresentativeSet::size() const { return (int)dags_.size(); }

void RepresentativeSet::reserve(std::size_t new_cap) { dags_.reserve(new_cap); }

std::vector<std::unique_ptr<CircuitSeq>> RepresentativeSet::extract() {
  return std::move(dags_);
}

void RepresentativeSet::set_dags(
    std::vector<std::unique_ptr<CircuitSeq>> dags) {
  dags_ = std::move(dags);
}

void RepresentativeSet::sort() {
  std::sort(dags_.begin(), dags_.end(), UniquePtrCircuitSeqComparator());
}

}  // namespace quartz
