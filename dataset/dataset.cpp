#include "dataset.h"

#include <fstream>

void Dataset::save_json(const std::string &file_name) {
  std::ofstream fout;
  fout.open(file_name, std::ofstream::out);
  fout << "{" << std::endl;
  bool start0 = true;
  for (auto &it : dataset) {
    if (start0) {
      start0 = false;
    } else {
      fout << ",";
    }
    fout << "\"" << std::hex << it.first << "\": [" << std::endl;
    bool start = true;
    for (auto &dag : it.second) {
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
}
