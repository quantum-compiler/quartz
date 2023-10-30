#include "qasm_parser.h"

namespace quartz {

void find_and_replace_all(std::string &data, const std::string &tofind,
                          const std::string &toreplace) {
  size_t pos = data.find(tofind);
  while (pos != std::string::npos) {
    data.replace(pos, tofind.size(), toreplace);
    pos = data.find(tofind, pos + toreplace.size());
  }
}

void find_and_replace_first(std::string &data, const std::string &tofind,
                            const std::string &toreplace) {
  size_t pos = data.find(tofind);
  if (pos != std::string::npos) {
    data.replace(pos, tofind.size(), toreplace);
  }
}

void find_and_replace_last(std::string &data, const std::string &tofind,
                           const std::string &toreplace) {
  size_t pos = data.rfind(tofind);
  if (pos != std::string::npos) {
    data.replace(pos, tofind.size(), toreplace);
  }
}

int string_to_number(const std::string &input) {
  int ret = -1;
  for (size_t i = 0; i < input.length(); i++) {
    if (input[i] >= '0' && input[i] <= '9') {
      if (ret == -1) {
        ret = 0;
      }
      ret = ret * 10 + input[i] - '0';
    }
  }
  return ret;
}

bool is_gate_string(const std::string &token, GateType &type) {
#define PER_GATE(x, XGate)                                                     \
  if (token == std::string(#x)) {                                              \
    type = GateType::x;                                                        \
    return true;                                                               \
  }

#include "../gate/gates.inc.h"

#undef PER_GATE
  return false;
}

std::string strip(const std::string &input) {
  auto st = input.begin();
  while (st != input.end() && std::isspace(*st))
    ++st;
  if (st == input.end()) {
    return std::string();
  }
  auto ed = input.rbegin();
  while (std::isspace(*ed))
    ++ed;
  return std::string(st, ed.base());
}

}  // namespace quartz
