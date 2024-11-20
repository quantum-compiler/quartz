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

int ParamParser::parse(std::string &token) {
  // Currently only support the format of
  // pi*0.123,
  // 0.123*pi,
  // 0.123*pi/2,
  // 0.123
  // pi
  // pi/2
  // 0.123/(2*pi)
  ParamType p = 0.0;
  bool negative = token[0] == '-';
  if (negative)
    token = token.substr(1);
  if (token.find("pi") == 0) {
    if (token == "pi") {
      // pi
      p = PI;
    } else {
      auto d = token.substr(3, std::string::npos);
      if (token[2] == '*') {
        // pi*0.123
        p = std::stod(d) * PI;
      } else if (token[2] == '/') {
        // pi/2
        p = PI / std::stod(d);
      } else {
        std::cerr << "Unsupported parameter format: " << token
                  << std::endl;
        assert(false);
      }
    }
  } else if (token.find("pi") != std::string::npos) {
    if (token.find('(') != std::string::npos) {
      assert(token.find('/') != std::string::npos);
      auto left_parenthesis_pos = token.find('(');
      // 0.123/(2*pi)
      p = std::stod(token.substr(0, token.find('/'))) / PI;
      p /= std::stod(
          token.substr(left_parenthesis_pos + 1,
                        token.find('*') - left_parenthesis_pos - 1));
    } else {
      // 0.123*pi
      auto d = token.substr(0, token.find('*'));
      p = std::stod(d) * PI;
      if (token.find('/') != std::string::npos) {
        // 0.123*pi/2
        p = p / std::stod(token.substr(token.find('/') + 1));
      }
    }
  } else {
    // 0.123
    p = std::stod(token);
  }
  if (negative)
    p = -p;
  if (parameters.count(p) == 0) {
    int param_id = ctx_->get_new_param_id(p);
    parameters[p] = param_id;
  }
  return parameters[p];
}

}  // namespace quartz
