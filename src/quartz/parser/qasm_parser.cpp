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

int ParamParser::parse_number(bool negative, ParamType p) {
  if (negative) {
    p = -p;
  }

  if (number_params_.count(p) == 0) {
    int param_id = ctx_->get_new_param_id(p);
    number_params_[p] = param_id;
  }

  return number_params_[p];
}

int ParamParser::parse_pi_expr(bool negative, ParamType n, ParamType d) {
  // If pi is not symbolic, then falls back to constants.
  if (!symbolic_pi_) {
    return parse_number(negative, n * PI / d);
  }

  // Handles negative coefficients.
  if (negative) {
    n = -n;
  }

  // Constructs the pi expression, if it does not already exist.
  if (pi_params_[n].count(d) == 0) {
    // Checks if fraction of pi already exists.
    // If (n == 1) then this will cache the final expression.
    if (pi_params_[1].count(d) == 0) {
      int id = parse_number(false, d);
      auto gate = ctx_->get_gate(GateType::pi);
      pi_params_[1][d] = ctx_->get_new_param_expression_id({id}, gate);
    }

    // Scales the fraction of pi when the numerator is not equal to 1.
    // If (n != 1), then this will cache the final expression.
    if (n != 1) {
      int nid = parse_number(false, n);
      int pid = pi_params_[1][d];
      auto gate = ctx_->get_gate(GateType::mult);
      pi_params_[n][d] = ctx_->get_new_param_expression_id({nid, pid}, gate);
    }
  }

  // Retrieves expression.
  return pi_params_[n][d];
}

bool ParamParser::parse_array_decl(std::stringstream &ss) {
  // The first two tokens the stream should be '[angle len]'. Recall that the
  // comma between angle and len has been replacedby a space.
  std::string type;
  ss >> type;
  type = strip(type);
  if (type != "[angle") {
    if (type.size() > 1) {
      type = type.substr(1);
    }
    std::cerr << "Unexpected array type: " << type << std::endl;
    assert(false);
    return false;
  }

  // Next, it is possible to retrieve the length of the array.
  std::string lstr;
  ss >> lstr;
  int len = string_to_number(lstr);
  if (len == -1) {
    std::cerr << "Invalid parameter array length: " << len << std::endl;
    assert(false);
    return false;
  }

  // The rest of the line should be the name of the array.
  std::string name;
  getline(ss, name);
  name = strip(name);

  // Ensures that the name is unique.
  if (symb_params_.count(name) > 0) {
    std::cerr << "Each param must have a unique name: " << name << std::endl;
    assert(false);
    return false;
  }

  // Allocates a symbolic parameter for each element of the array.
  for (int i = 0; i < len; ++i) {
    symb_params_[name][i] = ctx_->get_new_param_id();
  }
  return true;
}

int ParamParser::parse_expr(std::stringstream &ss) {
  // Extracts the parameter expression from the string stream.
  std::string token;
  ss >> token;

  // Determines if angle is negative or positive.
  bool negative = (token[0] == '-');
  if (negative) {
    token = token.substr(1);
  }

  // Currently only support the format of:
  //  pi*0.123,
  //  0.123*pi,
  //  0.123*pi/2,
  //  0.123
  //  pi
  //  pi/2
  //  0.123/(2*pi)
  if (token.find("pi") == 0) {
    if (token == "pi") {
      // Case: pi
      return parse_pi_expr(negative, 1.0, 1.0);
    } else {
      // Cases: pi*0.123 or pi/2
      auto d = token.substr(3, std::string::npos);
      if (token[2] == '*') {
        // Case: pi*0.123
        return parse_pi_expr(negative, std::stod(d), 1.0);
      } else if (token[2] == '/') {
        // Case: pi/2
        return parse_pi_expr(negative, 1.0, std::stod(d));
      } else {
        std::cerr << "Unsupported parameter format: " << token << std::endl;
        assert(false);
      }
    }
  } else if (token.find("pi") != std::string::npos) {
    if (token.find('(') != std::string::npos) {
      // Case: 0.123/(2*pi)
      assert(token.find('/') != std::string::npos);
      auto lparen_pos = token.find('(');
      auto mult_pos = token.find('*');

      ParamType p = std::stod(token.substr(0, token.find('/')));
      p /= PI;
      p /= std::stod(token.substr(lparen_pos + 1, mult_pos - lparen_pos - 1));
      return parse_number(negative, p);
    } else {
      // Case: 0.123*pi or 0.123*pi/2
      auto d = token.substr(0, token.find('*'));
      ParamType num = std::stod(d);
      ParamType denom = 1.0;
      if (token.find('/') != std::string::npos) {
        // Case: 0.123*pi/2
        denom = std::stod(token.substr(token.find('/') + 1));
      }
      return parse_pi_expr(negative, num, denom);
    }
  } else {
    // Case: 0.123
    return parse_number(negative, std::stod(token));
  }

  // This line should be unreachable.
  std::cerr << "Unknown parameter value expression: " << token << std::endl;
  assert(false);
  return -1;
}

bool QubitParser::parse_qasm2_decl(std::stringstream &ss) {
  std::string name;
  std::string len_str;
  getline(ss, name, '[');
  getline(ss, len_str, ' ');
  return add_decl(ss, name, len_str);
}

bool QubitParser::parse_qasm3_decl(std::stringstream &ss) {
  std::string name;
  std::string len_str;
  getline(ss, len_str, ']');
  getline(ss, name);
  return add_decl(ss, name, len_str);
}

bool QubitParser::add_decl(std::stringstream &ss, std::string &name,
                           std::string &lstr) {
  // Ensures qreg parsing is allowed.
  if (finalized_) {
    std::cerr << "Can only create qubit before finalization." << std::endl;
    assert(false);
    return false;
  }

  // Ensures this is the end of the line.
  if (!ss.eof()) {
    std::string tmp;
    getline(ss, tmp);
    tmp = strip(tmp);
    if (tmp != "") {
      std::cerr << "Unexpected tokens after qubit declaration." << std::endl;
      assert(false);
      return false;
    }
  }

  // Ensures the name is unique.
  name = strip(name);
  if (index_offset.count(name) > 0) {
    std::cerr << "Each qubit must have a unique name: " << name << std::endl;
    assert(false);
    return false;
  }

  // Ensures the index is valid.
  lstr = strip(lstr);
  int len = string_to_number(lstr);
  if (len == -1) {
    std::cerr << "Invalid qubit length: " << len << std::endl;
    assert(false);
    return false;
  }

  // Updates the index.
  index_offset[name] = len;
  return true;
}

int QubitParser::parse_access(std::stringstream &ss) {
  // Ensures access parsing is allowed.
  if (!finalized_) {
    std::cerr << "Can only access qubits after finalization." << std::endl;
    assert(false);
    return false;
  }

  // Gets qreg array name.
  std::string name;
  getline(ss, name, '[');
  name = strip(name);
  if (index_offset.count(name) == 0) {
    std::cerr << "Unknown qreg: " << name << std::endl;
    assert(false);
    return -1;
  }

  // Gets qreg array index.
  std::string token;
  ss >> token;
  int index = string_to_number(token);
  if (index == -1) {
    std::cerr << "Unknown qubit index: " << token << std::endl;
    assert(false);
    return -1;
  }

  // Computes the global qubit index.
  int offset = index_offset[name];
  return offset + index;
}

int QubitParser::finalize() {
  // Ensures finalization can only happen once.
  if (finalized_) {
    std::cerr << "Can only finalize qreg lookup once." << std::endl;
    assert(false);
    return false;
  }
  finalized_ = true;

  // Computes qubit indinces and totals.
  int num_qubits = 0;
  for (auto &qreg : index_offset) {
    int new_num_qubits = num_qubits + qreg.second;
    qreg.second = num_qubits;
    num_qubits = new_num_qubits;
  }
  return num_qubits;
}

}  // namespace quartz
