#include "string_utils.h"

#include "quartz/math/rational.h"

namespace quartz {

template <>
std::string quartz::to_string_with_precision(const Rational &val,
                                             int precision) {
  return "\"" + val.to_string() + "\"";
}

bool read_json_style_vector(std::istream &ss, std::vector<double> &vec) {
  char c;
  while (ss >> c) {
    if (c == '[') {
      // start of a vector
      break;
    } else if (!std::isspace(c)) {
      // not a vector, input corrupted
      return false;
    }
  }
  if (ss.eof()) {
    return false;
  }
  int vec_size;
  if (!(ss >> vec_size)) {
    return false;
  }
  vec.reserve(vec_size);
  vec.clear();
  while (ss >> c) {
    if (c == ',') {
      // start of the first item
      break;
    }
  }
  if (ss.eof()) {
    return false;
  }
  for (int i = 0; i < vec_size; i++) {
    std::string current;
    bool is_rational = false;
    while (ss >> c) {
      if (i == vec_size - 1 ? c == ']' : c == ',') {
        // separation of two items or end of vector
        break;
      } else if (!std::isspace(c)) {
        current += c;
        if (c == '/') {
          is_rational = true;
        }
      }
    }
    if (ss.eof()) {
      return false;
    }
    double item;
    if (is_rational) {
      Rational value(current);
      item = value.to_double();
    } else {
      try {
        item = std::stod(current);
      } catch (const std::invalid_argument &e) {
        std::cerr << "Invalid argument: " << e.what() << std::endl;
        return false;
      } catch (const std::out_of_range &e) {
        std::cerr << "Out of range: " << e.what() << std::endl;
        return false;
      }
    }
    vec.push_back(std::move(item));
  }
  if (ss.eof()) {
    return false;
  }
  return true;
}
}  // namespace quartz
