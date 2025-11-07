#pragma once

#include <cctype>
#include <sstream>
#include <string>
#include <vector>

namespace quartz {
template <typename T>
std::string to_string_with_precision(const T &val, int precision = 6) {
  std::ostringstream out;
  out.precision(precision);
  out << std::scientific << val;
  return out.str();
}

class Rational;
template <>
std::string to_string_with_precision(const Rational &val, int precision);

template <typename T>
std::string to_json_style_string(const std::vector<T> &vec) {
  std::string result = "[";
  result += std::to_string(vec.size());  // store the size for faster recovery
  for (const auto &item : vec) {
    result += ", ";
    result += std::to_string(item);
  }
  result += "]";
  return result;
}

template <typename T>
std::string to_json_style_string_with_precision(const std::vector<T> &vec,
                                                int precision = 6) {
  std::string result = "[";
  result += std::to_string(vec.size());  // store the size for faster recovery
  for (const auto &item : vec) {
    result += ", ";
    result += to_string_with_precision(item, precision);
  }
  result += "]";
  return result;
}

/**
 * Read a json array from an istream or stringstream into a vector.
 * @tparam S std::istream or std::stringstream.
 * @tparam T The type of the item in the vector.
 * @param ss The istream/stringstream with a json array at the beginning,
 * created by to_json_style_string(vec) above.
 * @param vec The returned vector. The original content is deleted.
 * @return True iff the read is successful.
 */
template <typename S, typename T>
bool read_json_style_vector(S &ss, std::vector<T> &vec) {
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
  for (int i = 0; i < vec_size; i++) {
    while (ss >> c) {
      if (c == ',') {
        // separation of two items
        break;
      } else if (!std::isspace(c)) {
        // not a vector, input corrupted
        return false;
      }
    }
    if (ss.eof()) {
      return false;
    }
    T item;
    ss >> item;
    vec.push_back(std::move(item));
  }
  while (ss >> c) {
    if (c == ']') {
      // end of a vector
      break;
    } else if (!std::isspace(c)) {
      // not a vector, input corrupted
      return false;
    }
  }
  if (ss.eof()) {
    return false;
  }
  return true;
}

/**
 * Specialization: also handle the case "a/b" when reading a floating-point
 * value from std::istream.
 * @param ss The istream with a json array at the beginning,
 * created by to_json_style_string(vec) above but potentially in Rational
 * instead of double.
 * @param vec The returned vector. The original content is deleted.
 * @return True iff the read is successful.
 */
bool read_json_style_vector(std::istream &ss, std::vector<double> &vec);

}  // namespace quartz
