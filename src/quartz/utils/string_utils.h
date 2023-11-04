#pragma once

#include <cctype>
#include <sstream>
#include <string>
#include <vector>

namespace quartz {
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

/**
 * Read a json array from a stringstream into a vector.
 * @tparam T The type of the item in the vector.
 * @param ss The stringstream with a json array at the beginning, created by
 * to_json_style_string(vec) above.
 * @param vec The returned vector. The original content is deleted.
 * @return True iff the read is successful.
 */
template <typename T>
bool read_json_style_vector(std::stringstream &ss, std::vector<T> &vec) {
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
  ss >> vec_size;
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

}  // namespace quartz
