#include "string_utils.h"

#include "quartz/math/rational.h"

namespace quartz {

template <>
std::string quartz::to_string_with_precision(const Rational &val,
                                             int precision) {
  return "\"" + val.to_string() + "\"";
}
}  // namespace quartz
