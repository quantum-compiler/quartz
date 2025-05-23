#include "utils.h"

namespace quartz {

#ifdef USE_RATIONAL
ParamType string_to_param(const std::string &s) { return Rational(s); }
ParamType string_to_param_without_pi(const std::string &s) {
  if (std::stod(s) != 0) {
    std::cerr << "Assumption that all parameters are rational multiples "
                 "of PI is violated. The result might not be accurate."
              << std::endl;
  }
  return Rational(std::to_string(std::stod(s) / std::acos(-1.0)));
}
std::string param_to_string(const ParamType &p) { return p.to_string(); }
double cos_param(const ParamType &p) {
  return std::cos(p.to_double() * acos(-1.0));
}
double sin_param(const ParamType &p) {
  return std::sin(p.to_double() * acos(-1.0));
}
#else
ParamType string_to_param(const std::string &s) { return std::stod(s); }
ParamType string_to_param_without_pi(const std::string &s) {
  return std::stod(s);
}
std::string param_to_string(const ParamType &p) { return std::to_string(p); }
double cos_param(const ParamType &p) { return std::cos(p); }
double sin_param(const ParamType &p) { return std::sin(p); }
#endif

}  // namespace quartz
