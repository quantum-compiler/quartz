#pragma once

#include <complex>
using ParamType = double;
using ComplexType = std::complex<double>;
using DAGHashType = unsigned long long;
using EquivalenceHashType = std::pair<unsigned long long, int>;

using namespace std::complex_literals;  // so that we can write stuff like 1.0i

struct PairHash {
 public:
  template<typename T, typename U>
  std::size_t operator()(const std::pair<T, U> &x) const {
    return std::hash<T>()(x.first) ^ std::hash<U>()(x.second);
  }
};
