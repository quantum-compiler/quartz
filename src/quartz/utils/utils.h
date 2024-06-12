#pragma once

#include <complex>
#include <filesystem>

using ParamType = double;
#ifdef USE_ARBLIB
#include "arb_complex.h"
using ComplexType = ArbComplex;
#else
using ComplexType = std::complex<double>;
#endif
using CircuitSeqHashType = unsigned long long;
using PhaseShiftIdType = int;
using EquivalenceHashType = std::pair<unsigned long long, int>;
using InputParamMaskType = unsigned long long;

using namespace std::complex_literals;  // so that we can write stuff like 1.0i

namespace quartz {
const ParamType PI = std::acos((ParamType)-1);

// Constants for CircuitSeq::hash()
constexpr double kCircuitSeqHashMaxError = 1e-15;
constexpr bool kFingerprintInvariantUnderPhaseShift = true;
// When |kFingerprintInvariantUnderPhaseShift| is false, the hash value is
// kCircuitSeqHashAlpha * real() + (1 - kCircuitSeqHashAlpha) * imag().
constexpr double kCircuitSeqHashAlpha = 0.233;
constexpr bool kCheckPhaseShiftInGenerator = false;
static_assert(!(kFingerprintInvariantUnderPhaseShift &&
                kCheckPhaseShiftInGenerator));
constexpr PhaseShiftIdType kNoPhaseShift = -1;
constexpr bool kCheckPhaseShiftOfPiOver4 = true;
constexpr int kCheckPhaseShiftOfPiOver4Index = 10000;  // not used now

constexpr bool kUseRowRepresentationToCompare = false;

constexpr int kDefaultQASMParamPrecision = 15;

const std::filesystem::path kQuartzRootPath =
    std::filesystem::canonical(__FILE__)
        .parent_path()
        .parent_path()
        .parent_path()
        .parent_path();

struct PairHash {
 public:
  template <typename T, typename U>
  std::size_t operator()(const std::pair<T, U> &x) const {
    return std::hash<T>()(x.first) ^ std::hash<U>()(x.second);
  }
};
}  // namespace quartz
