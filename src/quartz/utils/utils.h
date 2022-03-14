#pragma once

#include <complex>
using ParamType = double;
#ifdef USE_ARBLIB
#include "arb_complex.h"
using ComplexType = ArbComplex;
#else
using ComplexType = std::complex< double >;
#endif
using DAGHashType = unsigned long long;
using PhaseShiftIdType = int;
using EquivalenceHashType = std::pair< unsigned long long, int >;
using InputParamMaskType = unsigned long long;

using namespace std::complex_literals; // so that we can write stuff like 1.0i

namespace quartz {
	// Constants for DAG::hash()
	constexpr double kDAGHashMaxError = 1e-15;
	constexpr bool kFingerprintInvariantUnderPhaseShift = true;
	// When |kFingerprintInvariantUnderPhaseShift| is false, the hash value is
	// kDAGHashAlpha * real() + (1 - kDAGHashAlpha) * imag().
	constexpr double kDAGHashAlpha = 0.233;
	constexpr bool kCheckPhaseShiftInGenerator = false;
	static_assert(!(kFingerprintInvariantUnderPhaseShift &&
	                kCheckPhaseShiftInGenerator));
	constexpr PhaseShiftIdType kNoPhaseShift = -1;
	constexpr bool kCheckPhaseShiftOfPiOver4 = true;
	constexpr int kCheckPhaseShiftOfPiOver4Index = 10000; // not used now

	struct PairHash {
	public:
		template < typename T, typename U >
		std::size_t operator()(const std::pair< T, U > &x) const {
			return std::hash< T >()(x.first) ^ std::hash< U >()(x.second);
		}
	};

	template < typename T >
	std::string to_string_with_precision(const T &val, int precision = 6) {
		std::ostringstream out;
		out.precision(precision);
		out << std::scientific << val;
		return out.str();
	}

} // namespace quartz