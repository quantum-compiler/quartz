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

using namespace std::complex_literals; // so that we can write stuff like 1.0i

namespace quartz {
	// Constants for DAG::hash()
	constexpr int kDAGHashDiscardBits = 20;
	constexpr bool kFingerprintInvariantUnderPhaseShift = true;
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
