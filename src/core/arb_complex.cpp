#include "quartz/utils/arb_complex.h"

#ifdef USE_ARBLIB

double std::abs(const ArbComplex &val) {
	double re = val.real(), im = val.imag();
	return std::sqrt(re * re + im * im);
}

#endif
