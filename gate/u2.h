#pragma once

#include "gate.h"
#include "../math/matrix.h"
#include <assert.h>

class U2Gate : public Gate {
public:
  U2Gate() : Gate(GateType::u2, 1 /*num_qubits*/, 2 /*num_parameters*/) {}
  MatrixBase *get_matrix(const std::vector<ParamType> &params) override {
	assert(params.size() == 2);
	ParamType phi = params[0];
	ParamType lambda = params[1];
	auto mat = std::make_unique<Matrix<2>>(Matrix<2>(
	    {{1 / std::sqrt(2),
	      -1 / std::sqrt(2) * (cos(lambda) + 1.0i * sin(lambda))},
	     {1 / std::sqrt(2) * (cos(phi) + 1.0i * sin(phi)),
	      1 / std::sqrt(2) * (cos(phi + lambda) + 1.0i * sin(phi + lambda))}}));
	return mat.get();
  }
};
