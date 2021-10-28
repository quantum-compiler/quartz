#pragma once

#include "gate.h"
#include "../math/matrix.h"
#include <assert.h>

class U3Gate : public Gate {
public:
  U3Gate() : Gate(GateType::u3, 1 /*num_qubits*/, 3 /*num_parameters*/) {}
  MatrixBase *get_matrix(const std::vector<ParamType> &params) override {
	assert(params.size() == 3);
	ParamType theta = params[0];
	ParamType phi = params[1];
	ParamType lambda = params[2];
	auto mat = std::make_unique<Matrix<2>>(Matrix<2>(
	    {{cos(theta / 2), sin(theta / 2) * (cos(lambda) + 1.0i * sin(lambda))},
	     {sin(theta / 2) * (cos(phi) + 1.0i * sin(phi)),
	      cos(theta / 2) * (cos(phi + lambda) + 1.0i * sin(phi + lambda))}}));
	return mat.get();
  }
};
