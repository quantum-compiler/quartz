#pragma once

#include "gate.h"
namespace quartz {

	class SGate : public Gate {
	public:
		SGate()
		    : Gate(GateType::s, 1 /*num_qubits*/, 0 /*num_parameters*/),
		      mat({{1, 0}, {0, 1.0i}}) {}

		MatrixBase *get_matrix() override { return &mat; }
		Matrix< 2 > mat;
	};
} // namespace quartz