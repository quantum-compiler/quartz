#pragma once

#include "gate.h"

namespace quartz {

	class SDGGate : public Gate {
	public:
		SDGGate()
		    : Gate(GateType::sdg, 1 /*num_qubits*/, 0 /*num_parameters*/),
		      mat({{1, 0}, {0, -1.0i}}) {}

		MatrixBase *get_matrix() override { return &mat; }
		Matrix< 2 > mat;
	};
} // namespace quartz