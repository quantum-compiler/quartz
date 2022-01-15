#pragma once

#include "gate.h"

namespace quartz {
	class XGate : public Gate {
	public:
		XGate()
		    : Gate(GateType::x, 1 /*num_qubits*/, 0 /*num_parameters*/),
		      mat({{ComplexType(0), ComplexType(1)},
		           {ComplexType(1), ComplexType(0)}}) {}

		MatrixBase *get_matrix() override { return &mat; }
		Matrix< 2 > mat;
	};

} // namespace quartz