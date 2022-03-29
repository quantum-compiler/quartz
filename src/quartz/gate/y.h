#pragma once

#include "gate.h"

namespace quartz {
	class YGate : public Gate {
	public:
		YGate()
		    : Gate(GateType::y, 1 /*num_qubits*/, 0 /*num_parameters*/),
		      mat({{ComplexType(0), ComplexType(-1.0i)},
		           {ComplexType(1.0i), ComplexType(0)}}) {}
		MatrixBase *get_matrix() override { return &mat; }
		Matrix< 2 > mat;

        Z3ExprMat get_matrix(z3::context& z3ctx, const Z3ExprPairVec& params) override {
            using namespace z3Utils;
            return Z3ExprMat {
                    // TODO Colin
            };
        }
	};

} // namespace quartz