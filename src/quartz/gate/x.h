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

        Z3ExprMat get_matrix(z3::context& z3ctx, const Z3ExprPairVec& params) override {
            using namespace z3Utils;
            return Z3ExprMat {
                { c0(z3ctx), c1(z3ctx) },
                { c1(z3ctx), c0(z3ctx) },
            };
        }
	};

} // namespace quartz