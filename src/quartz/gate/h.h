#pragma once

#include "gate.h"

namespace quartz {
	class HGate : public Gate {
	public:
		HGate()
		    : Gate(GateType::h, 1 /*num_qubits*/, 0 /*num_parameters*/),
		      mat({{ComplexType(1 / std::sqrt(2)),
		            ComplexType(1 / std::sqrt(2))},
		           {ComplexType(1 / std::sqrt(2)),
		            ComplexType(-1 / std::sqrt(2))}}) {}

		MatrixBase *get_matrix() override { return &mat; }
		Matrix< 2 > mat;

        Z3ExprMat get_matrix(z3::context& z3ctx, const Z3ExprPairVec& params) override {
            using namespace z3Utils;
            return Z3ExprMat {
                { c1_over_sqrt2(z3ctx), c1_over_sqrt2(z3ctx) },
                { c1_over_sqrt2(z3ctx), cm1_over_sqrt2(z3ctx) },
            };
        }
	};

} // namespace quartz