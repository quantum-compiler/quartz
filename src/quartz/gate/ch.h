#pragma once

#include "gate.h"
#include "../math/matrix.h"
#include <assert.h>

namespace quartz {
	class CHGate : public Gate {
	public:
		CHGate()
		    : Gate(GateType::ch, 2 /*num_qubits*/, 0 /*num_parameters*/),
		      mat({{1, 0, 0, 0},
		           {0, 1 / std::sqrt(2), 0, 1 / std::sqrt(2)},
		           {0, 0, 1, 0},
		           {0, 1 / std::sqrt(2), 0, -1 / std::sqrt(2)}}) {}
		MatrixBase *get_matrix() override { return &mat; }
		Matrix< 4 > mat;

        Z3ExprMat get_matrix(z3::context& z3ctx, const Z3ExprPairVec& params) override {
            using namespace z3Utils;
            return Z3ExprMat {
                { c1(z3ctx), c0(z3ctx), c0(z3ctx), c0(z3ctx) },
                { c0(z3ctx), c1_over_sqrt2(z3ctx), c0(z3ctx), c1_over_sqrt2(z3ctx) },
                { c0(z3ctx), c0(z3ctx), c1(z3ctx), c0(z3ctx) },
                { c0(z3ctx), cm1_over_sqrt2(z3ctx), c0(z3ctx), cm1_over_sqrt2(z3ctx) },
            };
        }
	};

} // namespace quartz