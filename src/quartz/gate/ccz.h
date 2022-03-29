#pragma once

#include "gate.h"
#include "../math/matrix.h"
#include <assert.h>

namespace quartz {
	class CCZGate : public Gate {
	public:
		CCZGate()
		    : Gate(GateType::ccz, 3 /*num_qubits*/, 0 /*num_parameters*/),
		      mat({{1, 0, 0, 0, 0, 0, 0, 0},
		           {0, 1, 0, 0, 0, 0, 0, 0},
		           {0, 0, 1, 0, 0, 0, 0, 0},
		           {0, 0, 0, 1, 0, 0, 0, 0},
		           {0, 0, 0, 0, 1, 0, 0, 0},
		           {0, 0, 0, 0, 0, 1, 0, 0},
		           {0, 0, 0, 0, 0, 0, 1, 0},
		           {0, 0, 0, 0, 0, 0, 0, -1}}) {}
		MatrixBase *get_matrix() override { return &mat; }
		Matrix< 8 > mat;

        Z3ExprMat get_matrix(z3::context& z3ctx, const Z3ExprPairVec& params) override {
            using namespace z3Utils;
            return Z3ExprMat {
                {c1(z3ctx), c0(z3ctx), c0(z3ctx), c0(z3ctx), c0(z3ctx), c0(z3ctx), c0(
                        z3ctx), c0(
                        z3ctx) },
                {c0(z3ctx), c1(z3ctx), c0(z3ctx), c0(z3ctx), c0(z3ctx), c0(z3ctx), c0(
                        z3ctx),                                                                                                          c0(
                        z3ctx) },
                {c0(z3ctx), c0(z3ctx), c1(z3ctx), c0(z3ctx), c0(z3ctx), c0(z3ctx), c0(
                        z3ctx),                                                                                                          c0(
                        z3ctx) },
                {c0(z3ctx), c0(z3ctx), c0(z3ctx), c1(z3ctx), c0(z3ctx), c0(z3ctx), c0(
                        z3ctx),                                                               c0(
                        z3ctx) },
                {c0(z3ctx), c0(z3ctx), c0(z3ctx), c0(z3ctx), c1(z3ctx), c0(z3ctx), c0(
                        z3ctx),                                                               c0(
                        z3ctx) },
                {c0(z3ctx), c0(z3ctx), c0(z3ctx), c0(z3ctx), c0(z3ctx), c1(z3ctx), c0(
                        z3ctx),                                                               c0(
                        z3ctx) },
                {c0(z3ctx), c0(z3ctx), c0(z3ctx), c0(z3ctx), c0(z3ctx), c0(z3ctx), c1(z3ctx), c0(
                        z3ctx) },
                {c0(z3ctx), c0(z3ctx), c0(z3ctx), c0(z3ctx), c0(z3ctx), c0(z3ctx), c0(
                        z3ctx), cm1(z3ctx) },
            };
        }
	};

} // namespace quartz