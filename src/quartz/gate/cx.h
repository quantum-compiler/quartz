#pragma once

#include "gate.h"
#include "../math/matrix.h"
#include <assert.h>

namespace quartz {
	class CXGate : public Gate {
	public:
		CXGate()
		    : Gate(GateType::cx, 2 /*num_qubits*/, 0 /*num_parameters*/),
		      mat({{ComplexType(1), ComplexType(0), ComplexType(0),
		            ComplexType(0)},
		           {ComplexType(0), ComplexType(0), ComplexType(0),
		            ComplexType(1)},
		           {ComplexType(0), ComplexType(0), ComplexType(1),
		            ComplexType(0)},
		           {ComplexType(0), ComplexType(1), ComplexType(0),
		            ComplexType(0)}}) {}
		MatrixBase *get_matrix() override { return &mat; }
		Matrix< 4 > mat;

        Z3ExprMat get_matrix(z3::context& z3ctx, const Z3ExprPairVec& params) override {
            using namespace z3Utils;
            return Z3ExprMat {
                { c1(z3ctx), c0(z3ctx), c0(z3ctx), c0(z3ctx) },
                { c0(z3ctx), c0(z3ctx), c0(z3ctx), c1(z3ctx) },
                { c0(z3ctx), c0(z3ctx), c1(z3ctx), c0(z3ctx) },
                { c0(z3ctx), c1(z3ctx), c0(z3ctx), c0(z3ctx) },
            };
        }
	};

} // namespace quartz