#pragma once

#include "gate.h"
#include "../math/matrix.h"
#include <assert.h>

namespace quartz {
	class CZGate : public Gate {
	public:
		CZGate()
		    : Gate(GateType::cz, 2 /*num_qubits*/, 0 /*num_parameters*/),
		      mat({{1, 0, 0, 0}, {0, 1, 0, 0}, {0, 0, 1, 0}, {0, 0, 0, -1}}) {}
		MatrixBase *get_matrix() override { return &mat; }
		Matrix< 4 > mat;

        Z3ExprMat get_matrix(z3::context& z3ctx, const Z3ExprPairVec& params) override {
            using namespace z3Utils;
            return Z3ExprMat {
                // TODO Colin
            };
        }
	};

} // namespace quartz