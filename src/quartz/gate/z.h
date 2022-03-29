#pragma once

#include "gate.h"

namespace quartz {
	class ZGate : public Gate {
	public:
		ZGate()
		    : Gate(GateType::z, 1 /*num_qubits*/, 0 /*num_parameters*/),
		      mat({{1, 0}, {0, -1}}) {}

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