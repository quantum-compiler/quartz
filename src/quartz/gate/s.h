#pragma once

#include "gate.h"
namespace quartz {

	class SGate : public Gate {
	public:
		SGate()
		    : Gate(GateType::s, 1 /*num_qubits*/, 0 /*num_parameters*/),
		      mat({{1, 0}, {0, 1.0i}}) {}

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