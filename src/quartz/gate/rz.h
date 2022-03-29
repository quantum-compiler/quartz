#pragma once

#include "gate.h"
#include <assert.h>

namespace quartz {
	class RZGate : public Gate {
	public:
		RZGate() : Gate(GateType::rz, 1 /*num_qubits*/, 1 /*num_parameters*/) {}
		MatrixBase *
		get_matrix(const std::vector< ParamType > &params) override {
			assert(params.size() == 1);
			ParamType theta = params[0];
			if (cached_matrices.find(theta) == cached_matrices.end()) {
				// e ^ {i * theta} = cos theta + i sin theta
				auto mat = std::make_unique< Matrix< 2 > >(Matrix< 2 >(
				    {{ComplexType(cos(theta) - 1.0i * sin(theta)),
				      ComplexType(0)},
				     {ComplexType(0),
				      ComplexType(cos(theta) + 1.0i * sin(theta))}}));
				cached_matrices[theta] = std::move(mat);
			}
			return cached_matrices[theta].get();
		}
		std::unordered_map< float, std::unique_ptr< Matrix< 2 > > >
		    cached_matrices;

        Z3ExprMat get_matrix(z3::context& z3ctx, const Z3ExprPairVec& params) override {
            using namespace z3Utils;
            assert(params.size() == 1);
            const auto& cos = params[0].first;
            const auto& sin = params[0].first;
            return Z3ExprMat {
                { Z3ExprPair{cos, -sin}, c0(z3ctx) },
                { c0(z3ctx), Z3ExprPair{cos, sin} },
            };
        }
    };

} // namespace quartz