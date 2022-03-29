#pragma once

#include <cassert>

#include "gate.h"
#include "../math/matrix.h"

namespace quartz {
	class AddGate : public Gate {
	public:
		AddGate()
		    : Gate(GateType::add, 0 /*num_qubits*/, 2 /*num_parameters*/) {}
		bool is_commutative() const override { return true; }

        ParamType
        compute(const std::vector< ParamType > &input_params) override {
            assert(input_params.size() == 2);
            return input_params[0] + input_params[1];
        }

        Z3ExprPair compute(const Z3ExprPairVec& input_params) override {
            assert(input_params.size() == 2);
            return z3Utils::add(input_params[0], input_params[1]);
        }

	};

} // namespace quartz
