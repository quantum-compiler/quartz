#pragma once

#include "z3++.h"

#include "../context/context.h"
#include "../dataset/equivalence_set.h"
#include "../dag/dag.h"
#include "dataset/dataset.h"
#include "utils/z3Utils.h"

namespace quartz {
	// Verify if two circuits are equivalent and other things about DAGs.
	class Verifier {
	public:
		bool equivalent(
            const Context* ctx, const DAG* dag1, const DAG* dag2,
            const std::vector<ParamType>& params_for_fp, PhaseShiftIdType phase_shift_id,
            bool check_phase_shift_by_z3 = false, bool dont_invoke_z3 = false
        );
		// On-the-fly equivalence checking while generating circuits
		bool equivalent_on_the_fly(Context *ctx, DAG *circuit1, DAG *circuit2);

		// Check if the DAG is redundant (equivalence opportunities have already
		// been covered by smaller circuits).
		// This function assumes that two DAGs are equivalent iff they share the
		// same hash value.
		bool redundant(Context *ctx, const DAG* dag);

		// Check if the DAG is redundant (equivalence opportunities have already
		// been covered by smaller circuits).
		bool redundant(Context *ctx, const EquivalenceSet *eqs, const DAG* dag);

    private:
        Z3ExprPairVec phase_shift_by_id(z3::context& z3ctx, const Z3ExprPairVec& vec, const DAG* dag,
                                        PhaseShiftIdType phase_shift_id, const Z3ExprPairVec& all_params);

        std::vector<ParamType> gen_rand_params(int num_params);

        bool search_phase_factor(
            const Context* context, z3::context& z3ctx,
            const DAG* dag1, const DAG* dag2, const z3::expr& expression,
            const Z3ExprPairVec& output_vec1, const Z3ExprPairVec& output_vec2,
            bool dont_invoke_z3, const Z3ExprPairVec& params_symb,
            const std::vector<ParamType>& params_for_fp, int num_params,
            const ComplexType& goal_phase_factor, int cur_param_id,
            const Z3ExprPair& cur_phase_factor_symb,
            const ComplexType& cur_phase_factor_for_fp
        );
	};

} // namespace quartz