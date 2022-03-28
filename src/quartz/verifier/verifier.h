#pragma once

#include "../context/context.h"
#include "../dataset/equivalence_set.h"
#include "../dag/dag.h"
#include "dataset/dataset.h"
#include "z3++.h"

namespace quartz {
    typedef std::vector<z3::expr> Z3ExprVec;
    typedef std::pair<z3::expr_vector, z3::expr_vector> Z3ExprVecPair;
    typedef std::pair<z3::expr, z3::expr> Z3ExprPair;
    typedef std::vector<Z3ExprPair> Z3ExprPairVec;
	// Verify if two circuits are equivalent and other things about DAGs.
	class Verifier {
	public:
		bool equivalent(const Context* ctx, const DAG* dag1, const DAG* dag2,
                        PhaseShiftIdType phase_shift_id, bool check_phase_shift_by_z3);
		// On-the-fly equivalence checking while generating circuits
		bool equivalent_on_the_fly(Context *ctx, DAG *circuit1, DAG *circuit2);

		// Check if the DAG is redundant (equivalence opportunities have already
		// been covered by smaller circuits).
		// This function assumes that two DAGs are equivalent iff they share the
		// same hash value.
		bool redundant(Context *ctx, DAG *dag);

		// Check if the DAG is redundant (equivalence opportunities have already
		// been covered by smaller circuits).
		bool redundant(Context *ctx, const EquivalenceSet *eqs, DAG *dag);

    private:

        std::pair<Z3ExprPairVec, Z3ExprPairVec>
        evaluate_dag(const DAG* dag, const Z3ExprPairVec& input_dist,
                     const Z3ExprPairVec& input_params, bool use_z3 = true);

        Z3ExprPairVec phase_shift_by_id(const Z3ExprPairVec& vec, const DAG* dag,
                                        PhaseShiftIdType phase_shift_id, const Z3ExprPairVec all_params);

	};

    namespace z3Utils {

        std::pair<Z3ExprPairVec, z3::expr>
        input_dist_by_z3(z3::context& ctx, int num_qubits);

        std::pair<Z3ExprPairVec, z3::expr>
        input_params_by_z3(z3::context& ctx, int num_params);

        z3::expr angle(const z3::expr& cos, const z3::expr& sin);
        z3::expr angle(const Z3ExprPair& expr);
    }

} // namespace quartz