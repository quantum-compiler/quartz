#pragma once

#include "../context/context.h"
#include "../dataset/equivalence_set.h"
#include "../dag/dag.h"
#include "dataset/dataset.h"
#include "z3++.h"

namespace quartz {
    typedef std::pair<z3::expr_vector, z3::expr_vector> Z3ExprVecPair;
    typedef std::pair<z3::expr, z3::expr> Z3ExprPair;
    typedef std::vector<Z3ExprPair> Z3ExprPairVec;
	// Verify if two circuits are equivalent and other things about DAGs.
	class Verifier {
	public:
		bool equivalent(const Context* ctx, const DAG* dag1, const DAG* dag2,
                        PhaseShiftIdType phase_id, bool check_phase_shift_by_z3);
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

        auto evaluate_dag(const DAG* dag, const Z3ExprVecPair& input_dist,
                          const Z3ExprVecPair& input_params, bool use_z3);

	};

    namespace z3Utils {
        Z3ExprPairVec input_dist_by_z3(z3::context& ctx, z3::solver& solver, int num_qubits);
        Z3ExprPairVec input_params_by_z3(z3::context& ctx, z3::solver& solver, int num_params);

        z3::expr angle(const z3::expr& cos, const z3::expr& sin);
        z3::expr angle(const Z3ExprPair& expr);
    }

} // namespace quartz