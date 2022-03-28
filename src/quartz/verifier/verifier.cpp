#include "verifier.h"

#include <algorithm>

namespace quartz {
	bool Verifier::equivalent_on_the_fly(Context *ctx, DAG *circuit1,
	                                     DAG *circuit2) {
		// Disable the verifier.
		return false;
		// Aggressively assume circuits with the same hash values are
		// equivalent.
		return circuit1->hash(ctx) == circuit2->hash(ctx);
	}

	bool Verifier::redundant(Context *ctx, DAG *dag) {
		// Check if any suffix already exists.
		// This function assumes that two DAGs are equivalent iff they share the
		// same hash value.
		auto subgraph = std::make_unique< DAG >(*dag);
		while (subgraph->get_num_gates() > 0) {
			subgraph->remove_gate(subgraph->edges[0].get());
			if (subgraph->get_num_gates() == 0) {
				break;
			}
			subgraph->hash(ctx);
			const auto &rep = ctx->get_possible_representative(subgraph.get());
			if (rep && !subgraph->fully_equivalent(*rep)) {
				// |subgraph| already exists and is not the representative.
				return true;
			}
		}
		return false;
	}

	bool Verifier::redundant(Context *ctx, const EquivalenceSet *eqs,
	                         DAG *dag) {
		// Representative pruning.
		// Check if any suffix already exists.
		auto subgraph = std::make_unique< DAG >(*dag);
		while (subgraph->get_num_gates() > 0) {
			if (subgraph->remove_first_quantum_gate() == 0) {
				// Already no quantum gates
				break;
			}
			if (subgraph->get_num_gates() == 0) {
				break;
			}
			DAGHashType hash_value = subgraph->hash(ctx);
			auto possible_classes = eqs->get_possible_classes(hash_value);
			for (const auto &other_hash : subgraph->other_hash_values()) {
				auto more_possible_classes =
				    eqs->get_possible_classes(other_hash);
				possible_classes.insert(possible_classes.end(),
				                        more_possible_classes.begin(),
				                        more_possible_classes.end());
			}
			std::sort(possible_classes.begin(), possible_classes.end());
			auto last =
			    std::unique(possible_classes.begin(), possible_classes.end());
			possible_classes.erase(last, possible_classes.end());
			for (const auto &equiv_class : possible_classes) {
				if (equiv_class->contains(*subgraph)) {
					if (!subgraph->fully_equivalent(
					        *equiv_class->get_representative())) {
						// |subgraph| already exists and is not the
						// representative. So the whole |dag| is redundant.
						return true;
					}
					else {
						// |subgraph| already exists and is the representative.
						// So we need to check other subgraphs.
						break;
					}
				}
			}
		}
		return false;
	}

    bool Verifier::equivalent(
            const Context* const ctx, const DAG* const dag1, const DAG* const dag2,
            const PhaseShiftIdType phase_id, bool check_phase_shift_by_z3=false) {
        // check num_quibits
        if (dag1->get_num_qubits() != dag2->get_num_qubits())
            return false;

        const int num_qubits = dag1->get_num_qubits();
        const int num_params = std::max(dag1->get_num_input_parameters(), dag2->get_num_input_parameters());

        if (check_phase_shift_by_z3) {

        }
        else {
            z3::context ctx;
            z3::solver solver(ctx);
            auto input_dist_pair = z3Utils::input_dist_by_z3(ctx, solver, num_qubits);
            auto input_param_pair = z3Utils::input_params_by_z3(ctx, solver, num_params);

        }


        return false;
    }

    auto Verifier::evaluate_dag(const DAG* const dag, const Z3ExprVecPair& input_dist,
                      const Z3ExprVecPair& input_params, bool use_z3) {
        const int num_input_params = dag->get_num_input_parameters();
        const int num_tot_params = dag->get_num_total_parameters();
        assert(input_params.first.size() > num_input_params);

        // iterate edges to iterate (gate, output_nodes, input_nodes)
        for (const auto& edge : dag->edges) {
            // 1. read input
            for (const DAGNode* const input_node : edge->input_nodes) {

            }
            // 2. judge gate type, read output and compute
            if (edge->gate->is_parameter_gate()) {

            }
            else {
                assert(edge->gate->is_quantum_gate());

            }
        }

    }

    namespace z3Utils {
        Z3ExprPairVec input_dist_by_z3(
                z3::context& ctx, z3::solver& solver, const int num_qubits) {
            const int vec_size = (1 << num_qubits);
            Z3ExprPairVec input_dist;
            z3::expr sum_modulus = ctx.real_val(0);
            for (int i = 0; i < vec_size; i++) {
                const auto r_name = "r_" + std::to_string(i);
                const auto i_name = "i_" + std::to_string(i);
                input_dist.push_back(std::make_pair(
                    ctx.real_const(r_name.c_str()),
                    ctx.real_const(i_name.c_str())
                ));
                // A quantum state requires the sum of modulus of all numbers to be 1.
                sum_modulus = sum_modulus + input_dist.back().first * input_dist.back().first
                              + input_dist.back().second * input_dist.back().second;
            }
            solver.add(sum_modulus);
            return input_dist;
        }

        Z3ExprPairVec input_params_by_z3(
                z3::context& ctx, z3::solver& solver, const int num_params) {
            Z3ExprPairVec input_params;
            for (int i = 0; i < num_params; i++) {
                const auto cos_name = "cos_" + std::to_string(i);
                const auto sin_name = "sin_" + std::to_string(i);
                input_params.push_back(std::make_pair(
                    ctx.real_const(cos_name.c_str()),
                    ctx.real_const(sin_name.c_str())
                ));
                solver.add(angle(input_params.back()));
            }
            return input_params;
        }

        z3::expr angle(const z3::expr& cos, const z3::expr& sin) {
            return cos * cos + sin * sin == 1;
        }

        z3::expr angle(const Z3ExprPair& expr) {
            return expr.first * expr.first + expr.second * expr.second;
        }
    }





} // namespace quartz
