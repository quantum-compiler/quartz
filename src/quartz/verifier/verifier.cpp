#include "verifier.h"

#include <algorithm>
#include <cmath>

namespace quartz {
	bool Verifier::equivalent_on_the_fly(Context *ctx, DAG *circuit1,
	                                     DAG *circuit2) {
		// Disable the verifier.
		return false;
		// Aggressively assume circuits with the same hash values are
		// equivalent.
		return circuit1->hash(ctx) == circuit2->hash(ctx);
	}

	bool Verifier::redundant(Context *ctx, const DAG* const dag) {
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
	                         const DAG* const dag) {
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
        const Context* const context, const DAG* const dag1, const DAG* const dag2,
        const std::vector<ParamType>& params_for_fp, const PhaseShiftIdType phase_shift_id,
        const bool check_phase_shift_by_z3, const bool dont_invoke_z3
    ) {
        // check num_quibits
        if (dag1->get_num_qubits() != dag2->get_num_qubits())
            return false;

        const int num_qubits = dag1->get_num_qubits();
        const int num_params = std::max(dag1->get_num_input_parameters(), dag2->get_num_input_parameters());
        z3::context z3ctx;
        z3::solver solver(z3ctx);

        if (check_phase_shift_by_z3) {
            // TODO Colin
            assert(false);
        }
        else {
            z3::expr constraint(z3ctx.bool_val(true));
            auto [input_dist, cstr1] = z3Utils::input_dist_by_z3(z3ctx, num_qubits);
            constraint = constraint && cstr1;
            auto [input_params, cstr2] = z3Utils::input_params_by_z3(z3ctx, num_params);
            constraint = constraint && cstr2;
            auto [output_vec1, all_params] = dag1->evaluate(z3ctx, input_dist, input_params);
            const auto output_vec2 = dag2->evaluate(z3ctx, input_dist, input_params).first;
            if (phase_shift_id != kNoPhaseShift) {
                // Phase factor is provided in generator; We shift dag1 here
                output_vec1 = phase_shift_by_id(z3ctx, output_vec1, dag1, phase_shift_id, all_params);
                solver.add(constraint);
                solver.add(! z3Utils::eq(z3ctx, output_vec1, output_vec2));
            } // goto CHECK
            else {
                // Figure out the phase factor here
                assert(params_for_fp.size() >= num_params);
                const auto goal_phase_factor =
                        dag1->get_original_fingerprint() / dag2->get_original_fingerprint();
                const Z3ExprPair cur_phase_factor_symb{ z3ctx.real_val(1), z3ctx.real_val(0) };
                const ComplexType cur_phase_factor_for_fp = { 0, 0 };
auto start_search = std::chrono::high_resolution_clock::now();
                const bool res = search_phase_factor(
                    context, z3ctx, dag1, dag2, constraint,
                    output_vec1, output_vec2, dont_invoke_z3,
                    input_params, params_for_fp, num_params, goal_phase_factor,
                    0, cur_phase_factor_symb, cur_phase_factor_for_fp
                );
auto end_search = std::chrono::high_resolution_clock::now();
auto duration_search = std::chrono::duration_cast<std::chrono::milliseconds>(end_search - start_search).count();
//std::cout << "--------** search factor end in " << duration_search << "ms" << std::endl;
                return res;
            }
        }
        // CHECK
auto start_solver = std::chrono::high_resolution_clock::now();
        const auto res = solver.check();
auto end_solver = std::chrono::high_resolution_clock::now();
auto duration_solver = std::chrono::duration_cast<std::chrono::milliseconds>(end_solver - start_solver).count();
std::cout << "--------** solver checked end in " << duration_solver << "ms" << std::endl;
        assert(res != z3::unknown);
        return res == z3::unsat;
    }

    bool Verifier::search_phase_factor(
        const Context* const context, z3::context& z3ctx,
        const DAG* const dag1, const DAG* const dag2, const z3::expr& expression,
        const Z3ExprPairVec& output_vec1, const Z3ExprPairVec& output_vec2,
        const bool dont_invoke_z3, const Z3ExprPairVec& params_symb,
        const std::vector<ParamType>& params_for_fp, const int num_params,
        const ComplexType& goal_phase_factor, const int cur_param_id,
        const Z3ExprPair& cur_phase_factor_symb,
        const ComplexType& cur_phase_factor_for_fp
    ) {
        const Z3ExprVec kPhaseFactorConstantCosTable{
            z3ctx.real_val(1), z3ctx.real_val(1) / z3Utils::pow(z3ctx, "2", "1/2"),
            z3ctx.real_val(0), - z3ctx.real_val(1) / z3Utils::pow(z3ctx, "2", "1/2"),
            z3ctx.real_val(-1), - z3ctx.real_val(1) / z3Utils::pow(z3ctx, "2", "1/2"),
            z3ctx.real_val(0), z3ctx.real_val(1) / z3Utils::pow(z3ctx, "2", "1/2"),
        };
        const Z3ExprVec kPhaseFactorConstantSinTable{
            z3ctx.real_val(0), z3ctx.real_val(1) / z3Utils::pow(z3ctx, "2", "1/2"),
            z3ctx.real_val(1), z3ctx.real_val(1) / z3Utils::pow(z3ctx, "2", "1/2"),
            z3ctx.real_val(0), - z3ctx.real_val(1) / z3Utils::pow(z3ctx, "2", "1/2"),
            z3ctx.real_val(-1), - z3ctx.real_val(1) / z3Utils::pow(z3ctx, "2", "1/2"),
        };
        if (cur_param_id == num_params) {
            // Search for constants
            for (int const_coeff = kPhaseFactorConstantCoeffMin;
                 const_coeff <= kPhaseFactorConstantCoeffMax; const_coeff++) {
                const double const_val = const_coeff * kPhaseFactorConstant;
                const auto new_phase_factor_for_fp = cur_phase_factor_for_fp + const_val;
                const auto new_phase_factor_symb = z3Utils::add(
                    cur_phase_factor_symb,
                    {kPhaseFactorConstantCosTable[const_coeff], kPhaseFactorConstantSinTable[const_coeff]}
                );
                const bool res = search_phase_factor(
                    context, z3ctx, dag1, dag2, expression, output_vec1, output_vec2,
                    dont_invoke_z3, params_symb, params_for_fp, num_params,
                    goal_phase_factor, cur_param_id + 1,
                    new_phase_factor_symb, new_phase_factor_for_fp
                );
                if (res)
                    return true;
            } // end for const_coeff
            return false;
        }
        else if (cur_param_id == num_params + 1) {
            // Done searching, check for equivalence
            const ComplexType phase_factor_for_fp = {
                std::cos(cur_phase_factor_for_fp.real()),
                std::sin(cur_phase_factor_for_fp.imag())
            };
            const auto diff = phase_factor_for_fp - goal_phase_factor;
            if (std::abs(diff) > kPhaseFactorEpsilon)
                return false;
            if (dont_invoke_z3) {
                // Generate a random test to verify it.
                // ATTENTION Colin : is it ok to use generated things in context?
                const int num_qubits = dag1->get_num_qubits();
                const auto& input_dist = context->get_generated_input_dis(num_qubits);
                const int cur_num_params = std::max(dag1->get_num_input_parameters(), dag2->get_num_input_parameters());
                const auto input_params = context->get_generated_parameters(cur_num_params);
                const auto cur_output_vec1 = dag1->evaluate(input_dist, input_params).first;
                const auto cur_output_vec2 = dag2->evaluate(input_dist, input_params).first;
                for (size_t i = 0; i < (1 << num_qubits); i++) {
                    const auto& a = cur_output_vec1.at(i);
                    const auto& b = cur_output_vec2.at(i);
                    if (std::abs(a - b) > kPhaseFactorEpsilon)
                        return false;
                }
                return true;
            }
            else {
                z3::solver solver(z3ctx);
                solver.add(expression);
                const auto output_vec2_shifted = z3Utils::shift(output_vec2, cur_phase_factor_symb);
                solver.add(! z3Utils::eq(z3ctx, output_vec1, output_vec2_shifted));
                // ATTENTION Colin : try to reduce it!
                solver.set(":timeout", 30000u); // timeout after 30s
                const auto res = solver.check();
                assert(res != z3::unknown);
                return res == z3::unsat;
            }
        }
        // Search for the parameter |current_param_id|
        else {
            /* const std::vector<ComplexType> kPhaseFactorCoeffs{
                {0, 0}, {1, 0}, {-1, 0}, {2, 0}, {-2, 0}
            }; */
            const std::vector<int> kPhaseFactorCoeffs{ 0, 1, -1, 2, -2 };
            for (const auto& coeff : kPhaseFactorCoeffs) {
                auto new_phase_factor_for_fp = cur_phase_factor_for_fp;
                auto new_phase_factor_symb = cur_phase_factor_symb;
                if (coeff != kPhaseFactorCoeffs[0]) {
                    new_phase_factor_for_fp += coeff * params_for_fp[cur_param_id];
                }
                if (coeff == kPhaseFactorCoeffs[1]/* 1 */) {
                    new_phase_factor_symb = z3Utils::add(cur_phase_factor_symb, params_symb[cur_param_id]);
                }
                else if (coeff == kPhaseFactorCoeffs[2]/* -1 */) {
                    new_phase_factor_symb = z3Utils::sub(cur_phase_factor_symb, params_symb[cur_param_id]);
                }
                else if (coeff == kPhaseFactorCoeffs[3]/* 2 */) {
                    new_phase_factor_symb = z3Utils::add(cur_phase_factor_symb,
                                                         z3Utils::doub(params_symb[cur_param_id]));
                }
                else if (coeff == kPhaseFactorCoeffs[4]/* -2 */) {
                    new_phase_factor_symb = z3Utils::sub(cur_phase_factor_symb,
                                                         z3Utils::doub(params_symb[cur_param_id]));
                }
                const bool res = search_phase_factor(
                    context, z3ctx, dag1, dag2, expression, output_vec1, output_vec2,
                    dont_invoke_z3, params_symb, params_for_fp, num_params,
                    goal_phase_factor, cur_param_id + 1,
                    new_phase_factor_symb, new_phase_factor_for_fp
                );
                if (res)
                    return true;
            }
            return false;
        }
    }

    Z3ExprPairVec Verifier::phase_shift_by_id(
            z3::context& z3ctx, const Z3ExprPairVec& vec, const DAG* dag,
            PhaseShiftIdType phase_shift_id, const Z3ExprPairVec& all_params) {
        const int num_tot_params = dag->get_num_total_parameters();
        Z3ExprPair phase{ z3ctx.real_val(0), z3ctx.real_val(0) };
        if (kCheckPhaseShiftOfPiOver4Index < phase_shift_id && phase_shift_id < kCheckPhaseShiftOfPiOver4Index + 8) {
            const Z3ExprVec kPhaseFactorConstantCosTable{
                z3ctx.real_val(1), z3ctx.real_val(1) / z3Utils::pow(z3ctx, "2", "1/2"),
                z3ctx.real_val(0), - z3ctx.real_val(1) / z3Utils::pow(z3ctx, "2", "1/2"),
                z3ctx.real_val(-1), - z3ctx.real_val(1) / z3Utils::pow(z3ctx, "2", "1/2"),
                z3ctx.real_val(0), z3ctx.real_val(1) / z3Utils::pow(z3ctx, "2", "1/2"),
            };
            const Z3ExprVec kPhaseFactorConstantSinTable{
                z3ctx.real_val(0), z3ctx.real_val(1) / z3Utils::pow(z3ctx, "2", "1/2"),
                z3ctx.real_val(1), z3ctx.real_val(1) / z3Utils::pow(z3ctx, "2", "1/2"),
                z3ctx.real_val(0), - z3ctx.real_val(1) / z3Utils::pow(z3ctx, "2", "1/2"),
                z3ctx.real_val(-1), - z3ctx.real_val(1) / z3Utils::pow(z3ctx, "2", "1/2"),
            };
            const PhaseShiftIdType k = phase_shift_id - kCheckPhaseShiftOfPiOver4Index;
            phase = Z3ExprPair{ kPhaseFactorConstantCosTable[k], kPhaseFactorConstantSinTable[k] };
        }
        else if (phase_shift_id < num_tot_params) {
            phase = all_params.at(phase_shift_id);
        }
        else {
            phase = z3Utils::neg(all_params.at(phase_shift_id - num_tot_params));
        }
        return z3Utils::shift(vec, phase);
    }

    std::vector<ParamType> Verifier::gen_rand_params(const int num_params) {
        assert(num_params >= 0);
        std::vector<ParamType> rand_params;
        if (rand_params.size() < num_params) {
            static ParamType pi = std::acos((ParamType)-1.0);
            static std::uniform_real_distribution<ParamType> dis_real(-pi, pi);
            static std::mt19937 gen{0};
            while (rand_params.size() < num_params) {
                rand_params.emplace_back(dis_real(gen));
            }
        }
        return std::vector<ParamType>(
                rand_params.begin(),rand_params.begin() + num_params);
    }


} // namespace quartz
