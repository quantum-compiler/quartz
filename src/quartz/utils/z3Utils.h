#pragma once

namespace quartz {
    typedef std::vector<z3::expr> Z3ExprVec;
    typedef std::pair<z3::expr_vector, z3::expr_vector> Z3ExprVecPair;
    typedef std::pair<z3::expr, z3::expr> Z3ExprPair;
    typedef std::vector<Z3ExprPair> Z3ExprPairVec;

    namespace z3Utils {

        inline z3::expr angle(const z3::expr& cos, const z3::expr& sin) {
            return cos * cos + sin * sin == 1;
        }

        inline z3::expr angle(const Z3ExprPair& expr) {
            return expr.first * expr.first + expr.second * expr.second == 1;
        }

        inline z3::expr pow(z3::context& z3ctx, const char* _a, const char* _b) {
            return z3::pw(z3ctx.real_val(_a), z3ctx.real_val(_b));
        }

        inline Z3ExprPair add(const Z3ExprPair& _a, const Z3ExprPair& _b) {
            // cos_a * cos_b - sin_a * sin_b, sin_a * cos_b + cos_a * sin_b
            return { _a.first * _b.first - _a.second * _b.second,
                     _a.second * _b.first + _a.first * _b.second };
        }

        inline Z3ExprPair doub(const Z3ExprPair& _a) {
            return add(_a, _a);
        }

        inline Z3ExprPair neg(const Z3ExprPair& _a) {
            return { _a.first, - _a.second };
        }

        inline Z3ExprPair sub(const Z3ExprPair& _a, const Z3ExprPair& _b) {
            return add(_a, neg(_b));
        }

        inline Z3ExprPairVec shift(const Z3ExprPairVec& vec, const Z3ExprPair& phase) {
            Z3ExprPairVec output(vec);
            std::for_each(output.begin(), output.end(), [&](Z3ExprPair& item){
                item = { item.first * phase.first - item.second * phase.second,
                         item.second * phase.first + item.first * phase.second };
            });
            return output;
        }

        inline z3::expr eq(z3::context& z3ctx, const Z3ExprPairVec& a, const Z3ExprPairVec& b) {
            assert(a.size() == b.size());
            if (a.empty())
                return z3ctx.bool_val(true);
            else {
                auto res = z3ctx.bool_val(true);
                for (size_t i = 0; i < a.size(); i++) {
                    res = res && (a[i] == b[i]);
                }
                return res;
            }
        }

        inline std::pair<Z3ExprPairVec, z3::expr>
        input_dist_by_z3(z3::context& ctx, const int num_qubits) {
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
            z3::expr constraint(ctx);
            constraint = sum_modulus == 1;
            return std::make_pair(input_dist, constraint);
        }

        inline std::pair<Z3ExprPairVec, z3::expr>
        input_params_by_z3(z3::context& ctx, const int num_params) {
            z3::expr constraint(ctx.bool_val(true));
            Z3ExprPairVec input_params;
            for (int i = 0; i < num_params; i++) {
                const auto cos_name = "cos_" + std::to_string(i);
                const auto sin_name = "sin_" + std::to_string(i);
                input_params.push_back(std::make_pair(
                        ctx.real_const(cos_name.c_str()),
                        ctx.real_const(sin_name.c_str())
                ));
                constraint = constraint && angle(input_params.back());
            }
            return std::make_pair(input_params, constraint);
        }

    }

}
