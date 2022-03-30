#pragma once

#include <sys/wait.h>

#include <array>
#include <ostream>
#include <string>
#ifdef _WIN32
#include <stdio.h>
#endif

// https://github.com/RaymiiOrg/cpp-command-output
namespace raymii {

// Copyright (C) 2021 Remy van Elst
//
//     This program is free software: you can redistribute it and/or modify
//     it under the terms of the GNU General Public License as published by
//     the Free Software Foundation, either version 3 of the License, or
//     (at your option) any later version.
//
//     This program is distributed in the hope that it will be useful,
//     but WITHOUT ANY WARRANTY; without even the implied warranty of
//     MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
//     GNU General Public License for more details.
//
//     You should have received a copy of the GNU General Public License
//     along with this program.  If not, see <http://www.gnu.org/licenses/>.

    struct CommandResult {
        std::string output;
        int exitstatus;
        friend std::ostream &operator<<(std::ostream &os, const CommandResult &result) {
            os << "command exitstatus: " << result.exitstatus << " output: " << result.output;
            return os;
        }
        bool operator==(const CommandResult &rhs) const {
            return output == rhs.output &&
                   exitstatus == rhs.exitstatus;
        }
        bool operator!=(const CommandResult &rhs) const {
            return !(rhs == *this);
        }
    };

    class Command {
    public:
        /**
             * Execute system command and get STDOUT result.
             * Regular system() only gives back exit status, this gives back output as well.
             * @param command system command to execute
             * @return commandResult containing STDOUT (not stderr) output & exitstatus
             * of command. Empty if command failed (or has no output). If you want stderr,
             * use shell redirection (2&>1).
             */
        static CommandResult exec(const std::string &command) {
            int exitcode = 0;
            std::array<char, 1048576> buffer {};
            std::string result;
#ifdef _WIN32
#define popen _popen
#define pclose _pclose
#define WEXITSTATUS
#endif
            FILE *pipe = popen(command.c_str(), "r");
            if (pipe == nullptr) {
                throw std::runtime_error("popen() failed!");
            }
            try {
                std::size_t bytesread;
                while ((bytesread = std::fread(buffer.data(), sizeof(buffer.at(0)), sizeof(buffer), pipe)) != 0) {
                    result += std::string(buffer.data(), bytesread);
                }
            } catch (...) {
                pclose(pipe);
                throw;
            }
            const int ret_val = pclose(pipe);
            exitcode = WEXITSTATUS(ret_val);
            return CommandResult{result, exitcode};
        }

    };

}// namespace raymii

namespace quartz {
    typedef std::vector<z3::expr> Z3ExprVec;
    typedef std::pair<z3::expr_vector, z3::expr_vector> Z3ExprVecPair;
    typedef std::pair<z3::expr, z3::expr> Z3ExprPair;
    typedef std::vector<Z3ExprPair> Z3ExprPairVec;
    typedef std::vector<Z3ExprPairVec> Z3ExprMat;

    namespace z3Utils {

        inline Z3ExprPair c0(z3::context& z3ctx) {
            return { z3ctx.real_val(0), z3ctx.real_val(0) };
        }

        inline Z3ExprPair c1(z3::context& z3ctx) {
            return { z3ctx.real_val(1), z3ctx.real_val(0) };
        }

        inline Z3ExprPair cm1(z3::context& z3ctx) {
            return { z3ctx.real_val(-1), z3ctx.real_val(0) };
        }

        inline z3::expr angle(const z3::expr& cos, const z3::expr& sin) {
            return cos * cos + sin * sin == 1;
        }

        inline z3::expr angle(const Z3ExprPair& expr) {
            return expr.first * expr.first + expr.second * expr.second == 1;
        }

        inline z3::expr pow(z3::context& z3ctx, const char* _a, const char* _b) {
            return z3::pw(z3ctx.real_val(_a), z3ctx.real_val(_b));
        }

        inline Z3ExprPair c1_over_sqrt2(z3::context& z3ctx) {
            return Z3ExprPair{ z3ctx.real_val(1) / pow(z3ctx, "2", "1/2"),
                               z3ctx.real_val(0) };
        }

        inline Z3ExprPair cm1_over_sqrt2(z3::context& z3ctx) {
            return Z3ExprPair{ z3ctx.real_val(-1) / pow(z3ctx, "2", "1/2"),
                               z3ctx.real_val(0) };
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

        inline Z3ExprPairVec apply_matrix(
            z3::context& z3ctx, const Z3ExprPairVec& input,
            const Z3ExprMat& mat, const std::vector<int>& qubit_indices
        ) {
            const size_t num_state = input.size();
            assert(1 <= qubit_indices.size() && qubit_indices.size() <= 2);
            assert((1 << qubit_indices.size()) == mat.size());
            assert(mat.size() <= num_state);
            assert(num_state % mat.size() == 0);
            /* for (const auto& row : mat) {
                assert(row.size() == mat.size());
            } */
            /* for (const int index : qubit_indices) {
                assert(1 <= (1 << index) && (1 << index) < num_state);
            } */
            Z3ExprPairVec output(input);

            for (size_t i = 0; i < num_state; i++) {
                bool already_applied = false;
                for (const int index : qubit_indices) {
                    if (i & (1 << index))
                        already_applied = true;
                }
                if (!already_applied) {
                    std::vector<int> cur_indices;
                    for (size_t j = 0; j < mat.size(); j++) {
                        size_t cur_index = i;
                        for (size_t k = 0; k < qubit_indices.size(); k++) {
                            if (j & (1 << k))
                                cur_index ^= (1 << qubit_indices[k]);
                        }
                        cur_indices.emplace_back(cur_index);
                    } // end for j

                    for (size_t r = 0; r < mat.size(); r++) {
                        Z3ExprPair tmp(c0(z3ctx));
                        for (size_t k = 0; k < mat.size(); k++) {
                            tmp.first = tmp.first
                                    + mat[r][k].first * input[cur_indices[k]].first
                                    - mat[r][k].second * input[cur_indices[k]].second;
                            tmp.second = tmp.second
                                    + mat[r][k].first * input[cur_indices[k]].second
                                    - mat[r][k].second * input[cur_indices[k]].first;
                        } // end for k
                        output[cur_indices[r]] = tmp;
                    } // end for r
                } // end if
            } // end for i
            return output;
        }

        inline z3::check_result check_in_shell(const z3::solver& solver) {
            std::stringstream str_to_check;
            // echo "something" | z3 -in
            str_to_check << solver << "(check-sat)\n | z3 -in";
            const raymii::CommandResult cmd_res = raymii::Command::exec(str_to_check.str());
            if (cmd_res.exitstatus != 0) {
                // z3 exit with error!
                return z3::unknown;
            }
            else {
                if (cmd_res.output.front() == 'u')
                    return z3::unsat;
                else
                    return z3::sat;
            }
        }

    }

}
