#pragma once

#include <map>
#include <string>
#include <vector>
#include <istream>
#include <cmath>
#include <set>
#include <assert.h>
#include "../gate/gate_utils.h"
#include "context.h"

namespace quartz {
#define PI 3.14159265358979323846

	class Command {
	public:
		Command() {}
		Command(const Command &cmd) {
			tp = cmd.tp;
			qubit_idx = cmd.qubit_idx;
			param_idx = cmd.param_idx;
			constant_params = cmd.constant_params;
		}
		Command(const std::string &str_command) {
			std::istringstream iss(str_command);
			std::string gate_tp;
			while (gate_tp == "") {
				getline(iss, gate_tp, ' ');
			}
			tp = to_gate_type(gate_tp);

			std::string input;
			while (!iss.eof()) {
				getline(iss, input, ' ');
				if (input == "")
					continue;
				if (input[0] == 'q') {
					qubit_idx.push_back(
					    stoi(input.substr(1, input.size() - 1)));
				}
				else if (input[0] == 'p' && input.find("pi") == input.npos) {
					param_idx.push_back(
					    stoi(input.substr(1, input.size() - 1)));
				}
				else {
					auto pi_pos = input.find("pi");
					if (pi_pos == input.npos) {
						constant_params[param_idx.size()] = std::stod(input);
					}
					else if (pi_pos == 0) {
						constant_params[param_idx.size()] = PI;
					}
					else if (pi_pos == 1 && input[0] == '-') {
						constant_params[param_idx.size()] = -PI;
					}
					else {
						constant_params[param_idx.size()] =
						    PI * std::stod(input.substr(0, pi_pos));
					}
					param_idx.push_back(-1);
				}
			}
		}

		GateType get_gate_type() { return tp; }

		void print() {
			std::cout << gate_type_name(tp) << std::endl;
			for (auto idx : qubit_idx) {
				std::cout << idx << std::endl;
			}
			for (auto idx : param_idx) {
				std::cout << idx << std::endl;
			}
			for (auto p : constant_params) {
				std::cout << p.first << " " << p.second << std::endl;
			}
		}

		GateType tp;
		std::vector< int > qubit_idx;
		std::vector< int > param_idx;
		std::map< int, ParamType > constant_params;
	};

	class RuleParser {
	public:
		RuleParser(std::vector< std::string > rules_) {
			for (auto rule : rules_) {
				std::string gate_name;
				auto pos = rule.find("=");
				assert(pos != rule.npos);
				auto src_cmd = Command(rule.substr(0, rule.find("=")));
				GateType tp = src_cmd.get_gate_type();

				std::istringstream iss1(rule.substr(pos + 1));
				std::vector< Command > cmds;
				std::string input;
				while (!iss1.eof()) {
					getline(iss1, input, ';');
					// std::cout << input << std::endl;
					if (!input.empty()) {
						cmds.push_back(Command(input));
					}
				}
				std::set< GateType > tp_set;
				for (auto cmd : cmds) {
					tp_set.insert(cmd.get_gate_type());
				}

				if (rules.find(tp) == rules.end()) {
					std::vector< std::pair<
					    Command, std::pair< std::vector< Command >,
					                        std::set< GateType > > > >
					    tp_rules;
					tp_rules.push_back(
					    std::make_pair(src_cmd, std::make_pair(cmds, tp_set)));
					rules[tp] = tp_rules;
				}
				else {
					rules[tp].push_back(
					    std::make_pair(src_cmd, std::make_pair(cmds, tp_set)));
				}
			}
		}

		bool find_convert_commands(Context *ctx, const GateType tp,
		                           Command &src_cmd,
		                           std::vector< Command > &cmds) {
			cmds.clear();
			if (rules.find(tp) == rules.end()) {
				std::cout << "No rules found to fit gate to context"
				          << std::endl;
				return false;
			}

			std::set< GateType > supported_gate_tp_set(
			    ctx->get_supported_gates().begin(),
			    ctx->get_supported_gates().end());
			std::vector<
			    std::pair< Command, std::pair< std::vector< Command >,
			                                   std::set< GateType > > > >
			    cmds_list = rules[tp];
			for (auto cmds_info : cmds_list) {
				std::set< GateType > used_gate_tp_set = cmds_info.second.second;
				bool not_found = false;
				for (auto it = used_gate_tp_set.begin();
				     it != used_gate_tp_set.end(); ++it) {
					if (supported_gate_tp_set.find(*it) ==
					    supported_gate_tp_set.end()) {
						not_found = true;
						break;
					}
				}
				if (!not_found) {
					cmds = cmds_info.second.first;
					src_cmd = cmds_info.first;
					// for (auto cmd : cmds) {
					//   cmd.print();
					// }
					return true;
				}
			}
			std::cout << "No rules found to fit gate to context" << std::endl;
			return false;
		}

	public:
		static std::pair< RuleParser *, RuleParser * > ccz_cx_rz_rules() {
			RuleParser *rule_0 = new RuleParser(
			    {"ccz q0 q1 q2 = cx q1 q2; rz q2 -0.25pi; cx q0 q2; rz "
			     "q2 0.25pi; cx q1 q2; rz q2 -0.25pi; cx "
			     "q0 q2; cx q0 q1; rz q1 -0.25pi; cx q0 q1; rz q0 "
			     "0.25pi; rz q1 0.25pi; rz q2 0.25pi;"});
			RuleParser *rule_1 = new RuleParser(
			    {"ccz q0 q1 q2 = cx q1 q2; rz q2 0.25pi; cx q0 q2; rz "
			     "q2 -0.25pi; cx q1 q2; rz q2 0.25pi; cx "
			     "q0 q2; cx q0 q1; rz q1 0.25pi; cx q0 q1; rz q0 "
			     "-0.25pi; rz q1 -0.25pi; rz q2 -0.25pi;"});
			return std::make_pair(rule_0, rule_1);
		}

		// TODO: change all rz to u1
		static std::pair< RuleParser *, RuleParser * > ccz_cx_u1_rules() {
			RuleParser *rule_0 = new RuleParser(
			    {"ccz q0 q1 q2 = cx q1 q2; u1 q2 -0.25pi; cx q0 q2; u1 "
			     "q2 0.25pi; cx q1 q2; u1 q2 -0.25pi; cx "
			     "q0 q2; cx q0 q1; u1 q1 -0.25pi; cx q0 q1; u1 q0 "
			     "0.25pi; u1 q1 0.25pi; u1 q2 0.25pi;"});
			RuleParser *rule_1 = new RuleParser(
			    {"ccz q0 q1 q2 = cx q1 q2; u1 q2 0.25pi; cx q0 q2; u1 "
			     "q2 -0.25pi; cx q1 q2; u1 q2 0.25pi; cx "
			     "q0 q2; cx q0 q1; u1 q1 0.25pi; cx q0 q1; u1 q0 "
			     "-0.25pi; u1 q1 -0.25pi; u1 q2 -0.25pi;"});
			return std::make_pair(rule_0, rule_1);
		}

	private:
		std::map< GateType, std::vector< std::pair<
		                        Command, std::pair< std::vector< Command >,
		                                            std::set< GateType > > > > >
		    rules;
	};

} // namespace quartz
