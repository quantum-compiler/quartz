#pragma once

#include "../gate/gate_utils.h"
#include "context.h"

#include <assert.h>
#include <cmath>
#include <istream>
#include <map>
#include <set>
#include <string>
#include <vector>

namespace quartz {

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
    while (gate_tp.empty()) {
      getline(iss, gate_tp, ' ');
    }
    tp = to_gate_type(gate_tp);

    std::string input;
    while (!iss.eof()) {
      getline(iss, input, ' ');
      if (input.empty())
        continue;
      if (input[0] == 'q') {
        qubit_idx.push_back(stoi(input.substr(1, input.size() - 1)));
      } else if (input[0] == 'p' && input.find("pi") == input.npos) {
        param_idx.push_back(stoi(input.substr(1, input.size() - 1)));
      } else {
        auto pi_pos = input.find("pi");
        if (pi_pos == input.npos) {
          constant_params[param_idx.size()] = std::stod(input);
        } else if (pi_pos == 0) {
          constant_params[param_idx.size()] = PI;
        } else if (pi_pos == 1 && input[0] == '-') {
          constant_params[param_idx.size()] = -PI;
        } else {
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
  std::vector<int> qubit_idx;
  std::vector<int> param_idx;
  std::map<int, ParamType> constant_params;
};

class RuleParser {
 public:
  explicit RuleParser(const std::vector<std::string> &rules_) {
    for (const auto &rule : rules_) {
      std::string gate_name;
      auto pos = rule.find('=');
      assert(pos != rule.npos);
      auto src_cmd = Command(rule.substr(0, rule.find('=')));
      GateType tp = src_cmd.get_gate_type();

      std::istringstream iss1(rule.substr(pos + 1));
      std::vector<Command> cmds;
      std::string input;
      while (!iss1.eof()) {
        getline(iss1, input, ';');
        // std::cout << input << std::endl;
        if (!input.empty()) {
          cmds.emplace_back(input);
        }
      }
      std::set<GateType> tp_set;
      for (auto cmd : cmds) {
        tp_set.insert(cmd.get_gate_type());
      }

      if (rules.find(tp) == rules.end()) {
        std::vector<std::pair<
            Command, std::pair<std::vector<Command>, std::set<GateType>>>>
            tp_rules;
        tp_rules.emplace_back(src_cmd, std::make_pair(cmds, tp_set));
        rules[tp] = tp_rules;
      } else {
        rules[tp].emplace_back(src_cmd, std::make_pair(cmds, tp_set));
      }
    }
  }

  /**
   * Find all conversion commands for a gate type.
   * @param ctx The destination context.
   * @param tp The gate type.
   * @param src_cmd Return the source commands.
   * @param cmds Return the target commands.
   * @return The number of commands returned.
   */
  int find_convert_commands(Context *ctx, const GateType tp,
                            std::vector<Command> &src_cmd,
                            std::vector<std::vector<Command>> &cmds) {
    src_cmd.clear();
    cmds.clear();
    if (rules.find(tp) == rules.end()) {
      std::cerr
          << "No rules with the same gate type found to fit gate to context."
          << std::endl;
      return 0;
    }

    std::set<GateType> supported_gate_tp_set(ctx->get_supported_gates().begin(),
                                             ctx->get_supported_gates().end());
    std::vector<
        std::pair<Command, std::pair<std::vector<Command>, std::set<GateType>>>>
        cmds_list = rules[tp];
    for (const auto &cmds_info : cmds_list) {
      std::set<GateType> used_gate_tp_set = cmds_info.second.second;
      bool not_found = false;
      for (auto it : used_gate_tp_set) {
        if (supported_gate_tp_set.find(it) == supported_gate_tp_set.end()) {
          not_found = true;
          break;
        }
      }
      if (!not_found) {
        src_cmd.push_back(cmds_info.first);
        cmds.push_back(cmds_info.second.first);
      }
    }
    return (int)src_cmd.size();
  }

 public:
  static RuleParser ccz_cx_rz_rules() {
    return RuleParser({"ccz q0 q1 q2 = cx q1 q2; rz q2 -0.25pi; cx q0 q2; rz "
                       "q2 0.25pi; cx q1 q2; rz q2 -0.25pi; cx "
                       "q0 q2; cx q0 q1; rz q1 -0.25pi; cx q0 q1; rz q0 "
                       "0.25pi; rz q1 0.25pi; rz q2 0.25pi;",
                       "ccz q0 q1 q2 = cx q1 q2; rz q2 0.25pi; cx q0 q2; rz "
                       "q2 -0.25pi; cx q1 q2; rz q2 0.25pi; cx "
                       "q0 q2; cx q0 q1; rz q1 0.25pi; cx q0 q1; rz q0 "
                       "-0.25pi; rz q1 -0.25pi; rz q2 -0.25pi;"});
  }

  static RuleParser ccz_cx_u1_rules() {
    return RuleParser({"ccz q0 q1 q2 = cx q1 q2; u1 q2 -0.25pi; cx q0 q2; u1 "
                       "q2 0.25pi; cx q1 q2; u1 q2 -0.25pi; cx "
                       "q0 q2; cx q0 q1; u1 q1 -0.25pi; cx q0 q1; u1 q0 "
                       "0.25pi; u1 q1 0.25pi; u1 q2 0.25pi;",
                       "ccz q0 q1 q2 = cx q1 q2; u1 q2 0.25pi; cx q0 q2; u1 "
                       "q2 -0.25pi; cx q1 q2; u1 q2 0.25pi; cx "
                       "q0 q2; cx q0 q1; u1 q1 0.25pi; cx q0 q1; u1 q0 "
                       "-0.25pi; u1 q1 -0.25pi; u1 q2 -0.25pi;"});
  }

  static RuleParser ccz_cx_t_rules() {
    return RuleParser({"ccz q0 q1 q2 = cx q1 q2; tdg q2; cx q0 q2; t "
                       "q2; cx q1 q2; tdg q2; cx "
                       "q0 q2; cx q0 q1; tdg q1; cx q0 q1; t q0"
                       "; t q1; t q2;",
                       "ccz q0 q1 q2 = cx q1 q2; t q2; cx q0 q2; tdg "
                       "q2; cx q1 q2; t q2; cx "
                       "q0 q2; cx q0 q1; t q1; cx q0 q1; tdg q0"
                       "; tdg q1; tdg q2;"});
  }

 private:
  std::map<GateType,
           std::vector<std::pair<
               Command, std::pair<std::vector<Command>, std::set<GateType>>>>>
      rules;
};

}  // namespace quartz
