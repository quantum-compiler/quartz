#pragma once

#include "../context/context.h"
#include "quartz/circuitseq/circuitseq.h"

#include <cassert>
#include <cmath>
#include <fstream>
#include <map>

namespace quartz {
void find_and_replace_all(std::string &data, const std::string &tofind,
                          const std::string &toreplace);

void find_and_replace_first(std::string &data, const std::string &tofind,
                            const std::string &toreplace);

void find_and_replace_last(std::string &data, const std::string &tofind,
                           const std::string &toreplace);

int string_to_number(const std::string &input);

bool is_gate_string(const std::string &token, GateType &type);

std::string strip(const std::string &input);

class QASMParser {
 public:
  QASMParser(Context *ctx) : ctx_(ctx) {}

  template <class _CharT, class _Traits>
  bool load_qasm_stream(std::basic_istream<_CharT, _Traits> &qasm_stream,
                        CircuitSeq *&seq);

  bool load_qasm_str(const std::string &qasm_str, CircuitSeq *&seq) {
    std::stringstream sstream(qasm_str);
    return load_qasm_stream(sstream, seq);
  }

  bool load_qasm(const std::string &file_name, CircuitSeq *&seq) {
    std::ifstream fin;
    fin.open(file_name, std::ifstream::in);
    if (!fin.is_open()) {
      std::cerr << "QASMParser fails to open " << file_name << std::endl;
      return false;
    }
    const bool res = load_qasm_stream(fin, seq);
    fin.close();
    return res;
  }

 private:
  Context *ctx_;
};

// We cannot put this template function implementation in a .cpp file.
template <class _CharT, class _Traits>
bool QASMParser::load_qasm_stream(
    std::basic_istream<_CharT, _Traits> &qasm_stream, CircuitSeq *&seq) {
  seq = nullptr;
  std::string line;
  GateType gate_type;
  // At the beginning, |index_offset| stores the mapping from qreg names to
  // their sizes. After creating the CircuitSeq object, |index_offset| stores
  // the mapping from qreg names to the qubit index offset. The qregs are
  // ordered alphabetically.
  std::map<std::string, int> index_offset;
  std::unordered_map<ParamType, int> parameters;
  bool in_general_controlled_gate_block = false;
  std::vector<bool> general_control_flipped_qubits;
  int num_flipped_qubits;

  while (std::getline(qasm_stream, line, ';')) {
    if (line.find("//ctrl") != std::string::npos) {
      // Quartz's specific comment to enter a general control gate block
      assert(!in_general_controlled_gate_block);
      in_general_controlled_gate_block = true;
      general_control_flipped_qubits.clear();
      num_flipped_qubits = 0;
    }
    // Remove comments
    auto comment_position = line.find("//");
    while (comment_position != std::string::npos) {
      auto newline_position = line.find('\n', comment_position + 2 /*"//"*/);
      if (newline_position == std::string::npos) {
        // remove until the end
        line.resize(comment_position);
        break;
      }
      line.replace(comment_position, newline_position - comment_position, "");
      comment_position = line.find("//", comment_position);
    }
    // Replace comma with space
    find_and_replace_all(line, ",", " ");
    // Replace parentheses for parameterized gate with space
    find_and_replace_first(line, "(", " ");
    find_and_replace_last(line, ")", " ");
    // Ignore end of line
    find_and_replace_all(line, "\n", "");
    while (!line.empty() && line.front() == ' ') {
      line.erase(0, 1);
    }
    std::stringstream ss(line);
    std::string command;
    std::getline(ss, command, ' ');
    // Strip the command to avoid potential '\r'
    command = strip(command);
    // XXX: "u" is an alias of "u3".
    if (command == std::string("u")) {
      command = std::string("u3");
    }
    if (command.empty()) {
      continue;  // empty line, ignore this line
    } else if (command == "OPENQASM" || command == "OpenQASM") {
      continue;  // header, ignore this line
    } else if (command == "include") {
      continue;  // header, ignore this line
    } else if (command == "barrier") {
      continue;  // file end, ignore this line
    } else if (command == "measure") {
      continue;  // file end, ignore this line
    } else if (command == "creg") {
      continue;  // ignore this line
    } else if (command == "qreg") {
      std::string name;
      getline(ss, name, '[');
      name = strip(name);
      if (seq != nullptr) {
        std::cerr << "We only support creating qregs before all quantum gates."
                  << std::endl;
        return false;
      }
      std::string token;
      getline(ss, token, ' ');
      int num_qubits = string_to_number(token);
      // No two qregs have the same name.
      assert(index_offset.count(name) == 0);
      index_offset[name] = num_qubits;
      assert(!ss.good());
    } else if (is_gate_string(command, gate_type)) {
      if (seq == nullptr) {
        // End the phase of creating qregs.
        // Compute the total number of qubits, and let |index_offset| stores
        // the mapping from qreg names to the qubit index offset.
        int num_qubits = 0;
        for (auto &qreg : index_offset) {
          int new_num_qubits = num_qubits + qreg.second;
          qreg.second = num_qubits;
          num_qubits = new_num_qubits;
        }
        seq = new CircuitSeq(num_qubits);
      }
      Gate *gate = ctx_->get_gate(gate_type);
      if (!gate) {
        std::cerr << "Unsupported gate in current context: " << command
                  << std::endl;
        return false;
      }
      int num_qubits = ctx_->get_gate(gate_type)->num_qubits;
      int num_params = ctx_->get_gate(gate_type)->num_parameters;
      std::vector<int> qubit_indices(num_qubits);
      std::vector<int> param_indices(num_params);
      for (int i = 0; i < num_params; ++i) {
        assert(ss.good());
        std::string token;
        ss >> token;
        // Currently only support the format of
        // pi*0.123,
        // 0.123*pi,
        // 0.123*pi/2,
        // 0.123
        // pi
        // pi/2
        // 0.123/(2*pi)
        ParamType p = 0.0;
        bool negative = token[0] == '-';
        if (negative)
          token = token.substr(1);
        if (token.find("pi") == 0) {
          if (token == "pi") {
            // pi
            p = PI;
          } else {
            auto d = token.substr(3, std::string::npos);
            if (token[2] == '*') {
              // pi*0.123
              p = std::stod(d) * PI;
            } else if (token[2] == '/') {
              // pi/2
              p = PI / std::stod(d);
            } else {
              std::cerr << "Unsupported parameter format: " << token
                        << std::endl;
              assert(false);
            }
          }
        } else if (token.find("pi") != std::string::npos) {
          if (token.find('(') != std::string::npos) {
            assert(token.find('/') != std::string::npos);
            auto left_parenthesis_pos = token.find('(');
            // 0.123/(2*pi)
            p = std::stod(token.substr(0, token.find('/'))) / PI;
            p /= std::stod(
                token.substr(left_parenthesis_pos + 1,
                             token.find('*') - left_parenthesis_pos - 1));
          } else {
            // 0.123*pi
            auto d = token.substr(0, token.find('*'));
            p = std::stod(d) * PI;
            if (token.find('/') != std::string::npos) {
              // 0.123*pi/2
              p = p / std::stod(token.substr(token.find('/') + 1));
            }
          }
        } else {
          // 0.123
          p = std::stod(token);
        }
        if (negative)
          p = -p;
        if (parameters.count(p) == 0) {
          int param_id = ctx_->get_new_param_id(p);
          parameters[p] = param_id;
        }
        param_indices[i] = parameters[p];
      }
      for (int i = 0; i < num_qubits; ++i) {
        assert(ss.good());
        std::string token;
        std::string name;
        getline(ss, name, '[');
        name = strip(name);
        ss >> token;
        int index = string_to_number(token);
        if (index_offset.count(name) == 0) {
          std::cerr << "Unknown qreg: " << name << std::endl;
          return false;
        }
        if (index == -1) {
          std::cerr << "Unknown qubit index: " << token << std::endl;
          return false;
        }
        qubit_indices[i] = index_offset[name] + index;
      }
      if (in_general_controlled_gate_block) {
        if (gate_type == GateType::x) {
          // Flip a qubit.
          if (qubit_indices[0] >= (int)general_control_flipped_qubits.size()) {
            general_control_flipped_qubits.resize(qubit_indices[0] + 1);
          }
          if (general_control_flipped_qubits[qubit_indices[0]]) {
            // Already flipped, now flip it back.
            general_control_flipped_qubits[qubit_indices[0]] = false;
            num_flipped_qubits--;
            if (!num_flipped_qubits) {
              // Exit this general controlled gate block.
              in_general_controlled_gate_block = false;
            }
          } else {
            num_flipped_qubits++;
            general_control_flipped_qubits[qubit_indices[0]] = true;
          }
        } else if (gate->get_num_control_qubits() > 0) {
          // The general controlled gate.
          int num_control = gate->get_num_control_qubits();
          assert(num_control <= (int)qubit_indices.size());
          std::vector<bool> state(num_control, true);
          for (int i = 0; i < num_control; i++) {
            if (qubit_indices[i] < (int)general_control_flipped_qubits.size()) {
              // If flipped, set to 0 (default is 1).
              state[i] = !general_control_flipped_qubits[qubit_indices[i]];
            }
          }
          auto general_controlled_gate =
              ctx_->get_general_controlled_gate(gate_type, state);
          seq->add_gate(qubit_indices, param_indices, general_controlled_gate,
                        ctx_);
        } else {
          std::cerr << "Unexpected gate " << command
                    << " in general controlled gate block." << std::endl;
          assert(false);
        }
      } else {
        seq->add_gate(qubit_indices, param_indices, gate, ctx_);
      }
    } else {
      std::cerr << "Unknown gate: " << command << std::endl;
      assert(false);
    }
  }
  return true;
}

}  // namespace quartz
