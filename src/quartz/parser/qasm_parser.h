#pragma once

#include "../context/context.h"
#include "quartz/circuitseq/circuitseq.h"

#include <cassert>
#include <cmath>
#include <fstream>
#include <map>

namespace quartz {

// Replaces (in-place) all instances of tofind in data with toreplace.
void find_and_replace_all(std::string &data, const std::string &tofind,
                          const std::string &toreplace);

// Replaces (in-place) the first instance of tofind in data with toreplace.
void find_and_replace_first(std::string &data, const std::string &tofind,
                            const std::string &toreplace);

// Replaces (in-place) the last instance of tofind in data with toreplace.
void find_and_replace_last(std::string &data, const std::string &tofind,
                           const std::string &toreplace);

// Converts a string to a non-negative integer value. Requires that input is a
// valid non-negative integer consisting only of digits from 0 to 9.
int string_to_number(const std::string &input);

// If token is the name of a gate, then sets type to the gate type and returns
// true. Otherwise, false is returned.
bool is_gate_string(const std::string &token, GateType &type);

// Removes all trailing and leading spaces from input. The input remains
// unchanged and a new string is returned.
std::string strip(const std::string &input);

/**
 * Helper class to parse symbolic parameter declarations, and the parameters
 * passed to parameterized gates.
 */
class ParamParser {
 public:
  ParamParser(Context *ctx)
      : ctx_(ctx), symbolic_pi_(false), first_file_(true) {}

  /**
   * Adds an angle array declaration to the registry of symbolic parameters.
   * This entry will associate each cell of the array, as specified by a token
   * `[array,len] name`, with a unique symbolic parameter.
   * @param ss a string stream containing the token.
   * @return true if and only if the declaration is parsed successfully.
   */
  bool parse_array_decl(std::stringstream &ss);

  /**
   * Parses a stream which is known to contain a parameter expression. The
   * following formats are supported, where n and m are decimal literals, i is
   * an integer literal, and name is a string:
   *    | pi*n
   *    | n*pi
   *    | n*pi/m
   *    | n
   *    | pi/m
   *    | n/(m*pi)
   *    | name[i]
   * @param token the string stream which contains the parameter expression.
   * @return the parameter id for this expression in the current context.
   */
  int parse_expr(std::stringstream &token);

  /**
   * Calling this function allows for symbolic pi values to be enabled or
   * disabled. When symbolic pi values are enabled, each constant pi/n will be
   * replaced by the symbolic expression pi(n).
   * @param v if true, then symbolic pi values will be enabled.
   */
  void use_symbolic_pi(bool v) { symbolic_pi_ = v; }

  /**
   * Calling this function indicates that a file has been entirely parsed. In
   * particular, after the first file is parsed, only the names and indices of
   * existing symbolic variables may be used.
   */
  void end_file() { first_file_ = false; }

 private:
  /**
   * Implementation details for each term in parse_expr. The supported formats
   * for terms are as follows, where n and m are decimal literals, i is an
   * integer literal, and name is a string:
   *    | pi*n
   *    | n*pi
   *    | n*pi/m
   *    | n
   *    | pi/m
   *    | n/(m*pi)
   *    | name[i]
   * @param negative if true, then the parameter should be negative.
   * @param token the string which contains the parameter expression.
   * @return the parameter id for this term in the current context.
   */
  int parse_term(bool negative, std::string token);

  /**
   * Implementation details for parse_term when the term is a constant literal
   * value or of the form n/(m*pi).
   * @param negative if true, then the parameter should be negative.
   * @param p the literal value as a floating-point value.
   * @return the parameter id for this expression in the current context.
   */
  int parse_number(bool negative, ParamType p);

  /**
   * Implementation details for parse_term when the term is of the form pi*n,
   * n*pi, n*pi/m, or pi/m.
   * @param negative if true, then the parameter should be negative.
   * @param num either the value of n, or 1 if it is not in the format.
   * @param denom either the value of m, or 1 if it is not in the format.
   * @return the parameter id for this expression in the current context.
   */
  int parse_pi_term(bool negative, ParamType num, ParamType denom);

  /**
   * The context against which, all symbolic parameters are initialized, and
   * all expressions are evaluated.
   */
  Context *ctx_;

  /**
   * Maps parameter values to constant identifiers in the context of ctx_.
   */
  std::unordered_map<ParamType, int> number_params_;

  /**
   * Maps a pair (n, m) to the identifier of a symbolic expression, in the
   * context of ctx_, which corresponds to n*pi/m.
   * @see symbolic_pi_
   */
  std::unordered_map<ParamType, std::unordered_map<ParamType, int>> pi_params_;

  /**
   * Maps a parameter array name and index to the identifier of a symbolic
   * parameter, in the context of ctx_, which corresponds to the reference.
   */
  std::unordered_map<std::string, std::unordered_map<int, int>> symb_params_;

  /**
   * If true, then rational multiples of pi are evaluated exactly as symbolic
   * values. For example, 3*pi/2 becomes mult(3, pi(2)). If false, then 3*pi/2
   * is evaluated and stored as a floating-point constant.
   */
  bool symbolic_pi_;

  /**
   * If true, then this parameter parser has already parsed an OpenQASM 3 file.
   * When parsing the first file, it is expected that all parameter variables
   * are new. When parsing subsequent files, it is expected that all parameter
   * variables were defined in the original file.
   */
  bool first_file_;
};

/**
 * Helper class to parse qubit array declarations, and references to the cells
 * of these arrays.
 */
class QubitParser {
 public:
  QubitParser() : finalized_(false) {}

  /**
   * Adds a qreg declaration to the registry of qubit array declarations. This
   * entry will associate the name of the variable to the length of the array,
   * as specified by a token 'name[len] from a statement 'qreg name[len];'.
   * @param ss a string stream containing the token.
   * @return true if and only if the declaration is parsed successfully.
   * @warning requires that finalize() has not yet been called.
   */
  bool parse_qasm2_decl(std::stringstream &ss);

  /**
   * Adds a qubit declaration to the registry of qubit array declarations. This
   * entry will associate the name of the variable to the length of the array,
   * as specified by a token '[len] name from a statement 'qubit[len] name;'.
   * @param ss a string stream containing the token.
   * @return true if and only if the declaration is parsed successfully.
   * @warning requires that finalize() has not yet been called.
   */
  bool parse_qasm3_decl(std::stringstream &ss);

  /**
   * Determines the global qubit index for a qubit array access. This method
   * expects that the token is given in the form 'name[idx]', and that the
   * input stream may contain addition tokens after this refeerence.
   * @param ss a string stream containing the token.
   * @return the global qubit index for the reference, or -1 on failure.
   * @warning requires that finalize() has already been called.
   */
  int parse_access(std::stringstream &ss);

  /**
   * In CircuitSeq, qubits are modelled as a single global array. Calling this
   * method indicates that no more qreg arrays will be declared, and allows for
   * the mapping from qreg arrays to qubit indices to be finalized.
   * @return the total number of qubits in the global qubit array.
   * @warning After this method is called, all calls to parse_qasm2_decl and
   *   parse_qasm3_decl will fail.
   */
  int finalize();

 private:
  /**
   * Implementation details for parse_qasm2_decl and parse_qasm3_decl. This
   * method takes the name and length as strings, so that it is agnostic to
   * the syntax of OpenQASM 2 and OpenQASM 3.
   * @param ss the string stream from which the name are length are obtained.
   * @param name the name of the qubit array.
   * @param lstr the length of the qubit array, as a string.
   * @return the total number of qubits in the global qubit array.
   */
  bool add_decl(std::stringstream &ss, std::string &name, std::string &lstr);

  /**
   * When this flag, no more qubit declarations are allowed, and it is possible
   * to map qubit references to global array indices.
   * @see QubitParser::index_offset
   */
  bool finalized_;

  /**
   * At the beginning, |index_offset| stores the mapping from qreg names to
   * their sizes. After creating the CircuitSeq object, |index_offset| stores
   * the mapping from qreg names to the qubit index offset. The qregs are
   * ordered alphabetically.
   */
  std::map<std::string, int> index_offset;
};

// Parser from OpenQASM files to CircuitSeq objects.
class QASMParser {
 public:
  QASMParser(Context *ctx) : ctx_(ctx), param_parser_(ctx) {}

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

  void use_symbolic_pi(bool v) { param_parser_.use_symbolic_pi(v); }

 private:
  Context *ctx_;
  ParamParser param_parser_;
};

// We cannot put this template function implementation in a .cpp file.
template <class _CharT, class _Traits>
bool QASMParser::load_qasm_stream(
    std::basic_istream<_CharT, _Traits> &qasm_stream, CircuitSeq *&seq) {
  // Results and sub-parsers.
  seq = nullptr;
  QubitParser qubit_parser;

  // Generalized control data.
  bool in_general_controlled_gate_block = false;
  std::vector<bool> general_control_flipped_qubits;
  int num_flipped_qubits;

  // Parse each line of the file.
  std::string line;
  GateType gate_type;
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
    // Adds spaces before square brackets to support OpenQASM 3 declarations.
    // The spaces should not apply to the parameters passed to rotation gates.
    find_and_replace_all(line, "array[", "array [");
    find_and_replace_all(line, "qubit[", "qubit [");
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
      if (!qubit_parser.parse_qasm2_decl(ss)) {
        return false;
      }
    } else if (command == "qubit") {
      if (!qubit_parser.parse_qasm3_decl(ss)) {
        return false;
      }
    } else if (command == "input") {
      // This should be an array.
      std::string type;
      ss >> type;
      if (type != "array") {
        std::cout << "Unexpected input variable type: " << type << std::endl;
        assert(false);
        return false;
      }

      // Parses the parameter array.
      if (!param_parser_.parse_array_decl(ss)) {
        return false;
      }
    } else if (is_gate_string(command, gate_type)) {
      // End the phase of creating qregs.
      if (seq == nullptr) {
        int num_qubits = qubit_parser.finalize();
        seq = new CircuitSeq(num_qubits);
      }

      // Gate parsing.
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
        int index = param_parser_.parse_expr(ss);
        if (index == -1) {
          return false;
        }
        param_indices[i] = index;
      }
      for (int i = 0; i < num_qubits; ++i) {
        assert(ss.good());
        int index = qubit_parser.parse_access(ss);
        if (index == -1) {
          return false;
        }
        qubit_indices[i] = index;
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

  // Successfully parsed file.
  param_parser_.end_file();
  return true;
}

}  // namespace quartz
