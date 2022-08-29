#pragma once

#include "../context/context.h"
#include "../dag/dag.h"

#include <cassert>
#include <fstream>

namespace quartz {
void find_and_replace_all(std::string &data, const std::string &tofind,
                          const std::string &toreplace);

int string_to_number(const std::string &input);

bool is_gate_string(const std::string &token, GateType &type);

class QASMParser {
public:
  QASMParser(Context *ctx) : context(ctx) {}

  template <class _CharT, class _Traits>
  bool load_qasm_stream(std::basic_istream<_CharT, _Traits> &qasm_stream,
                        DAG *&dag) {
    dag = NULL;
    std::string line;
    GateType gate_type;
    while (std::getline(qasm_stream, line)) {
      // repleace comma with space
      find_and_replace_all(line, ",", " ");
      // ignore semicolon at the end
      find_and_replace_all(line, ";", "");
      // std::cout << line << std::endl;
      std::stringstream ss(line);
      std::string command;
      std::getline(ss, command, ' ');
      if (command == "OPENQASM") {
        continue; // ignore this line
      } else if (command == "include") {
        continue; // ignore this line
      } else if (command == "creg") {
        continue; // ignore this line
      } else if (command == "qreg") {
        std::string token;
        getline(ss, token, ' ');
        size_t num_qubits = string_to_number(token);
        // TODO: temporarily assume a program has at most 16
        // parameters
        assert(dag == NULL);
        dag = new DAG(num_qubits, 16);
        assert(!ss.good());
      } else if (is_gate_string(command, gate_type)) {
        Gate *gate = context->get_gate(gate_type);
        if (!gate) {
          std::cerr << "Unsupported gate in current context: " << command
                    << std::endl;
          return false;
        }
        // Currently don't support parameter gate
        assert(gate->is_quantum_gate());
        std::vector<int> qubit_indices, parameter_indices;
        while (ss.good()) {
          std::string token;
          //   std::getline(ss, token, ' ');
          ss >> token;
          int index = string_to_number(token);
          if (index != -1) {
            qubit_indices.push_back(index);
          }
        }
        assert(dag != NULL);
        bool ret = dag->add_gate(qubit_indices, parameter_indices, gate, NULL);
        assert(ret == true);
      } else {
        std::cout << "Unknown gate: " << command << std::endl;
        assert(false);
      }
    }
    return true;
  }

  bool load_qasm_str(const std::string &qasm_str, DAG *&dag) {
    std::stringstream sstream(qasm_str);
    return load_qasm_stream(sstream, dag);
  }

  bool load_qasm(const std::string &file_name, DAG *&dag) {
    std::ifstream fin;
    fin.open(file_name, std::ifstream::in);
    if (!fin.is_open()) {
      std::cerr << "QASMParser fails to open " << file_name << std::endl;
      return false;
    }
    const bool res = load_qasm_stream(fin, dag);
    fin.close();
    return res;
  }

private:
  Context *context;
};

} // namespace quartz
