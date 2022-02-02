#pragma once

#include <fstream>
#include <cassert>
#include "../context/context.h"
#include "../dag/dag.h"

namespace quartz {
	void find_and_replace_all(std::string &data, const std::string &tofind,
	                          const std::string &toreplace) {
		size_t pos = data.find(tofind);
		while (pos != std::string::npos) {
			data.replace(pos, tofind.size(), toreplace);
			pos = data.find(tofind, pos + toreplace.size());
		}
	}

	int string_to_number(const std::string &input) {
		int ret = -1;
		for (int i = 0; i < input.length(); i++) {
			if (input[i] >= '0' && input[i] <= '9') {
				if (ret == -1) {
					ret = 0;
				}
				ret = ret * 10 + input[i] - '0';
			}
		}
		return ret;
	}

	bool is_gate_string(const std::string &token, GateType &type) {

#define PER_GATE(x, XGate)                                                     \
	if (token == std::string(#x)) {                                            \
		type = GateType::x;                                                    \
		return true;                                                           \
	}

#include "../gate/gates.inc.h"

#undef PER_GATE
		return false;
	}

	class QASMParser {
	public:
		QASMParser(Context *ctx) : context(ctx) {}
		bool load_qasm(const std::string &file_name, DAG *&dag) {
			dag = NULL;
			std::ifstream fin;
			fin.open(file_name, std::ifstream::in);
			if (!fin.is_open()) {
				return false;
			}
			std::string line;
			GateType gate_type;
			while (std::getline(fin, line)) {
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
				}
				else if (command == "include") {
					continue; // ignore this line
				}
				else if (command == "creg") {
					continue; // ignore this line
				}
				else if (command == "qreg") {
					std::string token;
					getline(ss, token, ' ');
					size_t num_qubits = string_to_number(token);
					// TODO: temporarily assume a program has at most 16
					// parameters
					assert(dag == NULL);
					dag = new DAG(num_qubits, 16);
					assert(!ss.good());
				}
				else if (is_gate_string(command, gate_type)) {
					Gate *gate = context->get_gate(gate_type);
					// Currently don't support parameter gate
					assert(gate->is_quantum_gate());
					std::vector< int > qubit_indices, parameter_indices;
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
					bool ret = dag->add_gate(qubit_indices, parameter_indices,
					                         gate, NULL);
					assert(ret == true);
				}
				else {
					std::cout << "Unknown gate: " << command << std::endl;
					assert(false);
				}
			}
			fin.close();
			return true;
		}

	private:
		Context *context;
	};

} // namespace quartz