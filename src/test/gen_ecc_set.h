#pragma once

#include "quartz/context/context.h"
#include "quartz/generator/generator.h"

namespace quartz {
void gen_ecc_set(const std::vector<GateType> &supported_gates,
                 const std::string &file_prefix, bool unique_parameters,
                 int num_qubits, int num_input_parameters,
                 int max_num_quantum_gates, int max_num_param_gates = 1);
}
