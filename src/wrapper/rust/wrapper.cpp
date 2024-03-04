#include "wrapper.h"

#include "oracle.h"

#include <stdio.h>
rust::String optimize(rust::String circ_string, int n_qubits,
                      rust::String cost_func, rust::String ecc_path,
                      rust::String gateset, float timeout) {
  return optimize_(std::string(circ_string), n_qubits, std::string(cost_func),
                   std::string(ecc_path), std::string(gateset), timeout);
}