#include "wrapper.h"

#include <stdio.h>
rust::String optimize(rust::String circ_string, rust::String cost_func,
                      float timeout,
                      std::shared_ptr<SuperContext> super_context) {
  return optimize_(std::string(circ_string), std::string(cost_func), timeout,
                   super_context);
}

std::shared_ptr<SuperContext> get_context(rust::String gate_set, int n_qubits,
                                          rust::String ecc_path) {
  return get_context_(std::string(gate_set), n_qubits, std::string(ecc_path));
}