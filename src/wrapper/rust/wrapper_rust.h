#include "oracle.h"
#include "rust/cxx.h"
rust::String optimize(rust::String circ_string, rust::String cost_func,
                      float timeout,
                      std::shared_ptr<SuperContext> super_context);

std::shared_ptr<SuperContext> get_context(rust::String gate_set, int n_qubits,
                                          rust::String ecc_path);