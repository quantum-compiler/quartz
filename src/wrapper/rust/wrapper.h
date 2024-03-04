#include "rust/cxx.h"
rust::String optimize(rust::String circ_string, int n_qubits,
                      rust::String cost_func, rust::String ecc_path,
                      rust::String gateset, float timeout);