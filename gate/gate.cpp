#include "gate.h"

Gate::Gate(GateType tp, int num_qubits, int num_parameters)
    : tp(tp), num_qubits(num_qubits), num_parameters(num_parameters) {}

int Gate::get_num_qubits() const {
  return num_qubits;
}

int Gate::get_num_parameters() const {
  return num_parameters;
}
