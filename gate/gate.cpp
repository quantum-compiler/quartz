#include "gate.h"

Gate::Gate(int _num_qubits, int _num_parameters)
: num_qubits(_num_qubits), num_parameters(_num_parameters)
{}

int Gate::get_num_qubits() const
{
  return num_qubits;
}

int Gate::get_num_parameters() const
{
  return num_parameters;
}
