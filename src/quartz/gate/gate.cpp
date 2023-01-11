#include "gate.h"

namespace quartz {
Gate::Gate(GateType tp, int num_qubits, int num_parameters)
    : tp(tp), num_qubits(num_qubits), num_parameters(num_parameters) {}

MatrixBase *Gate::get_matrix() {
  // Default: no matrix (for parameter gates).
  return nullptr;
}

MatrixBase *Gate::get_matrix(const std::vector<ParamType> &params) {
  // Default: no parameters.
  return get_matrix();
}

ParamType Gate::compute(const std::vector<ParamType> &input_params) {
  // Default: do no computation (for quantum gates).
  return 0;
}

bool Gate::is_commutative() const { return false; }

bool Gate::is_symmetric() const { return false; }

bool Gate::is_sparse() const { return false; }

int Gate::get_num_control_qubits() const { return 0; }

int Gate::get_num_qubits() const { return num_qubits; }

int Gate::get_num_parameters() const { return num_parameters; }

bool Gate::is_parameter_gate() const {
  // Only arithmetic computation gates are count
  return num_qubits == 0 && tp != GateType::input_param &&
         tp != GateType::input_qubit;
}

bool Gate::is_quantum_gate() const { return num_qubits > 0; }

bool Gate::is_parametrized_gate() const {
  return num_qubits > 0 && num_parameters > 0;
}

bool Gate::is_toffoli_gate() const {
  // TODO: add other toffoli gates
  return tp == GateType::ccz;
}

} // namespace quartz
