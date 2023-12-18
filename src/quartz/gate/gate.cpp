#include "gate.h"

namespace quartz {
Gate::Gate(GateType tp, int num_qubits, int num_parameters)
    : tp(tp), num_qubits(num_qubits), num_parameters(num_parameters) {}

MatrixBase *Gate::get_matrix() {
  // Default: no matrix (for parameter gates).
  std::cerr << "Gate::get_matrix() called." << std::endl;
  if (is_quantum_gate()) {
    std::cerr << "Please double-check if you instantiated a \"Gate\" object."
              << std::endl;
  } else {
    std::cerr << "This object is not a quantum gate." << std::endl;
  }
  return nullptr;
}

MatrixBase *Gate::get_matrix(const std::vector<ParamType> &params) {
  // Default: no parameters.
  return get_matrix();
}

ParamType Gate::compute(const std::vector<ParamType> &input_params) {
  // Default: do no computation (for quantum gates).
  std::cerr
      << "Gate::compute(const std::vector<ParamType> &input_params) called."
      << std::endl;
  if (is_quantum_gate()) {
    std::cerr << "This object is not an arithmetic computation \"gate\"."
              << std::endl;
  } else {
    std::cerr << "Please double-check if you instantiated a \"Gate\" object."
              << std::endl;
  }
  return 0;
}

bool Gate::is_commutative() const { return false; }

bool Gate::is_symmetric() const { return false; }

bool Gate::is_sparse() const { return false; }

bool Gate::is_diagonal() const { return false; }

int Gate::get_num_control_qubits() const { return 0; }

std::vector<bool> Gate::get_control_state() const {
  return std::vector<bool>(get_num_control_qubits(), true);
}

int Gate::get_num_qubits() const { return num_qubits; }

int Gate::get_num_parameters() const { return num_parameters; }

bool Gate::is_parameter_gate() const {
  return num_qubits == 0 && tp != GateType::input_param &&
         tp != GateType::input_qubit;
}

bool Gate::is_quantum_gate() const { return num_qubits > 0; }

bool Gate::is_parametrized_gate() const {
  return num_qubits > 0 && num_parameters > 0;
}

bool Gate::is_toffoli_gate() const {
  return num_qubits == 3 && get_num_control_qubits() == 2;
}

}  // namespace quartz
