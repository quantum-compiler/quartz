#pragma once

#include "../math/matrix.h"

#include <iostream>
#include <memory>

class Gate {
public:
  virtual std::unique_ptr<MatrixBase> to_matrix() const {
    std::cerr << "Gate::to_matrix() called." << std::endl;
    return std::make_unique<MatrixBase>();
  }
  Gate(int num_qubits, int num_parameters);
  virtual bool is_parameter_gate() const = 0;
  virtual bool is_quantum_gate() const = 0;
  int get_num_qubits() const;
  int get_num_parameters() const;
  int num_qubits, num_parameters;
};
