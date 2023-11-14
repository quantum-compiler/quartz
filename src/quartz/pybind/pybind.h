#pragma once

#include <pybind11/embed.h>
#include <pybind11/pybind11.h>

namespace quartz {

// This function needed to be called exactly once before any PythonInterpreter
// object is constructed.
void init_python_interpreter();

// There can only be one alive PythonInterpreter object at any time.
class PythonInterpreter {
 public:
  std::vector<std::vector<int>>
  solve_ilp(const std::vector<std::vector<int>> &circuit_gate_qubits,
            const std::vector<int> &circuit_gate_executable_type,
            const std::vector<std::vector<int>> &out_gate, int num_qubits,
            int num_local_qubits, int num_iterations,
            bool print_solution = false);

  std::vector<std::vector<int>>
  solve_global_ilp(const std::vector<std::vector<int>> &circuit_gate_qubits,
                   const std::vector<int> &circuit_gate_executable_type,
                   const std::vector<std::vector<int>> &out_gate,
                   int num_qubits, int num_local_qubits, int num_global_qubits,
                   double global_cost_factor, int num_iterations,
                   bool print_solution = false);

 private:
  pybind11::scoped_interpreter guard_;
  pybind11::function solve_ilp_;
  pybind11::function solve_global_ilp_;
};

}  // namespace quartz
