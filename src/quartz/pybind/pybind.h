#pragma once

#include <pybind11/embed.h>
#include <pybind11/pybind11.h>

namespace quartz {

class PythonInterpreter {
public:
  std::vector<std::vector<int>>
  solve_ilp(const std::vector<std::vector<int>> &circuit_gate_qubits,
            const std::vector<bool> &circuit_gate_is_sparse,
            const std::vector<std::vector<int>> &out_gate, int num_qubits,
            int num_local_qubits, int num_iterations,
            bool print_solution = false);

private:
  pybind11::scoped_interpreter guard_;
  pybind11::function solve_ilp_;
};

} // namespace quartz
