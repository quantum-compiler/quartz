#include "pybind.h"
#include <pybind11/embed.h>
#include <pybind11/stl.h>

namespace quartz {

namespace py = pybind11;

std::vector<std::vector<int>> PythonInterpreter::solve_ilp(
    const std::vector<std::vector<int>> &circuit_gate_qubits,
    const std::vector<bool> &circuit_gate_is_sparse,
    const std::vector<std::vector<int>> &out_gate, int num_qubits,
    int num_local_qubits, int num_iterations, bool print_solution) {
  if (!solve_ilp_) {
    solve_ilp_ = py::reinterpret_steal<py::function>(
        py::module::import("src.python.simulator.ilp").attr("solve_ilp"));
  }
  auto result =
      solve_ilp_(circuit_gate_qubits, circuit_gate_is_sparse, out_gate,
                 num_qubits, num_local_qubits, num_iterations, print_solution);
  return result.cast<std::vector<std::vector<int>>>();
}
} // namespace quartz
