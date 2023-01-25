#include "pybind.h"

#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <pybind11/embed.h>
#include <pybind11/stl.h>

namespace quartz {

namespace py = pybind11;

void init_python_interpreter() {
  std::filesystem::path this_file_path(__FILE__);
  auto python_module_path =
      this_file_path.parent_path().parent_path().parent_path().append("python");
  char *current_python_path = getenv("PYTHONPATH");
  std::string new_python_path;
  if (current_python_path) {
    new_python_path = current_python_path;
    new_python_path += ":";
  }
  new_python_path += python_module_path;
  setenv("PYTHONPATH", new_python_path.c_str(), 1);
}

std::vector<std::vector<int>> PythonInterpreter::solve_ilp(
    const std::vector<std::vector<int>> &circuit_gate_qubits,
    const std::vector<bool> &circuit_gate_is_sparse,
    const std::vector<std::vector<int>> &out_gate, int num_qubits,
    int num_local_qubits, int num_iterations, bool print_solution) {
  if (!solve_ilp_) {
    solve_ilp_ = py::reinterpret_steal<py::function>(
        py::module::import("simulator.ilp").attr("solve_ilp"));
  }
  auto result =
      solve_ilp_(circuit_gate_qubits, circuit_gate_is_sparse, out_gate,
                 num_qubits, num_local_qubits, num_iterations, print_solution);
  return result.cast<std::vector<std::vector<int>>>();
}
} // namespace quartz
