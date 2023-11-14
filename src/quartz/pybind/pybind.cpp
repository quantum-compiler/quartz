#include "pybind.h"

#include <cstdlib>
#include <filesystem>
#include <fstream>
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
#ifdef WIN32
    new_python_path += ";";
#else
    new_python_path += ":";
#endif
  }
  new_python_path += python_module_path.string();
#ifdef WIN32
  system("python -c \"import sys;"
         "print(sys.prefix)\" > tmp.txt");
  std::ifstream fin("tmp.txt");
  std::string python_home_path;
  std::getline(fin, python_home_path);
  fin.close();
  std::string python_home_env = "PYTHONHOME=" + python_home_path;
  _putenv(python_home_env.c_str());
  std::string python_path = "PYTHONPATH=" + python_home_path +
                            "\\Lib\\site-packages;" + new_python_path;
  _putenv(python_path.c_str());
#else
  setenv("PYTHONPATH", new_python_path.c_str(), 1);
#endif
}

std::vector<std::vector<int>> PythonInterpreter::solve_ilp(
    const std::vector<std::vector<int>> &circuit_gate_qubits,
    const std::vector<int> &circuit_gate_executable_type,
    const std::vector<std::vector<int>> &out_gate, int num_qubits,
    int num_local_qubits, int num_iterations, bool print_solution) {
  if (!solve_ilp_) {
    solve_ilp_ = py::reinterpret_steal<py::function>(
        py::module::import("simulator.ilp").attr("solve_ilp"));
  }
  auto result =
      solve_ilp_(circuit_gate_qubits, circuit_gate_executable_type, out_gate,
                 num_qubits, num_local_qubits, num_iterations, print_solution);
  return result.cast<std::vector<std::vector<int>>>();
}

std::vector<std::vector<int>> PythonInterpreter::solve_global_ilp(
    const std::vector<std::vector<int>> &circuit_gate_qubits,
    const std::vector<int> &circuit_gate_executable_type,
    const std::vector<std::vector<int>> &out_gate, int num_qubits,
    int num_local_qubits, int num_global_qubits, double global_cost_factor,
    int num_iterations, bool print_solution) {
  if (!solve_global_ilp_) {
    solve_global_ilp_ = py::reinterpret_steal<py::function>(
        py::module::import("simulator.ilp").attr("solve_global_ilp"));
  }
  auto result = solve_global_ilp_(
      circuit_gate_qubits, circuit_gate_executable_type, out_gate, num_qubits,
      num_local_qubits, num_global_qubits, global_cost_factor, num_iterations,
      print_solution);
  return result.cast<std::vector<std::vector<int>>>();
}
}  // namespace quartz
