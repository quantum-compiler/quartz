#include <string.h>

#include "circuit.h"

using quartz::GateType;

int main(int argc, char *argv[]) {

  MPICHECK(MPI_Init(&argc, &argv));

  std::string circuit_file;
  unsigned nqubits;
  unsigned nlocal;
  int ndevice;
  bool use_ilp = false;
  for (int i = 1; i < argc; i++) {
    if (!strcmp(argv[i], "--import-circuit")) {
      circuit_file = std::string(argv[++i]);
      continue;
    }
    if (!strcmp(argv[i], "--n")) {
      nqubits = atoi(argv[++i]);
      continue;
    }
    if (!strcmp(argv[i], "--local")) {
      nlocal = atoi(argv[++i]);
      continue;
    }
    if (!strcmp(argv[i], "--device")) {
      ndevice = atoi(argv[++i]);
      continue;
    }
    if ((!strcmp(argv[i], "--use-ilp"))) {
      use_ilp = true;
      continue;
    }
  }

  quartz::init_python_interpreter();
  quartz::PythonInterpreter interpreter;
  quartz::Context ctx({GateType::input_qubit, GateType::input_param,
                       GateType::h, GateType::x, GateType::ry, GateType::u2,
                       GateType::u3, GateType::cx, GateType::cz, GateType::cp,
                       GateType::swap});
  auto seq = quartz::CircuitSeq::from_qasm_file(
      &ctx, std::string("/home/ubuntu/quartz-master/circuit/MQTBench_") +
                std::to_string(nqubits) + "q/" + circuit_file +
                "_indep_qiskit_" + std::to_string(nqubits) + ".qasm");
  sim::qcircuit::Circuit<double> circuit(nqubits, nlocal);
  circuit.compile(seq.get(), &ctx, &interpreter, use_ilp);
  // circuit.simulate(ndevice, true);

  MPICHECK(MPI_Finalize());

  return 0;
}
