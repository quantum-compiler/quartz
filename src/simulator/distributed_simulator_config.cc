#include "simulator.h"

#include "sim/circuit.h"

using namespace sim;

DSConfig::DSConfig() {
  data_type = DT_DOUBLE;
  gpus_per_node = 0;
  cpus_per_node = 1;
  num_local_qubits = 0;
  num_all_qubits = 0;
  num_state_vectors_on_gpu = 2;
  // Parse input arguments
  {
    InputArgs const &command_args = HighLevelRuntime::get_input_args();
    char **argv = command_args.argv;
    int argc = command_args.argc;
    parse_args(argv, argc);
  }
  assert(num_local_qubits > 0);
  assert(num_all_qubits > 0);
  assert(gpus_per_node > 0);
  // Use Real::Machine::get_address_space_count() to obtain the number of nodes
  num_nodes = Realm::Machine::get_machine().get_address_space_count();
  Runtime *runtime = Runtime::get_runtime();
  lg_hlr = runtime;
  lg_ctx = Runtime::get_context();
  field_space = runtime->create_field_space(lg_ctx);
}

void DSConfig::parse_args(char **argv, int argc) {
  for (int i = 1; i < argc; i++) {
    if (!strcmp(argv[i], "-ll:gpu")) {
      gpus_per_node = atoi(argv[++i]);
      continue;
    }
    if (!strcmp(argv[i], "-ll:cpu")) {
      cpus_per_node = atoi(argv[++i]);
      continue;
    }
    if (!strcmp(argv[i], "--local-qubits")) {
      num_local_qubits = atoi(argv[++i]);
      continue;
    }
    if (!strcmp(argv[i], "--all-qubits")) {
      num_all_qubits = atoi(argv[++i]);
      continue;
    }
  }
}

