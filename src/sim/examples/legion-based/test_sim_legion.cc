#include "distributed_simulator.h"

using namespace sim;
using namespace Legion;
// declare Legion names
using Legion::ArgumentMap;
using Legion::Context;
using Legion::coord_t;
using Legion::Domain;
using Legion::FutureMap;
using Legion::IndexLauncher;
using Legion::InlineLauncher;
using Legion::PhysicalRegion;
using Legion::Predicate;
using Legion::Rect;
using Legion::RegionRequirement;
using Legion::Runtime;
using Legion::Task;
using Legion::TaskArgument;
using Legion::TaskLauncher;

using quartz::GateType;

void sim::top_level_task(
    Task const *task, std::vector<PhysicalRegion> const &regions, Context ctx,
    Runtime *runtime) {
  DSConfig config;
  bool use_ilp = true;
  // quartz::init_python_interpreter();
  quartz::PythonInterpreter interpreter;
  quartz::Context qtz({GateType::input_qubit, GateType::input_param,
                       GateType::h, GateType::x, GateType::ry, GateType::u2,
                       GateType::u3, GateType::cx, GateType::cz, GateType::cp,
                       GateType::p, GateType::z, GateType::swap});
  // auto seq = quartz::CircuitSeq::from_qasm_file(
  //     &qtz, std::string("/home/ubuntu/quartz-master/circuit/MQTBench_") +
  //               std::to_string(config.num_all_qubits) + "q/qft"
  //               "_indep_qiskit_" + std::to_string(config.num_all_qubits) +
  //               ".qasm");
  auto seq = quartz::CircuitSeq::from_qasm_file(
      &qtz, std::string("/home/ubuntu/quartz-master/circuit/MQTBench_28") +
                "q/dj"
                "_indep_qiskit_28" +
                ".qasm");
  sim::qcircuit::Circuit<double> circuit(config.num_all_qubits,
                                         config.num_local_qubits, (1 << (config.num_all_qubits - config.num_local_qubits)), 0, 1);
  circuit.compile(seq.get(), &qtz, &interpreter, use_ilp);
  DistributedSimulator simulator(config, circuit);
  simulator.run();
}

DSConfig::DSConfig() {
  state_vec_data_type = DT_DOUBLE_COMPLEX;
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

int main(int argc, char **argv) {
  // This needs to be set, otherwise NCCL will try to use group kernel launches,
  // which are not compatible with the Realm CUDA hijack.
  setenv("NCCL_LAUNCH_MODE", "PARALLEL", true);
  setenv("PYTHONPATH", "/home/ubuntu/.local/lib/python3.8/site-packages", true /*overwrite*/);
  quartz::init_python_interpreter();

  char *python_path = getenv("PYTHONPATH");
  printf("python path: %s\n", python_path);

  Runtime::set_top_level_task_id(TOP_LEVEL_TASK_ID);
  {
    TaskVariantRegistrar registrar(TOP_LEVEL_TASK_ID, "top_level");
    registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
    registrar.set_replicable();
    Runtime::preregister_task_variant<sim::top_level_task>(
        registrar, "top_level");
  }
  {
    TaskVariantRegistrar registrar(CUDA_INIT_TASK_ID, "CUDA Init");
    registrar.add_constraint(ProcessorConstraint(Processor::TOC_PROC));
    registrar.set_leaf();
    Runtime::preregister_task_variant<DistributedSimulator::DSHandler, DistributedSimulator::cuda_init_task>(
        registrar, "CUDA_INIT_TASK");
  }
  {
    TaskVariantRegistrar registrar(GATE_COMP_TASK_ID, "Gate Compute");
    registrar.add_constraint(ProcessorConstraint(Processor::TOC_PROC));
    registrar.set_leaf();
    Runtime::preregister_task_variant<DistributedSimulator::sv_comp_task>(
        registrar, "Gate Compute Task");
  }
  {
    TaskVariantRegistrar registrar(SHUFFLE_TASK_ID, "Shuffle");
    registrar.add_constraint(ProcessorConstraint(Processor::TOC_PROC));
    registrar.set_leaf();
    Runtime::preregister_task_variant<DistributedSimulator::shuffle_task>(
        registrar, "Shuffle Task");
  }
  {
    TaskVariantRegistrar registrar(STORE_TASK_ID, "Store");
    registrar.add_constraint(ProcessorConstraint(Processor::TOC_PROC));
    registrar.set_leaf();
    Runtime::preregister_task_variant<DistributedSimulator::store_task>(
        registrar, "Store Task");
  }
  {
    TaskVariantRegistrar registrar(GPU_SV_INIT_TASK_ID, "GPU SV Init");
    registrar.add_constraint(ProcessorConstraint(Processor::TOC_PROC));
    registrar.set_leaf();
    Runtime::preregister_task_variant<DistributedSimulator::sv_init_task>(
        registrar, "GPU SV Init Task");
  }
  {
    TaskVariantRegistrar registrar(CPU_SV_INIT_TASK_ID, "CPU SV Init");
    registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
    registrar.set_leaf();
    Runtime::preregister_task_variant<DistributedSimulator::sv_init_task>(
        registrar, "CPU SV Init Task");
  }

  Runtime::add_registration_callback(FFMapper::update_mappers);
  return Runtime::start(argc, argv);
}
