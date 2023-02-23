#include "distributed_simulator.h"
#include "cuda_helper.h"

using namespace sim;
using namespace Legion;

DistributedSimulator::DSHandler
DistributedSimulator::cuda_init_task(Task const *task,
                                     std::vector<PhysicalRegion> const &regions,
                                     Context ctx, Runtime *runtime) {
  DSConfig const *config = (DSConfig *)task->args;
  DSHandler handle;
  handle.workSpaceSize = (size_t)1 * 1024 * 1024 * 1024; // 1GB work space
  handle.num_local_qubits = config->num_local_qubits;
  printf("Num_local_qubits = %lld\n", handle.num_local_qubits);
  custatevecCreate(&handle.statevec);
  {
    // allocate memory for workspace
    Memory gpu_mem = Machine::MemoryQuery(Machine::get_machine())
                         .only_kind(Memory::GPU_FB_MEM)
                         .best_affinity_to(task->target_proc)
                         .first();
    Realm::Rect<1, coord_t> bounds(
        Realm::Point<1, coord_t>(0),
        Realm::Point<1, coord_t>(handle.workSpaceSize - 1));
    std::vector<size_t> field_sizes;
    field_sizes.push_back(sizeof(char));
    Realm::RegionInstance workspaceInst;
    Realm::RegionInstance::create_instance(workspaceInst, gpu_mem, bounds,
                                           field_sizes, 0,
                                           Realm::ProfilingRequestSet())
        .wait();
    handle.workSpace = workspaceInst.pointer_untyped(0, sizeof(char));
  }
  handle.ncclComm = nullptr;
  return handle;
}

void DistributedSimulator::sv_init_task(
    Task const *task, std::vector<PhysicalRegion> const &regions, Context ctx,
    Runtime *runtime) {
  // TODO: implement this function
}

void DistributedSimulator::sv_comp_task(
    Task const *task, std::vector<PhysicalRegion> const &regions, Context ctx,
    Runtime *runtime) {
  DSHandler const *handler = *((DSHandler **)task->local_args);
  GateInfo const *info = (GateInfo *)task->args;
  assert(handler->vecDataType == DT_FLOAT || handler->vecDataType == DT_DOUBLE);
  cudaDataType_t data_type = handler->vecDataType == DT_FLOAT ? CUDA_C_32F : CUDA_C_64F;
  custatevecComputeType_t compute_type = handler->vecDataType == DT_FLOAT ? CUSTATEVEC_COMPUTE_32F : CUSTATEVEC_COMPUTE_64F;
  GenericTensorAccessorW state_vector = helperGetGenericTensorAccessorWO(
      handler->vecDataType, regions[0], task->regions[0], FID_DATA, ctx, runtime);

  // TODO: get target & control qubit idx from current perm[]
  std::vector<int> targets;
  std::vector<int> controls;

  unsigned const nIndexBits = handler->num_local_qubits;
  unsigned const nTargets = info->num_targets;
  unsigned const nControls = info->num_controls;
  int const adjoint = 0;
  // TODO: check if targets should be ordered
  printf("Targets: [");
  for (int i = 0; i < info->num_targets; i++) {
    int idx = 0;
    while (info->permutation[idx] != info->target[i])
      idx++;
    targets.push_back(idx);
    printf("(%d, %d) ", info->target[i], idx);
  }
  printf("]\n");

  for (int i = 0; i < info->num_controls; i++) {
    int idx = 0;
    while (info->permutation[idx] != info->target[i])
      idx++;
    controls.push_back(idx);
  }

  // apply gate
  custatevecApplyMatrix(
      /* custatevecHandle_t */ handler->statevec,
      /* void* */ state_vector.get_void_ptr(),
      /* cudaDataType_t */ data_type,
      /* const uint32_t */ nIndexBits,
      /* const void* */ info->matrix_data,
      /* cudaDataType_t */ data_type,
      /* custatevecMatrixLayout_t */ CUSTATEVEC_MATRIX_LAYOUT_ROW,
      /* const int32_t */ adjoint,
      /* const int32_t* */ targets.data(),
      /* const uint32_t */ nTargets,
      /* const int32_t* */ controls.data(),
      /* const int32_t* */ nullptr,
      /* const uint32_t */ nControls,
      /* custatevecComputeType_t */ compute_type,
      /* void* */ handler->workSpace,
      /* size_t */ handler->workSpaceSize);
}
