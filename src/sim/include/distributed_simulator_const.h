#ifndef _QUARTZ_SIMULATOR_CONST_H_
#define _QUARTZ_SIMULATOR_CONST_H_

#define MAX_NUM_WORKERS 128
#define MAX_NUM_QUBITS 64
#define MAX_CPU_SV_INPUT 128
#define MAX_TENSOR_DIM 4
#define MAX_GATE_MATRIX_SIZE (2 * 8 * (1 << 6) * (1 << 6))
// Pre-assigned const flags
#define MAP_TO_FB_MEMORY 0xABCD0000
#define MAP_TO_ZC_MEMORY 0xABCE0000

// Preserved IDs for the mapper
enum PreservedIDs {
  InvalidID = 0,
  DataParallelism_GPU = 1,
  // DataParallelism_GPU_2D = 2,
  // DataParallelism_GPU_3D = 3,
  // DataParallelism_GPU_4D = 4,
  // DataParallelism_GPU_5D = 5,
  DataParallelism_CPU = 11,
  // DataParallelism_CPU_2D = 12,
  // DataParallelism_CPU_3D = 13,
  // DataParallelism_CPU_4D = 14,
  // DataParallelism_CPU_5D = 15,
};

enum DataType {
  DT_BOOLEAN = 40,
  DT_INT32 = 41,
  DT_INT64 = 42,
  DT_HALF = 43,
  DT_FLOAT = 44,
  DT_DOUBLE = 45,
  DT_FLOAT_COMPLEX = 46,
  DT_DOUBLE_COMPLEX = 47,
  DT_NONE = 49,
};

enum TaskIDs {
  TOP_LEVEL_TASK_ID,
  CUDA_INIT_TASK_ID,
  CPU_SV_INIT_TASK_ID,
  GPU_SV_INIT_TASK_ID,
  GATE_COMP_TASK_ID,
  SHUFFLE_TASK_ID,
  STORE_TASK_ID,
  NCCL_GETUNIQUEID_TASK_ID,
  // Custom tasks
  CUSTOM_GPU_TASK_ID_FIRST,
  CUSTOM_GPU_TASK_ID_1,
  CUSTOM_GPU_TASK_ID_2,
  CUSTOM_GPU_TASK_ID_3,
  CUSTOM_GPU_TASK_ID_4,
  CUSTOM_GPU_TASK_ID_5,
  CUSTOM_GPU_TASK_ID_6,
  CUSTOM_GPU_TASK_ID_7,
  CUSTOM_GPU_TASK_ID_8,
  CUSTOM_GPU_TASK_ID_LAST,
  CUSTOM_CPU_TASK_ID_FIRST,
  CUSTOM_CPU_TASK_ID_1,
  CUSTOM_CPU_TASK_ID_2,
  CUSTOM_CPU_TASK_ID_3,
  CUSTOM_CPU_TASK_ID_4,
  CUSTOM_CPU_TASK_ID_5,
  CUSTOM_CPU_TASK_ID_6,
  CUSTOM_CPU_TASK_ID_7,
  CUSTOM_CPU_TASK_ID_LAST,
  // Make sure PYTHON_TOP_LEVEL_TASK_ID is
  // consistent with python/main.cc
  PYTHON_TOP_LEVEL_TASK_ID = 11111,
};

enum FieldIDs {
  FID_DATA,
};

#endif // _QUARTZ_SIMULATOR_CONST_H_
