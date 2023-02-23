#ifndef _QUARTZ_SIMULATOR_CONST_H_
#define _QUARTZ_SIMULATOR_CONST_H_
#include <custatevec.h>

#define MAX_NUM_WORKERS 128
#define MAX_NUM_QUBITS 64
#define MAX_GATE_MATRIX_SIZE (2 * 8 * (1 << 6) * (1 << 6))

enum DataType {
  DT_BOOLEAN = 40,
  DT_INT32 = 41,
  DT_INT64 = 42,
  DT_HALF = 43,
  DT_FLOAT = 44,
  DT_DOUBLE = 45,
  DT_NONE = 49,
};

enum TaskIDs {
  TOP_LEVEL_TASK_ID,
  CUDA_INIT_TASK_ID,
  GATE_COMP_TASK_ID,
};

enum FieldIDs {
  FID_DATA,
};

#endif // _QUARTZ_SIMULATOR_CONST_H_
