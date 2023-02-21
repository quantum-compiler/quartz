#ifndef _CONST_H_
#define _CONST_H_

#include "mpi.h"
#include "nccl.h"
#include "helper.h" // HANDLE_ERROR, HANDLE_CUDA_ERROR
#include <cuComplex.h>        // cuDoubleComplex
#include <cuda_runtime_api.h> // cudaMalloc, cudaMemcpy, etc.
#include <custatevec.h>       // custatevecApplyMatrix

// form HyQuas
#ifdef USE_FLOAT
typedef float qreal;
typedef long long qindex;
typedef cuFloatComplex qComplex;
#define make_qComplex make_cuFloatComplex
#define MPI_Complex MPI_C_COMPLEX
#define NCCL_FLOAT_TYPE ncclFloat
#else
typedef double qreal;
typedef long long qindex;
typedef cuDoubleComplex qComplex;
#define make_qComplex make_cuDoubleComplex
#define MPI_Complex MPI_C_DOUBLE_COMPLEX
#define NCCL_FLOAT_TYPE ncclDouble
#endif

#define SHARED_MEM_SIZE 10
#define MAX_GATE 600
#define THREAD_DEP 7

enum class KernelGateType {
    CCX, CNOT, CY, CZ, CRX, CRY, CU1, CRZ, U1, U2, U3, H, X, Y, Z, S, SDG, T, TDG, RX, RY, RZ, TOTAL, ID, GII, GZZ, GOC, GCC 
};

enum SimGateType {
  CONTROL,
  NORMAL,
  FUSED,
  SHUFFLE,
};

enum CompileMode {
  HEUR,
  ILP,
};

#endif // _CONST_H_
