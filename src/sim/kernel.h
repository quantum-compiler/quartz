#pragma once

#include "simgate.h"
using namespace sim;

void copyGatesToSymbol(KernelGate* hostGates, int numGates, cudaStream_t& stream, int gpuID);
void initControlIdx(int n_devices, cudaStream_t* stream);
// call cudaSetDevice() before this function
void ApplyGatesSHM(int gridDim, qComplex* deviceStateVec, unsigned int* threadBias, int numLocalQubits, int numGates, unsigned int blockHot, unsigned int enumerate, cudaStream_t& stream, int gpuID);