#ifndef HeterogeneousCore_MPICode_cudaOffload_h
#define HeterogeneousCore_MPICode_cudaOffload_h

#include <vector>

void cudaVectorAdd(std::vector<float>& vectorWorkers1, const std::vector<float>& vectorWorkers2);

#endif