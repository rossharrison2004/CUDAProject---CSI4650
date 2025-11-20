// luitjens_atomic.cu
#include "utils/check_cuda.cuh"
__global__ void atomicExample(int *data) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    atomicAdd(&data[0], 1);
}


