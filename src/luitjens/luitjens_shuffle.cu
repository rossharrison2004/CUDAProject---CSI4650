// luitjens_shuffle.cu
#include "utils/check_cuda.cuh"
__global__ void shuffleExample(int *data) {
    int lane = threadIdx.x % 32;
    int val = data[threadIdx.x];
    int shuffled = __shfl_xor_sync(0xFFFFFFFF, val, 1);
    data[threadIdx.x] = shuffled;
}


