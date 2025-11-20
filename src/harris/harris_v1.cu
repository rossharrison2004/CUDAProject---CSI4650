// harris_v1.cu
#include "utils/check_cuda.cuh"
#include "utils/timer.cuh"

#include <cuda_runtime.h>
#include <iostream>
#include "harris_v1.cuh"

__global__ void harrisKernel_v1(const float* d_img, float* d_dst,
                                int width, int height, float k) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        int idx = y * width + x;
        d_dst[idx] = d_img[idx] * k;  // dummy Harris op
    }
}

void harrisGPU_v1(const float* h_img, float* h_dst,
                  int width, int height, float k) {
    size_t size = static_cast<size_t>(width) * height * sizeof(float);

    float *d_img = nullptr;
    float *d_dst = nullptr;
    CHECK_CUDA(cudaMalloc(&d_img, size));
    CHECK_CUDA(cudaMalloc(&d_dst, size));

    CHECK_CUDA(cudaMemcpy(d_img, h_img, size, cudaMemcpyHostToDevice));

    dim3 block(16, 16);
    dim3 grid((width  + block.x - 1) / block.x,
              (height + block.y - 1) / block.y);

    GPUTimer timer;
    timer.start();
    harrisKernel_v1<<<grid, block>>>(d_img, d_dst, width, height, k);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());
    timer.stop();
    std::cout << "v1 kernel elapsed: " << timer.elapsedMillis() << " ms\n";

    CHECK_CUDA(cudaMemcpy(h_dst, d_dst, size, cudaMemcpyDeviceToHost));

    CHECK_CUDA(cudaFree(d_img));
    CHECK_CUDA(cudaFree(d_dst));
}
