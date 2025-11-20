// harris_v1.cu
#include "utils/check_cuda.cuh"
#include "utils/timer.cuh"
#include <cuda_runtime.h>
#include <opencv2/opencv.hpp>

__global__ void harrisKernel(float *d_img, float *d_dst, int width, int height, float k) {
    // simple placeholder kernel
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        d_dst[y * width + x] = d_img[y * width + x] * k; // dummy operation
    }
}

void harrisGPU_v1(const cv::Mat &img, cv::Mat &dst, float k = 0.04f) {
    int width = img.cols;
    int height = img.rows;
    size_t size = width * height * sizeof(float);

    float *d_img, *d_dst;
    CHECK_CUDA(cudaMalloc(&d_img, size));
    CHECK_CUDA(cudaMalloc(&d_dst, size));

    CHECK_CUDA(cudaMemcpy(d_img, img.ptr<float>(), size, cudaMemcpyHostToDevice));

    dim3 block(16, 16);
    dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);

    GPUTimer timer;
    timer.start();
    harrisKernel<<<grid, block>>>(d_img, d_dst, width, height, k);
    CHECK_CUDA(cudaDeviceSynchronize());
    timer.stop();
    std::cout << "v1 kernel elapsed: " << timer.elapsedMillis() << " ms\n";

    dst.create(height, width, CV_32F);
    CHECK_CUDA(cudaMemcpy(dst.ptr<float>(), d_dst, size, cudaMemcpyDeviceToHost));

    CHECK_CUDA(cudaFree(d_img));
    CHECK_CUDA(cudaFree(d_dst));
}


