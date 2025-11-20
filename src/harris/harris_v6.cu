## harris_v6
__global__ void harrisKernel_v6(const float *d_img, float *d_dst,
                                int total, float k) {
    int idx4   = blockIdx.x * blockDim.x + threadIdx.x;   // index in float4 units
    int stride4 = blockDim.x * gridDim.x;

    int total4 = total / 4;  // number of full float4s

    // Process in float4 chunks
    const float4 *src4 = reinterpret_cast<const float4*>(d_img);
    float4 *dst4       = reinterpret_cast<float4*>(d_dst);

    for (int i = idx4; i < total4; i += stride4) {
        float4 v = src4[i];
        v.x *= k;
        v.y *= k;
        v.z *= k;
        v.w *= k;
        dst4[i] = v;
    }

    // Handle remaining elements (tail) – let first warp do it
    if (idx4 == 0) {
        for (int i = total4 * 4; i < total; ++i) {
            d_dst[i] = d_img[i] * k;
        }
    }
}

void harrisGPU_v6(const cv::Mat &img, cv::Mat &dst, float k /*= 0.04f*/) {
    CV_Assert(img.type() == CV_32FC1);
    CV_Assert(img.isContinuous());

    int width  = img.cols;
    int height = img.rows;
    int total  = width * height;
    size_t size = static_cast<size_t>(total) * sizeof(float);

    float *d_img = nullptr;
    float *d_dst = nullptr;
    CHECK_CUDA(cudaMalloc(&d_img, size));
    CHECK_CUDA(cudaMalloc(&d_dst, size));

    CHECK_CUDA(cudaMemcpy(d_img, img.ptr<float>(), size, cudaMemcpyHostToDevice));

    int blockSize = 256;
    int gridSize  = ( (total / 4) + blockSize - 1 ) / blockSize;

    GPUTimer timer;
    timer.start();
    harrisKernel_v6<<<gridSize, blockSize>>>(d_img, d_dst, total, k);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());
    timer.stop();
    std::cout << "v6 kernel elapsed: " << timer.elapsedMillis() << " ms\n";

    dst.create(height, width, CV_32FC1);
    CHECK_CUDA(cudaMemcpy(dst.ptr<float>(), d_dst, size, cudaMemcpyDeviceToHost));

    CHECK_CUDA(cudaFree(d_img));
    CHECK_CUDA(cudaFree(d_dst));
}
