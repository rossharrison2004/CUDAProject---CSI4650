__global__ void harrisKernel_v2(const float *d_img, float *d_dst,
                                int width, int height, float k) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = width * height;
    int stride = blockDim.x * gridDim.x;

    for (int i = idx; i < total; i += stride) {
        d_dst[i] = d_img[i] * k;
    }
}

void harrisGPU_v2(const cv::Mat &img, cv::Mat &dst, float k /*= 0.04f*/) {
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
    int gridSize  = (total + blockSize - 1) / blockSize;

    GPUTimer timer;
    timer.start();
    harrisKernel_v2<<<gridSize, blockSize>>>(d_img, d_dst, width, height, k);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());
    timer.stop();
    std::cout << "v2 kernel elapsed: " << timer.elapsedMillis() << " ms\n";

    dst.create(height, width, CV_32FC1);
    CHECK_CUDA(cudaMemcpy(dst.ptr<float>(), d_dst, size, cudaMemcpyDeviceToHost));

    CHECK_CUDA(cudaFree(d_img));
    CHECK_CUDA(cudaFree(d_dst));
}
