## harris_v3
template<int BLOCK_W, int BLOCK_H>
__global__ void harrisKernel_v3(const float *d_img, float *d_dst,
                                int width, int height, float k) {
    __shared__ float tile[BLOCK_H][BLOCK_W];

    int x = blockIdx.x * BLOCK_W + threadIdx.x;
    int y = blockIdx.y * BLOCK_H + threadIdx.y;

    // Load from global to shared if in range
    if (x < width && y < height) {
        tile[threadIdx.y][threadIdx.x] = d_img[y * width + x];
    } else {
        tile[threadIdx.y][threadIdx.x] = 0.0f;
    }
    __syncthreads();

    // Process and write back
    if (x < width && y < height) {
        float v = tile[threadIdx.y][threadIdx.x] * k;
        d_dst[y * width + x] = v;
    }
}

void harrisGPU_v3(const cv::Mat &img, cv::Mat &dst, float k /*= 0.04f*/) {
    CV_Assert(img.type() == CV_32FC1);
    CV_Assert(img.isContinuous());

    int width  = img.cols;
    int height = img.rows;
    size_t size = static_cast<size_t>(width) * height * sizeof(float);

    float *d_img = nullptr;
    float *d_dst = nullptr;
    CHECK_CUDA(cudaMalloc(&d_img, size));
    CHECK_CUDA(cudaMalloc(&d_dst, size));

    CHECK_CUDA(cudaMemcpy(d_img, img.ptr<float>(), size, cudaMemcpyHostToDevice));

    constexpr int BW = 16;
    constexpr int BH = 16;
    dim3 block(BW, BH);
    dim3 grid((width  + BW - 1) / BW,
              (height + BH - 1) / BH);

    GPUTimer timer;
    timer.start();
    harrisKernel_v3<BW, BH><<<grid, block>>>(d_img, d_dst, width, height, k);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());
    timer.stop();
    std::cout << "v3 kernel elapsed: " << timer.elapsedMillis() << " ms\n";

    dst.create(height, width, CV_32FC1);
    CHECK_CUDA(cudaMemcpy(dst.ptr<float>(), d_dst, size, cudaMemcpyDeviceToHost));

    CHECK_CUDA(cudaFree(d_img));
    CHECK_CUDA(cudaFree(d_dst));
}
