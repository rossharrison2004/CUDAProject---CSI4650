## harris_v5
__global__ void harrisKernel_v5(cudaTextureObject_t texObj, float *d_dst,
                                int width, int height, float k) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        float v = tex2D<float>(texObj, static_cast<float>(x) + 0.5f,
                                         static_cast<float>(y) + 0.5f);
        d_dst[y * width + x] = v * k;
    }
}

void harrisGPU_v5(const cv::Mat &img, cv::Mat &dst, float k /*= 0.04f*/) {
    CV_Assert(img.type() == CV_32FC1);
    CV_Assert(img.isContinuous());

    int width  = img.cols;
    int height = img.rows;
    size_t size = static_cast<size_t>(width) * height * sizeof(float);

    // Allocate CUDA array for texture
    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float>();
    cudaArray_t d_array;
    CHECK_CUDA(cudaMallocArray(&d_array, &channelDesc, width, height));

    CHECK_CUDA(cudaMemcpy2DToArray(
        d_array, 0, 0,
        img.ptr<float>(), width * sizeof(float),
        width * sizeof(float), height,
        cudaMemcpyHostToDevice));

    // Create texture object
    cudaResourceDesc resDesc{};
    resDesc.resType = cudaResourceTypeArray;
    resDesc.res.array.array = d_array;

    cudaTextureDesc texDesc{};
    texDesc.addressMode[0]   = cudaAddressModeClamp;
    texDesc.addressMode[1]   = cudaAddressModeClamp;
    texDesc.filterMode       = cudaFilterModePoint;
    texDesc.readMode         = cudaReadModeElementType;
    texDesc.normalizedCoords = 0;

    cudaTextureObject_t texObj = 0;
    CHECK_CUDA(cudaCreateTextureObject(&texObj, &resDesc, &texDesc, nullptr));

    float *d_dst = nullptr;
    CHECK_CUDA(cudaMalloc(&d_dst, size));

    dim3 block(16, 16);
    dim3 grid((width  + block.x - 1) / block.x,
              (height + block.y - 1) / block.y);

    GPUTimer timer;
    timer.start();
    harrisKernel_v5<<<grid, block>>>(texObj, d_dst, width, height, k);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());
    timer.stop();
    std::cout << "v5 kernel elapsed: " << timer.elapsedMillis() << " ms\n";

    dst.create(height, width, CV_32FC1);
    CHECK_CUDA(cudaMemcpy(dst.ptr<float>(), d_dst, size, cudaMemcpyDeviceToHost));

    CHECK_CUDA(cudaDestroyTextureObject(texObj));
    CHECK_CUDA(cudaFreeArray(d_array));
    CHECK_CUDA(cudaFree(d_dst));
}
