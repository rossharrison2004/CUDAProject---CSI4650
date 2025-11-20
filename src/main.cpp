// main.cpp
#include "cpu_baseline.hpp"
#include "harris_v1.cuh"

#include <iostream>
#include <vector>
#include <cmath>
#include <fstream>

int main() {
    const int width  = 1024;
    const int height = 768;
    const float k = 0.04f;

    std::vector<float> img(width * height);
    std::vector<float> dst_cpu(width * height);
    std::vector<float> dst_gpu(width * height);

    // Simple synthetic image: horizontal gradient 0..1
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            img[y * width + x] = static_cast<float>(x) / (width - 1);
        }
    }

    std::cout << "Running CPU baseline...\n";
    harrisCPU(img.data(), dst_cpu.data(), width, height, k);

    std::cout << "Running GPU v1...\n";
    harrisGPU_v1(img.data(), dst_gpu.data(), width, height, k);

    // Compare CPU and GPU
    float maxDiff = 0.0f;
    for (int i = 0; i < width * height; ++i) {
        float diff = std::fabs(dst_cpu[i] - dst_gpu[i]);
        if (diff > maxDiff) maxDiff = diff;
    }
    std::cout << "Max |CPU - GPU| = " << maxDiff << "\n";

    // Optional: write output image as PGM (no OpenCV)
    std::ofstream ofs("output_gpu.pgm", std::ios::binary);
    if (!ofs) {
        std::cerr << "Failed to open output_gpu.pgm for writing\n";
        return 1;
    }
    ofs << "P5\n" << width << " " << height << "\n255\n";
    for (int i = 0; i < width * height; ++i) {
        float v = dst_gpu[i];
        if (v < 0.0f) v = 0.0f;
        if (v > 1.0f) v = 1.0f;
        unsigned char byte = static_cast<unsigned char>(v * 255.0f + 0.5f);
        ofs.write(reinterpret_cast<char*>(&byte), 1);
    }
    ofs.close();
    std::cout << "Wrote output_gpu.pgm\n";

    return 0;
}

