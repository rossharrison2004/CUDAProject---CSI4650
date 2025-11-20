// cpu_baseline.cpp
#include "cpu_baseline.hpp"

// Simple dummy CPU baseline: dst = src * k
void harrisCPU(const float* src, float* dst, int width, int height, float k) {
    int total = width * height;
    for (int i = 0; i < total; ++i) {
        dst[i] = src[i] * k;
    }
}
