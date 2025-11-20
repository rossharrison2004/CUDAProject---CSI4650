// cpu_baseline.hpp
#pragma once

// CPU baseline Harris dummy: dst = k * src
void harrisCPU(const float* src, float* dst, int width, int height, float k = 0.04f);

