// harris_v1.cuh
#pragma once

// GPU Harris dummy: dst = k * src
void harrisGPU_v1(const float* h_img, float* h_dst,
                  int width, int height, float k = 0.04f);

                  