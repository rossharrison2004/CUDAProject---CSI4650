#include "cpu_baseline.cpp"
#include <iostream>

int main() {
    cv::Mat img = cv::imread("input.png", cv::IMREAD_GRAYSCALE);
    img.convertTo(img, CV_32F, 1.0/255.0);

    cv::Mat dst_cpu, dst_gpu;

    std::cout << "Running CPU baseline...\n";
    harrisCPU(img, dst_cpu);

    std::cout << "Running GPU v1...\n";
    harrisGPU_v1(img, dst_gpu);

    cv::imwrite("output_cpu.png", dst_cpu * 255);
    cv::imwrite("output_gpu.png", dst_gpu * 255);

    return 0;
}


