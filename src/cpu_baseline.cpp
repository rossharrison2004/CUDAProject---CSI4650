//code here
#include <opencv2/opencv.hpp>
#include <vector>
#include <cmath>
#include "utils.h"

void harrisCPU(const cv::Mat &img, cv::Mat &dst, float k = 0.04, int window_size = 3) {
    cv::Mat Ix, Iy;
    cv::Sobel(img, Ix, CV_32F, 1, 0);
    cv::Sobel(img, Iy, CV_32F, 0, 1);

    cv::Mat Ix2 = Ix.mul(Ix);
    cv::Mat Iy2 = Iy.mul(Iy);
    cv::Mat Ixy = Ix.mul(Iy);

    dst = cv::Mat::zeros(img.size(), CV_32F);

    int offset = window_size / 2;
    for (int y = offset; y < img.rows - offset; y++) {
        for (int x = offset; x < img.cols - offset; x++) {
            float sumIx2 = 0, sumIy2 = 0, sumIxy = 0;
            for (int dy = -offset; dy <= offset; dy++) {
                for (int dx = -offset; dx <= offset; dx++) {
                    sumIx2 += Ix2.at<float>(y+dy, x+dx);
                    sumIy2 += Iy2.at<float>(y+dy, x+dx);
                    sumIxy += Ixy.at<float>(y+dy, x+dx);
                }
            }
            float det = sumIx2*sumIy2 - sumIxy*sumIxy;
            float trace = sumIx2 + sumIy2;
            dst.at<float>(y,x) = det - k*trace*trace;
        }
    }
}
