#pragma once
#include <cuda_runtime.h>
#include <iostream>

class GPUTimer {
private:
    cudaEvent_t startEvent, stopEvent;
public:
    GPUTimer() {
        cudaEventCreate(&startEvent);
        cudaEventCreate(&stopEvent);
    }

    ~GPUTimer() {
        cudaEventDestroy(startEvent);
        cudaEventDestroy(stopEvent);
    }

    void start() { cudaEventRecord(startEvent, 0); }
    void stop() { cudaEventRecord(stopEvent, 0); cudaEventSynchronize(stopEvent); }

    float elapsedMillis() {
        float ms = 0;
        cudaEventElapsedTime(&ms, startEvent, stopEvent);
        return ms;
    }
};


