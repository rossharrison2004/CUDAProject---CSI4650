#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <cmath>

using T = float;

// =============================
// Error Macro
// =============================
#define CHECK_CUDA(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error %s:%d: %s\n", \
                __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
} while(0)


// =============================
// CPU Reference Reduction
// =============================
T cpu_reduce(const T* data, int n) {
    double sum = 0.0;
    for (int i = 0; i < n; ++i)
        sum += data[i];
    return (T)sum;
}

bool nearly_equal(T a, T b, float tol = 1e-5f) {
    return fabs(a - b) <= tol * fmax(fabs(a), fabs(b)) + 1e-6f;
}



// ============================================================
// HARRIS KERNEL 3 – Sequential Addressing (Shared Memory Tree)
// ============================================================
__global__ void reduce_sequential(const T* __restrict__ g_idata,
                                  T* __restrict__ g_odata,
                                  int n) {
    extern __shared__ T sdata[];

    unsigned int tid = threadIdx.x;
    unsigned int i   = blockIdx.x * blockDim.x * 2 + tid;

    T mySum = 0;
    if (i < n)               mySum += g_idata[i];
    if (i + blockDim.x < n)  mySum += g_idata[i + blockDim.x];

    sdata[tid] = mySum;
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s)
            sdata[tid] += sdata[tid + s];
        __syncthreads();
    }

    if (tid == 0)
        g_odata[blockIdx.x] = sdata[0];
}



// ==================================================================
// HARRIS KERNEL 5 – Unroll Last Warp (Avoid Final Syncthreads)
// ==================================================================
__device__ void warp_reduce_volatile(volatile T* sdata, unsigned int tid) {
    sdata[tid] += sdata[tid + 32];
    sdata[tid] += sdata[tid + 16];
    sdata[tid] += sdata[tid + 8];
    sdata[tid] += sdata[tid + 4];
    sdata[tid] += sdata[tid + 2];
    sdata[tid] += sdata[tid + 1];
}

__global__ void reduce_unroll_last_warp(const T* __restrict__ g_idata,
                                        T* __restrict__ g_odata,
                                        int n) {
    extern __shared__ T sdata[];

    unsigned int tid = threadIdx.x;
    unsigned int i   = blockIdx.x * blockDim.x * 2 + tid;

    T mySum = 0;
    if (i < n)               mySum += g_idata[i];
    if (i + blockDim.x < n)  mySum += g_idata[i + blockDim.x];

    sdata[tid] = mySum;
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 32; s >>= 1) {
        if (tid < s)
            sdata[tid] += sdata[tid + s];
        __syncthreads();
    }

    if (tid < 32) {
        volatile T* vmem = sdata;
        warp_reduce_volatile(vmem, tid);
    }

    if (tid == 0)
        g_odata[blockIdx.x] = sdata[0];
}



// ============================================================
// LUITJENS – WARP SHUFFLE + BLOCK REDUCTION
// ============================================================
__inline__ __device__
T warp_reduce_shfl(T val) {
    unsigned int mask = 0xffffffff;
    for (int offset = 16; offset > 0; offset >>= 1)
        val += __shfl_down_sync(mask, val, offset);
    return val;
}

__inline__ __device__
T block_reduce_shfl(T val) {
    static __shared__ T shared[32];

    int lane = threadIdx.x & 31;
    int wid  = threadIdx.x >> 5;

    val = warp_reduce_shfl(val);

    if (lane == 0)
        shared[wid] = val;
    __syncthreads();

    val = (threadIdx.x < blockDim.x / 32) ? shared[lane] : 0.0f;

    if (wid == 0)
        val = warp_reduce_shfl(val);

    return val;
}



// ============================================================
// LUITJENS – BLOCK REDUCE + ATOMIC ADD (Single Pass)
// ============================================================
__global__
void reduce_block_atomic(const T* __restrict__ g_idata,
                         T* __restrict__ g_odata,
                         int n) {
    T sum = 0;

    for (int i = blockIdx.x * blockDim.x + threadIdx.x;
         i < n;
         i += blockDim.x * gridDim.x) {
        sum += g_idata[i];
    }

    sum = block_reduce_shfl(sum);

    if (threadIdx.x == 0)
        atomicAdd(g_odata, sum);
}



// ============================================================
// TIMING HELPER
// ============================================================
T run_harris_kernel(const char* name,
                    const T* d_in,
                    T* d_tmp,
                    int N,
                    int blockSize,
                    void (*kernel)(const T*, T*, int)) {

    int gridSize = (N + blockSize * 2 - 1) / (blockSize * 2);

    T* h_tmp = (T*)malloc(gridSize * sizeof(T));
    CHECK_CUDA(cudaMalloc(&d_tmp, gridSize * sizeof(T)));

    int smem = blockSize * sizeof(T);

    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    CHECK_CUDA(cudaEventRecord(start));
    kernel<<<gridSize, blockSize, smem>>>(d_in, d_tmp, N);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float ms;
    CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));

    CHECK_CUDA(cudaMemcpy(h_tmp, d_tmp, gridSize * sizeof(T), cudaMemcpyDeviceToHost));

    double finalSum = 0;
    for (int i = 0; i < gridSize; i++)
        finalSum += h_tmp[i];
    free(h_tmp);

    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));

    printf("%s: %.3f ms, result = %.6f\n", name, ms, (T)finalSum);

    return (T)finalSum;
}



// ============================================================
// MAIN
// ============================================================
int main() {
    const int N = 1 << 24;   // ~16 million
    const int blockSize = 256;
    const int gridSizeAtomic = 256;

    printf("Allocating %d floats...\n", N);

    T* h_in = (T*)malloc(N * sizeof(T));
    for (int i = 0; i < N; i++)
        h_in[i] = 1.0f;

    T cpuSum = cpu_reduce(h_in, N);
    printf("CPU reference = %.6f\n\n", cpuSum);

    T *d_in, *d_tmp, *d_out;
    CHECK_CUDA(cudaMalloc(&d_in,  N * sizeof(T)));
    CHECK_CUDA(cudaMalloc(&d_out, sizeof(T)));

    CHECK_CUDA(cudaMemcpy(d_in, h_in, N * sizeof(T),
                          cudaMemcpyHostToDevice));

    // ======================
    // Run Harris Kernel 3
    // ======================
    T result1 = run_harris_kernel(
        "Harris Kernel 3 (Sequential)",
        d_in, d_tmp, N, blockSize,
        reduce_sequential
    );

    // ======================
    // Run Harris Kernel 5
    // ======================
    T result2 = run_harris_kernel(
        "Harris Kernel 5 (Unroll Last Warp)",
        d_in, d_tmp, N, blockSize,
        reduce_unroll_last_warp
    );

    // ======================
    // Run Luitjens Block-Atomic
    // ======================
    CHECK_CUDA(cudaMemset(d_out, 0, sizeof(T)));

    cudaEvent_t s, e;
    CHECK_CUDA(cudaEventCreate(&s));
    CHECK_CUDA(cudaEventCreate(&e));

    CHECK_CUDA(cudaEventRecord(s));
    reduce_block_atomic<<<gridSizeAtomic, blockSize>>>(d_in, d_out, N);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaEventRecord(e));
    CHECK_CUDA(cudaEventSynchronize(e));

    float msAtomic;
    CHECK_CUDA(cudaEventElapsedTime(&msAtomic, s, e));

    T result3;
    CHECK_CUDA(cudaMemcpy(&result3, d_out, sizeof(T),
                          cudaMemcpyDeviceToHost));

    printf("Luitjens Block-Atomic: %.3f ms, result = %.6f\n",
           msAtomic, result3);

    printf("\n=========================\n");
    printf("Correctness Check:\n");
    printf("=========================\n");
    printf("Harris 3:   %s\n", nearly_equal(result1, cpuSum) ? "OK" : "FAIL");
    printf("Harris 5:   %s\n", nearly_equal(result2, cpuSum) ? "OK" : "FAIL");
    printf("Luitjens:   %s\n", nearly_equal(result3, cpuSum) ? "OK" : "FAIL");

    free(h_in);
    cudaFree(d_in);
    cudaFree(d_tmp);
    cudaFree(d_out);

    return 0;
}


