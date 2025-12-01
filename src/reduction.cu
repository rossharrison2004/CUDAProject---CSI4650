#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <algorithm>
#include <cuda_runtime.h>

using T = float;

// ===========================================================
// Error checking helper
// ===========================================================
#define CHECK_CUDA(call) do {                                      \
    cudaError_t err = call;                                        \
    if (err != cudaSuccess) {                                      \
        fprintf(stderr, "CUDA error %s:%d: %s\n",                  \
                __FILE__, __LINE__, cudaGetErrorString(err));      \
        exit(EXIT_FAILURE);                                        \
    }                                                              \
} while (0)

// ===========================================================
// CPU sequential reduction (baseline)
// ===========================================================
T cpu_reduce(const T* data, int n) {
    double sum = 0.0;
    for (int i = 0; i < n; ++i) {
        sum += data[i];
    }
    return static_cast<T>(sum);
}

bool nearly_equal(T a, T b, float tol = 1e-5f) {
    float diff = std::fabs(a - b);
    float norm = std::max(std::fabs(a), std::fabs(b));
    return diff <= tol * norm + 1e-6f;
}

// ===========================================================
// HARRIS KERNEL 1 – Interleaved addressing with divergence
// ===========================================================
__global__ void reduce_harris_v1(const T* __restrict__ g_idata,
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

    // Interleaved addressing, divergent
    for (unsigned int s = 1; s < blockDim.x; s <<= 1) {
        if ((tid % (2 * s)) == 0 && (tid + s) < blockDim.x) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0)
        g_odata[blockIdx.x] = sdata[0];
}

// ===========================================================
// HARRIS KERNEL 2 – Interleaved addressing, less divergence
// ===========================================================
__global__ void reduce_harris_v2(const T* __restrict__ g_idata,
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

    // Interleaved addressing, but avoid "if (tid % ..)" divergence
    for (unsigned int s = 1; s < blockDim.x; s <<= 1) {
        unsigned int index = 2 * s * tid;
        if (index < blockDim.x) {
            sdata[index] += sdata[index + s];
        }
        __syncthreads();
    }

    if (tid == 0)
        g_odata[blockIdx.x] = sdata[0];
}

// ===========================================================
// HARRIS KERNEL 3 – Sequential addressing (shared memory tree)
// ===========================================================
__global__ void reduce_harris_v3(const T* __restrict__ g_idata,
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

    // Sequential addressing
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0)
        g_odata[blockIdx.x] = sdata[0];
}

// ===========================================================
// HARRIS KERNEL 4 – Sequential addressing + partial unrolling
// ===========================================================
__global__ void reduce_harris_v4(const T* __restrict__ g_idata,
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

    // Manually unroll a few first steps (assuming blockDim.x >= 64)
    if (blockDim.x >= 512) {
        if (tid < 256) sdata[tid] += sdata[tid + 256];
        __syncthreads();
    }
    if (blockDim.x >= 256) {
        if (tid < 128) sdata[tid] += sdata[tid + 128];
        __syncthreads();
    }
    if (blockDim.x >= 128) {
        if (tid < 64) sdata[tid] += sdata[tid + 64];
        __syncthreads();
    }

    // Finish with a loop for the last stages
    for (unsigned int s = 32; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0)
        g_odata[blockIdx.x] = sdata[0];
}

// ===========================================================
// Helper for warp unrolling (used by v5 & v6)
// ===========================================================
__device__ void warp_reduce_volatile(volatile T* sdata, unsigned int tid) {
    sdata[tid] += sdata[tid + 32];
    sdata[tid] += sdata[tid + 16];
    sdata[tid] += sdata[tid + 8];
    sdata[tid] += sdata[tid + 4];
    sdata[tid] += sdata[tid + 2];
    sdata[tid] += sdata[tid + 1];
}

// ===========================================================
// HARRIS KERNEL 5 – Unroll last warp (avoid final syncthreads)
// ===========================================================
__global__ void reduce_harris_v5(const T* __restrict__ g_idata,
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

    // Sequential addressing down to 32
    for (unsigned int s = blockDim.x / 2; s > 32; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    // Unroll last warp (no __syncthreads needed)
    if (tid < 32) {
        volatile T* vmem = sdata;
        warp_reduce_volatile(vmem, tid);
    }

    if (tid == 0)
        g_odata[blockIdx.x] = sdata[0];
}

// ===========================================================
// HARRIS KERNEL 6 – Fully unrolled reduction (blockDim-dependent)
// ===========================================================
__global__ void reduce_harris_v6(const T* __restrict__ g_idata,
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

    // This is the classic "fully unrolled" pattern assuming
    // blockDim.x is a power of two up to 1024.
    if (blockDim.x >= 1024) { if (tid < 512) sdata[tid] += sdata[tid + 512]; __syncthreads(); }
    if (blockDim.x >= 512 ) { if (tid < 256) sdata[tid] += sdata[tid + 256]; __syncthreads(); }
    if (blockDim.x >= 256 ) { if (tid < 128) sdata[tid] += sdata[tid + 128]; __syncthreads(); }
    if (blockDim.x >= 128 ) { if (tid <  64) sdata[tid] += sdata[tid +  64]; __syncthreads(); }

    if (tid < 32) {
        volatile T* vmem = sdata;
        vmem[tid] += vmem[tid + 32];
        vmem[tid] += vmem[tid + 16];
        vmem[tid] += vmem[tid + 8];
        vmem[tid] += vmem[tid + 4];
        vmem[tid] += vmem[tid + 2];
        vmem[tid] += vmem[tid + 1];
    }

    if (tid == 0)
        g_odata[blockIdx.x] = sdata[0];
}

// ===========================================================
// LUITJENS – Warp shuffle helpers
// ===========================================================
__inline__ __device__
T warp_reduce_shfl(T val) {
    unsigned int mask = 0xffffffff;
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(mask, val, offset);
    }
    return val;
}

__inline__ __device__
T block_reduce_shfl(T val) {
    static __shared__ T shared[32];   // one entry per warp

    int lane = threadIdx.x & 31;
    int wid  = threadIdx.x >> 5;

    val = warp_reduce_shfl(val);      // warp-level reduction

    if (lane == 0) {
        shared[wid] = val;
    }
    __syncthreads();

    val = (threadIdx.x < blockDim.x / 32) ? shared[lane] : 0.0f;
    if (wid == 0) {
        val = warp_reduce_shfl(val);
    }
    return val;
}

// ===========================================================
// LUITJENS – Block reduction + atomic add (single pass)
// ===========================================================
__global__
void reduce_luitjens_atomic(const T* __restrict__ g_idata,
                            T* __restrict__ g_odata,
                            int n) {
    T sum = 0;

    // Grid-stride loop over input
    for (int i = blockIdx.x * blockDim.x + threadIdx.x;
         i < n;
         i += blockDim.x * gridDim.x) {
        sum += g_idata[i];
    }

    sum = block_reduce_shfl(sum);

    if (threadIdx.x == 0) {
        atomicAdd(g_odata, sum);
    }
}

// ===========================================================
// Timing + runner for Harris-style kernels
// ===========================================================
template <typename Kernel>
T run_reduction_kernel(const char* name,
                       Kernel kernel,
                       const T* d_in,
                       T* d_tmp,
                       int N,
                       int blockSize) {
    int gridSize = (N + blockSize * 2 - 1) / (blockSize * 2);
    int smem     = blockSize * sizeof(T);

    T* h_tmp = (T*)std::malloc(gridSize * sizeof(T));
    if (!h_tmp) {
        fprintf(stderr, "Host malloc failed\n");
        exit(EXIT_FAILURE);
    }

    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    CHECK_CUDA(cudaEventRecord(start));
    kernel<<<gridSize, blockSize, smem>>>(d_in, d_tmp, N);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float ms = 0.0f;
    CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));

    CHECK_CUDA(cudaMemcpy(h_tmp, d_tmp,
                          gridSize * sizeof(T),
                          cudaMemcpyDeviceToHost));

    double finalSum = 0.0;
    for (int i = 0; i < gridSize; ++i) {
        finalSum += h_tmp[i];
    }

    std::free(h_tmp);
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));

    printf("%s: %.3f ms, result = %.6f\n", name, ms, (T)finalSum);
    return static_cast<T>(finalSum);
}

// ===========================================================
// MAIN
// ===========================================================
int main() {
    const int N             = 1 << 24;   // ~16 million elements
    const int blockSize     = 256;
    const int gridSize      = (N + blockSize * 2 - 1) / (blockSize * 2);
    const int gridSizeAtomic = 256;      // for Luitjens atomic kernel

    printf("Allocating %d floats...\n", N);

    // Host input
    T* h_in = (T*)std::malloc(N * sizeof(T));
    if (!h_in) {
        fprintf(stderr, "Host malloc failed\n");
        return 1;
    }

    for (int i = 0; i < N; ++i) {
        h_in[i] = 1.0f;   // easy correctness check: sum = N
    }

    // CPU reference
    T cpuSum = cpu_reduce(h_in, N);
    printf("CPU reference = %.6f\n\n", cpuSum);

    // Device buffers
    T *d_in = nullptr, *d_tmp = nullptr, *d_out = nullptr;
    CHECK_CUDA(cudaMalloc(&d_in,  N * sizeof(T)));
    CHECK_CUDA(cudaMalloc(&d_tmp, gridSize * sizeof(T)));
    CHECK_CUDA(cudaMalloc(&d_out, sizeof(T)));

    CHECK_CUDA(cudaMemcpy(d_in, h_in,
                          N * sizeof(T),
                          cudaMemcpyHostToDevice));

    // Run Harris kernels 1–6
    T r1 = run_reduction_kernel("Harris v1 (interleaved, divergent)",
                                reduce_harris_v1,
                                d_in, d_tmp, N, blockSize);

    T r2 = run_reduction_kernel("Harris v2 (interleaved, less divergence)",
                                reduce_harris_v2,
                                d_in, d_tmp, N, blockSize);

    T r3 = run_reduction_kernel("Harris v3 (sequential addressing)",
                                reduce_harris_v3,
                                d_in, d_tmp, N, blockSize);

    T r4 = run_reduction_kernel("Harris v4 (partial unrolling)",
                                reduce_harris_v4,
                                d_in, d_tmp, N, blockSize);

    T r5 = run_reduction_kernel("Harris v5 (unroll last warp)",
                                reduce_harris_v5,
                                d_in, d_tmp, N, blockSize);

    T r6 = run_reduction_kernel("Harris v6 (fully unrolled)",
                                reduce_harris_v6,
                                d_in, d_tmp, N, blockSize);

    // Luitjens atomic + shuffle
    CHECK_CUDA(cudaMemset(d_out, 0, sizeof(T)));

    cudaEvent_t s, e;
    CHECK_CUDA(cudaEventCreate(&s));
    CHECK_CUDA(cudaEventCreate(&e));

    CHECK_CUDA(cudaEventRecord(s));
    reduce_luitjens_atomic<<<gridSizeAtomic, blockSize>>>(d_in, d_out, N);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaEventRecord(e));
    CHECK_CUDA(cudaEventSynchronize(e));

    float msAtomic = 0.0f;
    CHECK_CUDA(cudaEventElapsedTime(&msAtomic, s, e));

    T rL;
    CHECK_CUDA(cudaMemcpy(&rL, d_out, sizeof(T),
                          cudaMemcpyDeviceToHost));

    printf("Luitjens Block-Atomic: %.3f ms, result = %.6f\n",
           msAtomic, rL);

    // Correctness summary
    printf("\n=========================\n");
    printf("Correctness Check:\n");
    printf("=========================\n");
    printf("Harris v1:  %s\n", nearly_equal(r1, cpuSum) ? "OK" : "FAIL");
    printf("Harris v2:  %s\n", nearly_equal(r2, cpuSum) ? "OK" : "FAIL");
    printf("Harris v3:  %s\n", nearly_equal(r3, cpuSum) ? "OK" : "FAIL");
    printf("Harris v4:  %s\n", nearly_equal(r4, cpuSum) ? "OK" : "FAIL");
    printf("Harris v5:  %s\n", nearly_equal(r5, cpuSum) ? "OK" : "FAIL");
    printf("Harris v6:  %s\n", nearly_equal(r6, cpuSum) ? "OK" : "FAIL");
    printf("Luitjens:   %s\n", nearly_equal(rL, cpuSum) ? "OK" : "FAIL");

    // Cleanup
    std::free(h_in);
    cudaFree(d_in);
    cudaFree(d_tmp);
    cudaFree(d_out);

    return 0;
}

