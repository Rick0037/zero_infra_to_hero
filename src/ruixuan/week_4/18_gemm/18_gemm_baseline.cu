#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#define CHECK(call)                                                         \
    do {                                                                    \
        const cudaError_t error_code = call;                                \
        if (error_code != cudaSuccess) {                                    \
            printf("CUDA Error:\n");                                        \
            printf("    File:       %s\n", __FILE__);                       \
            printf("    Line:       %d\n", __LINE__);                       \
            printf("    Error code: %d\n", error_code);                     \
            printf("    Error text: %s\n", cudaGetErrorString(error_code)); \
            exit(1);                                                        \
        }                                                                   \
    } while (0)

#define WARMUP 10
#define ITERS 100

__global__ void GemmBaseline(const float* A, const float* B, float* C, int M, int N, int K) {
    //* 两个线程负责C种的一个元素

    int tid_x = threadIdx.x + blockDim.x * blockIdx.x;
    int tid_y = threadIdx.y + blockDim.y * blockIdx.y;
    if ((tid_x < M) && (tid_y < N)) {
        float sum{0.0};
        for (int i = 0; i < K; i++) {
            sum += A[tid_x * K + i] * B[i * N + tid_y];
        }
        C[tid_x * N + tid_y] = sum;
    }

    return;
}

// CPU 参考: C = A * B, A[M,K] x B[K,N] -> C[M,N]
void gemm_cpu_ref(const float* A, const float* B, float* C, int M, int N, int K) {
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            float sum = 0.0f;
            for (int k = 0; k < K; k++) {
                sum += A[i * K + k] * B[k * N + j];
            }
            C[i * N + j] = sum;
        }
    }
    return;
}

int check_gemm_res(int M, int N, float* h_out, float* h_gt, float* max_err) {
    int bad = 0;
    *max_err = 0.0f;
    for (int i = 0; i < M * N; i++) {
        float diff = fabsf(h_out[i] - h_gt[i]);
        if (diff > *max_err) {
            *max_err = diff;
        }
        // 写成 !(diff <= tol), 写成 diff > tol 时 NaN 会被判成通过
        if (!(diff <= 1e-3f)) {
            bad++;
        }
    }
    return bad;
}

int main() {
    printf("-----gemm baseline----\n");

    const int M = 512;
    const int N = 512;
    const int K = 512;
    const size_t a_bytes = (size_t)M * K * sizeof(float);
    const size_t b_bytes = (size_t)K * N * sizeof(float);
    const size_t c_bytes = (size_t)M * N * sizeof(float);

    // CPU 数据
    float* h_A = (float*)malloc(a_bytes);
    float* h_B = (float*)malloc(b_bytes);
    float* h_C = (float*)malloc(c_bytes);
    float* h_gt = (float*)malloc(c_bytes);
    for (int i = 0; i < M * K; i++) {
        h_A[i] = (float)(i % 7) - 3.0f;  // -3 ~ 3
    }
    for (int i = 0; i < K * N; i++) {
        h_B[i] = (float)(i % 5) - 2.0f;  // -2 ~ 2
    }
    gemm_cpu_ref(h_A, h_B, h_gt, M, N, K);

    // GPU 数据
    float *d_A, *d_B, *d_C;
    CHECK(cudaMalloc((void**)&d_A, a_bytes));
    CHECK(cudaMalloc((void**)&d_B, b_bytes));
    CHECK(cudaMalloc((void**)&d_C, c_bytes));
    CHECK(cudaMemcpy(d_A, h_A, a_bytes, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_B, h_B, b_bytes, cudaMemcpyHostToDevice));

    dim3 block(16, 16);
    dim3 grid((M + block.x - 1) / block.x, (N + block.y - 1) / block.y);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float ms = 0.0f;

    for (int i = 0; i < WARMUP; i++) {
        GemmBaseline<<<grid, block>>>(d_A, d_B, d_C, M, N, K);
    }
    cudaDeviceSynchronize();
    cudaEventRecord(start);
    for (int i = 0; i < ITERS; i++) {
        GemmBaseline<<<grid, block>>>(d_A, d_B, d_C, M, N, K);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ms, start, stop);
    float avg_ms = ms / ITERS;
    printf("gemm baseline [%d,%d]x[%d,%d] avg = %.4f ms\n", M, N, N, K, avg_ms);

    // 算力: 2*M*N*K FLOPs(MAC = mul+add)
    double gflops = 2.0 * M * N * K / (avg_ms * 1e-3) / 1e9;
    printf("gemm baseline perf = %.1f GFLOPS\n", gflops);

    CHECK(cudaMemcpy(h_C, d_C, c_bytes, cudaMemcpyDeviceToHost));
    float max_err = 0.0f;
    int bad = check_gemm_res(M, N, h_C, h_gt, &max_err);
    printf("gemm mismatch = %d / %d, max_abs_err = %.3e\n", bad, M * N, max_err);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    CHECK(cudaFree(d_A));
    CHECK(cudaFree(d_B));
    CHECK(cudaFree(d_C));
    free(h_A);
    free(h_B);
    free(h_C);
    free(h_gt);
    return 0;
}
