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

//*-------------------> x 方向
//*|
//*|
//*|
//*|
//*v
//* 列方向
template <int BM, int BN, int BK, int BLOCK_SIZE>
__global__ void GemmTiling(const float* A, const float* B, float* C, int M, int N, int K) {
    // must english

    int tid = threadIdx.x;
    //! 必须乘以 BM, BN 才是global index
    int tile_row = blockIdx.y * BM;
    int tile_col = blockIdx.x * BN;

    // one block must be C [bm, bn], A [bm, bk], b[bk, bn]

    __shared__ float smem_a[BM][BK];
    __shared__ float smem_b[BK][BN];

    // for a
    constexpr int A_LOOP_X = BK;
    constexpr int A_LOOP_Y = BLOCK_SIZE / BK;
    int a_shape_row = tid / A_LOOP_X;
    int a_shape_col = tid % A_LOOP_X;

    // for b
    constexpr int B_LOOP_X = BLOCK_SIZE / BK;
    constexpr int B_LOOP_Y = BK;
    int b_shape_row = tid / B_LOOP_X;
    int b_shape_col = tid % B_LOOP_X;

    // for c
    constexpr int C_LOOP_X = 16;
    constexpr int C_LOOP_Y = BLOCK_SIZE / C_LOOP_X;
    int c_shape_row = tid / C_LOOP_X;
    int c_shape_col = tid % C_LOOP_X;

    // thread cache
    const int TM = BM / C_LOOP_Y;
    const int TN = BN / C_LOOP_X;

    float cache[TM][TN] = {};

    const int loop_size = K / BK;
    for (int d_k = 0; d_k < loop_size; d_k++) {
        // load a to smem
        // 针对[bm, bk] 来说 线程在a中排布如何遍历
        for (int i = a_shape_row; i < BM; i += A_LOOP_Y) {
            int global_row_in_a = tile_row + i;
            // !  不是 tile_col + a_shape_col, 而是 d_k * BK + a_shape_col
            int global_col_in_a = d_k * BK + a_shape_col;
            smem_a[i][a_shape_col] = A[global_row_in_a * K + global_col_in_a];
        }

        // load b to smem
        for (int i = b_shape_col; i < BN; i += B_LOOP_X) {
            // !  不是 tile_row + b_shape_row, 而是 d_k * BK + b_shape_row
            int global_row_in_b = d_k * BK + b_shape_row;
            int global_col_in_b = tile_col + i;
            smem_b[b_shape_row][i] = B[global_row_in_b * N + global_col_in_b];
        }

        __syncthreads();

        // smema * smemb -> result store in cache
        // each thread will cache [BM, BN] / [blocksize.x, blocksize.y]
        //* for [BM * BK] * [BK, BN] loop
        //* for mat mal look at result matrix

        for (int i = c_shape_row; i < BM; i += C_LOOP_Y) {
            int cache_row = (i - c_shape_row) / C_LOOP_Y;
            for (int j = c_shape_col; j < BN; j += C_LOOP_X) {
                int cache_col = (j - c_shape_col) / C_LOOP_X;
                for (int p = 0; p < BK; p++) {
                    cache[cache_row][cache_col] += smem_a[i][p] * smem_b[p][j];
                }
            }
        }

        __syncthreads();
    }
    // 写入C
    for (int i = 0; i < TM; i++) {
        int global_row_in_c = tile_row + i * C_LOOP_Y + c_shape_row;
        for (int j = 0; j < TN; j++) {
            int global_col_in_c = tile_col + j * C_LOOP_X + c_shape_col;
            C[global_row_in_c * N + global_col_in_c] = cache[i][j];
        }
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

    dim3 block_2(256);
    dim3 gird_2(4, 4);

    for (int i = 0; i < WARMUP; i++) {
        // GemmBaseline<<<grid, block>>>(d_A, d_B, d_C, M, N, K);
        GemmTiling<128, 128, 8, 256><<<gird_2, block_2>>>(d_A, d_B, d_C, M, N, K);
    }
    cudaDeviceSynchronize();
    cudaEventRecord(start);
    for (int i = 0; i < ITERS; i++) {
        // GemmBaseline<<<grid, block>>>(d_A, d_B, d_C, M, N, K);
        GemmTiling<128, 128, 8, 256><<<gird_2, block_2>>>(d_A, d_B, d_C, M, N, K);
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
