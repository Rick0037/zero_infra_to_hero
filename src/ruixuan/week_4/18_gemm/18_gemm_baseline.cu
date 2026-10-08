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

// template <int BM, int BN, int BK, int BLOCK_SIZE>
// __global__ void GemmTiling(const float* A, const float* B, float* C, int M, int N, int K) {
//     //* one block own [bm][bn] in matrix C
//     __shared__ float smem_a[BM][BK];
//     __shared__ float smem_b[BK][BN];

//     // 当前在哪个block, 只管block内的东西
//     int tid = threadIdx.x;
//     // 起始位置的坐标 需要M/BM * N/BN个block 来处理

//     int tile_row = blockIdx.y * BM;
//     int tile_col = blockIdx.x * BN;

//     // 一次性加载A 到显存
//     // 对应线程在a中的排布
//     constexpr int thread_in_load_a_x = BK;
//     constexpr int thread_in_load_a_y = BLOCK_SIZE / BK;
//     const int a_col = tid % thread_in_load_a_x;
//     const int a_row = tid / thread_in_load_a_x;

//     // 一次性加载B 到显存
//     constexpr int thread_in_load_b_x = BLOCK_SIZE / BK;  // warpsize
//     constexpr int thread_in_load_b_y = BK;
//     const int b_row = tid / thread_in_load_b_x;
//     const int b_col = tid % thread_in_load_b_x;

//     // 一次在写入C 矩阵的时候进行重新排布
//     constexpr int thread_in_perocess_c_x = 16;
//     constexpr int thread_in_perocess_c_y = BLOCK_SIZE / thread_in_perocess_c_x;
//     const int c_row = tid / thread_in_perocess_c_x;
//     const int c_col = tid % thread_in_perocess_c_x;

//     constexpr int Tm = BM / thread_in_perocess_c_x;
//     constexpr int Tn = BN / thread_in_perocess_c_y;

//     // 每个线程只能保存这么多东西
//     float local_cache[Tm][Tn] = {};

//     // 外层遍历K/BK
//     //* 每个block 只管自己的线程怎么遍历, 对于c来说仅仅是[BM, BN]
//     //* 对应在A 中只负责 [BM, BK]
//     //* 对应在B 中只负责 [BK, BN]
//     const int k_loop_count = K / BK;
//     for (int dk = 0; dk < k_loop_count; dk++) {
//         //* 每个线程从a_row 开始每次跨步thread_in_load_a_y, 填充完 一个[[BM, BK]] 就ok了
//         //!  i < BM
//         for (int i = a_row; i < BM; i += thread_in_load_a_y) {
//             // 线程的列都是订好的
//             int thread_global_row = tile_row + i;
//             int thread_global_col = tile_col + a_col;
//             smem_a[i][a_col] = ((thread_global_row < M) && (thread_global_col < K))
//                                    ? A[thread_global_row * K + thread_global_col]
//                                    : 0.0;
//         }
//         //* 每个线程从b_col 开始每次跨步 thread_in_load_b_x, fill a [BK, BN]
//         //!  i < BN
//         for (int i = b_col; i < BN; i += thread_in_load_b_x) {
//             int thread_global_row = tile_row + b_row;
//             int thread_global_col = tile_col + i;
//             smem_b[b_row][i] = ((thread_global_row < K) && (thread_global_col < N))
//                                    ? B[thread_global_row * N + thread_global_col]
//                                    : 0.0;
//         }
//         __syncthreads();

//         // [BM, BK] * [BK, BN] 实际线程中只能处理 [bm, bn] 中的一部分数据
//         // !这么多[bm,bn]的数据只能让[16 * 16]的block 处理多次来实现
//         // !对应每个线程只能处理[tm, tn]次并最终写入

//         //* 内积写法 -> 以每个线程只能存储到local cache为目的
//         //! 遍历也可以是TM. TN 也可以是从大往输出取遍历, 只是说方便看就从Tm开始遍历了
//         //! 不好使用除法 可能性能有问题, 我感觉这也手写不了啊, 太恶心太容易错了

//         // 恶心到我了 还要外积写法不可能:
//         // 没一个
//         // [BM, BK] * [BK, BN], 但是分线程了
//         // 写入local cache
//         for (int i = c_row; i < BM; i += thread_in_perocess_c_y) {
//             int cache_row = (i - c_row) / thread_in_perocess_c_y;
//             for (int j = c_col; j < BN; j += thread_in_perocess_c_x) {
//                 int cache_col = (j - c_col) / thread_in_perocess_c_x;
//                 for (int p = 0; p < BK; p++) {
//                     local_cache[cache_row][cache_col] += smem_a[i][p] * smem_b[p][j];
//                 }
//             }
//         }
//         __syncthreads();
//     }

//     // 把local chache的内容写回去
//     for (int i = 0; i < Tm; i++) {
//         int global_row = tile_row + i * thread_in_perocess_c_y + c_col;
//         for (int j = 0; j < Tn; j++) {
//             int global_col = tile_col + j * thread_in_perocess_c_x + c_row;
//             if (global_row < M && global_col < N)
//                 C[global_row * BM + global_col] = local_cache[i][j];
//         }
//     }

//     return;
// }

// for (int row = c_local_row; row < BM; row += C_BLOCK_Y) {
//     int i = (row - c_local_row) / C_BLOCK_Y;

//     for (int col = c_local_col; col < BN; col += C_BLOCK_X) {
//         int j = (col - c_local_col) / C_BLOCK_X;

//         float sum = 0.0f;

//         for (int p = 0; p < BK; ++p) {
//             sum += As[row][p] * Bs[p][col];
//         }

//         accum[i][j] += sum;
//     }
// }

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
