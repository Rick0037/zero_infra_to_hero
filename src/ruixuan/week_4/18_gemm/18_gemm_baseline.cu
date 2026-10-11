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

__global__ void GemmNative(const float* A, const float* B, float* C, int M, int N, int K) {
    int gid = threadIdx.x + blockDim.x * blockIdx.x;
    int loop_size = blockDim.x * gridDim.x;
    for (int i = gid; i < M * N; i += loop_size) {
        int row = i / N;
        int col = i % N;
        float sum = 0.0;
        for (int j = 0; j < K; j++) {
            sum += A[row * K + j] * B[j * N + col];
        }
        C[row * N + col] = sum;
    }
    return;
}

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

template <int BM, int BN, int BK, int BLOCK_SIZE>
__global__ void GemmTilingSharedOpt(const float* A, const float* B, float* C, int M, int N, int K) {
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
    float cache_a[TM] = {};
    float cache_b[TN] = {};

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

        // ---------method 4----------
        for (int p = 0; p < BK; p++) {
            for (int i = 0; i < TM; i++) {
                //! 记得自己在小方块中永远是什么位置 i * C_LOOP_Y + c_shape_row
                cache_a[i] = smem_a[i * C_LOOP_Y + c_shape_row][p];
            }
            for (int i = 0; i < TN; i++) {
                //! 记得自己在小方块中永远是什么位置 i * C_LOOP_X + c_shape_col
                cache_b[i] = smem_b[p][i * C_LOOP_X + c_shape_col];
            }
            for (int i = 0; i < TM; i++) {
                for (int j = 0; j < TN; j++) {
                    cache[i][j] += cache_a[i] * cache_b[j];
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
template <int BM, int BN, int BK, int BLOCK_SIZE>
__global__ void GemmTilingSplitKLearing(const float* A, const float* B, float* part_of_k_result,
                                        int M, int N, int K, int split_k_iters) {
    int bidx = blockIdx.x;
    int bidy = blockIdx.y;
    int bidz = blockIdx.z;  // for split k
    int tid = threadIdx.x;

    //* 多个block 把 k 分成了几段, 每一段 K / split_k_iters
    //* 每一个小段让bk 去循环 K / split_k_iters
    const int k_iter = (K / split_k_iters) / BK;
    const int k_start = (K / split_k_iters) * bidz;

    // 每个block 负责的 数值范围更小了
    __shared__ float a_smem[BM][BK];
    __shared__ float b_smem[BK][BN];

    // block in a 排布
    constexpr int a_Y_dic_thread_number = 16;
    constexpr int a_X_dic_thread_number = BLOCK_SIZE / a_Y_dic_thread_number;
    int thread_in_a_row = tid / a_X_dic_thread_number;
    int thread_in_a_col = tid % a_X_dic_thread_number;

    // block in c 排布
    constexpr int b_Y_dic_thread_number = 32;
    constexpr int b_X_dic_thread_number = BLOCK_SIZE / b_Y_dic_thread_number;
    int thread_in_b_row = tid / b_X_dic_thread_number;
    int thread_in_b_col = tid % b_X_dic_thread_number;

    // block in c 排布
    constexpr int Y_dic_thread_number = 16;
    constexpr int X_dic_thread_number = BLOCK_SIZE / Y_dic_thread_number;
    int thread_in_c_row = tid / X_dic_thread_number;
    int thread_in_c_col = tid % X_dic_thread_number;

    constexpr int TN = BN / X_dic_thread_number;
    constexpr int TM = BM / Y_dic_thread_number;
    float cache[TM][TN] = {};

    for (int k_0 = 0; k_0 < k_iter; k_0++) {
        int current_k_iter_start_offset = k_start + k_0 * BK;
        int i_max = BM / a_Y_dic_thread_number;
        int j_max = BK / a_X_dic_thread_number;
        for (int i = 0; i < i_max; i++) {
            int row = i * a_Y_dic_thread_number + thread_in_a_row;
            int a_global_row = bidy * BM + row;
            for (int j = 0; j < j_max; j++) {
                int col = j * a_X_dic_thread_number + thread_in_a_col;
                int a_global_col = current_k_iter_start_offset + col;
                a_smem[row][col] = A[a_global_row * K + a_global_col];
            }
        }

        i_max = BK / b_Y_dic_thread_number;
        j_max = BN / b_X_dic_thread_number;
        for (int i = 0; i < i_max; i++) {
            int row = i * b_Y_dic_thread_number + thread_in_b_row;
            int b_global_row = current_k_iter_start_offset + row;
            for (int j = 0; j < j_max; j++) {
                int col = j * b_X_dic_thread_number + thread_in_b_col;
                int b_global_col = bidx * BN + col;
                b_smem[row][col] = B[b_global_row * N + b_global_col];
            }
        }
        __syncthreads();

        for (int i = 0; i < TM; i++) {
            // 这里实际上是在BM,BN中的位置
            int row = i * Y_dic_thread_number + thread_in_c_row;
            for (int j = 0; j < TN; j++) {
                int col = j * X_dic_thread_number + thread_in_c_col;
                for (int p = 0; p < BK; p++) {
                    cache[i][j] += a_smem[row][p] * b_smem[p][col];
                }
            }
        }
        __syncthreads();
    }

    for (int i = 0; i < TM; i++) {
        int global_row = bidy * BM + i * Y_dic_thread_number + thread_in_c_row;
        for (int j = 0; j < TN; j++) {
            int global_col = bidx * BN + j * X_dic_thread_number + thread_in_c_col;
            //! bidz * M * N z方向的结果
            part_of_k_result[bidz * M * N + global_row * N + global_col] = cache[i][j];
        }
    }

    return;
}

// 把 split_k 段的部分和沿 z 方向累加成最终 C
// part_of_k_result 布局: [split_id][M][N] 展平, 即 split_id * M * N + row * N + col
__global__ void ReduceSplitKLearing(const float* part_of_k_result, float* C, int M, int N,
                                    int split_k_iters) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    int loop_size = gridDim.x * blockDim.x;
    int total_elements = M * N;

    for (int idx = gid; idx < total_elements; idx += loop_size) {
        float sum = 0.0f;
        for (int split_id = 0; split_id < split_k_iters; split_id++) {
            sum += part_of_k_result[split_id * total_elements + idx];
        }
        C[idx] = sum;
    }
}

template <int BM, int BN, int BK, int BLOCK_SIZE>
__global__ void GemmTilingLearing(const float* A, const float* B, float* C, int M, int N, int K) {
    // block is two dim, thread is one dim
    int bidx = blockIdx.x;
    int bidy = blockIdx.y;
    int tid = threadIdx.x;

    const int k_iter = K / BK;

    __shared__ float a_smem[BM][BK];
    __shared__ float b_smem[BK][BN];
    // __shared__ float c_smem[BM][BN];

    // 线程在 c上的分布, 外一 线程数量小于 一条边长怎么办呢
    constexpr int Y_dic_thread_number = 16;
    constexpr int X_dic_thread_number = BLOCK_SIZE / Y_dic_thread_number;
    int thread_in_c_row = tid / X_dic_thread_number;
    int thread_in_c_col = tid % X_dic_thread_number;

    // 实际上每个线程需要处理这么多数据
    constexpr int TN = BN / X_dic_thread_number;
    constexpr int TM = BM / Y_dic_thread_number;
    float cache[TM][TN] = {};

    for (int k_0 = 0; k_0 < k_iter; k_0++) {
        // copy from a to a shared
        for (int i = tid; i < BM * BK; i += BLOCK_SIZE) {
            int row = i / BK;
            int col = i % BK;
            int a_global_row = bidy * BM + row;
            int a_global_col = k_0 * BK + col;
            a_smem[row][col] = A[a_global_row * K + a_global_col];
        }
        // copy from b to shared
        for (int i = tid; i < BK * BN; i += BLOCK_SIZE) {
            int row = i / BN;
            int col = i % BN;
            int b_global_row = k_0 * BK + row;
            int b_global_col = bidx * BN + col;
            b_smem[row][col] = B[b_global_row * N + b_global_col];
        }
        __syncthreads();

        for (int i = 0; i < TM; i++) {
            // 这里实际上是在BM,BN中的位置
            int row = i * Y_dic_thread_number + thread_in_c_row;
            for (int j = 0; j < TN; j++) {
                int col = j * X_dic_thread_number + thread_in_c_col;
                for (int p = 0; p < BK; p++) {
                    cache[i][j] += a_smem[row][p] * b_smem[p][col];
                }
            }
        }
        __syncthreads();
    }
    for (int i = 0; i < TM; i++) {
        int global_row = bidy * BM + i * Y_dic_thread_number + thread_in_c_row;
        for (int j = 0; j < TN; j++) {
            int global_col = bidx * BN + j * X_dic_thread_number + thread_in_c_col;
            C[global_row * N + global_col] = cache[i][j];
        }
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

        //-----------method 1-----------
        // for (int i = c_shape_row; i < BM; i += C_LOOP_Y) {
        //     int cache_row = (i - c_shape_row) / C_LOOP_Y;
        //     for (int j = c_shape_col; j < BN; j += C_LOOP_X) {
        //         int cache_col = (j - c_shape_col) / C_LOOP_X;
        //         for (int p = 0; p < BK; p++) {
        //             cache[cache_row][cache_col] += smem_a[i][p] * smem_b[p][j];
        //         }
        //     }
        // }

        //-----------method 2-----------
        // for (int i = 0; i < TM; i++) {
        //     int row = i * C_LOOP_Y + c_shape_row;
        //     for (int j = 0; j < TN; j++) {
        //         int col = j * C_LOOP_X + c_shape_col;
        //         for (int p = 0; p < BK; p++) {
        //             cache[i][j] += smem_a[row][p] * smem_b[p][col];
        //         }
        //     }
        // }
        // ---------method 3----------
        for (int p = 0; p < BK; p++) {
            for (int i = 0; i < TM; i++) {
                int row = i * C_LOOP_Y + c_shape_row;
                for (int j = 0; j < TN; j++) {
                    int col = j * C_LOOP_X + c_shape_col;
                    cache[i][j] += smem_a[row][p] * smem_b[p][col];
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

    // GemmTilingSplitKLearing 测试
    // BM=128 BN=128 BK=32, split_k=4: K/split_k=128, k_iter=128/32=4
    // 约束: BK%32==0, K%(split_k*BK)==0, M%BM==0, N%BN==0
    const int split_k_iters = 4;
    const size_t partial_bytes = (size_t)split_k_iters * M * N * sizeof(float);
    float* d_partial;
    CHECK(cudaMalloc((void**)&d_partial, partial_bytes));

    dim3 block_2(256);
    dim3 gird_2(4, 4);     // m 方向, 按照 M/ bm, n方向 按照N/bn 方向切分
    dim3 gird_3(4, 4, 4);  // x: N/BN, y: M/BM, z: split_k

    for (int i = 0; i < WARMUP; i++) {
        // GemmBaseline<<<grid, block>>>(d_A, d_B, d_C, M, N, K);
        // GemmTiling<128, 128, 8, 256><<<gird_2, block_2>>>(d_A, d_B, d_C, M, N, K);
        GemmTilingSplitKLearing<128, 128, 32, 256>
            <<<gird_3, block_2>>>(d_A, d_B, d_partial, M, N, K, split_k_iters);
        ReduceSplitKLearing<<<(M * N + 255) / 256, 256>>>(d_partial, d_C, M, N, split_k_iters);
    }
    cudaDeviceSynchronize();
    cudaEventRecord(start);
    for (int i = 0; i < ITERS; i++) {
        // GemmBaseline<<<grid, block>>>(d_A, d_B, d_C, M, N, K);
        // GemmTiling<128, 128, 8, 256><<<gird_2, block_2>>>(d_A, d_B, d_C, M, N, K);
        GemmTilingSplitKLearing<128, 128, 32, 256>
            <<<gird_3, block_2>>>(d_A, d_B, d_partial, M, N, K, split_k_iters);
        ReduceSplitKLearing<<<(M * N + 255) / 256, 256>>>(d_partial, d_C, M, N, split_k_iters);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ms, start, stop);
    float avg_ms = ms / ITERS;
    printf("gemm tiling split-k [%d,%d]x[%d,%d] avg = %.4f ms\n", M, N, N, K, avg_ms);

    // 算力: 2*M*N*K FLOPs(MAC = mul+add)
    double gflops = 2.0 * M * N * K / (avg_ms * 1e-3) / 1e9;
    printf("gemm tiling split-k perf = %.1f GFLOPS\n", gflops);

    CHECK(cudaMemcpy(h_C, d_C, c_bytes, cudaMemcpyDeviceToHost));
    float max_err = 0.0f;
    int bad = check_gemm_res(M, N, h_C, h_gt, &max_err);
    printf("gemm mismatch = %d / %d, max_abs_err = %.3e\n", bad, M * N, max_err);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    CHECK(cudaFree(d_A));
    CHECK(cudaFree(d_B));
    CHECK(cudaFree(d_C));
    CHECK(cudaFree(d_partial));
    free(h_A);
    free(h_B);
    free(h_C);
    free(h_gt);
    return 0;
}
