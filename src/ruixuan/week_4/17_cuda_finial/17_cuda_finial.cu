#include <cuda.h>
#include <cuda_runtime.h>
#include <float.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include <algorithm>

#include "17_cuda.cuh"

#define WarpSize 32
#define M 1024

// *------N dispatch------

// #define N 512
#define N 4096
// #define N 16384

__device__ float warp_level_sum(float sum) {
    for (int i = 16; i > 0; i >>= 1) {
        sum += __shfl_down_sync(0xffffffff, sum, i);
    }
    return sum;
}
__device__ float warp_level_max(float max_value) {
    for (int i = 16; i > 0; i >>= 1) {
        max_value = fmax(max_value, __shfl_down_sync(0xffffffff, max_value, i));
    }
    return max_value;
}

__global__ void SoftmaxBaseline(int m, int n, float* d_in, float* d_out) {
    //(a) baseline 三遍 pass（max→sum→normalize），一个 block 处理一行
    // 先求最大值, 求和, 向量化读取
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    int thread_index_in_warp = tid % WarpSize;
    int warp_idx = tid / WarpSize;

    extern __shared__ float smem[];
    for (int index = bid; index < m; index += gridDim.x) {
        float* start = d_in + n * index;
        float* out = d_out + n * index;
        //----------max-----------
        float max_value = -FLT_MAX;
        for (int j = tid; j < n; j += blockDim.x) {
            max_value = fmax(start[j], max_value);
        }

        max_value = warp_level_max(max_value);
        if (thread_index_in_warp == 0) {
            smem[warp_idx] = max_value;
        }
        __syncthreads();

        if (warp_idx == 0) {
            max_value = (tid < blockDim.x / WarpSize) ? smem[tid] : -FLT_MAX;
            max_value = warp_level_max(max_value);
        }

        if (tid == 0) {
            smem[tid] = max_value;
        }
        __syncthreads();
        max_value = smem[0];
        __syncthreads();
        //----------sum-----------
        float sum = 0.0;
        for (int j = tid; j < n; j += blockDim.x) {
            sum += std::exp(start[j] - max_value);
        }

        sum = warp_level_sum(sum);
        if (thread_index_in_warp == 0) {
            smem[warp_idx] = sum;
        }
        __syncthreads();

        if (warp_idx == 0) {
            sum = (tid < blockDim.x / WarpSize) ? smem[tid] : 0.0;
            sum = warp_level_sum(sum);
        }

        if (tid == 0) {
            smem[tid] = sum;
        }
        __syncthreads();
        sum = smem[0];
        __syncthreads();

        for (int j = tid; j < n; j += blockDim.x) {
            out[j] = std::exp(start[j] - max_value) / sum;
        }
    }
    return;
}

__device__ void warp_level_sum_and_max(float& max_value, float& sum) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        float new_max_value = __shfl_down_sync(0xffffffff, max_value, offset);
        float new_sum = __shfl_down_sync(0xffffffff, sum, offset);
        float temp = fmax(new_max_value, max_value);
        sum = sum * std::exp(max_value - temp) + new_sum * std::exp(new_max_value - temp);
        max_value = temp;
    }
}

template <int BlockSize, int ColNumber>
__global__ void SoftmaxOnline(int m, int n, float* d_in, float* d_out) {
    // (b) online 单 pass 版（flash attention 的 online softmax 递推
    // one pass, 寄存器, 一次归约
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    int thread_index_in_warp = tid % WarpSize;
    int warp_idx = tid / WarpSize;
    constexpr int ThreadLoopCount = (ColNumber / 4) / BlockSize;

    __shared__ unsigned char smem_online[BlockSize * 2 * 4];
    float* max_smem = reinterpret_cast<float*>(smem_online);
    float* sum_smem = reinterpret_cast<float*>(smem_online) + blockDim.x;

    for (int index = bid; index < m; index += gridDim.x) {
        float4* start = reinterpret_cast<float4*>(d_in) + index * n / 4;
        float4* out_start = reinterpret_cast<float4*>(d_out) + index * n / 4;
        float max_value = -FLT_MAX;
        float sum = 0.0;
        float new_max_value;
        float4 temp_value_array[ThreadLoopCount];
#pragma unroll
        for (int i = 0; i < ThreadLoopCount; i++) {
            int j = tid + i * BlockSize;
            temp_value_array[i] = start[j];
            float temp_array[4] = {temp_value_array[i].x, temp_value_array[i].y,
                                   temp_value_array[i].z, temp_value_array[i].w};
            for (int k = 0; k < 4; k++) {
                new_max_value = fmax(temp_array[k], max_value);
                sum = sum * std::exp(max_value - new_max_value) +
                      std::exp(temp_array[k] - new_max_value);
                max_value = new_max_value;
            }
        }

        warp_level_sum_and_max(max_value, sum);
        if (thread_index_in_warp == 0) {
            max_smem[warp_idx] = max_value;
            sum_smem[warp_idx] = sum;
        }
        __syncthreads();

        if (warp_idx == 0) {
            max_value = (tid < blockDim.x / WarpSize) ? max_smem[tid] : -FLT_MAX;

            sum = (tid < blockDim.x / WarpSize) ? sum_smem[tid] : 0.0;
            warp_level_sum_and_max(max_value, sum);
        }

        if (tid == 0) {
            max_smem[0] = max_value;
            sum_smem[0] = sum;
        }
        __syncthreads();
        max_value = max_smem[0];
        sum = sum_smem[0];
        __syncthreads();
#pragma unroll
        for (int i = 0; i < ThreadLoopCount; i++) {
            int j = tid + i * BlockSize;
            float4 output_to_write{};
            output_to_write.x = std::exp(temp_value_array[i].x - max_value) / sum;
            output_to_write.y = std::exp(temp_value_array[i].y - max_value) / sum;
            output_to_write.w = std::exp(temp_value_array[i].w - max_value) / sum;
            output_to_write.z = std::exp(temp_value_array[i].z - max_value) / sum;
            out_start[j] = output_to_write;
        }
    }
    return;
}

__device__ __forceinline__ float getnearbyint(float val) { return std::nearbyint(val); }

__device__ int8_t scaleandclip(float value, float scale, int8_t max_int8, int8_t min_int8) {
    float temp_value = getnearbyint(value / scale);
    temp_value = (temp_value > max_int8) ? max_int8 : temp_value;
    temp_value = (temp_value < min_int8) ? min_int8 : temp_value;
    return (int8_t)temp_value;
}

template <int BlockSize, int ColNumber>
__global__ void QuantizePerTokenSymmetric(int m, int n, const float* d_in, float* d_scale,
                                          int8_t* d_out, int8_t max_int8, int8_t min_int8) {
    // per-token 对称 INT8 quantize kernel：输入 [m=1024, n=4096]
    // absmax→scale=absmax/127→round+clip
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    int thread_index_in_warp = tid % WarpSize;
    int warp_idx = tid / WarpSize;
    constexpr int ThreadLoopCount = (ColNumber / 4) / BlockSize;

    __shared__ float smem_quant[BlockSize];
    __shared__ float shared_scale;
    //
    for (int index = bid; index < m; index += gridDim.x) {
        float max_value = -FLT_MAX;
        float min_value = FLT_MAX;
        const float4* start = reinterpret_cast<const float4*>(d_in) + index * n / 4;
        char4* q_out = reinterpret_cast<char4*>(d_out) + index * n / 4;
        float4 temp_value_array[ThreadLoopCount];
#pragma unroll
        for (int i = 0; i < ThreadLoopCount; i++) {
            int j = tid + i * BlockSize;
            temp_value_array[i] = start[j];
            max_value = fmax(temp_value_array[i].x, max_value);
            max_value = fmax(temp_value_array[i].y, max_value);
            max_value = fmax(temp_value_array[i].z, max_value);
            max_value = fmax(temp_value_array[i].w, max_value);
            min_value = fmin(temp_value_array[i].x, min_value);
            min_value = fmin(temp_value_array[i].y, min_value);
            min_value = fmin(temp_value_array[i].z, min_value);
            min_value = fmin(temp_value_array[i].w, min_value);
        }
        max_value = fmax(fabsf(min_value), fabsf(max_value));

        max_value = warp_level_max(max_value);
        if (thread_index_in_warp == 0) {
            smem_quant[warp_idx] = max_value;
        }
        __syncthreads();

        if (warp_idx == 0) {
            max_value = (tid < blockDim.x / WarpSize) ? smem_quant[tid] : -FLT_MAX;
            max_value = warp_level_max(max_value);
        }
        if (tid == 0) {
            smem_quant[tid] = max_value;
            shared_scale = max_value / max_int8;  // 赋值
            d_scale[index] = shared_scale;
        }
        __syncthreads();
        //* 自己yy的写法, 看看对不对
#pragma unroll
        for (int i = 0; i < ThreadLoopCount; i++) {
            int j = tid + i * BlockSize;
            float4 temp_value = temp_value_array[i];
            char4 q = make_char4(scaleandclip(temp_value.x, shared_scale, max_int8, min_int8),
                                 scaleandclip(temp_value.y, shared_scale, max_int8, min_int8),
                                 scaleandclip(temp_value.z, shared_scale, max_int8, min_int8),
                                 scaleandclip(temp_value.w, shared_scale, max_int8, min_int8));
            // 赋值
            q_out[j] = q;
        }
    }
    return;
}

template <int BlockSize>
__global__ void GemvHalf(int m, int n, const half* d_mat, const half* d_vec, half* d_out) {
    // gemv（W4A16 之前的热身）：y = A · x，A shape [4096, 4096] fp16，x [4096]，一个 warp/block
    // 负责若干行，用 float4/half2 向量化访存
    // 先乘积, 乘积之后进行reduce
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    int thread_index_in_warp = tid % WarpSize;
    int warp_idx = tid / WarpSize;
    const float4* vec_start = reinterpret_cast<const float4*>(d_vec);
    __shared__ float smem_gemv[BlockSize];
    for (int index = bid; index < m; index += gridDim.x) {
        //* 读取还是使用float4, 实际计算还是使用half2
        const float4* start = reinterpret_cast<const float4*>(d_mat) + index * n / 8;
        half2 thread_local_sum_value{0.0, 0.0};
        for (int j = tid; j < n / 8; j += blockDim.x) {
            float4 temp_value = start[j];
            float4 temp_vec_value = vec_start[j];
            //! half2 * half2 -> half2  __hmul2
            thread_local_sum_value +=
                __hmul2(*(half2*)(&temp_value.x), *(half2*)(&temp_vec_value.x));
            thread_local_sum_value +=
                __hmul2(*(half2*)(&temp_value.y), *(half2*)(&temp_vec_value.y));
            thread_local_sum_value +=
                __hmul2(*(half2*)(&temp_value.z), *(half2*)(&temp_vec_value.z));
            thread_local_sum_value +=
                __hmul2(*(half2*)(&temp_value.w), *(half2*)(&temp_vec_value.w));
        }
        float sum = thread_local_sum_value.x + thread_local_sum_value.y;

        sum = warp_level_sum(sum);

        if (thread_index_in_warp == 0) {
            smem_gemv[warp_idx] = sum;
        }
        __syncthreads();

        if (warp_idx == 0) {
            sum = (tid < blockDim.x / WarpSize) ? smem_gemv[tid] : 0.0;
            sum = warp_level_sum(sum);
        }
        if (tid == 0) {
            smem_gemv[tid] = sum;
            d_out[index] = (half)sum;
        }
        __syncthreads();
    }

    return;
}

template <int BlockSize, int ColNumber>
__global__ void FuseSoftmaxQuant(int m, int n, const float* d_in, float* d_scale, int8_t* d_out,
                                 int8_t max_int8, int8_t min_int8) {
    // 可选加分：把 quantize kernel 融合进 softmax 的输出端（softmax→int8
    // 一次写回），对比融合前后的总耗时，说明省掉了几次 HBM 往返。
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    int thread_index_in_warp = tid % WarpSize;
    int warp_idx = tid / WarpSize;
    constexpr int ThreadLoopCount = (ColNumber / 4) / BlockSize;

    __shared__ unsigned char smem_online[BlockSize * 2 * 4];
    float* max_smem = reinterpret_cast<float*>(smem_online);
    float* sum_smem = reinterpret_cast<float*>(smem_online) + blockDim.x;

    for (int index = bid; index < m; index += gridDim.x) {
        const float4* start = reinterpret_cast<const float4*>(d_in) + index * n / 4;
        char4* q_out = reinterpret_cast<char4*>(d_out) + index * n / 4;
        float max_value = -FLT_MAX;
        float sum = 0.0;
        float new_max_value;
        float4 temp_value_array[ThreadLoopCount];
#pragma unroll
        for (int i = 0; i < ThreadLoopCount; i++) {
            int j = tid + i * BlockSize;
            temp_value_array[i] = start[j];
            float temp_array[4] = {temp_value_array[i].x, temp_value_array[i].y,
                                   temp_value_array[i].z, temp_value_array[i].w};
            for (int k = 0; k < 4; k++) {
                new_max_value = fmax(temp_array[k], max_value);
                sum = sum * std::exp(max_value - new_max_value) +
                      std::exp(temp_array[k] - new_max_value);
                max_value = new_max_value;
            }
        }

        warp_level_sum_and_max(max_value, sum);
        if (thread_index_in_warp == 0) {
            max_smem[warp_idx] = max_value;
            sum_smem[warp_idx] = sum;
        }
        __syncthreads();

        if (warp_idx == 0) {
            max_value = (tid < blockDim.x / WarpSize) ? max_smem[tid] : -FLT_MAX;

            sum = (tid < blockDim.x / WarpSize) ? sum_smem[tid] : 0.0;
            warp_level_sum_and_max(max_value, sum);
        }

        if (tid == 0) {
            max_smem[0] = max_value;
            sum_smem[0] = sum;
            d_scale[index] = 1 / (sum * max_int8);
        }
        __syncthreads();
        max_value = max_smem[0];
        sum = sum_smem[0];
        __syncthreads();
        // 公式堆导:
        // max(softmax) = std::exp(max - max) / sum = 1 / sum
        // scale = max(softmax) / int8_max = 1 / (sum * int8_max);
#pragma unroll
        for (int i = 0; i < ThreadLoopCount; i++) {
            int j = tid + i * BlockSize;
            char4 q =
                make_char4(scaleandclip(std::exp(temp_value_array[i].x - max_value) * max_int8, 1,
                                        max_int8, min_int8),
                           scaleandclip(std::exp(temp_value_array[i].y - max_value) * max_int8, 1,
                                        max_int8, min_int8),
                           scaleandclip(std::exp(temp_value_array[i].z - max_value) * max_int8, 1,
                                        max_int8, min_int8),
                           scaleandclip(std::exp(temp_value_array[i].w - max_value) * max_int8, 1,
                                        max_int8, min_int8));
            // 赋值
            q_out[j] = q;
        }
    }
    return;
}

// ---------------------------------------------------------
void SoftmaxTest(int m, const int n) {
    printf("=======SoftmaxTest=======\n");
    // 1. online safe softmax：输入 [m=1024, n=4096] fp32，实现两版 —— (a) baseline 三遍
    // pass（max→sum→normalize），一个 block 处理一行；(b) online 单 pass 版（flash attention 的
    // online softmax 递推）。与 torch.softmax(dim=-1) 对齐 atol=1e-4。
    constexpr int BlockSize = 256;
    // constexpr int BlockSize = 128;

    dim3 block(BlockSize);
    dim3 grid(m);
    const size_t bytes = (size_t)m * n * sizeof(float);

    // CPU 数据
    float* h_in = (float*)malloc(bytes);
    float* h_out = (float*)malloc(bytes);
    float* h_gt = (float*)malloc(bytes);
    softmax_cpu_init(m, n, h_in, h_out, h_gt);

    // GPU 数据
    float *d_in, *d_out;
    CHECK(cudaMalloc((void**)&d_in, bytes));
    CHECK(cudaMalloc((void**)&d_out, bytes));
    CHECK(cudaMemcpy(d_in, h_in, bytes, cudaMemcpyHostToDevice));

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float ms = 0.0f;

    // ---- (a) baseline 三遍 pass ----
    for (int i = 0; i < g_warmup; i++) {
        SoftmaxBaseline<<<grid, block, (block.x / WarpSize) * sizeof(float)>>>(m, n, d_in, d_out);
    }
    cudaDeviceSynchronize();
    ms = 0.0f;
    if (g_flush_buf == nullptr) {
        // 热模式: 连续 launch 一起计时, launch 间隙被流水线掩盖
        cudaEventRecord(start);
        for (int i = 0; i < ITERS; i++) {
            SoftmaxBaseline<<<grid, block, (block.x / WarpSize) * sizeof(float)>>>(m, n, d_in,
                                                                                   d_out);
        }
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&ms, start, stop);
    } else {
        // 冷模式: 每次 launch 前冲 L2; start 事件在 memset 之后入队, memset 不计入时间
        for (int i = 0; i < ITERS; i++) {
            flush_l2();
            cudaEventRecord(start);
            SoftmaxBaseline<<<grid, block, (block.x / WarpSize) * sizeof(float)>>>(m, n, d_in,
                                                                                   d_out);
            cudaEventRecord(stop);
            cudaEventSynchronize(stop);
            float one = 0.0f;
            cudaEventElapsedTime(&one, start, stop);
            ms += one;
        }
    }
    printf("baseline(3-pass) avg = %.4f ms\n", ms / ITERS);
    // DRAM 流量: 1R+1W = 2*bytes(第 2/3 遍 pass 读的是 L2 里的输入)
    // FLOPs: max 1 + sum 3 + norm 2 = 6/元素
    print_perf("softmax baseline", 2.0 * bytes, 6.0 * m * n, ms / ITERS, 0);

    CHECK(cudaMemcpy(h_out, d_out, bytes, cudaMemcpyDeviceToHost));
    dump_bin("softmax_baseline.bin", h_out, bytes);
    float max_err = 0.0f;
    int bad = check_softmax_res(m, n, h_out, h_gt, &max_err);
    printf("baseline mismatch = %d / %d, max_abs_err = %.3e\n", bad, m * n, max_err);

    // ---- (b) online 单 pass ----
    for (int i = 0; i < g_warmup; i++) {
        SoftmaxOnline<BlockSize, N><<<grid, block>>>(m, n, d_in, d_out);
    }
    cudaDeviceSynchronize();
    ms = 0.0f;
    if (g_flush_buf == nullptr) {
        cudaEventRecord(start);
        for (int i = 0; i < ITERS; i++) {
            SoftmaxOnline<BlockSize, N><<<grid, block>>>(m, n, d_in, d_out);
        }
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&ms, start, stop);
    } else {
        for (int i = 0; i < ITERS; i++) {
            flush_l2();
            cudaEventRecord(start);
            SoftmaxOnline<BlockSize, N><<<grid, block>>>(m, n, d_in, d_out);
            cudaEventRecord(stop);
            cudaEventSynchronize(stop);
            float one = 0.0f;
            cudaEventElapsedTime(&one, start, stop);
            ms += one;
        }
    }
    printf("online(1-pass) avg = %.4f ms\n", ms / ITERS);
    // DRAM 流量: 1R+1W = 2*bytes; FLOPs: 递推摊每元素 ~2 + norm 3 = 4
    print_perf("softmax online", 2.0 * bytes, 4.0 * m * n, ms / ITERS, 0);
    g_online_ms = ms / ITERS;  // 留给融合版对比

    CHECK(cudaMemcpy(h_out, d_out, bytes, cudaMemcpyDeviceToHost));
    dump_bin("softmax_online.bin", h_out, bytes);
    bad = check_softmax_res(m, n, h_out, h_gt, &max_err);
    printf("online mismatch = %d / %d, max_abs_err = %.3e\n", bad, m * n, max_err);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    CHECK(cudaFree(d_in));
    CHECK(cudaFree(d_out));
    free(h_in);
    free(h_out);
    free(h_gt);
}

void QuantizedTest(int m, int n) {
    printf("=======QuantizedTest=======\n");
    // 2. per-token 对称 INT8 quantize kernel：输入 [m=1024, n=4096]，每行算
    // absmax→scale=absmax/127→round+clip，输出 int8 张量 + fp32 scale 数组。与 numpy reference
    // 逐元素比对（允许 ±1 个量化级差）。
    constexpr int BlockSize = 256;
    // constexpr int BlockSize = 128;

    dim3 block(BlockSize);
    dim3 grid(m);
    const size_t f_bytes = (size_t)m * n * sizeof(float);
    const size_t i8_bytes = (size_t)m * n * sizeof(int8_t);
    const size_t s_bytes = (size_t)m * sizeof(float);

    // CPU 数据
    float* h_in = (float*)malloc(f_bytes);
    int8_t* h_q = (int8_t*)malloc(i8_bytes);
    float* h_scale = (float*)malloc(s_bytes);
    int8_t* h_q_gt = (int8_t*)malloc(i8_bytes);
    float* h_scale_gt = (float*)malloc(s_bytes);
    quantize_cpu_init(m, n, h_in, h_q_gt, h_scale_gt);

    // GPU 数据
    float* d_in;
    int8_t* d_q;
    float* d_scale;
    CHECK(cudaMalloc((void**)&d_in, f_bytes));
    CHECK(cudaMalloc((void**)&d_q, i8_bytes));
    CHECK(cudaMalloc((void**)&d_scale, s_bytes));
    CHECK(cudaMemcpy(d_in, h_in, f_bytes, cudaMemcpyHostToDevice));

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float ms = 0.0f;

    for (int i = 0; i < g_warmup; i++) {
        QuantizePerTokenSymmetric<BlockSize, N>
            <<<grid, block>>>(m, n, d_in, d_scale, d_q, 127, -128);
    }
    cudaDeviceSynchronize();
    ms = 0.0f;
    if (g_flush_buf == nullptr) {
        cudaEventRecord(start);
        for (int i = 0; i < ITERS; i++) {
            QuantizePerTokenSymmetric<BlockSize, N>
                <<<grid, block>>>(m, n, d_in, d_scale, d_q, 127, -128);
        }
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&ms, start, stop);
    } else {
        for (int i = 0; i < ITERS; i++) {
            flush_l2();
            cudaEventRecord(start);
            QuantizePerTokenSymmetric<BlockSize, N>
                <<<grid, block>>>(m, n, d_in, d_scale, d_q, 127, -128);
            cudaEventRecord(stop);
            cudaEventSynchronize(stop);
            float one = 0.0f;
            cudaEventElapsedTime(&one, start, stop);
            ms += one;
        }
    }
    printf("per-token int8 quantize avg = %.4f ms\n", ms / ITERS);
    // DRAM 流量: 读 fp32 输入 + 写 int8 输出 + 每行一个 scale
    // FLOPs: fabs/fmax 归约 ~1 + div 1 + round 1 = 3/元素(取整和比较各算 1)
    print_perf("quantize", f_bytes + i8_bytes + s_bytes, 3.0 * m * n, ms / ITERS, 0);
    g_quant_ms = ms / ITERS;  // 留给融合版对比

    CHECK(cudaMemcpy(h_q, d_q, i8_bytes, cudaMemcpyDeviceToHost));
    dump_bin("quant_out.bin", h_q, i8_bytes);
    int max_err_q = 0;
    int bad_q = check_quantize_res(m, n, h_q, h_q_gt, &max_err_q);
    printf("quantize mismatch = %d / %d, max_err = %d\n", bad_q, m * n, max_err_q);
    CHECK(cudaMemcpy(h_scale, d_scale, s_bytes, cudaMemcpyDeviceToHost));
    dump_bin("scale_out.bin", h_scale, s_bytes);
    float max_err_s = 0.0f;
    int bad_s = check_scale_res(m, h_scale, h_scale_gt, &max_err_s);
    printf("scale mismatch = %d / %d, max_abs_err = %.3e\n", bad_s, m, max_err_s);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    CHECK(cudaFree(d_in));
    CHECK(cudaFree(d_q));
    CHECK(cudaFree(d_scale));
    free(h_in);
    free(h_q);
    free(h_scale);
    free(h_q_gt);
    free(h_scale_gt);
}

void GemvTest(int m, int n) {
    printf("=======GemvTest=======\n");
    // 3. gemv（W4A16 之前的热身）：y = A · x，A shape [4096, 4096] fp16，x [4096]，一个 warp/block
    // 负责若干行，用 float4/half2 向量化访存。与 torch.mv 对齐 rtol=1e-2。
    constexpr int BlockSize = 256;
    // constexpr int BlockSize = 128;
    dim3 block(BlockSize);
    dim3 grid(m);
    const size_t a_bytes = (size_t)m * n * sizeof(half);
    const size_t x_bytes = (size_t)n * sizeof(half);
    const size_t y_bytes = (size_t)m * sizeof(half);

    // CPU 数据
    half* h_A = (half*)malloc(a_bytes);
    half* h_x = (half*)malloc(x_bytes);
    half* h_y = (half*)malloc(y_bytes);
    float* h_gt = (float*)malloc((size_t)m * sizeof(float));
    gemv_cpu_init(m, n, h_A, h_x, h_gt);

    // GPU 数据
    half *d_A, *d_x, *d_y;
    CHECK(cudaMalloc((void**)&d_A, a_bytes));
    CHECK(cudaMalloc((void**)&d_x, x_bytes));
    CHECK(cudaMalloc((void**)&d_y, y_bytes));
    CHECK(cudaMemcpy(d_A, h_A, a_bytes, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_x, h_x, x_bytes, cudaMemcpyHostToDevice));

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float ms = 0.0f;

    // 一个 block 负责一行, block = 256(8 个 warp) -> grid = m / 8
    for (int i = 0; i < g_warmup; i++) {
        GemvHalf<BlockSize><<<grid, block>>>(m, n, d_A, d_x, d_y);
    }
    cudaDeviceSynchronize();
    ms = 0.0f;
    if (g_flush_buf == nullptr) {
        cudaEventRecord(start);
        for (int i = 0; i < ITERS; i++) {
            GemvHalf<BlockSize><<<grid, block>>>(m, n, d_A, d_x, d_y);
        }
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&ms, start, stop);
    } else {
        for (int i = 0; i < ITERS; i++) {
            flush_l2();
            cudaEventRecord(start);
            GemvHalf<BlockSize><<<grid, block>>>(m, n, d_A, d_x, d_y);
            cudaEventRecord(stop);
            cudaEventSynchronize(stop);
            float one = 0.0f;
            cudaEventElapsedTime(&one, start, stop);
            ms += one;
        }
    }
    printf("gemv [%d, %d] fp16 avg = %.4f ms\n", m, n, ms / ITERS);
    // DRAM 流量: 读矩阵 + 读向量 + 写结果; FLOPs: MAC = mul+add = 2*m*n
    print_perf("gemv", a_bytes + x_bytes + y_bytes, 2.0 * m * n, ms / ITERS, 1);

    CHECK(cudaMemcpy(h_y, d_y, y_bytes, cudaMemcpyDeviceToHost));
    dump_bin("gemv_y.bin", h_y, y_bytes);
    float max_abs_err = 0.0f, max_rel_err = 0.0f;
    int bad = check_gemv_res(m, h_y, h_gt, &max_abs_err, &max_rel_err);
    printf("gemv mismatch = %d / %d, max_abs_err = %.3e, max_rel_err = %.3e\n", bad, m, max_abs_err,
           max_rel_err);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    CHECK(cudaFree(d_A));
    CHECK(cudaFree(d_x));
    CHECK(cudaFree(d_y));
    free(h_A);
    free(h_x);
    free(h_y);
    free(h_gt);
}

void FuseSoftamxAndQuantTest(int m, int n) {
    printf("=======FuseSoftmaxAndQuantTest=======\n");
    // 可选加分：把 quantize kernel 融合进 softmax 的输出端（softmax→int8
    // 一次写回），对比融合前后的总耗时，说明省掉了几次 HBM 往返。
    constexpr int BlockSize = 256;
    // constexpr int BlockSize = 128;

    dim3 block(BlockSize);
    dim3 grid(m);
    const size_t f_bytes = (size_t)m * n * sizeof(float);
    const size_t i8_bytes = (size_t)m * n * sizeof(int8_t);
    const size_t s_bytes = (size_t)m * sizeof(float);

    // CPU 数据
    float* h_in = (float*)malloc(f_bytes);
    int8_t* h_q = (int8_t*)malloc(i8_bytes);
    float* h_scale = (float*)malloc(s_bytes);
    int8_t* h_q_gt = (int8_t*)malloc(i8_bytes);
    float* h_scale_gt = (float*)malloc(s_bytes);
    fuse_cpu_init(m, n, h_in, h_q_gt, h_scale_gt);

    // GPU 数据
    float* d_in;
    int8_t* d_q;
    float* d_scale;
    CHECK(cudaMalloc((void**)&d_in, f_bytes));
    CHECK(cudaMalloc((void**)&d_q, i8_bytes));
    CHECK(cudaMalloc((void**)&d_scale, s_bytes));
    CHECK(cudaMemcpy(d_in, h_in, f_bytes, cudaMemcpyHostToDevice));

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float ms = 0.0f;
    for (int i = 0; i < g_warmup; i++) {
        FuseSoftmaxQuant<BlockSize, N>
            <<<grid, block>>>(m, n, d_in, d_scale, d_q, INT8_MAX, INT8_MIN);
    }
    cudaDeviceSynchronize();
    ms = 0.0f;
    if (g_flush_buf == nullptr) {
        cudaEventRecord(start);
        for (int i = 0; i < ITERS; i++) {
            FuseSoftmaxQuant<BlockSize, N>
                <<<grid, block>>>(m, n, d_in, d_scale, d_q, INT8_MAX, INT8_MIN);
        }
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&ms, start, stop);
    } else {
        for (int i = 0; i < ITERS; i++) {
            flush_l2();
            cudaEventRecord(start);
            FuseSoftmaxQuant<BlockSize, N>
                <<<grid, block>>>(m, n, d_in, d_scale, d_q, INT8_MAX, INT8_MIN);

            cudaEventRecord(stop);
            cudaEventSynchronize(stop);
            float one = 0.0f;
            cudaEventElapsedTime(&one, start, stop);
            ms += one;
        }
    }
    // 融合后 DRAM 流量: 1R(fp32) + 1W(int8) + scale
    // 不融合: softmax(1R+1W fp32) + quantize(再 1R fp32 + 1W int8), 多一次 fp32 写+读
    float fused_ms = ms / ITERS;
    printf("fused softmax+quant avg = %.4f ms\n", fused_ms);
    // FLOPs: softmax 递推+归一化 4 + quant 除法取整 2 = 6/元素
    print_perf("fused softmax+quant", f_bytes + i8_bytes + s_bytes, 6.0 * m * n, fused_ms, 0);

    CHECK(cudaMemcpy(h_q, d_q, i8_bytes, cudaMemcpyDeviceToHost));
    dump_bin("fused_quant_out.bin", h_q, i8_bytes);
    int max_err_q = 0;
    int bad_q = check_quantize_res(m, n, h_q, h_q_gt, &max_err_q);
    printf("fused quantize mismatch = %d / %d, max_err = %d\n", bad_q, m * n, max_err_q);
    CHECK(cudaMemcpy(h_scale, d_scale, s_bytes, cudaMemcpyDeviceToHost));
    dump_bin("fused_scale_out.bin", h_scale, s_bytes);
    float max_err_s = 0.0f;
    int bad_s = check_scale_res(m, h_scale, h_scale_gt, &max_err_s);
    printf("fused scale mismatch = %d / %d, max_abs_err = %.3e\n", bad_s, m, max_err_s);

    // 对比: 融合前 = online softmax + 独立 quantize 两段耗时之和
    printf("unfused(online+quant) = %.4f ms, fused = %.4f ms, speedup = %.2fx\n",
           g_online_ms + g_quant_ms, fused_ms, (g_online_ms + g_quant_ms) / fused_ms);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    CHECK(cudaFree(d_in));
    CHECK(cudaFree(d_q));
    CHECK(cudaFree(d_scale));
    free(h_in);
    free(h_q);
    free(h_scale);
    free(h_q_gt);
    free(h_scale_gt);
}

int main(int argc, char** argv) {
    init_peak_info();
    // 用法: ./17_cuda_finial [--dump] [--cold]
    //   --dump: 把 GPU 输出写成 bin, 之后跑 compare_ref.py 出对齐表
    //   --cold: 跳过预热, 每次计时 launch 前 memset 冲掉 L2, 测真实 DRAM 带宽
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--dump") == 0) {
            g_dump_bin = 1;
        } else if (strcmp(argv[i], "--cold") == 0) {
            g_warmup = 0;
            CHECK(cudaMalloc((void**)&g_flush_buf, g_flush_bytes));
            CHECK(cudaMemset(g_flush_buf, 0, g_flush_bytes));
            cudaDeviceSynchronize();
        }
    }

    //----------------------
    SoftmaxTest(M, N);
    //----------------------
    QuantizedTest(M, N);
    //----------------------
    GemvTest(N, N);
    //----------------------
    FuseSoftamxAndQuantTest(M, N);
    if (g_flush_buf != nullptr) {
        CHECK(cudaFree(g_flush_buf));
    }
    return 0;
}