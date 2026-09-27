#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <stdio.h>

#include <cmath>

#define WarpSize 32

// V0: 朴素实现，一个线程处理一行, 要求线程数量大于n
__global__ void softmax_v0(float* in, float* out, int n, int m) {
    int gid = threadIdx.x + blockDim.x * blockIdx.x;
    if (gid >= n) {
        return;
    }

    // 行指针
    int row{gid};
    float* input_row = in + row * m;
    float* output_row = out + row * m;
    float max_val = -INFINITY;
    for (int i = 0; i < m; i++) {
        if (input_row[i] > max_val) {
            max_val = input_row[i];
        }
    }
    float sum{0.0};
    for (int i = 0; i < m; i++) {
        sum += std::exp(input_row[i] - max_val);
    }

    for (int i = 0; i < m; i++) {
        output_row[i] = std::exp(input_row[i] - max_val) / sum;
    }

    return;
}

// 一个block 处理一行, 使用block 内进行 归约 sum 以及max

template <int BlockSize>
__global__ void softmax_v1(float* in, float* out, int m, int n) {
    // 单个block内的shared memory
    __shared__ float smem[BlockSize];
    // 单个block 干多个行
    int tid = threadIdx.x;
    for (int index = blockIdx.x; index < m; index += gridDim.x) {
        float* x = in + index * n;
        float* y = out + index * n;

        float max_val = -INFINITY;
        // 单个thread 加载多个数字
        // * 同一个内存访问就是连续的
        for (int i = tid; i < n; i += blockDim.x) {
            if (x[i] > max_val) {
                max_val = x[i];
            }
        }
        // 一定在外面 在里面就炸了
        smem[tid] = max_val;
        __syncthreads();

        // 归约求最大数值
        for (int step = BlockSize / 2; step > 0; step >>= 1) {
            //! 只有一部分在工作
            if (tid < step) {
                smem[tid] = (smem[tid] > smem[tid + step]) ? smem[tid] : smem[tid + step];
                //* 在里面
            }
            //! 放在if 外面
            __syncthreads();
        }
        max_val = smem[0];
        __syncthreads();
        // 这是所有thread都拿到了 max = max_val;

        // 求归约的和,
        float sum = 0;
        for (int i = tid; i < n; i += blockDim.x) {
            sum += std::exp(x[i] - max_val);
        }
        smem[tid] = sum;
        __syncthreads();
        // block 内部归约
        for (int step = BlockSize / 2; step > 0; step >>= 1) {
            if (tid < step) {
                smem[tid] += smem[tid + step];
            }
            //! 放在if 外面
            __syncthreads();
        }
        sum = smem[0];
        __syncthreads();

        // elements wise
        for (int i = tid; i < n; i += blockDim.x) {
            y[i] = std::exp(x[i] - max_val) / sum;
        }
    }
}

// V1: 一个 Block 处理一行，Block 内并行规约
//! 如果griddim < m 就完了
// __global__ void softmax_v1(float* input, float* output, int M, int N) {
//     extern __shared__ float smem[];  // 用于规约

//     int row = blockIdx.x;
//     int tid = threadIdx.x;

//     float* x = input + row * N;
//     float* y = output + row * N;

//     // Pass 1: 并行求最大值
//     float max_val = -INFINITY;
//     for (int i = tid; i < N; i += blockDim.x) {
//         max_val = fmaxf(max_val, x[i]);
//     }
//     smem[tid] = max_val;
//     __syncthreads();

//     // Shared Memory 规约求全局最大值
//     for (int s = blockDim.x / 2; s > 0; s >>= 1) {
//         if (tid < s) {
//             smem[tid] = fmaxf(smem[tid], smem[tid + s]);
//         }
//         __syncthreads();
//     }
//     max_val = smem[0];
//     __syncthreads();

//     // Pass 2: 并行求指数和
//     float sum = 0.0f;
//     for (int i = tid; i < N; i += blockDim.x) {
//         sum += expf(x[i] - max_val);
//     }
//     smem[tid] = sum;
//     __syncthreads();

//     // Shared Memory 规约求总和
//     for (int s = blockDim.x / 2; s > 0; s >>= 1) {
//         if (tid < s) {
//             smem[tid] += smem[tid + s];
//         }
//         __syncthreads();
//     }
//     sum = smem[0];
//     __syncthreads();

//     // Pass 3: 归一化写出
//     float inv_sum = 1.0f / sum;
//     for (int i = tid; i < N; i += blockDim.x) {
//         y[i] = expf(x[i] - max_val) * inv_sum;
//     }
// }

// warp level shuff + first warp shared memory
__device__ float warp_level_max(float value) {
    for (int index = 16; index > 0; index >>= 1) {
        value = fmax(value, __shfl_down_sync(0xffffffff, value, index));
    }
    return value;
}

__device__ float warp_level_sum(float sum) {
    for (int index = 16; index > 0; index >>= 1) {
        sum += __shfl_down_sync(0xffffffff, sum, index);
    }
    return sum;
}

template <int BlockSize>
__device__ float block_level_max(float max_value) {
    int tid = threadIdx.x;
    int warp_idx = threadIdx.x / 32;
    int thread_index_in_warp = threadIdx.x % 32;
    __shared__ float smem[BlockSize / 32];

    max_value = warp_level_max(max_value);
    if (thread_index_in_warp == 0) {
        smem[warp_idx] = max_value;
    }
    __syncthreads();

    if (warp_idx == 0) {
        if (tid < BlockSize / 32) {
            max_value = smem[tid];
        }
        max_value = warp_level_max(max_value);
    }

    // 广播结果给所有线程
    __shared__ float block_result;
    if (tid == 0) block_result = max_value;
    __syncthreads();
    return block_result;
}

template <int BlockSize>
__device__ float block_level_sum(float sum, float max_value) {
    int tid = threadIdx.x;
    int warp_idx = threadIdx.x / 32;
    int thread_index_in_warp = threadIdx.x % 32;
    __shared__ float smem[BlockSize / 32];

    sum = warp_level_sum(sum);
    if (thread_index_in_warp == 0) {
        smem[warp_idx] = sum;
    }
    __syncthreads();

    if (warp_idx == 0) {
        //! 错了 没有用的也需要赋值
        // if (tid < BlockSize / 32) {
        //     sum = smem[tid];
        // }

        sum = (tid < BlockSize / 32) ? smem[tid] : 0.0;

        sum = warp_level_sum(sum);
    }

    // 广播结果给所有线程
    __shared__ float block_result;
    if (tid == 0) block_result = sum;
    __syncthreads();
    return block_result;
}

// 三次read 一次 write
template <int BlockSize>
__global__ void softmax_v2(float* in, float* out, int m, int n) {
    // 单个block 干多个行
    int tid = threadIdx.x;
    for (int index = blockIdx.x; index < m; index += gridDim.x) {
        float* x = in + index * n;
        float* y = out + index * n;
        // 一个block 内使用 多个warp 全干完
        //!!!! 先每个线程求自己的在归约
        //! 在没有线程具体的数值,之前不用考虑 warp级别的遍历,先把线程的寄存器填满
        float max_value = -INFINITY;
        for (int j = tid; j < m; j += blockDim.x) {
            max_value = fmax(max_value, x[j]);
        }

        max_value = block_level_max<BlockSize>(max_value);

        float sum = 0.0;
        // 把每个线程填满
        for (int j = tid; j < m; j += blockDim.x) {
            sum += std::exp(x[j] - max_value);
        }

        sum = block_level_sum<BlockSize>(sum, max_value);

        for (int j = tid; j < m; j += blockDim.x) {
            y[j] = std::exp(x[j] - max_value) / sum;
        }
    }
}

// 使用向量化 加载在加一波指令的简化
// * 如果不能被float4整除的化 还得想想办法,加一个收尾得工作
// * float = 4 byte   float4 = 16 byte  = 16 * 8 bit = 一次内存事务
template <int BlockSize>
__global__ void softmax_v3(float* in, float* out, int m, int n) {
    int tid = threadIdx.x;
    for (int index = blockIdx.x; index < m; index += gridDim.x) {
        // 直接把一行 搞成float4的一行了
        float4* x = reinterpret_cast<float4*>(in + index * n);
        float4* y = reinterpret_cast<float4*>(out + index * n);

        float max_value = -INFINITY;

        for (int j = tid; j < n / 4; j += blockDim.x) {
            max_value = fmax(max_value, x[j].w);
            max_value = fmax(max_value, x[j].x);
            max_value = fmax(max_value, x[j].y);
            max_value = fmax(max_value, x[j].z);
        }

        max_value = block_level_max<BlockSize>(max_value);

        //
        float sum{0.0};
        for (int j = tid; j < n / 4; j += blockDim.x) {
            sum += std::exp(x[j].w - max_value);
            sum += std::exp(x[j].x - max_value);
            sum += std::exp(x[j].y - max_value);
            sum += std::exp(x[j].z - max_value);
        }
        sum = block_level_sum<BlockSize>(sum, max_value);

        for (int j = tid; j < n / 4; j += blockDim.x) {
            float4 temp_res;
            temp_res.x = std::exp(x[j].x - max_value) / sum;
            temp_res.y = std::exp(x[j].y - max_value) / sum;
            temp_res.z = std::exp(x[j].z - max_value) / sum;
            temp_res.w = std::exp(x[j].w - max_value) / sum;
            y[j] = temp_res;
        }
    }
}

// 先办法 搞一个中间的sharememory 去把 exp(xi - max)的数值保存一下,后面就不同算了
//! 但是实际上shared memory 支撑不了太大的这个中间结果的计算
template <int BlockSize>
__global__ void softmax_v4(float* in, float* out, int m, int n) {
    int tid = threadIdx.x;
    for (int index = blockIdx.x; index < m; index += gridDim.x) {
        // 直接把一行 搞成float4的一行了
        float4* x = reinterpret_cast<float4*>(in + index * n);
        float4* y = reinterpret_cast<float4*>(out + index * n);

        float max_value = -INFINITY;
        for (int j = tid; j < n / 4; j += blockDim.x) {
            max_value = fmax(max_value, x[j].w);
            max_value = fmax(max_value, x[j].x);
            max_value = fmax(max_value, x[j].y);
            max_value = fmax(max_value, x[j].z);
        }

        max_value = block_level_max<BlockSize>(max_value);

        //
        float sum{0.0};
        extern __shared__ float4 smem4[];  // 大小 = N（暂存 exp 值）
        for (int j = tid; j < n / 4; j += blockDim.x) {
            smem4[j].w = std::exp(x[j].w - max_value);
            smem4[j].x = std::exp(x[j].x - max_value);
            smem4[j].y = std::exp(x[j].y - max_value);
            smem4[j].z = std::exp(x[j].z - max_value);
            sum = sum + smem4[j].w + smem4[j].x + smem4[j].y + smem4[j].z;
        }
        sum = block_level_sum<BlockSize>(sum, max_value);

        for (int j = tid; j < n / 4; j += blockDim.x) {
            float4 temp_res;
            temp_res.x = smem4[j].x / sum;
            temp_res.y = smem4[j].y / sum;
            temp_res.z = smem4[j].z / sum;
            temp_res.w = smem4[j].w / sum;
            y[j] = temp_res;
        }
    }
}

int main() {
    // n 行, 每行 m 列
    const int n = 4096;
    const int m = 4096;
    const int N = n * m;

    // CPU 数据
    float* h_in = (float*)malloc(N * sizeof(float));
    float* h_out = (float*)malloc(N * sizeof(float));
    float* h_gt = (float*)malloc(N * sizeof(float));
    for (int i = 0; i < N; i++) {
        h_in[i] = (float)(i % 10) - 5.0f;  // -5 ~ 4, 包含负数可测 max 下溢
    }

    // CPU 参考
    for (int j = 0; j < n; j++) {
        float max_v = -INFINITY, total = 0.0f;
        for (int i = 0; i < m; i++) {
            max_v = std::max(h_in[j * m + i], max_v);
        }
        for (int i = 0; i < m; i++) {
            total += std::exp(h_in[j * m + i] - max_v);
        }
        for (int i = 0; i < m; i++) {
            h_gt[j * m + i] = std::exp(h_in[j * m + i] - max_v) / total;
        }
    }

    // GPU 数据
    float *d_in, *d_out;
    cudaMalloc((void**)&d_in, N * sizeof(float));
    cudaMalloc((void**)&d_out, N * sizeof(float));
    cudaMemcpy(d_in, h_in, N * sizeof(float), cudaMemcpyHostToDevice);

    // 一个线程一行, 线程数 >= n
    const int block = 256;
    const int grid = (n + block - 1) / block;

    //*-----耗时测试: 多轮计时取平均----
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    const int iters = 10;
    float milliseconds = 0;

    //*-----预热: 先跑几轮不计时, 排除 JIT/频率爬升的影响----
    for (int it = 0; it < 3; ++it) {
        softmax_v4<block><<<n, block, (m / 4) * sizeof(float4)>>>(d_in, d_out, n, m);
    }
    cudaDeviceSynchronize();

    //*-----L2 冲刷: 每轮计时前用 256MB memset 把 72MB L2 逐出, memset 不计时----
    size_t flush_size = 256ull * 1024 * 1024;
    char* d_flush;
    cudaMalloc((void**)&d_flush, flush_size);

    for (int it = 0; it < iters; ++it) {
        cudaMemset(d_flush, 0, flush_size);
        cudaEventRecord(start);
        softmax_v0<<<grid, block>>>(d_in, d_out, n, m);
        // softmax_v1<block><<<grid, block>>>(d_in, d_out, n, m);
        // softmax_v1<block><<<n, block>>>(d_in, d_out, n, m);
        // softmax_v2<block><<<n, block>>>(d_in, d_out, n, m);
        // softmax_v3<block><<<n, block>>>(d_in, d_out, n, m);
        // softmax_v4<block><<<n, block, (m / 4) * sizeof(float4)>>>(d_in, d_out, n, m);

        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float ms = 0;
        cudaEventElapsedTime(&ms, start, stop);
        milliseconds += ms;
    }
    cudaFree(d_flush);
    milliseconds /= iters;
    printf("softmax_v0 latency = %f ms\n", milliseconds);

    cudaMemcpy(h_out, d_out, N * sizeof(float), cudaMemcpyDeviceToHost);

    // 校验: 每个元素与 CPU 参考的相对误差 < 1e-5, 且每行和为 1
    bool ok = true;
    for (int i = 0; i < N; i++) {
        if (std::fabs(h_out[i] - h_gt[i]) > 1e-5f * std::fabs(h_gt[i])) {
            ok = false;
            printf("mismatch at [%d]: gpu=%f cpu=%f\n", i, h_out[i], h_gt[i]);
            break;
        }
    }
    if (ok) {
        printf("softmax_v0 result is right\n");
    } else {
        printf("softmax_v0 result is wrong\n");
    }

    cudaFree(d_in);
    cudaFree(d_out);
    free(h_in);
    free(h_out);
    free(h_gt);
    // softmax_v0 latency = 1.824154 ms
    // softmax_v1 latency = 1.797734 ms (grid size = 16)
    // softmax_v1 latency = 0.155293 ms (grid size = 4096)
    // softmax_v2 latency = 0.154784 ms (grid size = 4096)
    // softmax_v3 latency = 0.147971 ms (grid size = 4096)
    // softmax_v4 latency = 0.145603 ms (grid size = 4096)
    return 0;
}