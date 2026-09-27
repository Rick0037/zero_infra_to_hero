#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <stdio.h>

#include <cmath>

#define WarpSize 32

// online softmax 一个线程负责一行
__global__ void online_softmax_v0(float* in, float* out, int m, int n) {
    int gid = threadIdx.x + blockDim.x * blockIdx.x;
    int grid_loop_size = blockDim.x * gridDim.x;
    for (int i = gid; i < m; i += grid_loop_size) {
        // 一个线程对应一个thread
        float* x = in + i * n;
        float* y = out + i * n;

        float max_value = -INFINITY;
        float sum = 0.0;

        for (int j = 0; j < n; j++) {
            float new_max_value = fmax(max_value, x[j]);
            // if (max_value != -INFINITY) {
            //     sum *= std::exp(max_value - new_max_value);
            // }
            // * 似乎可以不考虑边界情况, 因为sum 上来是0.0 可以不考虑边界
            sum *= std::exp(max_value - new_max_value);
            sum += std::exp(x[j] - new_max_value);
            max_value = new_max_value;
        }

        for (int j = 0; j < n; j++) {
            y[j] = std::exp(x[j] - max_value) / sum;
        }
    }
    return;
}

__device__ void warp_level_online_max_sum(float& max_value, float& sum_value) {
    for (int i = 16; i > 0; i >>= 1) {
        float max_value_2 = __shfl_down_sync(0xffffffff, max_value, i);
        float sum_2 = __shfl_down_sync(0xffffffff, sum_value, i);
        // if (max_value > max_value_2) {
        // }
        //! 不能if else 搞出divergence, 直接强请要求两边都进行更新
        float max_finial = fmax(max_value_2, max_value);
        sum_value = sum_value * std::exp(max_value - max_finial) +
                    sum_2 * std::exp(max_value_2 - max_finial);
        max_value = max_finial;
    }
}

// online softmax 一个 block 处理一行的数据
// 之后使用warp 来进行处理, 之后写入share memory + block 层面 最后一次归约
template <int BlockSize>
__global__ void online_softmax_v1(float* in, float* out, int m, int n) {
    // 每个block 处理一行
    for (int i = blockIdx.x; i < m; i += gridDim.x) {
        float* x = in + i * n;
        float* y = out + i * n;

        // 每个block 内的线程先干多个活
        float sum{0.0};
        float max_value = -INFINITY;
        for (int j = threadIdx.x; j < n; j += blockDim.x) {
            //! 这里实际上满足连续访问内存的
            float value = x[j];
            float new_max_value = fmax(max_value, value);
            // * 注意要加新的数值
            sum = sum * std::exp(max_value - new_max_value) + std::exp(value - new_max_value);
            max_value = new_max_value;
        }
        // block 内部归约
        warp_level_online_max_sum(max_value, sum);

        int warp_idx = threadIdx.x / 32;
        int thread_index_in_warp = threadIdx.x % 32;

        __shared__ float smem[BlockSize / 32];
        __shared__ float smem_max[BlockSize / 32];
        if (thread_index_in_warp == 0) {
            smem[warp_idx] = sum;
            smem_max[warp_idx] = max_value;
        }
        __syncthreads();

        if (warp_idx == 0) {
            sum = threadIdx.x < (blockDim.x / 32) ? smem[threadIdx.x] : 0.0;
            //! -INFINIT
            max_value = threadIdx.x < (blockDim.x / 32) ? smem_max[threadIdx.x] : -INFINITY;
            warp_level_online_max_sum(max_value, sum);
        }

        __shared__ float max_value_shared;
        __shared__ float sum_shared;
        if (threadIdx.x == 0) {
            max_value_shared = max_value;
            sum_shared = sum;
        }
        // ! shared 读写之后加
        __syncthreads();
        for (int j = threadIdx.x; j < n; j += blockDim.x) {
            y[j] = std::exp(x[j] - max_value_shared) / sum_shared;
        }
    }
}

// 向量化进行加载
template <int BlockSize>
__global__ void online_softmax_v2(float* in, float* out, int m, int n) {
    // 只在单行上面进行向量化加载
    // 每个block 处理一行
    for (int i = blockIdx.x; i < m; i += gridDim.x) {
        float4* x = reinterpret_cast<float4*>(in + i * n);
        float4* y = reinterpret_cast<float4*>(out + i * n);

        float max_value{-INFINITY};
        float sum{0.0};
        float new_max_value{-INFINITY};
        for (int j = threadIdx.x; j < n / 4; j += blockDim.x) {
            float4 value = x[j];
            new_max_value = fmax(value.x, max_value);
            sum = sum * std::exp(max_value - new_max_value) + std::exp(value.x - new_max_value);
            //! 必须更新 value
            max_value = new_max_value;
            new_max_value = fmax(value.y, max_value);
            sum = sum * std::exp(max_value - new_max_value) + std::exp(value.y - new_max_value);
            max_value = new_max_value;

            new_max_value = fmax(value.z, max_value);
            sum = sum * std::exp(max_value - new_max_value) + std::exp(value.z - new_max_value);
            max_value = new_max_value;

            new_max_value = fmax(value.w, max_value);
            sum = sum * std::exp(max_value - new_max_value) + std::exp(value.w - new_max_value);
            max_value = new_max_value;
        }

        //! 尾部处理
        for (int i = (n / 4) * 4 + threadIdx.x; i < n; i += blockDim.x) {
            float xi = (in + i * n)[i];
            float m_new = fmaxf(max_value, xi);
            sum = sum * expf(max_value - m_new) + expf(xi - m_new);
            max_value = m_new;
        }
        // block 内部归约
        warp_level_online_max_sum(max_value, sum);

        int warp_idx = threadIdx.x / 32;
        int thread_index_in_warp = threadIdx.x % 32;

        __shared__ float smem[BlockSize / 32];
        __shared__ float smem_max[BlockSize / 32];
        if (thread_index_in_warp == 0) {
            smem[warp_idx] = sum;
            smem_max[warp_idx] = max_value;
        }
        __syncthreads();

        if (warp_idx == 0) {
            sum = threadIdx.x < (blockDim.x / 32) ? smem[threadIdx.x] : 0.0;
            //! -INFINIT
            max_value = threadIdx.x < (blockDim.x / 32) ? smem_max[threadIdx.x] : -INFINITY;
            warp_level_online_max_sum(max_value, sum);
        }

        __shared__ float max_value_shared;
        __shared__ float sum_shared;
        if (threadIdx.x == 0) {
            max_value_shared = max_value;
            sum_shared = sum;
        }
        // ! shared 读写之后加
        __syncthreads();

        float4 temp{};
        for (int j = threadIdx.x; j < n / 4; j += blockDim.x) {
            temp.x = std::exp(x[j].x - max_value_shared) / sum_shared;
            temp.y = std::exp(x[j].y - max_value_shared) / sum_shared;
            temp.z = std::exp(x[j].z - max_value_shared) / sum_shared;
            temp.w = std::exp(x[j].w - max_value_shared) / sum_shared;
            y[j] = temp;
        }
        __syncthreads();  // 确保 Shared Memory 在下一行之前被重置
    }
}

// `value_cahche[kPacks]` ：4 × float4 × 4 分量 16
// 指针`x` 、`y` （64-bit 各占 2 个）+ 索引`i` 、`j` /`k` ~6
// online 标量`max_value` 、`sum` 、`new_max_value` 3
// 归约段临时（shuffle mask/lane、warp_idx、smem 下标） ~3
// exp 计算临时 + 写回时 load slot ~4-6

// 使用临时变量寄存器 防止读取过回
template <int BlockSize, int N>
__global__ void online_softmax_v3(float* in, float* out, int m, int n) {
    // 只在单行上面进行向量化加载
    // 每个block 处理一行
    constexpr int kPacks = ((N / 4) + BlockSize - 1) / BlockSize;
    float4 value_cahche[kPacks];

    for (int i = blockIdx.x; i < m; i += gridDim.x) {
        float4* x = reinterpret_cast<float4*>(in + i * n);
        float4* y = reinterpret_cast<float4*>(out + i * n);

        float max_value{-INFINITY};
        float sum{0.0};
        float new_max_value{-INFINITY};
//! 下标 k 为编译期常量, 配合展开使 value_cahche 驻留寄存器
#pragma unroll
        for (int k = 0; k < kPacks; ++k) {
            int j = threadIdx.x + k * BlockSize;
            value_cahche[k] = x[j];
            new_max_value = fmax(value_cahche[k].x, max_value);
            sum = sum * std::exp(max_value - new_max_value) +
                  std::exp(value_cahche[k].x - new_max_value);
            max_value = new_max_value;
            new_max_value = fmax(value_cahche[k].y, max_value);
            sum = sum * std::exp(max_value - new_max_value) +
                  std::exp(value_cahche[k].y - new_max_value);
            max_value = new_max_value;

            new_max_value = fmax(value_cahche[k].z, max_value);
            sum = sum * std::exp(max_value - new_max_value) +
                  std::exp(value_cahche[k].z - new_max_value);
            max_value = new_max_value;

            new_max_value = fmax(value_cahche[k].w, max_value);
            sum = sum * std::exp(max_value - new_max_value) +
                  std::exp(value_cahche[k].w - new_max_value);
            max_value = new_max_value;
        }

        // 实际上寄存器 应该时小的 不应该直接float4 当作寄存器
        // //! 尾部处理
        // for (int i = (n / 4) * 4 + threadIdx.x; i < n; i += blockDim.x) {
        //     float xi = (in + i * n)[i];
        //     float m_new = fmaxf(max_value, xi);
        //     sum = sum * expf(max_value - m_new) + expf(xi - m_new);
        //     max_value = m_new;
        // }
        // block 内部归约
        warp_level_online_max_sum(max_value, sum);

        int warp_idx = threadIdx.x / 32;
        int thread_index_in_warp = threadIdx.x % 32;

        __shared__ float smem[BlockSize / 32];
        __shared__ float smem_max[BlockSize / 32];
        if (thread_index_in_warp == 0) {
            smem[warp_idx] = sum;
            smem_max[warp_idx] = max_value;
        }
        __syncthreads();

        if (warp_idx == 0) {
            sum = threadIdx.x < (blockDim.x / 32) ? smem[threadIdx.x] : 0.0;
            //! -INFINIT
            max_value = threadIdx.x < (blockDim.x / 32) ? smem_max[threadIdx.x] : -INFINITY;
            warp_level_online_max_sum(max_value, sum);
        }

        __shared__ float max_value_shared;
        __shared__ float sum_shared;
        if (threadIdx.x == 0) {
            max_value_shared = max_value;
            sum_shared = sum;
        }
        // ! shared 读写之后加
        __syncthreads();

        float4 temp{};
#pragma unroll
        for (int k = 0; k < kPacks; ++k) {
            int j = threadIdx.x + k * BlockSize;
            temp.x = std::exp(value_cahche[k].x - max_value_shared) / sum_shared;
            temp.y = std::exp(value_cahche[k].y - max_value_shared) / sum_shared;
            temp.z = std::exp(value_cahche[k].z - max_value_shared) / sum_shared;
            temp.w = std::exp(value_cahche[k].w - max_value_shared) / sum_shared;
            y[j] = temp;
        }
        __syncthreads();  // 确保 Shared Memory 在下一行之前被重置
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

    //*------最大gpu 利用率测试
    int num_sms;
    cudaDeviceGetAttribute(&num_sms, cudaDevAttrMultiProcessorCount, 0);
    int grid_size = fmin(n, num_sms * 4);  // 最大化 GPU 利用率
    int block_size = 256;
    // online_softmax_v4<<<grid_size, block_size>>>(d_input, d_output, M, N);

    //*-----耗时测试: 多轮计时取平均----
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    const int iters = 10;
    float milliseconds = 0;

    //*-----预热: 先跑几轮不计时, 排除 JIT/频率爬升的影响----
    for (int it = 0; it < 3; ++it) {
        // online_softmax_v0<<<grid, block>>>(d_in, d_out, n, m);
        // online_softmax_v1<block><<<grid, block>>>(d_in, d_out, n, m);
        // online_softmax_v1<block><<<n, block>>>(d_in, d_out, n, m);
        // online_softmax_v2<block><<<n, block>>>(d_in, d_out, n, m);
        online_softmax_v3<block, m><<<n, block>>>(d_in, d_out, n, m);
    }
    cudaDeviceSynchronize();

    //*-----L2 冲刷: 每轮计时前用 256MB memset 把 72MB L2 逐出, memset 不计时----
    size_t flush_size = 256ull * 1024 * 1024;
    char* d_flush;
    cudaMalloc((void**)&d_flush, flush_size);

    for (int it = 0; it < iters; ++it) {
        cudaMemset(d_flush, 0, flush_size);
        cudaEventRecord(start);
        // softmax_v0<<<grid, block>>>(d_in, d_out, n, m);
        // softmax_v1<block><<<grid, block>>>(d_in, d_out, n, m);
        // softmax_v1<block><<<n, block>>>(d_in, d_out, n, m);
        // softmax_v2<block><<<n, block>>>(d_in, d_out, n, m);
        // softmax_v3<block><<<n, block>>>(d_in, d_out, n, m);
        // softmax_v4<block><<<n, block, (m / 4) * sizeof(float4)>>>(d_in, d_out, n, m);
        // ------------------
        // online_softmax_v0<<<grid, block>>>(d_in, d_out, n, m);
        // online_softmax_v1<block><<<grid, block>>>(d_in, d_out, n, m);
        // online_softmax_v1<block><<<n, block>>>(d_in, d_out, n, m);
        // online_softmax_v2<block><<<n, block>>>(d_in, d_out, n, m);
        // online_softmax_v3<block, m><<<n, block>>>(d_in, d_out, n, m);
        online_softmax_v3<block, m><<<n, block>>>(d_in, d_out, n, m);

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
            printf("mismatch at [%d]: gpu=%e cpu=%e\n", i, h_out[i], h_gt[i]);
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
    // -------------------------------------
    // online_softmax_v0 latency = 1.500480 ms
    // online_softmax_v1 latency = 1.852535 ms (grid size = 16)
    // online_softmax_v1 latency = 0.153507 ms (grid size = 4096)
    // online_softmax_v2 latency = 0.148582 ms (grid size = 4096)
    // online_softmax_v3 latency = 0.161402 ms (没有使用寄存器 还是使用的local memory 或者stack)
    // online_softmax_v3 latency = 0.145382 ms (使用了寄存器)
    // online_softmax_v4  实际和上面v2 基本上一样就是加了一个最后的sync threads 控制
    // * 如何判断是否使用了寄存器

    /*
    nvcc -arch=sm_89 -Xptxas -v -c ./src/ruixuan/week_4/11_online_softmax.cu -o test.o
    ./src/ruixuan/week_4/11_online_softmax.cu(308): warning #177-D: variable "grid" was declared but
    never referenced const int grid = (n + block - 1) / block;
                    ^

    Remark: The warnings can be suppressed with "-diag-suppress <warning-number>"

    ptxas info    : 0 bytes gmem
    ptxas info    : Compiling entry function '_Z17online_softmax_v3ILi256ELi4096EEvPfS0_ii' for
    'sm_89' ptxas info    : Function properties for _Z17online_softmax_v3ILi256ELi4096EEvPfS0_ii 64
    bytes stack frame, 0 bytes spill stores, 0 bytes spill loads ptxas info    : Used 38 registers,
    used 1 barriers, 64 bytes cumulative stack size, 72 bytes smem, 376 bytes cmem[0] ptxas info :
    Compile time = 48.895 ms ptxas info    : Compiling entry function '_Z17online_softmax_v0PfS_ii'
    for 'sm_89' ptxas info    : Function properties for _Z17online_softmax_v0PfS_ii 0 bytes stack
    frame, 0 bytes spill stores, 0 bytes spill loads ptxas info    : Used 36 registers, used 0
    barriers, 376 bytes cmem[0] ptxas info    : Compile time = 28.801 ms
    */

    /*
     nvcc -arch=sm_89 -Xptxas -v -c ./src/ruixuan/week_4/11_online_softmax.cu -o test.o
    ./src/ruixuan/week_4/11_online_softmax.cu(314): warning #177-D: variable "grid" was declared but
    never referenced const int grid = (n + block - 1) / block;
    Remark: The warnings can be suppressed with "-diag-suppress <warning-number>"

    ptxas info    : 0 bytes gmem
    ptxas info    : Compiling entry function '_Z17online_softmax_v3ILi256ELi4096EEvPfS0_ii' for
    'sm_89' ptxas info    : Function properties for _Z17online_softmax_v3ILi256ELi4096EEvPfS0_ii 0
    bytes stack frame, 0 bytes spill stores, 0 bytes spill loads ptxas info    : Used 40 registers,
    used 1 barriers, 72 bytes smem, 376 bytes cmem[0] ptxas info    : Compile time = 76.941 ms ptxas
    info    : Compiling entry function '_Z17online_softmax_v0PfS_ii' for 'sm_89' ptxas info    :
    Function properties for _Z17online_softmax_v0PfS_ii 0 bytes stack frame, 0 bytes spill stores, 0
    bytes spill loads ptxas info    : Used 36 registers, used 0 barriers, 376 bytes cmem[0] ptxas
    info    : Compile time = 28.794 ms
    */

    return 0;
}