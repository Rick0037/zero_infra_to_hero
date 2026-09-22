#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>

__global__ void hierarchical_reduce(float* input, float* output, int N) {
    __shared__ float block_sum;
    if (threadIdx.x == 0) block_sum = 0.0f;
    __syncthreads();

    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    float val = (gid < N) ? input[gid] : 0.0f;

    // 第一层：Warp 内归约（无需原子操作）
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xFFFFFFFF, val, offset);
    }

    // 第二层：每个 Warp 的 lane 0 原子加到共享内存
    // 只有 blockDim.x/32 个线程竞争 → 冲突很小
    int thread_warp_idx = threadIdx.x % 32;
    // block 内的warp index
    int warp_idx = threadIdx.x / 32;
    if (thread_warp_idx == 0) {
        atomicAdd(&block_sum, val);
    }
    __syncthreads();

    // 第三层：每个 Block 的 thread 0 原子加到全局结果
    // 只有 gridDim.x 个线程竞争（远少于总线程数）
    if (threadIdx.x == 0) {
        atomicAdd(output, block_sum);
    }
}

__global__ void privatized_histogram(int* data, int* hist, int N) {
    // 每个线程维护私有计数器（适用于 bin 数很少的情况）
    int private_hist[4] = {0, 0, 0, 0};  // 假设只有4个 bin

    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = tid; i < N; i += stride) {
        private_hist[data[i]]++;  // 纯寄存器操作，无冲突
    }

    // 最终写回（大幅减少原子操作次数）
    for (int bin = 0; bin < 4; bin++) {
        if (private_hist[bin] > 0) {
            atomicAdd(&hist[bin], private_hist[bin]);
        }
    }
}

__global__ void hello_cuda() {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    printf("block id = [ %d ], thread id = [ %d ] hello cuda\n", blockIdx.x, idx);
    return;
}

int main() {
    hello_cuda<<<3, 2>>>();  // 启动 kernel
    cudaDeviceSynchronize();
    return 0;
}
