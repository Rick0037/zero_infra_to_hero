#include <bits/stdc++.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>

#include <cmath>
#include <numeric>

// __global__ void GridLoop(float* d_in, size_t N) {
//     int grid_loop_size = blockDim.x * gridDim.x;
//     int gid = blockDim.x * blockIdx.x + threadIdx.x;
//     float sum{0.0};
//     for (int index = gid; index < N; index += grid_loop_size) {
//         sum += d_in[index];
//     }
//     return;
// }

__global__ void reduce_v1(float* in, float* out, size_t n) {
    double res{0.0};
    for (size_t i = 0; i < n; i++) {
        res += in[i];
    }
    out[0] = res;
    return;
}

//! 说实话这个记起来感觉完全每必要,还增加自己的复杂度
template <int BlockSzie>
__global__ void reduce_v2(float* in, float* out, size_t n) {
    __shared__ float smem[BlockSzie];
    int tid = threadIdx.x;
    int gid = threadIdx.x + blockDim.x * blockIdx.x;
    //! 需要加 (gid < n)
    smem[tid] = (gid < n) ? in[gid] : 0.0;
    __syncthreads();

    // 这里是步长
    for (int step = 1; step < blockDim.x; step *= 2) {
        if ((tid & (2 * step - 1)) == 0) {
            smem[tid] += smem[tid + step];
        }
        __syncthreads();
    }

    if (tid == 0) {
        out[blockIdx.x] = smem[0];
    }
}

/** v3 block 内每次只有一半的线程启动
 */
template <int BlockSzie>
__global__ void reduce_v3(float* in, float* out, size_t n) {
    __shared__ float smem[BlockSzie];
    int tid = threadIdx.x;
    int gid = threadIdx.x + blockDim.x * blockIdx.x;
    smem[tid] = (gid < n) ? in[gid] : 0.0;

    __syncthreads();

    for (int index = BlockSzie / 2; index > 0; index >>= 1) {
        //* 每次只有一半的线程启动
        if (tid < index) {
            smem[tid] += smem[tid + index];
        }
        __syncthreads();
    }

    if (tid == 0) {
        out[blockIdx.x] = smem[0];
    }
}

/** v4 一个线程干两个活
 */
template <int BlockSzie>
__global__ void reduce_v4(float* in, float* out, size_t n) {
    int tid = threadIdx.x;
    //* 这里乘2 只是回到原来的位置
    int gid = threadIdx.x + blockDim.x * blockIdx.x * 2;

    __shared__ float smem[BlockSzie];
    // 还得在加一个block size 才能真正的开始干两个活
    smem[tid] = in[gid] + in[gid + BlockSzie];

    __syncthreads();

    for (int i = BlockSzie / 2; i > 0; i >>= 1) {
        if (tid < i) {
            smem[tid] += smem[tid + i];
        }
        __syncthreads();
    }
    if (tid == 0) {
        out[blockIdx.x] = smem[0];
    }
}

// v5 warp level
__device__ float warp_level_add(float sum) {
    for (int i = 16; i > 0; i >>= 1) {
        sum += __shfl_down_sync(0xffffffff, sum, i);
    }
    return sum;
}
template <int BlockSzie>
__global__ void reduce_v5(float* in, float* out, size_t n) {
    float sum = 0.0;
    int grid_size = blockDim.x * gridDim.x;
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    for (int i = gid; i < n; i += grid_size) {
        sum += in[i];
    }

    sum = warp_level_add(sum);

    // 有多少个warp
    __shared__ float smem[BlockSzie / 32];

    int warp_index = threadIdx.x / 32;
    int index_in_warp = threadIdx.x % 32;
    if (index_in_warp == 0) {
        smem[warp_index] = sum;
    }
    __syncthreads();

    if (warp_index == 0) {
        sum = (threadIdx.x < BlockSzie / 32) ? smem[threadIdx.x] : 0.0;
        sum = warp_level_add(sum);
    }

    if (threadIdx.x == 0) {
        out[blockIdx.x] = sum;
    }
}

void TestKernel(size_t N) {
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, 0);
    constexpr int BlockSize = 256;
    size_t blocks = (N + BlockSize - 1) / BlockSize;
    int GridSize = (int)std::min<size_t>(blocks, (size_t)deviceProp.maxGridSize[0]);

    // init cpu data
    float* h_in = (float*)malloc(N * sizeof(float));
    for (size_t i = 0; i < N; i++) {
        h_in[i] = 1.0;
    }
    float* h_out = (float*)malloc(sizeof(float));

    // init gpu data
    float *d_in, *d_middle_value, *d_out;
    cudaMalloc((void**)&d_in, N * sizeof(float));
    cudaMalloc((void**)&d_middle_value, GridSize * sizeof(float));
    cudaMalloc((void**)&d_out, sizeof(float));

    cudaMemcpy(d_in, h_in, N * sizeof(float), cudaMemcpyHostToDevice);

    float milliseconds = 0;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    //*-----多轮计时: 每轮先用大块 memset 把 L2 冲掉, memset 不在计时窗口内----
    const int iters = 10;
    size_t flush_size = 256ull * 1024 * 1024;  // 256MB > 4090D 的 72MB L2
    char* d_flush;
    cudaMalloc((void**)&d_flush, flush_size);

    for (int it = 0; it < iters; ++it) {
        cudaMemset(d_flush, 0, flush_size);
        cudaEventRecord(start);
        //*-----kernel----
        // v1 朴素
        // reduce_v1<<<1, 1>>>(d_in, d_out, N);

        // v2 交错寻址但是有bank
        reduce_v2<BlockSize><<<GridSize, BlockSize>>>(d_in, d_middle_value, N);

        // v3 sequential addressing
        // reduce_v3<BlockSize><<<GridSize, BlockSize>>>(d_in, d_middle_value, N);

        // v4 提前进行加载干一半的活动
        // reduce_v4<BlockSize / 2><<<GridSize, BlockSize / 2>>>(d_in, d_middle_value, N);

        // v5 warp level
        // reduce_v5<BlockSize><<<GridSize, BlockSize>>>(d_in, d_middle_value, N);

        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float ms = 0;
        cudaEventElapsedTime(&ms, start, stop);
        milliseconds += ms;
    }
    cudaFree(d_flush);
    milliseconds /= iters;

    //*-----出来之后在进行reduce----
    // v2 第二级交给 CPU：把每个 block 的局部和拷回主机求和
    float* h_mid = (float*)malloc(GridSize * sizeof(float));
    cudaMemcpy(h_mid, d_middle_value, GridSize * sizeof(float), cudaMemcpyDeviceToHost);
    double gpu_sum = std::accumulate(h_mid, h_mid + GridSize, 0.0);
    free(h_mid);

    // 如果第二层级交给gpu
    cudaMemcpy(h_out, d_out, sizeof(float), cudaMemcpyDeviceToHost);

    // print bandwidth
    size_t byte_size = N * 4;
    /* 测量显存带宽时,根据实际读写的数组个数, 指定下行是1*(float)N还是2*(float)N 还是 3*(float)N*/
    printf("Mem BW= %f (GB/sec)\n", byte_size / milliseconds / 1e6);

    // print value
    double cpu_sum = std::accumulate(h_in, h_in + N, 0.0);

    if (std::fabs(gpu_sum - cpu_sum) < 1e-3) {
        printf("result is true\n");
    } else {
        printf("result is false\n");
        printf("cpp sum is %lf, gpu sum is %lf \n", cpu_sum, h_out);
    }

    free(h_in);
    free(h_out);
    cudaFree(d_in);
    cudaFree(d_middle_value);
    cudaFree(d_out);

    return;
}

int main() {
    // 1024
    size_t test_a = 1024;
    // 1M
    size_t test_b = 1 * 1024 * 1024;
    // 32M
    size_t test_c = 32 * 1024 * 1024;
    // test
    printf("-----test with sum of %llu\n", test_a);
    TestKernel(test_a);
    printf("-----test with sum of %llu\n", test_b);
    TestKernel(test_b);
    printf("-----test with sum of %llu\n", test_c);
    TestKernel(test_c);

    return 0;
}
