// warp shuffle
#include <bits/stdc++.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define WarpSize 32

__device__ float WarplevelMax(float value) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        float temp = __shfl_down_sync(0xffffffff, value, offset);
        value = (value > temp) ? value : temp;
    }
    return value;
}

__device__ float WarplevelMin(float value) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        float temp = __shfl_down_sync(0xffffffff, value, offset);
        value = (value < temp) ? value : temp;
    }
    return value;
}

// 全部归约到0
__device__ float WarpLevelShffle(float sum) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    }
    return sum;
}

__global__ void BlockLevelShffle(float *d_in, float *d_out, size_t N) {
    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    int total_thread_num = blockDim.x * gridDim.x;

    // warp size 的归约
    __shared__ float smem[WarpSize];
    float sum = (gid < N) ? d_in[gid] : 0;

    sum = WarpLevelShffle(sum);

    int thread_warp_idx = tid % 32;
    // block 内的warp index
    int warp_idx = tid / 32;

    if (thread_warp_idx == 0) {
        smem[warp_idx] = sum;
    }
    __syncthreads();

    // 第一个warp 来操作所有的
    if (warp_idx == 0) {
        //* 给第一个warp 中每个线程的 sum 都赋上值
        sum = (thread_warp_idx < blockDim.x / WarpSize) ? smem[thread_warp_idx] : 0.0f;
        sum = WarpLevelShffle(sum);
    }
    if (tid == 0) {
        d_out[blockIdx.x] = sum;
    }

    return;
}

// latency: 1.254ms
// *一个warp 内的递推
// ? 深究全是问题
__device__ float WarpShuffle(float sum) {
    // __shfl_down_sync：前面的thread向后面的thread要数据
    // __shfl_up_sync: 后面的thread向前面的thread要数据
    // 1. 返回前面的thread向后面的thread要的数据，比如__shfl_down_sync(0xffffffff, sum,
    // 16)那就是返回16号线程，17号线程的数据
    // 2. 使用warp shuffle指令的数据交换不会出现warp在shared
    // memory上交换数据时的不一致现象，这一点是由GPU driver完成，故无需任何sync, 比如syncwarp
    // 3. 原先15-19行有5个if判断block size的大小，目前已经被移除，确认了一下__shfl_down_sync等warp
    // shuffle指令可以handle一个block或一个warp的线程数量<32，不足32会自动填充0
    //* 因为这个sync得后缀导致了 __shfl_down_sync 必须是同步得要指令 yes !
    sum += __shfl_down_sync(0xffffffff, sum, 16);  // 0-16, 1-17, 2-18, etc.
    sum += __shfl_down_sync(0xffffffff, sum, 8);   // 0-8, 1-9, 2-10, etc.
    sum += __shfl_down_sync(0xffffffff, sum, 4);   // 0-4, 1-5, 2-6, etc.
    sum += __shfl_down_sync(0xffffffff, sum, 2);   // 0-2, 1-3, 4-6, 5-7, etc.
    sum += __shfl_down_sync(0xffffffff, sum, 1);   // 0-1, 2-3, 4-5, etc.
    return sum;
}

template <int blockSize>
__global__ void reduce_warp_level(float *d_in, float *d_out, unsigned int n) {
    float sum = 0;  // 当前线程的私有寄存器，即每个线程都会拥有一个sum寄存器

    unsigned int tid = threadIdx.x;
    unsigned int gtid = blockIdx.x * blockSize + threadIdx.x;
    // 分配的线程总数
    unsigned int total_thread_num = blockSize * gridDim.x;
    // 基于v5的改进：不用显式指定一个线程处理2个元素，而是通过L30的for循环来自动确定每个线程处理的元素个数
    for (int i = gtid; i < n; i += total_thread_num) {
        sum += d_in[i];
    }

    // 用于存储每个warp 内的reduce的结果，一共有block dim / warpdim 个 warp
    __shared__ float WarpSums[blockSize / WarpSize];
    // 当前线程在其所在warp内的ID
    const int laneId = tid % WarpSize;
    // 当前线程所在warp在所有warp范围内的ID
    const int warpId = tid / WarpSize;
    // 对当前线程所在warp作warpshuffle操作，直接交换warp内线程间的寄存器数据
    // ! 硬件视角只能操作 warp， 所以注定是一起发生得
    // ! SM 调度和发射指令的对象是 warp，不是单个线程
    sum = WarpShuffle(sum);
    // * 只有lane id ==0 的时候拿到的才是真正warp 内的数值
    if (laneId == 0) {
        WarpSums[warpId] = sum;
    }
    //* 对 WarpSums 进行了写入，__shared__写入 需要做线程同步
    __syncthreads();
    // 至此，得到了每个warp的reduce sum结果
    // 接下来，再使用第一个warp(laneId=0-31)对每个warp的reduce sum结果求和
    // 首先，把warpsums存入前blockDim.x / WarpSize个线程的sum寄存器中
    // 接着，继续warpshuffle
    // * 目前正在并行sum 数值是每一个 thread都有的一个变量，针对每一个tid 拿到自己得sum
    sum = (tid < blockSize / WarpSize) ? WarpSums[laneId] : 0;
    // Final reduce using first warp
    //* 第一个warp 内的线程都会执行这个，warp 级别的原语 要进就一起进去，要不就都不进去
    //* 一句话： warp 级集体指令的参与粒度就是"warp"，`warpId == 0` 已经是最细的正确粒度；warp
    //* 内部不该再用 if 切，该用填零让它无害。
    if (warpId == 0 /**&& laneId < 8 mask 中有这个部分，但是没写属于是未定义行为 */) {
        sum = WarpShuffle(sum);
    }
    // store: 哪里来回哪里去，把reduce结果写回显存
    if (tid == 0) {
        d_out[blockIdx.x] = sum;
    }
}

bool CheckResult(float *out, float groudtruth, int n) {
    float res = 0;
    for (int i = 0; i < n; i++) {
        res += out[i];
    }
    if (res != groudtruth) {
        return false;
    }
    return true;
}

void TestMaxSize(const int N) {
    int blockSize;
    int minGridSize;

    // 自动计算使占用率最大化的 Block 大小
    cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize,
                                       reduce_warp_level<256>,  // kernel 函数指针
                                       0,                       // 动态共享内存大小
                                       0  // Block 大小上限（0 = 不限制）
    );

    int gridSize = (N + blockSize - 1) / blockSize;
    // myKernel<<<gridSize, blockSize>>>(args);
    printf("blockSize is %ld, minGridSize is %ld, gridSize is %ld\n", blockSize, minGridSize,
           gridSize);
    return;
}

int main() {
    float milliseconds = 0;
    const int N = 25600000;
    TestMaxSize(N);
    cudaSetDevice(0);
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, 0);
    const int blockSize = 256;
    int GridSize = std::min((N + 256 - 1) / 256, deviceProp.maxGridSize[0]);
    // // yy d_final
    // int GridSize = 1000;
    float *a = (float *)malloc(N * sizeof(float));
    float *d_a;
    cudaMalloc((void **)&d_a, N * sizeof(float));

    float *out = (float *)malloc((GridSize) * sizeof(float));
    float *d_out;
    cudaMalloc((void **)&d_out, (GridSize) * sizeof(float));

    // // yy d_final
    // float *final = (float *)malloc(sizeof(float));
    // float *d_final;
    // cudaMalloc((void **)&d_final, sizeof(float));

    for (int i = 0; i < N; i++) {
        a[i] = 1.0f;
    }

    float groudtruth = N * 1.0f;

    cudaMemcpy(d_a, a, N * sizeof(float), cudaMemcpyHostToDevice);

    dim3 Grid(GridSize);
    dim3 Block(blockSize);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    reduce_warp_level<blockSize><<<Grid, Block>>>(d_a, d_out, N);
    // YY 第二趟：1000 个部分和 → 1 个总和（只开 1 个 block！）
    // reduce_warp_level<blockSize><<<1, Block>>>(d_out, d_final, GridSize);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    // // yy
    // cudaMemcpy(final, d_final, sizeof(float), cudaMemcpyDeviceToHost);
    // printf("d_final is %f\n", *final);

    cudaMemcpy(out, d_out, GridSize * sizeof(float), cudaMemcpyDeviceToHost);
    printf("allcated %d blocks, data counts are %d \n", GridSize, N);

    bool is_right = CheckResult(out, groudtruth, GridSize);
    if (is_right) {
        printf("the ans is right\n");
    } else {
        printf("the ans is wrong\n");
        for (int i = 0; i < GridSize; i++) {
            printf("resPerBlock : %lf ", out[i]);
        }
        printf("\n");
        printf("groudtruth is: %f \n", groudtruth);
    }
    printf("reduce_warp_level latency = %f ms\n", milliseconds);

    cudaFree(d_a);
    cudaFree(d_out);
    // // yy
    // free(final);
    // cudaFree(d_final);
    free(a);
    free(out);
}