#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>

__global__ void reduce_baseline(const int *input, int *output, size_t n) {
    // 由于只分配了1个block和thread,此时cuda程序相当于串行程序
    int sum = 0;
    // 累加
    for (size_t i = 0; i < n; ++i) {
        sum += input[i];
    }
    // 累加结果写回显存
    *output = sum;
}

// 普通的
template <int BlockSzie>
__global__ void reduce_v0(int *input, int *out, size_t N) {
    __shared__ int smem[BlockSzie];
    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + threadIdx.x;

    smem[tid] = input[gid];
    __syncthreads();

    for (int i = 1; i < blockDim.x; i *= 2) {
        if (tid % (2 * i) == 0) {
            smem[tid] += smem[tid + i];
        }
        //! 必须在外面加这个sync
        __syncthreads();
    }

    if (tid == 0) {
        out[blockIdx.x] = smem[0];
    }

    return;
}

//

// V1: strided index 方式，减少 Warp Divergence
//* 一个warp 内的还是会出现 bank conflict， 线程0 从1-2，线程16 32-33 同一个warp 访问同一个地址了
__global__ void reduce_v1(float *input, float *output, int n) {
    extern __shared__ float smem[];

    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + threadIdx.x;

    smem[tid] = (gid < n) ? input[gid] : 0.0f;
    __syncthreads();

    // 步长从 1 开始逐步翻倍，但用 strided index 映射活跃线程
    for (unsigned int s = 1; s < blockDim.x; s *= 2) {
        int index = threadIdx.x * 2 * s;
        if (index < blockDim.x) {
            //* 还是相邻的两个数相加，但是只让tid在前面的线程干活
            smem[index] += smem[index + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        output[blockIdx.x] = smem[0];
    }
}

__global__ void reduce_v2(float *input, float *output, int n) {
    extern __shared__ float smem[];

    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + threadIdx.x;

    smem[tid] = (gid < n) ? input[gid] : 0.0f;
    __syncthreads();

    for (int index = blockDim.x / 2; index > 0; index >>= 1) {
        // * 每次实际上只有index 个线程进行工作
        if (tid < index) {
            smem[tid] += smem[tid + index];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        output[blockIdx.x] = smem[0];
    }

    return;
}

// V3: 每线程处理 2 个元素，减少空闲线程
__global__ void reduce_v3(float *input, float *output, int n) {
    extern __shared__ float smem[];

    // int tid = threadIdx.x;
    // int gid = blockDim.x * blockIdx.x + threadIdx.x;
    // int grid_loop_size = blockDim.x * gridDim.x;

    // float sum{};
    // for (int i = gid; i < n; i += grid_loop_size) {
    //     sum += input[i];
    // }
    // smem[tid] = sum;
    // __syncthreads();

    int tid = threadIdx.x;
    int gid = blockDim.x * blockIdx.x * 2 + threadIdx.x;

    smem[tid] = input[gid] + input[gid + blockDim.x];
    __syncthreads();

    // 每个线程加载 2 个相距 blockDim.x 的元素并求和
    // float val = 0.0f;
    // if (gid < n) val += input[gid];
    // if (gid + blockDim.x < n) val += input[gid + blockDim.x];
    // smem[tid] = val;
    // __syncthreads();

    // 步长从大到小的规约（同 V2）
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            smem[tid] += smem[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        output[blockIdx.x] = smem[0];
    }
}

__device__ void lastadd(volatile float *smem, int tid) {
    float x = smem[tid];
    if (blockDim.x >= 64) {
        x += smem[tid + 32];
        __syncwarp();
        smem[tid] = x;
        __syncwarp();
    }

    x += smem[tid + 16];
    __syncwarp();
    smem[tid] = x;
    __syncwarp();

    x += smem[tid + 8];
    __syncwarp();
    smem[tid] = x;
    __syncwarp();

    x += smem[tid + 4];
    __syncwarp();
    smem[tid] = x;
    __syncwarp();

    x += smem[tid + 2];
    __syncwarp();
    smem[tid] = x;
    __syncwarp();

    x += smem[tid + 1];
    __syncwarp();
    smem[tid] = x;
    __syncwarp();
    // 之前v100 之前是线程安全的
    // smem[tid] += smem[tid + 32];
    // smem[tid] += smem[tid + 16];
    // smem[tid] += smem[tid + 8];
    // smem[tid] += smem[tid + 4];
    // smem[tid] += smem[tid + 2];
    // smem[tid] += smem[tid + 1];
}

//
__global__ void reduce_v4(float *input, float *output, int n) {
    extern __shared__ float smem[];

    int tid = threadIdx.x;
    int gid = blockDim.x * blockIdx.x * 2 + threadIdx.x;

    smem[tid] = input[gid] + input[gid + blockDim.x];
    __syncthreads();

    // 缩小到64的时候不做了
    for (unsigned int s = blockDim.x / 2; s > 32; s >>= 1) {
        if (tid < s) {
            smem[tid] += smem[tid + s];
        }
        __syncthreads();
    }
    if (threadIdx.x < 32) {
        lastadd(smem, tid);
    }

    if (tid == 0) {
        output[blockIdx.x] = smem[0];
    }
}

// 直接把循环拆掉
template <int blockSize>
__global__ void reduce_v5(float *input, float *output, int n) {
    extern __shared__ float smem[];

    int tid = threadIdx.x;
    int gid = blockDim.x * blockIdx.x * 2 + threadIdx.x;

    smem[tid] = input[gid] + input[gid + blockDim.x];
    __syncthreads();

    // 缩小到64的时候不做了
    // for (unsigned int s = blockDim.x / 2; s > 32; s >>= 1) {
    //     if (tid < s) {
    //         smem[tid] += smem[tid + s];
    //     }
    //     __syncthreads();
    // }

    if (blockSize >= 512) {
        // ! 必须添加tid 的判断
        if (tid < 256) {
            smem[tid] += smem[tid + 256];
        }
        __syncthreads();
    }
    if (blockSize >= 256) {
        if (tid < 128) {
            smem[tid] += smem[tid + 128];
        }
        __syncthreads();
    }
    if (blockSize >= 128) {
        if (tid < 64) {
            smem[tid] += smem[tid + 64];
        }
        __syncthreads();
    }

    if (threadIdx.x < 32) {
        lastadd(smem, tid);
    }

    if (tid == 0) {
        output[blockIdx.x] = smem[0];
    }
}

__device__ float warplevelreduce(float sum) {
    // sum += __shfl_down_sync(0xffffffff, sum, 16);
    // sum += __shfl_down_sync(0xffffffff, sum, 8);
    // sum += __shfl_down_sync(0xffffffff, sum, 4);
    // sum += __shfl_down_sync(0xffffffff, sum, 2);
    // sum += __shfl_down_sync(0xffffffff, sum, 1);

    for (int i = 16; i > 0; i >>= 1) {
        sum += __shfl_down_sync(0xffffffff, sum, i);
    }

    return sum;
}

__global__ void reduce_v6(float *input, float *output, int n) {
    // 实际上block 内的一个warp 给到一个数值中
    extern __shared__ float smem[];

    float sum = 0.0;
    int gid = threadIdx.x + blockDim.x * blockIdx.x;
    sum = input[gid];

    sum = warplevelreduce(sum);

    int warp_idx = threadIdx.x / 32;
    int index_in_warp = threadIdx.x % 32;

    if (index_in_warp == 0) {
        smem[warp_idx] = sum;
    }
    //! 别忘记
    __syncthreads();

    // 可以移动到下面取
    // sum = (threadIdx.x < blockDim.x / 32) ? smem[threadIdx.x] : 0.0f;

    if (warp_idx == 0) {
        sum = (threadIdx.x < blockDim.x / 32) ? smem[threadIdx.x] : 0.0f;
        sum = warplevelreduce(sum);
    }

    if (threadIdx.x == 0) {
        output[blockIdx.x] = sum;
    }
}

// float4 强行加载
// grid size loop
__global__ void reduce_v7(float *in, float *out, size_t N) {
    int tid = threadIdx.x;
    int gid = threadIdx.x + blockDim.x * blockIdx.x;
    int n_4 = N / 4;

    // int loop_size = (blockDim.x * gridDim.x) / 4;
    //! 不是loop size 和使用什么数字没关系
    size_t loop_size = (blockDim.x * gridDim.x);

    float4 *d_in_4 = reinterpret_cast<float4 *>(in);

    float sum{0.0};

    //* 把grid loop 想象成 一维的空间去把他们填上
    for (size_t index = gid; index < n_4; index += loop_size) {
        sum += d_in_4[index].w + d_in_4[index].x + d_in_4[index].y + d_in_4[index].z;
    }

    //! 处理 n 不是 4 的倍数时的尾部元素
    int tail_start = n_4 * 4;
    //* 整个 grid 角度的前一个 thread
    for (int idx = tail_start + blockIdx.x * blockDim.x + tid; idx < N;
         idx += gridDim.x * blockDim.x) {
        sum += in[idx];
    }

    sum = warplevelreduce(sum);
    __shared__ float smem[32];

    int warp_ids = tid / 32;
    int thread_index_in_warp = tid % 32;
    if (thread_index_in_warp == 0) {
        smem[warp_ids] = sum;
    }
    __syncthreads();

    if (warp_ids == 0) {
        sum = (tid < blockDim.x / 32 /*单个block 内有多少的warp*/) ? smem[tid] : 0.0;
        sum = warplevelreduce(sum);
    }

    if (tid == 0) {
        out[blockIdx.x] = sum;
    }

    return;
}

bool CheckResult(int *out, int groudtruth, int n) {
    if (*out != groudtruth) {
        return false;
    }
    return true;
}

int main() {
    float milliseconds = 0;
    // const int N = 32 * 1024 * 1024;
    const int N = 25600000;
    cudaSetDevice(0);
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, 0);
    // const int blockSize = 256;
    const int blockSize = 1;
    // int GridSize = std::min((N + 256 - 1) / 256, deviceProp.maxGridSize[0]);//used later
    int GridSize = 1;
    // 分配内存和显存并初始化数据
    int *a = (int *)malloc(N * sizeof(int));
    int *d_a;
    cudaMalloc((void **)&d_a, N * sizeof(int));

    int *out = (int *)malloc((GridSize) * sizeof(int));
    int *d_out;
    cudaMalloc((void **)&d_out, (GridSize) * sizeof(int));

    for (int i = 0; i < N; i++) {
        a[i] = 1;
    }

    int groudtruth = N * 1;
    // 把初始化后的数据拷贝到GPU
    cudaMemcpy(d_a, a, N * sizeof(int), cudaMemcpyHostToDevice);
    // 定义分配的block数量和threads数量
    dim3 Grid(GridSize);
    dim3 Block(blockSize);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    // 分配1个block和1个thread
    reduce_baseline<<<1, 1>>>(d_a, d_out, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    // 将结果拷回CPU并check正确性
    cudaMemcpy(out, d_out, GridSize * sizeof(int), cudaMemcpyDeviceToHost);
    printf("allcated %d blocks, data counts are %d", GridSize, N);
    bool is_right = CheckResult(out, groudtruth, GridSize);
    if (is_right) {
        printf("the ans is right\n");
    } else {
        printf("the ans is wrong\n");
        for (int i = 0; i < GridSize; i++) {
            printf("res per block : %lf ", out[i]);
        }
        printf("\n");
        printf("groudtruth is: %f \n", groudtruth);
    }
    printf("reduce_baseline latency = %f ms\n", milliseconds);

    cudaFree(d_a);
    cudaFree(d_out);
    free(a);
    free(out);
    return 0;
}