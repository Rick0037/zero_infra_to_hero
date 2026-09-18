#include <bits/stdc++.h>
#include <cuda.h>

#include "cuda_runtime.h"

// v4: 只有最后一个warp 的情况下去优化__syncthreads();

// reduce_v3 latency = 0.210624 ms
__device__ void WarpSharedMemReduce(volatile float *smem, int tid) {
    // CUDA不保证所有的shared memory读操作都能在写操作之前完成，因此存在竞争关系，可能导致结果错误
    // 比如smem[tid] += smem[tid + 16] => smem[0] += smem[16], smem[16] += smem[32]
    // 此时L9中smem[16]的读和写到底谁在前谁在后，这是不确定的，
    // 所以在Volta架构后最后加入中间寄存器(L11)配合syncwarp和volatile(使得不会看见其他线程更新smem上的结果)保证读写依赖
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
}

// * `template <int blockSize>` 里的`blockSize` 不是类型，也不是对象，而是一个"编译期常量占位符"
// * ——这叫非类型模板参数（non-type template parameter） 。
// * 编译时 进行模板实例化

template <int blockSize>
__global__ void reduce_v4(float *d_in, float *d_out) {
    __shared__ float smem[blockSize];

    int tid = threadIdx.x;

    int gid = blockIdx.x * blockSize * 2 + threadIdx.x;

    smem[tid] = d_in[gid] + d_in[gid + blockSize];
    __syncthreads();

    // 上来就从64开始了
    for (int index = blockSize / 2; index > 32; index >>= 1) {
        if (tid < index) {
            smem[tid] += smem[tid + index];
        }
        __syncthreads();
    }

    //   剩下最后一个warp 不要搞 __syncthreads
    if (tid < 32) {
        WarpSharedMemReduce(smem, tid);
    }

    // 最后等于0 的这个线程去干这个事情
    if (tid == 0) {
        d_out[blockIdx.x] = smem[0];
    }

    return;
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

int main() {
    float milliseconds = 0;
    // const int N = 32 * 1024 * 1024;
    const int N = 25600000;
    cudaSetDevice(0);
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, 0);
    const int blockSize = 256;
    int GridSize = std::min((N + 256 - 1) / 256, deviceProp.maxGridSize[0]);
    // int GridSize = 100000;
    float *a = (float *)malloc(N * sizeof(float));
    float *d_a;
    cudaMalloc((void **)&d_a, N * sizeof(float));

    float *out = (float *)malloc((GridSize) * sizeof(float));
    float *d_out;
    cudaMalloc((void **)&d_out, (GridSize) * sizeof(float));

    for (int i = 0; i < N; i++) {
        a[i] = 1.0f;
    }

    float groudtruth = N * 1.0f;

    cudaMemcpy(d_a, a, N * sizeof(float), cudaMemcpyHostToDevice);

    dim3 Grid(GridSize);
    //! blockSize / 2
    dim3 Block(blockSize / 2);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    reduce_v4<blockSize / 2><<<Grid, Block>>>(d_a, d_out);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);

    cudaMemcpy(out, d_out, GridSize * sizeof(float), cudaMemcpyDeviceToHost);
    printf("allcated %d blocks, data counts are %d", GridSize, N);
    bool is_right = CheckResult(out, groudtruth, GridSize);
    if (is_right) {
        printf("the ans is right\n");
    } else {
        printf("the ans is wrong\n");
        // for(int i = 0; i < GridSize;i++){
        // printf("res per block : %lf ",out[i]);
        //}
        // printf("\n");
        printf("groudtruth is: %f \n", groudtruth);
    }
    printf("reduce_v4 latency = %f ms\n", milliseconds);

    cudaFree(d_a);
    cudaFree(d_out);
    free(a);
    free(out);
}