#include <bits/stdc++.h>
#include <cuda.h>

#include "cuda_runtime.h"

// latency: 0.656ms

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
// v5：循环展开
template <int blockSize>
__device__ void BlockSharedMemReduce(float *smem) {
    //! 注意这里有等于号  >= 1024， 1024 也需要操作，这里是size
    if (blockSize >= 1024) {
        if (threadIdx.x < 512) {
            smem[threadIdx.x] += smem[threadIdx.x + 512];
        }
        __syncthreads();
    }
    if (blockSize >= 512) {
        if (threadIdx.x < 128) {
            smem[threadIdx.x] += smem[threadIdx.x + 128];
        }
        __syncthreads();
    }
    if (blockSize >= 128) {
        if (threadIdx.x < 64) {
            smem[threadIdx.x] += smem[threadIdx.x + 64];
        }
        __syncthreads();
    }
    // if (blockSize > 64) {
    //     if (threadIdx.x < 32) {
    //         smem[threadIdx.x] += smem[threadIdx.x + 32];
    //     }
    //     __syncthreads();
    // }
    // the final warp
    if (threadIdx.x < 32) {
        // WarpSharedMemReduce(smem, threadIdx.x);
        //! `volatile` 的字面意思是"易变的"——告诉编译器：
        //! 这个内存位置可能被程序代码之外的因素改变，每次访问都必须真的去读写内存，禁止缓存和优化
        //! 有点浪费 线程都在跑
        volatile float *vshm = smem;
        if (blockDim.x >= 64) {
            vshm[threadIdx.x] += vshm[threadIdx.x + 32];
        }
        vshm[threadIdx.x] += vshm[threadIdx.x + 16];
        vshm[threadIdx.x] += vshm[threadIdx.x + 8];
        vshm[threadIdx.x] += vshm[threadIdx.x + 4];
        vshm[threadIdx.x] += vshm[threadIdx.x + 2];
        vshm[threadIdx.x] += vshm[threadIdx.x + 1];
    }
}

template <int blockSize>
__global__ void reduce_v5(float *d_in, float *d_out) {
    __shared__ float smem[blockSize];
    // 泛指当前线程在其block内的id
    unsigned int tid = threadIdx.x;
    // 泛指当前线程在所有block范围内的全局id, *2代表当前block要处理2*blocksize的数据
    // ep. blocksize = 2, blockIdx.x = 1, when tid = 0, gtid = 4, gtid + blockSize = 6; when tid =
    // 1, gtid = 5, gtid + blockSize = 7 ep. blocksize = 2, blockIdx.x = 0, when tid = 0, gtid = 0,
    // gtid + blockSize = 2; when tid = 1, gtid = 1, gtid + blockSize = 3 so, we can understand L59,
    // one thread handle data located in tid and tid + blockSize
    unsigned int i = blockIdx.x * (blockDim.x * 2) + threadIdx.x;
    // load: 每个线程加载两个元素到shared mem对应位置
    smem[tid] = d_in[i] + d_in[i + blockDim.x];
    __syncthreads();
    // compute: reduce in shared mem
    BlockSharedMemReduce<blockSize>(smem);

    // store: 哪里来回哪里去，把reduce结果写回显存
    // GridSize个block内部的reduce sum已得出，保存到d_out的每个索引位置
    if (tid == 0) {
        d_out[blockIdx.x] = smem[0];
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

int main() {
    float milliseconds = 0;

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
    dim3 Block(blockSize / 2);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    reduce_v5<blockSize / 2><<<Grid, Block>>>(d_a, d_out);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);

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
    printf("reduce_v5 latency = %f ms\n", milliseconds);

    cudaFree(d_a);
    cudaFree(d_out);
    free(a);
    free(out);
}