#include <bits/stdc++.h>
#include <cuda.h>

#include "cuda_runtime.h"

template <int blockSize>
__global__ void reduce_v0(float *d_in, float *d_out) {
    // block 内的共享指针
    __shared__ float smem[blockSize];
    // 当前执行的线程
    int tid = threadIdx.x;
    // 对应在全局中的编号
    int gid = threadIdx.x + blockIdx.x * blockDim.x;
    // copy 到共享内存中
    smem[tid] = d_in[gid];
    // 涉及到对shared memory的读写最好都加上__syncthreads
    __syncthreads();

    // block 到哪里的逻辑
    for (int i = 1; i < blockDim.x; i *= 2) {
        // 注意！v0并没有wa？rp divergence，因为没有else分支，视频目前这里讲错
        // 现在的v0和v1性能大体相似
        // v0 reduce_v0 latency = 0.595120 ms
        // if (tid % (2 * i) == 0) {
        //     smem[tid] += smem[tid + i];
        // }
        // v1 reduce_v1 latency = 0.465120 ms
        if ((tid & (2 * i - 1)) == 0) {
            smem[tid] += smem[tid + i];
        }

        // * 线程执行上一句之后一起停留在这里, 有些线程当时没有执行上一句
        // * 有些执行了上一句了。一起在这里等着放行
        __syncthreads();
    }

    if (tid == 0) {
        d_out[blockIdx.x] = smem[0];
    }
    return;
}

// v0: naive版本
// latency: 3.835ms
// blockSize作为模板参数的效果主要用于静态shared memory的申请需要传入编译期常量指定大小（L10)
// template <int blockSize>
// __global__ void reduce_v0(float *d_in, float *d_out) {
//     __shared__ float smem[blockSize];
//     // 泛指当前线程在其block内的id
//     int tid = threadIdx.x;
//     // 泛指当前线程在所有block范围内的全局id
//     int gtid = blockIdx.x * blockSize + threadIdx.x;
//     // load: 每个线程加载一个元素到shared mem对应位置
//     smem[tid] = d_in[gtid];
//     // 涉及到对shared memory的读写最好都加上__syncthreads
//     __syncthreads();

//     // 每个线程在shared memory上跨index加另一个元素，直到跨度>线程数量
//     // 此时一个block对d_in这块数据的reduce sum结果保存在id为0的线程上面
//     for (int index = 1; index < blockDim.x; index *= 2) {
//         // 注意！v0并没有warp divergence，因为没有else分支，视频目前这里讲错
//         // 现在的v0和v1性能大体相似
//         //
//         v0慢的原因在于下一行使用了除余%，除余%是个非常耗时的指令，我会在下个版本对这里进一步修正
//         // 可尝试把下一行替换为`if ((tid & (2 * index - 1)) == 0) {`, 性能大概可以提升30%～50%
//         if (tid % (2 * index) == 0) {
//             smem[tid] += smem[tid + index];
//         }
//         __syncthreads();
//     }

//     // store: 哪里来回哪里去，把reduce结果写回显存
//     if (tid == 0) {
//         d_out[blockIdx.x] = smem[0];
//     }
// }
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
    const int blockDim = 256;
    // 使用 防止n/256 需要多少个block，但是为了防止除0，给出的这个
    int GridDim = std::min((N + 256 - 1) / 256, deviceProp.maxGridSize[0]);
    // int GridSize = 100000;

    float *a = (float *)malloc(N * sizeof(float));
    float *d_a;
    cudaMalloc((void **)&d_a, N * sizeof(float));

    // 每个block 有一个数值，也就是说上来先并行的把结果从改成每个block 内部分的求和
    float *out = (float *)malloc((GridDim) * sizeof(float));
    float *d_out;
    cudaMalloc((void **)&d_out, (GridDim) * sizeof(float));

    for (int i = 0; i < N; i++) {
        a[i] = 1.0f;
    }

    float groudtruth = N * 1.0f;

    cudaMemcpy(d_a, a, N * sizeof(float), cudaMemcpyHostToDevice);
    // 有默认构造参数
    dim3 Grid(GridDim, 1, 1);
    dim3 Block(blockDim, 1, 1);
    std::cout << "GridDim is " << GridDim << " blockDim is " << blockDim
              << " deviceProp.maxGridSize[0] is " << deviceProp.maxGridSize[0] << std::endl;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    reduce_v0<blockDim><<<Grid, Block>>>(d_a, d_out);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);

    cudaMemcpy(out, d_out, GridDim * sizeof(float), cudaMemcpyDeviceToHost);
    printf("allcated %d blocks, data counts are %d", GridDim, N);
    bool is_right = CheckResult(out, groudtruth, GridDim);
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
    printf("reduce_v0 latency = %f ms\n", milliseconds);

    cudaFree(d_a);
    cudaFree(d_out);
    free(a);
    free(out);
}