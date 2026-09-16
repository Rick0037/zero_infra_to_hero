#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>

__global__ void EachThreadAdd(float* x) {
    int block_id = blockIdx.x;
    int thread_id = threadIdx.x;
    int global_index = blockIdx.x * blockDim.x + threadIdx.x;
    printf("current block is %d, thread id is %d, global_index is %d\n", block_id, thread_id,
           global_index);
    x[global_index] += 1;
    return;
}

int main() {
    int n = 32;
    int nbtypes = n * sizeof(float);
    // device_x = GPU
    // host_x = CPU
    float *device_x, *host_x;
    //! 这里为什么是二级指针
    cudaError_t error_code = cudaMalloc((void**)&device_x, nbtypes);
    printf("error_code is %d", error_code);
    // malloc 之后return 的是void*
    host_x = (float*)malloc(nbtypes);
    printf("-----------------\n");
    for (int i = 0; i < n; i++) {
        host_x[i] = i;
        printf("i is %f \n", host_x[i]);
    }
    //
    cudaMemcpy(device_x, host_x, nbtypes, cudaMemcpyHostToDevice);

    EachThreadAdd<<<2, n / 2>>>(device_x);
    //
    //* 这里隐含了一个 同步等到gpu 进行返回 同步了
    cudaMemcpy(host_x, device_x, nbtypes, cudaMemcpyDeviceToHost);

    printf("--------after---------\n");
    for (int i = 0; i < n; i++) {
        printf("i is %f \n", host_x[i]);
    }

    return 0;
}
