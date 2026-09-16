#include <__clang_cuda_builtin_vars.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>

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
