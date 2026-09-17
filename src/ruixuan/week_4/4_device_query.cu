#include <cuda.h>
#include <cuda_runtime.h>

#include <iostream>

int main() {
    int deviceCount = 0;
    // 获取当前机器的GPU数量
    cudaGetDeviceCount(&deviceCount);

    std::cout << "deviceCount is " << deviceCount << std::endl;

    // cudaError_t error_id = cudaGetDeviceCount(&deviceCount);
    // if (deviceCount == 0) {
    //     printf("There are no available device(s) that support CUDA\n");
    // } else {
    //     printf("Detected %d CUDA Capable device(s)\n", deviceCount);
    // }
    cudaSetDevice(0);
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, 0);
    printf("\nDevice %d: \"%s\"\n", 0, deviceProp.name);

    // 显存容量
    printf(
        "  Total amount of global memory:                 %.0f MBytes "
        "(%llu bytes)\n",
        static_cast<float>(deviceProp.totalGlobalMem / 1048576.0f),
        (unsigned long long)deviceProp.totalGlobalMem);
    // l2 cache
    printf("  L2 Cache Size:%d bytes\n", deviceProp.l2CacheSize);
    //* 单个block 的share memory
    printf("  Total amount of shared memory per block:       %zu bytes\n",
           deviceProp.sharedMemPerBlock);
    //* 一个sm 最大的memory
    printf("  Total shared memory per multiprocessor:        %zu bytes\n",
           deviceProp.sharedMemPerMultiprocessor);
    //* regsPerBlock -> 一个 block 能分到的寄存器总数上限（= SM 寄存器堆 64K）
    //*  约束: threads_per_block * registers_per_thread <= regsPerBlock
    printf("  Total number of registers available per block: %d\n", deviceProp.regsPerBlock);
    printf("  Warp size:                                     %d\n", deviceProp.warpSize);
    // sm 上的 线程
    printf("  Maximum number of threads per multiprocessor:  %d\n",
           deviceProp.maxThreadsPerMultiProcessor);
    printf("  Maximum number of threads per block:           %d\n", deviceProp.maxThreadsPerBlock);
    // * kernel<<<gridDim (limit by maxGridSize), blockDim (limit by maxThreadsDim)>>>(...);
    printf("  Max dimension size of a block size (x,y,z): (%d, %d, %d)\n",
           deviceProp.maxThreadsDim[0], deviceProp.maxThreadsDim[1], deviceProp.maxThreadsDim[2]);
    printf("  Max dimension size of a grid size    (x,y,z): (%d, %d, %d)\n",
           deviceProp.maxGridSize[0], deviceProp.maxGridSize[1], deviceProp.maxGridSize[2]);

    // for (int dev = 0; dev < deviceCount; ++dev) {
    //     cudaSetDevice(dev);
    //     // 初始化当前device的属性获取对象
    //     cudaDeviceProp deviceProp;
    //     cudaGetDeviceProperties(&deviceProp, dev);

    //     printf("\nDevice %d: \"%s\"\n", dev, deviceProp.name);
    //     // 显存容量
    //     printf(
    //         "  Total amount of global memory:                 %.0f MBytes "
    //         "(%llu bytes)\n",
    //         static_cast<float>(deviceProp.totalGlobalMem / 1048576.0f),
    //         (unsigned long long)deviceProp.totalGlobalMem);
    //     // 时钟频率
    //     printf(
    //         "  GPU Max Clock rate:                            %.0f MHz (%0.2f "
    //         "GHz)\n",
    //         deviceProp.clockRate * 1e-3f, deviceProp.clockRate * 1e-6f);
    //     // L2 cache大小
    //     printf("  L2 Cache Size:                                 %d bytes\n",
    //            deviceProp.l2CacheSize);
    //     // high-frequent used
    //     // 注释见每个printf内的字符串
    //     printf("  Total amount of shared memory per block:       %zu bytes\n",
    //            deviceProp.sharedMemPerBlock);
    //     printf("  Total shared memory per multiprocessor:        %zu bytes\n",
    //            deviceProp.sharedMemPerMultiprocessor);
    //     printf("  Total number of registers available per block: %d\n", deviceProp.regsPerBlock);
    //     printf("  Warp size:                                     %d\n", deviceProp.warpSize);
    //     printf("  Maximum number of threads per multiprocessor:  %d\n",
    //            deviceProp.maxThreadsPerMultiProcessor);
    //     printf("  Maximum number of threads per block:           %d\n",
    //            deviceProp.maxThreadsPerBlock);
    //     printf("  Max dimension size of a block size (x,y,z): (%d, %d, %d)\n",
    //            deviceProp.maxThreadsDim[0], deviceProp.maxThreadsDim[1],
    //            deviceProp.maxThreadsDim[2]);
    //     printf("  Max dimension size of a grid size    (x,y,z): (%d, %d, %d)\n",
    //            deviceProp.maxGridSize[0], deviceProp.maxGridSize[1], deviceProp.maxGridSize[2]);
    // }
    return 0;
}