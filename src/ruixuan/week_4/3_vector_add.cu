#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>
#include <stdio.h>

// 宏定义没用
#define GpuErrorCheck(ans) \
    { GpuAssert((ans), __FILE__, __LINE__); }
inline void GpuAssert(cudaError_t code, const char* file, int line, bool abort = true) {
    if (code != cudaSuccess) {
        fprintf(stderr, "GPU assert in %s, %s, %d \n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
    return;
}

__global__ void VectorAdd(float* x, float* y, float* z, int size) {
    // 2D grid
    int index = (gridDim.x * blockIdx.y + blockIdx.x) * blockDim.x + threadIdx.x;
    // --------------->  x
    // |
    // |
    // |   [*]
    // |
    // v
    //
    // Y

    // block_index = blockidx.y * griddim.x + blockidx.x (前两行)
    // 线程 一个维度的话
    // thread_index = block_index * block_size + thread_index.x

    // 1D grid
    // int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < size) {
        z[index] = x[index] + y[index];
    }

    return;
}

void VecAddCpu(float* x, float* y, float* z, int size) {
    for (int i = 0; i < size; i++) {
        z[i] = x[i] + y[i];
    }
    return;
}

int main() {
    int size = 10000;
    int byte_size = size * sizeof(float);

    int block_dim = 256;

    /* 2D grid */
    // 10000 / 256 开跟号原因是2维
    int s = ceil(sqrt((size + block_dim - 1.) / block_dim));
    dim3 grid_dim(s, s);

    /* 1D grid */
    // 没有开根号
    // int s = ceil((size + block_dim - 1.) / block_dim);
    // dim3 grid(s);

    // malloc
    float *device_x, *host_x;
    float *device_y, *host_y;
    float *device_z, *host_z;

    GpuErrorCheck(cudaMalloc((void**)&device_x, byte_size));
    GpuErrorCheck(cudaMalloc((void**)&device_y, byte_size));
    GpuErrorCheck(cudaMalloc((void**)&device_z, byte_size));

    float millsecond = 0;

    host_x = (float*)malloc(byte_size);
    host_y = (float*)malloc(byte_size);
    host_z = (float*)malloc(byte_size);

    for (int i = 0; i < size; i++) {
        host_x[i] = i;
        host_y[i] = i;
    }

    //
    GpuErrorCheck(cudaMemcpy(device_x, host_x, byte_size, cudaMemcpyHostToDevice));
    GpuErrorCheck(cudaMemcpy(device_y, host_y, byte_size, cudaMemcpyHostToDevice));

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    VectorAdd<<<grid_dim, block_dim>>>(device_x, device_y, device_z, size);
    GpuErrorCheck(cudaGetLastError());
    cudaEventRecord(stop);
    //* 与 cudaDeviceSynchronize 基本上一致，cpu 等待stop 结束才进行记录
    cudaEventSynchronize(stop);
    // 写入millsecond
    cudaEventElapsedTime(&millsecond, start, stop);

    //
    GpuErrorCheck(cudaMemcpy(host_z, device_z, byte_size, cudaMemcpyDeviceToHost));

    float* cpu_res = (float*)malloc(byte_size);
    VecAddCpu(host_x, host_y, cpu_res, size);

    // printf("cpu res is %lf", cpu_res);

    for (int i = 0; i < size; i++) {
        printf("cpu_res is %f, host is %f \n", cpu_res[i], host_z[i]);
        if (fabs(cpu_res[i] - host_z[i]) > 1e-5) {
            printf("cpu_res and host_z is diff in i: %d", i);
        }
    }
    printf("mem %f (gb/sec)", (float)(size * 4 * 3 / millsecond / 1e6));
    cudaFree(device_x);
    cudaFree(device_y);
    cudaFree(device_z);

    free(host_x);
    free(host_y);
    free(host_z);
    free(cpu_res);

    return 0;
}