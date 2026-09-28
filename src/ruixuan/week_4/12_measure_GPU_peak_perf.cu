#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <stdio.h>

#define LOOP_TIMES 1000

__global__ void FP32FLOPStest(int *start, int *stop, float *x, float *y, float *result) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    float x1 = x[tid];
    float y1 = y[tid];
    float res = 0;

    int start_time = 0;

    asm volatile("mov.u32 %0, %%clock;" : "=r"(start_time)::"memory");
    for (int i = 0; i < LOOP_TIMES; i++) {
        res += x1 * y1;
        res += x1 * y1;
        res += x1 * y1;
    }
    asm volatile("bar.sync 0;");  // sync all threads
    int stop_time = 0;
    asm volatile("mov.u32 %0, %%clock;" : "=r"(stop_time)::"memory");
    start[tid] = start_time;
    stop[tid] = stop_time;

    return;
}

//* T4 fp32: 8.08 TFLOPS   -> FP32 cuda core = 64
//* 4090D fp32 : 73 TFLOPS T = 1e9 -> FP32 cuda core = 128
__global__ void FP32FLOPS(int *start, int *stop, float *x, float *y, float *result) {
    int gtid = blockDim.x * blockIdx.x + threadIdx.x;
    float d1 = x[gtid];
    float d2 = y[gtid];
    float res = 0;
    int start_time = 0;
    // only measure the computation time, eliminate the memory access time
    asm volatile("mov.u32 %0, %%clock;" : "=r"(start_time)::"memory");
    // Q1: why use 4 fma instruction to get GPU peak performance?
    // A1: we use >2(3or4) fma instruction to hide for loop comparsion and addition instruction
    // overhead Q2: why use 4 dependant fma instruction to get GPU peak performance, can we use 4
    // independant ones? A2: yes, we can use 2/3/4 independant ones
    for (int i = 0; i < LOOP_TIMES; i++) {
        // asm volatile ("{\n\t""fma.rn.f32 %0, %1, %2 , %0; \n\t"
        //                      "fma.rn.f32 %0, %1, %2 , %0; \n\t"
        //                      "fma.rn.f32 %0, %1, %2 , %0; \n\t"
        //                      "fma.rn.f32 %0, %1, %2 , %0; \n\t"
        //                      "}" : "+f"(res), "+f"(d1),"+f"(d2)); // res + d1 * d2 = res
        res = d1 * d2 + res;
        res = d1 * d2 + res;
        res = d1 * d2 + res;
        res = d1 * d2 + res;
    }
    asm volatile("bar.sync 0;");  // sync all threads

    int stop_time = 0;
    asm volatile("mov.u32 %0, %%clock;" : "=r"(stop_time)::"memory");
    start[gtid] = start_time;
    stop[gtid] = stop_time;
    result[gtid] = res;
}

int main() {
    int N = 1024;
    float *x = (float *)malloc(N * sizeof(float));
    float *y = (float *)malloc(N * sizeof(float));
    float *d_x;
    float *d_y;
    cudaMalloc((void **)&d_x, N * sizeof(float));
    cudaMalloc((void **)&d_y, N * sizeof(float));
    for (int i = 0; i < 1024; i++) {
        x[i] = static_cast<float>(i);
        y[i] = static_cast<float>(i);
    }
    cudaMemcpy(d_x, x, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, y, N * sizeof(float), cudaMemcpyHostToDevice);

    float *d_result;
    int *startClock = (int *)malloc(N * sizeof(int));
    int *stopClock = (int *)malloc(N * sizeof(int));
    int *d_startClock;
    int *d_stopClock;
    cudaMalloc((void **)&d_result, N * sizeof(float));
    cudaMalloc((void **)&d_startClock, N * sizeof(int));
    cudaMalloc((void **)&d_stopClock, N * sizeof(int));
    // confirm launch max threads of SM = 1024 to do FMA to saturate SM resource

    // * T4 芯片 只有64 cuda core,
    // * 4090 有128 个cuda core 但是最多 32 * 4 = 128 线程发射, 不一定能打满算力
    FP32FLOPS<<<1, 1024>>>(d_startClock, d_stopClock, d_x, d_y, d_result);
    cudaMemcpy(startClock, d_startClock, N * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(stopClock, d_stopClock, N * sizeof(int), cudaMemcpyDeviceToHost);

    cudaDeviceProp props;
    cudaGetDeviceProperties(&props, 0);

    int ThreadsPerSM = props.maxThreadsPerMultiProcessor;
    printf("ThreadsPerSM is %ld, maxThreadsPerBlock is %ld \n", ThreadsPerSM,
           props.maxThreadsPerBlock);
    float FLOPS = (LOOP_TIMES * 4 * 2 * 1024) / (static_cast<float>(stopClock[0] - startClock[0]));
    printf("  GPU Max Clock rate: %0.2f GHz\n", props.clockRate * 1e-6f);
    printf(" SM counts is %d \n", props.multiProcessorCount);
    printf("actual %s peak FLOPS is %f (TFLOPS) \n", props.name,
           FLOPS * props.clockRate * 1e-9 * props.multiProcessorCount);
    free(x);
    free(y);
    free(startClock);
    free(stopClock);
    cudaFree(d_x);
    cudaFree(d_y);
    cudaFree(d_result);
    cudaFree(d_startClock);
    cudaFree(d_stopClock);
}