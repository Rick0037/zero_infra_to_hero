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

//* fp16 CUDA Core 峰值: 一条 __hfma2 同时对 2 个 half 做乘加 = 4 FLOPs
__global__ void FP16FLOPS(int *start, int *stop, half2 *x, half2 *y, half2 *result) {
    int gtid = blockDim.x * blockIdx.x + threadIdx.x;
    half2 d1 = x[gtid];
    half2 d2 = y[gtid];
    half2 res = __float2half2_rn(0.0f);  //! 必须显式清零, 否则栈上垃圾导致 inf/nan
    int start_time = 0;
    asm volatile("mov.u32 %0, %%clock;" : "=r"(start_time)::"memory");
    for (int i = 0; i < LOOP_TIMES; i++) {
        //! 必须用 __hfma2 单指令 FMA; res += d1*d2 会退化成 HMUL2+HADD2 两条指令
        res = __hfma2(d1, d2, res);
        res = __hfma2(d1, d2, res);
        res = __hfma2(d1, d2, res);
        res = __hfma2(d1, d2, res);
    }
    asm volatile("bar.sync 0;");
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
    //* props.clockRate 的单位不是 Hz，而是 kHz
    printf("  GPU Max Clock rate: %0.2f GHz\n", props.clockRate * 1e-6f);
    printf(" SM counts is %d \n", props.multiProcessorCount);
    printf("actual %s fp32 peak FLOPS is %f (TFLOPS) \n", props.name,
           FLOPS * props.clockRate * 1e-9 * props.multiProcessorCount);
    free(x);
    free(y);
    cudaFree(d_x);
    cudaFree(d_y);
    cudaFree(d_result);
    //! startClock/stopClock 和 d_startClock/d_stopClock 还要给 fp16 段复用, 不能在这里释放

    //* ---- fp16 CUDA Core 峰值 (half2 FMA) ----
    half2 *hx = (half2 *)malloc(N * sizeof(half2));
    half2 *hy = (half2 *)malloc(N * sizeof(half2));
    half2 *d_hx;
    half2 *d_hy;
    half2 *d_hresult;
    cudaMalloc((void **)&d_hx, N * sizeof(half2));
    cudaMalloc((void **)&d_hy, N * sizeof(half2));
    cudaMalloc((void **)&d_hresult, N * sizeof(half2));
    // 填 1: 每线程累计 LOOP*4 = 4000, 不超 half 上限 65504
    for (int i = 0; i < N; i++) {
        hx[i] = __float2half2_rn(1.0f);
        hy[i] = __float2half2_rn(1.0f);
    }
    cudaMemcpy(d_hx, hx, N * sizeof(half2), cudaMemcpyHostToDevice);
    cudaMemcpy(d_hy, hy, N * sizeof(half2), cudaMemcpyHostToDevice);

    FP16FLOPS<<<1, 1024>>>(d_startClock, d_stopClock, d_hx, d_hy, d_hresult);
    cudaMemcpy(startClock, d_startClock, N * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(stopClock, d_stopClock, N * sizeof(int), cudaMemcpyDeviceToHost);

    //* 一条 __hfma2 = 2 half 乘 + 2 half 加 = 4 FLOPs
    float FP16_FLOPS =
        (LOOP_TIMES * 4 * 4 * 1024) / (static_cast<float>(stopClock[0] - startClock[0]));
    printf("actual %s fp16 (CUDA core, half2) peak FLOPS is %f (TFLOPS) \n", props.name,
           FP16_FLOPS * props.clockRate * 1e-9 * props.multiProcessorCount);

    free(hx);
    free(hy);
    free(startClock);
    free(stopClock);
    cudaFree(d_hx);
    cudaFree(d_hy);
    cudaFree(d_hresult);
    cudaFree(d_startClock);
    cudaFree(d_stopClock);
}