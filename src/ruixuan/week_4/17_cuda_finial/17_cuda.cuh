#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include <algorithm>

#define CHECK(call)                                                         \
    do {                                                                    \
        const cudaError_t error_code = call;                                \
        if (error_code != cudaSuccess) {                                    \
            printf("CUDA Error:\n");                                        \
            printf("    File:       %s\n", __FILE__);                       \
            printf("    Line:       %d\n", __LINE__);                       \
            printf("    Error code: %d\n", error_code);                     \
            printf("    Error text: %s\n", cudaGetErrorString(error_code)); \
            exit(1);                                                        \
        }                                                                   \
    } while (0)

// 预热次数, 采样次数
#define WARMUP 10
#define ITERS 100

//
void softmax_cpu_init(int M, int N, float* h_in, float* h_out, float* h_gt) {
    for (int i = 0; i < M * N; i++) {
        h_in[i] = (float)(i % 10) - 5.0f;  // -5 ~ 4, 包含负数可测 max 下溢
    }

    // CPU 参考
    for (int j = 0; j < M; j++) {
        float max_v = -INFINITY, total = 0.0f;
        for (int i = 0; i < N; i++) {
            max_v = std::max(h_in[j * N + i], max_v);
        }
        for (int i = 0; i < N; i++) {
            total += std::exp(h_in[j * N + i] - max_v);
        }
        for (int i = 0; i < N; i++) {
            h_gt[j * N + i] = std::exp(h_in[j * N + i] - max_v) / total;
        }
    }
    return;
}

//
int check_softmax_res(int M, int N, float* h_out, float* h_gt) {
    int bad = 0;
    for (int i = 0; i < M * N; i++) {
        // 写成 !(diff <= atol), 写成 diff > atol 时 NaN 会被判成通过
        if (!(fabsf(h_out[i] - h_gt[i]) <= 1e-4f)) {
            bad++;
        }
    }
    return bad;
}

//
void quantize_cpu_init(int M, int N, float* h_in, int8_t* h_q, float* h_scale) {
    for (int i = 0; i < M * N; i++) {
        h_in[i] = (float)(i % 10) - 5.0f;  // -5 ~ 4
    }

    // CPU 参考: 每行 absmax -> scale = absmax / 127 -> round + clip
    for (int j = 0; j < M; j++) {
        float absmax = 0.0f;
        for (int i = 0; i < N; i++) {
            absmax = std::max(fabsf(h_in[j * N + i]), absmax);
        }
        float scale = absmax / 127.0f;  // 2^(8-1) - 1
        h_scale[j] = scale;
        for (int i = 0; i < N; i++) {
            float q = nearbyintf(h_in[j * N + i] / scale);
            q = std::min(std::max(q, -128.0f), 127.0f);
            h_q[j * N + i] = (int8_t)q;
        }
    }
    return;
}

// 量化值允许 ±1 个量化级差
int check_quantize_res(int M, int N, int8_t* h_out, int8_t* h_gt) {
    int bad = 0;
    for (int i = 0; i < M * N; i++) {
        if (abs((int)h_out[i] - (int)h_gt[i]) > 1) {
            bad++;
        }
    }
    return bad;
}

// scale 校验: 两边算的 absmax 相同, scale 应当一致, 留一点浮点余量
int check_scale_res(int M, float* h_out, float* h_gt) {
    int bad = 0;
    for (int j = 0; j < M; j++) {
        // 写成 !(diff <= tol), 写成 diff > tol 时 NaN 会被判成通过
        if (!(fabsf(h_out[j] - h_gt[j]) <= 1e-6f)) {
            bad++;
        }
    }
    return bad;
}

//
void gemv_cpu_init(int M, int N, half* h_A, half* h_x, float* h_gt) {
    for (int i = 0; i < M * N; i++) {
        h_A[i] = __float2half((float)(i % 7) - 3.0f);  // -3 ~ 3
    }
    for (int j = 0; j < N; j++) {
        h_x[j] = __float2half((float)(j % 5) - 2.0f);  // -2 ~ 2
    }

    // CPU 参考: y[i] = sum(A[i][j] * x[j]), 用 float 累加
    for (int i = 0; i < M; i++) {
        float sum = 0.0f;
        for (int j = 0; j < N; j++) {
            sum += __half2float(h_A[i * N + j]) * __half2float(h_x[j]);
        }
        h_gt[i] = sum;
    }
    return;
}

// 与 torch.mv 对齐, 允许 1% 的相对误差
int check_gemv_res(int M, half* h_out, float* h_gt) {
    int bad = 0;
    for (int i = 0; i < M; i++) {
        float out = __half2float(h_out[i]);
        // 写成 !(diff <= tol), 写成 diff > tol 时 NaN 会被判成通过
        if (!(fabsf(out - h_gt[i]) <= 1e-2f * fabsf(h_gt[i]))) {
            bad++;
        }
    }
    return bad;
}