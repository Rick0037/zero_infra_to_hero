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

// 运行期开关, 由 main 解析命令行参数设置:
//   --dump: 把 GPU 输出写成 bin, 供 compare_ref.py 与 torch/numpy 对齐
//   --cold: 跳过预热, 每次计时 launch 前 memset 冲掉 L2, 测真实 DRAM 带宽
int g_dump_bin = 0;
int g_warmup = WARMUP;
char* g_flush_buf = nullptr;
const size_t g_flush_bytes = 192ull * 1024ull * 1024ull;  // 明显大于 L2(4090D 72MB)

void flush_l2() {
    if (g_flush_buf != nullptr) {
        CHECK(cudaMemset(g_flush_buf, 0, g_flush_bytes));
    }
}

void dump_bin(const char* path, const void* ptr, size_t bytes) {
    if (!g_dump_bin) {
        return;
    }
    FILE* f = fopen(path, "wb");
    fwrite(ptr, 1, bytes, f);
    fclose(f);
    printf("dumped %s (%zu bytes)\n", path, bytes);
}

// 峰值参照: 启动时通过 CUDA API 查询设备参数计算, 不写死
//   带宽 = 2(DDR 双沿) x 显存频率 x 位宽/8
//   FLOPS = SM 数 x 每 SM CUDA core 数 x 2(FMA) x 主频; fp16 CUDA core 按 fp32 的 2 倍
struct PeakInfo {
    double bw_gbps;
    double fp32_gflops;
    double fp16_gflops;
};
PeakInfo g_peak;

void init_peak_info() {
    cudaDeviceProp prop;
    CHECK(cudaGetDeviceProperties(&prop, 0));
    int clock_khz = 0;
    CHECK(cudaDeviceGetAttribute(&clock_khz, cudaDevAttrClockRate, 0));

    // API 查不到每 SM 的 CUDA core 数, 按计算能力给: Ampere 及以后 128, Volta/Turing 64
    int cores_per_sm = (prop.major >= 8) ? 128 : 64;

    g_peak.bw_gbps = 2.0 * prop.memoryClockRate * 1e3 * (prop.memoryBusWidth / 8.0) / 1e9;
    g_peak.fp32_gflops = prop.multiProcessorCount * cores_per_sm * 2.0 * clock_khz * 1e3 / 1e9;
    g_peak.fp16_gflops = 2.0 * g_peak.fp32_gflops;

    printf("device: %s, SM=%d, mem=%d MHz x %d bit, clock=%d MHz\n", prop.name,
           prop.multiProcessorCount, prop.memoryClockRate / 1000, prop.memoryBusWidth,
           clock_khz / 1000);
    printf("peak: BW=%.1f GB/s, fp32=%.1f GFLOPS, fp16=%.1f GFLOPS\n", g_peak.bw_gbps,
           g_peak.fp32_gflops, g_peak.fp16_gflops);
}

// 按搬运字节数 / 算术指令数, 打印实测带宽、算力及各自占峰值百分比
// bytes: 单次调用的 DRAM 流量(字节); flops: 单次调用算术指令数; avg_ms: 单次平均耗时
// is_fp16: 算力峰值参照选 fp16 还是 fp32
void print_perf(const char* name, double bytes, double flops, double avg_ms, int is_fp16) {
    double sec = avg_ms * 1e-3;
    double gbps = bytes / sec / 1e9;
    double gflops = flops / sec / 1e9;
    double peak_flops = is_fp16 ? g_peak.fp16_gflops : g_peak.fp32_gflops;
    printf("%s: BW = %.1f GB/s (%.1f%% of peak), perf = %.1f GFLOPS (%.2f%% of peak)\n", name, gbps,
           gbps / g_peak.bw_gbps * 100.0, gflops, gflops / peak_flops * 100.0);
}

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
int check_softmax_res(int M, int N, float* h_out, float* h_gt, float* max_err) {
    int bad = 0;
    *max_err = 0.0f;
    for (int i = 0; i < M * N; i++) {
        float diff = fabsf(h_out[i] - h_gt[i]);
        if (diff > *max_err) {
            *max_err = diff;
        }
        // 写成 !(diff <= atol), 写成 diff > atol 时 NaN 会被判成通过
        if (!(diff <= 1e-4f)) {
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
int check_quantize_res(int M, int N, int8_t* h_out, int8_t* h_gt, int* max_err) {
    int bad = 0;
    *max_err = 0;
    for (int i = 0; i < M * N; i++) {
        int diff = abs((int)h_out[i] - (int)h_gt[i]);
        if (diff > *max_err) {
            *max_err = diff;
        }
        if (diff > 1) {
            bad++;
        }
    }
    return bad;
}

// scale 校验: 两边算的 absmax 相同, scale 应当一致, 留一点浮点余量
int check_scale_res(int M, float* h_out, float* h_gt, float* max_err) {
    int bad = 0;
    *max_err = 0.0f;
    for (int j = 0; j < M; j++) {
        float diff = fabsf(h_out[j] - h_gt[j]);
        if (diff > *max_err) {
            *max_err = diff;
        }
        // 写成 !(diff <= tol), 写成 diff > tol 时 NaN 会被判成通过
        if (!(diff <= 1e-6f)) {
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
int check_gemv_res(int M, half* h_out, float* h_gt, float* max_abs_err, float* max_rel_err) {
    int bad = 0;
    *max_abs_err = 0.0f;
    *max_rel_err = 0.0f;
    for (int i = 0; i < M; i++) {
        float out = __half2float(h_out[i]);
        float diff = fabsf(out - h_gt[i]);
        if (diff > *max_abs_err) {
            *max_abs_err = diff;
        }
        // gt 为 0 时相对误差没有意义, 跳过
        if (fabsf(h_gt[i]) > 0.0f) {
            float rel = diff / fabsf(h_gt[i]);
            if (rel > *max_rel_err) {
                *max_rel_err = rel;
            }
        }
        // 写成 !(diff <= tol), 写成 diff > tol 时 NaN 会被判成通过
        if (!(diff <= 1e-2f * fabsf(h_gt[i]))) {
            bad++;
        }
    }
    return bad;
}