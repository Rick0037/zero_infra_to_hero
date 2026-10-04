#include <type_traits>

#include "15_gemv.cuh"

template <typename T>
void gemvCPU(T *mat, T *vec, T *dst, int M, int N) {
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            dst[i] += mat[i * N + j] * vec[j];
        }
        if (i < 5) {
            //! __half 传给 printf 的变参不会自动转 float, 必须显式转
            printf("cpu res = %f\n", (float)dst[i]);
        }
    }
}

// half 专用参考实现: CPU 没有原生 half 运算, 逐元素转 float 后用 float 累加才是数学真值
void gemvCPUHalf(half *mat, half *vec, float *dst, int M, int N) {
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            dst[i] += __half2float(mat[i * N + j]) * __half2float(vec[j]);
        }
        if (i < 5) {
            printf("cpu res = %f\n", dst[i]);
        }
    }
}

// half 的 GPU 结果与 float 参考之间有舍入差, 不能用 !=, 用相对容差比较
bool CheckResultHalf(float *out, float *groudtruth, int M) {
    for (int i = 0; i < M; i++) {
        if (i == 0) {
            printf("1st comparsion: %f and %f \n", out[i], groudtruth[i]);
        }
        if (fabs(out[i] - groudtruth[i]) > 1e-2 * fabs(groudtruth[i])) {
            printf("%dth res is wrong: %f and %f \n", i, out[i], groudtruth[i]);
            return false;
        }
    }
    return true;
}

// float 版本: 全 1 数据下两边都是精确累加, 可以用 != 精确比较
bool CheckResult(float *out, float *groudtruth, int M) {
    for (int i = 0; i < M; i++) {
        if (i == 0) {
            printf("1st comparsion: %f and %f \n", out[i], groudtruth[i]);
        }
        if (out[i] != groudtruth[i]) {
            printf("%dth res is wrong: %f and %f \n", i, out[i], groudtruth[i]);
            return false;
        }
    }
    return true;
}

void learning_gemv_kernel(half *vec, half *d_vec, half *mat, half *d_mat, half *dst, half *d_dst) {
    constexpr int N = 2048;  // 256 * 8
    constexpr int M = 256;

    //    initialize<T>(vec, d_vec, mat, d_mat, dst, d_dst, M, N);
    vec = (half *)malloc(N * sizeof(half));
    cudaMalloc((void **)&d_vec, N * sizeof(half));

    mat = (half *)malloc(M * N * sizeof(half));
    cudaMalloc((void **)&d_mat, M * N * sizeof(half));

    dst = (half *)malloc(M * sizeof(half));
    //! dst 在 gemvCPU 里是 += 累加, 必须先清零
    memset(dst, 0, M * sizeof(half));
    cudaMalloc((void **)&d_dst, M * sizeof(half));

    for (int i = 0; i < N; i++) {
        vec[i] = (half)1;
    }
    for (int i = 0; i < N * M; i++) {
        mat[i] = (half)1;
    }

    gemvCPU(mat, vec, dst, M, N);

    cudaMemcpy(d_vec, vec, N * sizeof(half), cudaMemcpyHostToDevice);

    cudaMemcpy(d_mat, mat, M * N * sizeof(half), cudaMemcpyHostToDevice);

    constexpr int THREAD_NUMS = 256;
    constexpr int VEC_SIZE = Vec<half>::size;
    // constexpr int VECS_PER_THREAD = (N / THREAD_NUMS) / VEC_SIZE;  // 1 for half, 2 for fp32
    // // *实际上是模板类种的静态函数, 中间必须有template
    // DispatchLauncher<VECS_PER_THREAD, VEC_SIZE, THREAD_NUMS>::template launcher<T>(d_mat, d_vec,
    //    d_dst, M, N);

    dim3 Grid(M);
    dim3 Block(THREAD_NUMS);
    float milliseconds = 0;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    printf("calling\n");
    learning_gemv<THREAD_NUMS, VEC_SIZE><<<Grid, Block>>>(d_mat, d_vec, d_dst, M, N);
    cudaError_t result = cudaGetLastError();
    if (result) {
        throw std::runtime_error(std::string("[ERROR] CUDA runtime error: ") +
                                 (_cudaGetErrorEnum(result)) + " " + __FILE__ + ":" +
                                 std::to_string(__LINE__) + " \n");
    }
    printf("called\n");
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("gemv latency = %f ms\n", milliseconds);

    CHECK(cudaMemcpy(dst, d_dst, M * sizeof(half), cudaMemcpyDeviceToHost));
    bool is_right;
    //! half: 参考用 float 累加, GPU 结果逐元素转 float 后按容差比较
    float *groudtruth = (float *)malloc(sizeof(float) * M);
    memset(groudtruth, 0, sizeof(float) * M);
    gemvCPUHalf(mat, vec, groudtruth, M, N);
    float *dst_f = (float *)malloc(sizeof(float) * M);
    for (int i = 0; i < M; i++) {
        dst_f[i] = __half2float(dst[i]);
    }
    is_right = CheckResultHalf(dst_f, groudtruth, M);
    free(groudtruth);
    free(dst_f);
    if (is_right) {
        printf("the ans is right\n");
    } else {
        printf("the ans is wrong\n");
    }
    cudaFree(d_vec);
    cudaFree(d_mat);
    cudaFree(d_dst);
    free(vec);
    free(mat);
    free(dst);
}

template <typename T>
void gemv_kernel(T *vec, T *d_vec, T *mat, T *d_mat, T *dst, T *d_dst) {
    constexpr int N = 131072;  // 128MiB 矩阵, 超出 L2(72MiB); 每线程 512 标量 / 128 个 float4
    constexpr int M = 256;

    //    initialize<T>(vec, d_vec, mat, d_mat, dst, d_dst, M, N);
    vec = (T *)malloc(N * sizeof(T));
    cudaMalloc((void **)&d_vec, N * sizeof(T));

    mat = (T *)malloc(M * N * sizeof(T));
    cudaMalloc((void **)&d_mat, M * N * sizeof(T));

    dst = (T *)malloc(M * sizeof(T));
    //! dst 在 gemvCPU 里是 += 累加, 必须先清零
    memset(dst, 0, M * sizeof(T));
    cudaMalloc((void **)&d_dst, M * sizeof(T));

    for (int i = 0; i < N; i++) {
        vec[i] = (T)1;
    }
    for (int i = 0; i < N * M; i++) {
        mat[i] = (T)1;
    }

    gemvCPU(mat, vec, dst, M, N);

    cudaMemcpy(d_vec, vec, N * sizeof(T), cudaMemcpyHostToDevice);

    cudaMemcpy(d_mat, mat, M * N * sizeof(T), cudaMemcpyHostToDevice);
    constexpr int THREAD_NUMS = 256;
    constexpr int VEC_SIZE = Vec<T>::size;
    constexpr int VECS_PER_THREAD = (N / THREAD_NUMS) / VEC_SIZE;  // 1 for half, 2 for fp32
    // *实际上是模板类种的静态函数, 中间必须有template
    DispatchLauncher<VECS_PER_THREAD, VEC_SIZE, THREAD_NUMS>::template launcher<T>(d_mat, d_vec,
                                                                                   d_dst, M, N);

    //! 仅 fp32: 向量化(float4) vs 标量(float) 对照, grid/block/归约完全相同, 只改加载宽度
    if constexpr (std::is_same_v<T, float>) {
        T *d_dst_scalar = nullptr;
        cudaMalloc((void **)&d_dst_scalar, M * sizeof(T));

        constexpr int WARMUP = 10;
        constexpr int ITERS = 100;
        for (int k = 0; k < WARMUP; k++) {
            gemv<VECS_PER_THREAD, VEC_SIZE><<<M, THREAD_NUMS>>>(d_mat, d_vec, d_dst, N);
            gemv_scalar<<<M, THREAD_NUMS>>>(d_mat, d_vec, d_dst_scalar, N);
        }
        cudaDeviceSynchronize();

        cudaEvent_t vs, ve, ss, se;
        cudaEventCreate(&vs);
        cudaEventCreate(&ve);
        cudaEventCreate(&ss);
        cudaEventCreate(&se);

        cudaEventRecord(vs);
        for (int k = 0; k < ITERS; k++) {
            gemv<VECS_PER_THREAD, VEC_SIZE><<<M, THREAD_NUMS>>>(d_mat, d_vec, d_dst, N);
        }
        cudaEventRecord(ve);

        cudaEventRecord(ss);
        for (int k = 0; k < ITERS; k++) {
            gemv_scalar<<<M, THREAD_NUMS>>>(d_mat, d_vec, d_dst_scalar, N);
        }
        cudaEventRecord(se);

        cudaEventSynchronize(ve);
        cudaEventSynchronize(se);
        float t_vec = 0, t_scalar = 0;
        cudaEventElapsedTime(&t_vec, vs, ve);
        cudaEventElapsedTime(&t_scalar, ss, se);
        printf("vectorized(float4) avg latency = %f ms\n", t_vec / ITERS);
        printf("scalar(float)      avg latency = %f ms, speedup = %.2fx\n", t_scalar / ITERS,
               t_scalar / t_vec);

        // 此时 dst 里还是 CPU 参考结果, 直接校验标量版
        T *scalar_host = (T *)malloc(M * sizeof(T));
        CHECK(cudaMemcpy(scalar_host, d_dst_scalar, M * sizeof(T), cudaMemcpyDeviceToHost));
        bool scalar_right = true;
        for (int k = 0; k < M; k++) {
            if (scalar_host[k] != dst[k]) {
                printf("scalar %dth res is wrong: %f and %f\n", k, scalar_host[k], dst[k]);
                scalar_right = false;
                break;
            }
        }
        printf("scalar kernel ans is %s\n", scalar_right ? "right" : "wrong");

        //! 路线B: 一行拆 SPLIT 个 block (grid=SPLIT*M=768), occupancy 可到 100%; atomicAdd
        //! 前必须清零
        constexpr int SPLIT = 3;
        dim3 split_grid(SPLIT, M);
        for (int k = 0; k < WARMUP; k++) {
            cudaMemset(d_dst, 0, M * sizeof(T));
            gemv_split<<<split_grid, THREAD_NUMS>>>(d_mat, d_vec, d_dst, M, N);
        }
        cudaDeviceSynchronize();

        cudaEventRecord(ss);
        for (int k = 0; k < ITERS; k++) {
            cudaMemset(d_dst, 0, M * sizeof(T));  // 若分离计时可把 memset 挪出, 这里保守保证正确
            gemv_split<<<split_grid, THREAD_NUMS>>>(d_mat, d_vec, d_dst, M, N);
        }
        cudaEventRecord(se);
        cudaEventSynchronize(se);
        float t_split = 0;
        cudaEventElapsedTime(&t_split, ss, se);
        printf("split(3 blocks/row) avg latency = %f ms, vs float4 speedup = %.2fx\n",
               t_split / ITERS, t_vec / t_split);

        T *split_host = (T *)malloc(M * sizeof(T));
        CHECK(cudaMemcpy(split_host, d_dst, M * sizeof(T), cudaMemcpyDeviceToHost));
        bool split_right = true;
        for (int k = 0; k < M; k++) {
            if (split_host[k] != dst[k]) {
                printf("split %dth res is wrong: %f and %f\n", k, split_host[k], dst[k]);
                split_right = false;
                break;
            }
        }
        printf("split kernel ans is %s\n", split_right ? "right" : "wrong");

        free(split_host);
        free(scalar_host);
        cudaFree(d_dst_scalar);
        cudaEventDestroy(vs);
        cudaEventDestroy(ve);
        cudaEventDestroy(ss);
        cudaEventDestroy(se);
    }

    CHECK(cudaMemcpy(dst, d_dst, M * sizeof(T), cudaMemcpyDeviceToHost));
    bool is_right;
    if constexpr (std::is_same_v<T, half>) {
        //! half: 参考用 float 累加, GPU 结果逐元素转 float 后按容差比较
        float *groudtruth = (float *)malloc(sizeof(float) * M);
        memset(groudtruth, 0, sizeof(float) * M);
        gemvCPUHalf(mat, vec, groudtruth, M, N);
        float *dst_f = (float *)malloc(sizeof(float) * M);
        for (int i = 0; i < M; i++) {
            dst_f[i] = __half2float(dst[i]);
        }
        is_right = CheckResultHalf(dst_f, groudtruth, M);
        free(groudtruth);
        free(dst_f);
    } else {
        float *groudtruth = (float *)malloc(sizeof(float) * M);
        memset(groudtruth, 0, sizeof(float) * M);
        gemvCPU(mat, vec, groudtruth, M, N);
        is_right = CheckResult(dst, groudtruth, M);
        free(groudtruth);
    }
    if (is_right) {
        printf("the ans is right\n");
    } else {
        printf("the ans is wrong\n");
    }
    cudaFree(d_vec);
    cudaFree(d_mat);
    cudaFree(d_dst);
    free(vec);
    free(mat);
    free(dst);
}

//* 模板的实例化, 因为实现写在了cu里面, 不是cuh
template void gemv_kernel<float>(float *, float *, float *, float *, float *, float *);
template void gemv_kernel<half>(half *, half *, half *, half *, half *, half *);

int main() {
    if (true) {
        float *vec;
        float *d_vec;
        float *mat;
        float *d_mat;
        float *dst;
        float *d_dst;
        gemv_kernel<float>(vec, d_vec, mat, d_mat, dst, d_dst);
    } else {
        half *vec;
        half *d_vec;
        half *mat;
        half *d_mat;
        half *dst;
        half *d_dst;
        // gemv_kernel<half>(vec, d_vec, mat, d_mat, dst, d_dst);
        learning_gemv_kernel(vec, d_vec, mat, d_mat, dst, d_dst);
    }
}