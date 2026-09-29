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

template <typename T>
void gemv_kernel(T *vec, T *d_vec, T *mat, T *d_mat, T *dst, T *d_dst) {
    constexpr int N = 2048;  // 256 * 8
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
    if (false) {
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
        gemv_kernel<half>(vec, d_vec, mat, d_mat, dst, d_dst);
    }
}