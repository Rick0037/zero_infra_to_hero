#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <stdio.h>

#include <stdexcept>
#include <string>

//* check cuda error
static const char* _cudaGetErrorEnum(cudaError_t error) { return cudaGetErrorString(error); }
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

// 实际上是预加载 的size 都是是多少
template <typename T>
struct Vec {
    static constexpr int size = 4;
};
template <>
struct Vec<half> {
    static constexpr int size = 8;
};

template <typename T>
struct SumOp {
    __device__ __forceinline__ T operator()(const T& a, const T& b) const { return a + b; }
};

template <>
struct SumOp<half> {
    __device__ __forceinline__ half operator()(const half& a, const half& b) const {
        return __hadd(a, b);
    }
};

//** warp level reduce
//  */
template <template <typename> class ReductionOp, typename T>
__device__ __forceinline__ T warpReduce(T val) {
    for (int mask = 16; mask > 0; mask >>= 1) {
        val = ReductionOp<T>()(val, __shfl_xor_sync(0xffffffff, val, mask));
    }
    return val;
}

// 把block reduce拆分为多个warp reduce来计算
template <template <typename> class ReductionOp, typename T>
__device__ __forceinline__ T blockReduce(T val) {
    int tid = threadIdx.x;
    int warp_id = tid / 32;
    int lane_id = tid % 32;
    //* 向上进1，以防分配的线程数量小于32导致warp nums为0
    int warp_nums = (blockDim.x + 31) / 32;
    // block 内 32 个warps
    static __shared__ float warp_smem[64];
    // block内每个warp reduce的结果，该结果保存在每个warp内的0号线程，所以L65用0号线程写入warp
    T res = warpReduce<ReductionOp, T>(val);
    if (lane_id == 0) {
        warp_smem[warp_id] = res;
    }
    __syncthreads();

    val = (tid < warp_nums) ? warp_smem[tid] : 0.0f;
    return warpReduce<ReductionOp, T>(val);
}

//* 一个block 计算一行, 行内实际上由block 内的thread来计算
//* VECS_PER_THREAD, VEC_SIZE 实际上再外面就已经准备好了
// mat * vec = {M, N} * {N, 1}/{1, N}
template <int VECS_PER_THREAD, int VEC_SIZE>
__global__ void gemv(float* matrix, float* vector, float* res, int cols) {
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    float thread_local_sum = 0.0f;
    //* 不需要担心什么, thread number 以及vec per thread 一定会覆盖cols, 因为再外面确定好了
    for (int i = 0; i < VECS_PER_THREAD; i++) {
        // 只考虑单行的 thread处理时候的模型. 实际上每次TID 为0, 255, 512
        //* 当前访问列数是 tid + i * blockDim.x
        //* 当前访问行数是 bid
        //* 再float4的 matrix 上是这样的
        float4 mat_val =
            reinterpret_cast<float4*>(matrix)[bid * (cols / VEC_SIZE) + tid + i * blockDim.x];
        float4 vec_val = reinterpret_cast<float4*>(vector)[tid + i * blockDim.x];
        //* 一个block 内的可以都加给 thread_local_sum
        thread_local_sum += mat_val.x * vec_val.x;
        thread_local_sum += mat_val.y * vec_val.y;
        thread_local_sum += mat_val.z * vec_val.z;
        thread_local_sum += mat_val.w * vec_val.w;
    }
    // reduce to get the final val
    // 以上仅得到了每个向量的内部乘加结果，故还需要reduce得到matrix的一行乘加vector的最终结果
    //* 先乘加 再集体做reduce
    float reduce_res = blockReduce<SumOp, float>(thread_local_sum);
    // store to gmem
    if (tid == 0) {
        res[bid] = reduce_res;
    }
}

__device__ float warp_level_sum_reduce(float sum_value) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        sum_value += __shfl_down_sync(0xffffffff, sum_value, offset);
    }
    return sum_value;
}

// block to m,
template <int BLOCK_SIZE, int PACKAGE_SIZE>
__global__ void learning_gemv(half* matrix, half* vector, half* res, int rows, int cols) {
    int bid = blockIdx.x;
    int tid = threadIdx.x;
    int warp_index = tid / 32;
    int thread_index_in_warp = tid % 32;
    __shared__ float smem[BLOCK_SIZE / 32];
    for (int i = bid; i < rows; i += gridDim.x) {
        // 当前行
        float4* start = reinterpret_cast<float4*>(matrix) + i * cols / PACKAGE_SIZE;
        float4* vect_start = reinterpret_cast<float4*>(vector);
        // float4* res_start = reinterpret_cast<float4*>(res + bid);
        half2 thread_sum_value{0.0, 0.0};
        float4 temp_mat;
        float4 temp_vec;
        //* 可以线程内先求一波和, 一会儿来一波warp level + block level reduce
        for (int j = tid; j < cols / PACKAGE_SIZE; j += blockDim.x) {
            temp_mat = start[j];
            temp_vec = vect_start[j];
            half2* mat_val_1 = (half2*)&temp_mat.x;
            half2* mat_val_2 = (half2*)&temp_mat.y;
            half2* mat_val_3 = (half2*)&temp_mat.z;
            half2* mat_val_4 = (half2*)&temp_mat.w;

            half2* vec_val_1 = (half2*)&temp_vec.x;
            half2* vec_val_2 = (half2*)&temp_vec.y;
            half2* vec_val_3 = (half2*)&temp_vec.z;
            half2* vec_val_4 = (half2*)&temp_vec.w;
            thread_sum_value += __hmul2(*mat_val_1, *vec_val_1);
            thread_sum_value += __hmul2(*mat_val_2, *vec_val_2);
            thread_sum_value += __hmul2(*mat_val_3, *vec_val_3);
            thread_sum_value += __hmul2(*mat_val_4, *vec_val_4);
        }

        float sum_value = thread_sum_value.x + thread_sum_value.y;
        sum_value = warp_level_sum_reduce(sum_value);

        if (thread_index_in_warp == 0) {
            smem[warp_index] = sum_value;
        }
        __syncthreads();

        if (warp_index == 0) {
            sum_value = tid < BLOCK_SIZE / 32 ? smem[tid] : 0.0;
            sum_value = warp_level_sum_reduce(sum_value);
        }

        if (tid == 0) {
            res[i] = sum_value;
        }
        __syncthreads();  // !保护跨行迭代: 下一轮写 smem 前, 确保本轮 warp 0 已读完 smem
    }
    return;
}

template <int VECS_PER_THREAD, int VEC_SIZE>
__global__ void gemv(half* matrix, half* vector, half* res, int cols) {
    int tid = threadIdx.x;
    int gid = blockIdx.x;

    half2 thread_local_sum_x{0, 0};
    for (int i = 0; i < VECS_PER_THREAD; i++) {
        //! 注意这里的 matrix 实际上已经是half 类型了
        float4 temp_mat =
            reinterpret_cast<float4*>(matrix)[gid * (cols / VEC_SIZE) + i * blockDim.x + tid];
        float4 temp_vec = reinterpret_cast<float4*>(vector)[i * blockDim.x + tid];

        /** half2 = float*/
        half2* mat_val_1 = (half2*)&temp_mat.x;
        half2* mat_val_2 = (half2*)&temp_mat.y;
        half2* mat_val_3 = (half2*)&temp_mat.z;
        half2* mat_val_4 = (half2*)&temp_mat.w;

        half2* vec_val_1 = (half2*)&temp_vec.x;
        half2* vec_val_2 = (half2*)&temp_vec.y;
        half2* vec_val_3 = (half2*)&temp_vec.z;
        half2* vec_val_4 = (half2*)&temp_vec.w;

        thread_local_sum_x += __hmul2(*mat_val_1, *vec_val_1);
        thread_local_sum_x += __hmul2(*mat_val_2, *vec_val_2);
        thread_local_sum_x += __hmul2(*mat_val_3, *vec_val_3);
        thread_local_sum_x += __hmul2(*mat_val_4, *vec_val_4);
    }
    half thread_local_sum = __hadd(thread_local_sum_x.x, thread_local_sum_x.y);
    half reduce_res = blockReduce<SumOp, half>(thread_local_sum);
    // store to gmem
    if (tid == 0) {
        printf("block reduce_res = %f\n", (float)reduce_res);
        res[blockIdx.x] = reduce_res;
    }
    // int tid = threadIdx.x;
    // int bid = blockIdx.x;

    // // float thread_local_sum = 0.0f;
    // half thread_local_sum = 0;
    // for (int i = 0; i < VECS_PER_THREAD; i++) {
    //     float4 mat4 = reinterpret_cast<float4*>(
    //         matrix)[bid * (cols / VEC_SIZE) + i * blockDim.x + tid];  // 4 * half2
    //     float4 vec4 = reinterpret_cast<float4*>(vector)[i * blockDim.x + tid];
    //     // 与fp32的gemv不同点在于，向量宽度由4变为8，满足128bit的CUDA线程最大读写宽度
    //     //
    //     所以依然可以用float4表示读取的偏移宽度，half也OK，只是CUDA没有half8这个内置类型，需要自定义half8这个struct，见下文190行左右
    //     // 然后再转成half2，调用half2 intrinsic做计算
    //     half2* vec_h1 = (half2*)&vec4.x;
    //     half2* vec_h2 = (half2*)&vec4.y;
    //     half2* vec_h3 = (half2*)&vec4.z;
    //     half2* vec_h4 = (half2*)&vec4.w;
    //     half2* mat_h1 = (half2*)&mat4.x;
    //     half2* mat_h2 = (half2*)&mat4.y;
    //     half2* mat_h3 = (half2*)&mat4.z;
    //     half2* mat_h4 = (half2*)&mat4.w;
    //     half2 res1 = __hmul2(*mat_h1, *vec_h1);
    //     half2 res2 = __hmul2(*mat_h2, *vec_h2);
    //     half2 res3 = __hmul2(*mat_h3, *vec_h3);
    //     half2 res4 = __hmul2(*mat_h4, *vec_h4);
    //     half2 res = __hadd2(__hadd2(__hadd2(res1, res2), res3), res4);
    //     thread_local_sum = __hadd(res.x, res.y);
    //     // float2 res1 = __half22float2(__hmul2(*mat_h1, *vec_h1));
    //     // float2 res2 = __half22float2(__hmul2(*mat_h2, *vec_h2));
    //     // float2 res3 = __half22float2(__hmul2(*mat_h3, *vec_h3));
    //     // float2 res4 = __half22float2(__hmul2(*mat_h4, *vec_h4));
    //     // thread_local_sum += res1.x;
    //     // thread_local_sum += res1.y;
    //     // thread_local_sum += res2.x;
    //     // thread_local_sum += res2.y;
    //     // thread_local_sum += res3.x;
    //     // thread_local_sum += res3.y;
    //     // thread_local_sum += res4.x;
    //     // thread_local_sum += res4.y;
    //     // if(i == 0 && tid == 0 && bid == 0) {
    //     // printf("thread sum = %f\n", (float)thread_local_sum); // 8
    //     // printf("res1.x = %f\n", res1.x); // 1
    //     //}
    // }
    // // reduce to get the final val
    // //  以上仅得到了每个向量的内部乘加结果，故还需要reduce得到matrix的一行乘加vector的最终结果
    // half reduce_res = blockReduce<SumOp, half>(thread_local_sum);
    // // store to gmem
    // if (tid == 0) {
    //     printf("block reduce_res = %f\n", (float)reduce_res);
    //     res[blockIdx.x] = reduce_res;
    // }
    // __syncthreads();
}

template <int VECS_PER_THREAD, int VEC_SIZE, int THREAD_NUMS>
struct DispatchLauncher {
    template <typename T>
    static void launcher(T* d_mat, T* d_vec, T* d_dst, int M, int N) {
        dim3 Grid(M);
        dim3 Block(THREAD_NUMS);
        float milliseconds = 0;
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start);
        printf("calling\n");
        gemv<VECS_PER_THREAD, VEC_SIZE><<<Grid, Block>>>(d_mat, d_vec, d_dst, N);
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
    }
};

// vec * mat, mat is row major
// [1, N] * [N, M]
// logits * v
// 有关fp32/fp16 fma和add的各种重载操作
namespace gemv2 {
struct half8 {
    half2 h1;
    half2 h2;
    half2 h3;
    half2 h4;

    __device__ half8& operator=(half8 h8) {
        h1 = h8.h1;
        h2 = h8.h2;
        h3 = h8.h3;
        h4 = h8.h4;
        return *this;
    }
};

//* 实际上是M行的数据, 如果是T 是fp32 就是每行总共有M/4的数据等待被处理
//* 如果是T 是fp16 就是每行总共有M/8的数据等待被处理

template <int M, typename T>
struct get_threads_per_mat_row {
    static const int value = M * sizeof(T) / 16;
};

inline __device__ float add(float a, float b) { return a + b; }

inline __device__ float4 add(float4 a, float4 b) {
    float4 c;
    c.x = gemv2::add(a.x, b.x);
    c.y = gemv2::add(a.y, b.y);
    c.z = gemv2::add(a.z, b.z);
    c.w = gemv2::add(a.w, b.w);
    return c;
}
inline __device__ half add(half a, half b) {
    // return __hadd(a, b);
    // if use L216, half+half is not really adding, its so weird, which  cause our result is 32,not
    // 256
    return (half)((float)a + (float)b);
}

inline __device__ half2 add(half2 a, half2 b) {
    half2 res;
    res.x = gemv2::add(a.x, b.x);
    res.y = gemv2::add(a.y, b.y);
    return res;
}

inline __device__ half8 add(half8 a, half8 b) {
    half8 c;
    c.h1 = gemv2::add(a.h1, b.h1);
    c.h2 = gemv2::add(a.h2, b.h2);
    c.h3 = gemv2::add(a.h3, b.h3);
    c.h4 = gemv2::add(a.h4, b.h4);
    return c;
}

inline __device__ half fma(half a, half b, half c) {
    // 有的编译器会不认识half intrinsic 例如__hmul或者__hadd，这很奇怪
    // 所以粗暴转成fp32计算再转回fp16
    return __float2half((float)a * (float)b + (float)c);
}

inline __device__ half2 fma(half a, half2 b, half2 c) {
    half2 res;
    res.x = gemv2::fma(a, b.x, c.x);
    res.y = gemv2::fma(a, b.y, c.y);
    return res;
}

inline __device__ half8 fma(half a, half8 b, half8 c) {
    half8 d;
    d.h1 = gemv2::fma(a, b.h1, c.h1);
    d.h2 = gemv2::fma(a, b.h2, c.h2);
    d.h3 = gemv2::fma(a, b.h3, c.h3);
    d.h4 = gemv2::fma(a, b.h4, c.h4);
    return d;
}

inline __device__ float fma(float a, float b, float c) { return a * b + c; }

inline __device__ float4 fma(float a, float4 b, float4 c) {
    float4 d;
    d.x = gemv2::fma(a, b.x, c.x);
    d.y = gemv2::fma(a, b.y, c.y);
    d.z = gemv2::fma(a, b.z, c.z);
    d.w = gemv2::fma(a, b.w, c.w);
    return d;
}
}  // namespace gemv2

// for fp32: <64, M * sizeof(T) / 16 = M / 4, 4>
template <int THREADS_PER_BLOCK, int THREADS_PER_VALUE, int VEC_SIZE>
__global__ void gemv2_kernel(float* matrix, float* vector, float* res, int N, int M) {
    int tid = threadIdx.x;
    int row_idx = tid / THREADS_PER_VALUE;
    int col_idx = tid % THREADS_PER_VALUE;

    // 一共循环处理几次, 针对一个矩阵
    // 或者求出每次循环得间隔
    constexpr int loop_iter = THREADS_PER_BLOCK / THREADS_PER_VALUE;
    // * smem size =  THREADS_PER_VALUE * (THREADS_PER_BLOCK / THREADS_PER_VALUE)
    __shared__ float4 smem[THREADS_PER_BLOCK];
    //* 这里得out 每个线程得out 就是自己负责得float4 对应的一个变量,
    float4 out{0.0, 0.0, 0.0, 0.0};
    //* 实际上计算得时候已经按照float4 在计算index了
    for (int j = row_idx; j < N; j += loop_iter) {
        // 可以使用matrix 换成float4 再去寻找地址, 也可以先换成float在寻找地图
        float4 mat_value = reinterpret_cast<float4*>(matrix)[j * THREADS_PER_VALUE + col_idx];
        float vec_value = vector[j];
        out = gemv2::fma(vec_value, mat_value, out);

        // vector 不需要float4, 需要得是一个数值
        // float4 vec_value = reinterpret_cast<float4*>(vector)[row_idx];
    }
    smem[tid] = out;
    __syncthreads();

    //
    for (int i = loop_iter / 2; i > 0; i >>= 1) {
        int count = loop_iter / i;
        int offset = THREADS_PER_BLOCK / count;
        if (tid < offset) {
            smem[tid] = gemv2::add(smem[tid], smem[tid + offset]);
        }
        __syncthreads();
    }
    if (tid < THREADS_PER_VALUE) {
        //! @注意这里最好就是向量化的写!
        reinterpret_cast<float4*>(res)[col_idx] = smem[tid];
    }
}

// // 1个block处理一个[1, M], 循环处理完[N, M]
// // for fp32: <64, M * sizeof(T) / 16 = M / 4, 4>
// template <int THREADS_PER_BLOCK, int THREADS_PER_VALUE, int VEC_SIZE>
// __global__ void gemv2_kernel(float* matrix, float* vector, float* res, int N, int M) {
//     // 根据编译期常量获取每个thread处理的行列号
//     int tid = threadIdx.x;
//     // 每个线程负责数据所在行号
//     int mat_o = tid / THREADS_PER_VALUE;
//     // 每个线程负责数据所在向量号
//     int mat_i = tid % THREADS_PER_VALUE * VEC_SIZE;
//     // 一个block处理的行数
//     constexpr int ROW_PER_ITER = THREADS_PER_BLOCK / THREADS_PER_VALUE;
//     __shared__ float out_smem[512];
//     float4 out;
//     // 点乘或fma，inter-block循环累加
//     for (int ti = mat_o; ti < N; ti += ROW_PER_ITER) {
//         float4 mat = *reinterpret_cast<float4*>(&matrix[ti * M + mat_i]);
//         float logits = vector[ti];
//         // fused mul and add: d = a * b + c
//         out = gemv2::fma(logits, mat, out);
//     }
//     // intra-block二分法相加得最终结果
//     for (int ROWS_PER_BLOCK = ROW_PER_ITER; ROWS_PER_BLOCK >= 2; ROWS_PER_BLOCK /= 2) {
//         int midpoint = ROWS_PER_BLOCK / 2;
//         if (mat_o >= midpoint && mat_o < ROWS_PER_BLOCK) {
//             *reinterpret_cast<float4*>(&out_smem[(mat_o - midpoint) * M + mat_i]) = out;
//         }
//         __syncthreads();
//         if (mat_o < midpoint) {
//             // ROW_PER_ITER中上半部分out和下半部分out相加
//             out = gemv2::add(*reinterpret_cast<float4*>(&out_smem[mat_o * M + mat_i]), out);
//         }
//         __syncthreads();
//     }
//     // 二分法最终结果存在首行，写回显存
//     if (mat_o == 0) {
//         *reinterpret_cast<float4*>(&res[mat_i]) = out;
//     }
// }

// for fp16: <64, M * sizeof(T) / 16 = M / 8, 8>
template <int THREADS_PER_BLOCK, int THREADS_PER_VALUE, int VEC_SIZE>
__global__ void gemv2_kernel(half* matrix, half* vector, half* res, int N, int M) {
    int tid = threadIdx.x;
    int mat_o = tid / THREADS_PER_VALUE;
    int mat_i = tid % THREADS_PER_VALUE * VEC_SIZE;
    constexpr int ROW_PER_ITER = THREADS_PER_BLOCK / THREADS_PER_VALUE;
    __shared__ half out_smem[2048];
    //! out 必须清零, 否则 fma 在栈垃圾上累加, 结果是 nan/inf
    half2 zero_h2 = __float2half2_rn(0.0f);
    gemv2::half8 out{zero_h2, zero_h2, zero_h2, zero_h2};
    for (int ti = mat_o; ti < N; ti += ROW_PER_ITER) {
        gemv2::half8 mat = *reinterpret_cast<gemv2::half8*>(&matrix[ti * M + mat_i]);
        half logits = vector[ti];
        out = gemv2::fma(logits, mat, out);
    }
    for (int ROWS_PER_BLOCK = ROW_PER_ITER; ROWS_PER_BLOCK >= 2; ROWS_PER_BLOCK /= 2) {
        int midpoint = ROWS_PER_BLOCK / 2;
        if (mat_o >= midpoint && mat_o < ROWS_PER_BLOCK) {
            *reinterpret_cast<gemv2::half8*>(&out_smem[(mat_o - midpoint) * M + mat_i]) = out;
        }
        __syncthreads();

        if (mat_o < midpoint) {
            // ROW_PER_ITER中上半部分out和下半部分out相加
            out = gemv2::add(*reinterpret_cast<gemv2::half8*>(&out_smem[mat_o * M + mat_i]), out);
        }
        __syncthreads();
    }
    if (mat_o == 0) {
        *reinterpret_cast<gemv2::half8*>(&res[mat_i]) = out;
    }
}
// // TODO: 修改float4部分为可以泛化表示float4和half8类型的代码,
// // 而后此模板函数可以取代以上fp32和fp16的gemv2
// template <int THREADS_PER_BLOCK, int THREADS_PER_VALUE, int VEC_SIZE, typename T>
// __global__ void gemv2_kernel_template(T* matrix, T* vector, T* res, int N, int M) {
//     int tid = threadIdx.x;
//     int mat_o = tid / THREADS_PER_VALUE;
//     int mat_i = tid % THREADS_PER_VALUE * VEC_SIZE;
//     constexpr int ROW_PER_ITER = THREADS_PER_BLOCK / THREADS_PER_VALUE;
//     __shared__ T out_smem[512];
//     float4 out;  // TODO
//     for (int ti = mat_o; ti < N; ti += ROW_PER_ITER) {
//         float4 mat = *reinterpret_cast<float4*>(&matrix[ti * M + mat_i]);  // TODO
//         T logits = vector[ti];
//         out = gemv2::fma(logits, mat, out);
//     }
//     for (int ROWS_PER_BLOCK = ROW_PER_ITER; ROWS_PER_BLOCK >= 2; ROWS_PER_BLOCK /= 2) {
//         int midpoint = ROWS_PER_BLOCK / 2;
//         if (mat_o >= midpoint && mat_o < ROWS_PER_BLOCK) {
//             *reinterpret_cast<float4*>(&out_smem[(mat_o - midpoint) * M + mat_i]) = out;  // TODO
//         }
//         __syncthreads();
//         if (mat_o < midpoint) {
//             // ROW_PER_ITER中上半部分out和下半部分out相加
//             out =
//                 gemv2::add(*reinterpret_cast<float4*>(&out_smem[mat_o * M + mat_i]), out);  //
//                 TODO
//         }
//         __syncthreads();
//     }
//     if (mat_o == 0) {
//         *reinterpret_cast<float4*>(&res[mat_i]) = out;  // TODO
//     }
// }

//? THREADS_PER_BLOCK thread nums in block
//? THREADS_PER_VALUE each row number wants to solve
template <int THREADS_PER_BLOCK, int THREADS_PER_VALUE, int VEC_SIZE>
struct DispatchLauncher2 {
    template <typename T>
    static void launcher(T* d_mat, T* d_vec, T* d_dst, int M, int N) {
        dim3 Grid(1);
        dim3 Block(THREADS_PER_BLOCK);
        float milliseconds = 0;
        // 使用cudaevent计时，开销最小
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start);
        printf("calling\n");
        // 启动cuda kernel
        //* 这里默认了 THREADS_PER_BLOCK > THREADS_PER_VALUE, 一个block thread扫一遍能处理很多行
        gemv2_kernel<THREADS_PER_BLOCK, THREADS_PER_VALUE, VEC_SIZE>
            <<<Grid, Block>>>(d_mat, d_vec, d_dst, N, M);
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
    }
};