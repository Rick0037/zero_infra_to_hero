#include <bits/stdc++.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <float.h>

#include <cfenv>
#include <cmath>
#include <random>

/**
 */
bool CheckResult(float* out, float* groudtruth, int nums) {
    for (int i = 0; i < nums; i++) {
        if (out[i] != groudtruth[i]) {
            printf("out and groudtruth is diff %ld \n", i);
            return false;
        }
    }

    return true;
}

/**
 */
template <typename T>
void GenScalePerTensorSymmetricCPU(const T* in_ptr, const int quantization_bit,
                                   const int num_elements, T* scale, T* zero_point) {
    // 对称的per-tensor 量化
    T max_val = *std::max_element(in_ptr, in_ptr + num_elements);
    T min_val = *std::min_element(in_ptr, in_ptr + num_elements);
    T value = std::max(std::abs(max_val), std::abs(min_val));
    T max_int_value = static_cast<T>(std::pow(2.0, quantization_bit - 1) - 1);
    *scale = value / max_int_value;
    *zero_point = 0;
    return;
}

/**
 */
template <typename T>
void QuantizationPerTensorSymmetricCPU(const T* in_ptr, const T scale, const int quantization_bit,
                                       const int num_elements, T* out_ptr) {
    // 使用 scale 去进行量化
    T upper_bound = static_cast<T>(std::pow(2.0, quantization_bit - 1) - 1);
    T lower_bound = -upper_bound - 1;
    T temp_value;
    for (int i = 0; i < num_elements; i++) {
        //! 必须要进行一波round
        temp_value = std::nearbyint(in_ptr[i] / scale);
        temp_value = temp_value > upper_bound ? upper_bound : temp_value;
        temp_value = temp_value < lower_bound ? lower_bound : temp_value;
        out_ptr[i] = temp_value;
    }
    return;
}

/**
 */
template <typename T>
void GenScalePerChannelSymmetricCPU(const T* in_ptr, const int quantization_bit, const int HW,
                                    const int channel, const int num_elements, T* scale,
                                    T* zero_point) {
    T max_int_value = static_cast<T>(std::pow(2.0, quantization_bit - 1) - 1);
    for (int i = 0; i < channel; i++) {
        const T* start = in_ptr + i * HW;
        const T* end = in_ptr + (i + 1) * HW;
        T max_value = *std::max_element(start, end);
        T min_value = *std::min_element(start, end);
        T finial_value = std::max(std::abs(max_value), std::abs(min_value));
        scale[i] = finial_value / max_int_value;
        zero_point[i] = 0.0;
    }

    return;
}

template <typename T>
void GenScalePerChannelAsymmetricCPU(const T* in_ptr, const int quantization_bit, const int HW,
                                     const int channel, const int num_elements, T* scale,
                                     T* zero_point) {
    T max_int_value = static_cast<T>(std::pow(2.0, quantization_bit) - 1);

    for (int i = 0; i < channel; i++) {
        const T* start = in_ptr + i * HW;
        const T* end = in_ptr + (i + 1) * HW;
        T max_value = *std::max_element(start, end);
        T min_value = *std::min_element(start, end);
        scale[i] = (max_value - min_value) / max_int_value;
        //! @ 一定需要加round
        zero_point[i] = std::nearbyint(-min_value / scale);
    }

    return;
}

template <typename T>
void QuantizationPerChannelSymmetricCPU(const T* in_ptr, const T* scale, const int quantization_bit,
                                        const int HW, const int num_elements, T* out_ptr) {
    T upper_bound = static_cast<T>(std::pow(2.0, quantization_bit - 1) - 1);
    T lower_bound = -upper_bound - 1;

    for (int i = 0; i < num_elements / HW; i++) {
        const T* start = in_ptr + i * HW;
        T* out = out_ptr + i * HW;
        for (int j = 0; j < HW; j++) {
            out[j] = std::nearbyint(start[j] / scale[i]);
            out[j] = out[j] > upper_bound ? upper_bound : out[j];
            out[j] = out[j] < lower_bound ? lower_bound : out[j];
        }
    }
    return;
}

// * --------------------------GPU function------------------------
__device__ float getnearbyint(float value) { return std::nearbyint(value); }

inline __device__ float atomicMax(float* address, float val) {
    int* address_as_i = (int*)address;
    int old = *address_as_i;
    int assumed = 0;
    do {
        assumed = old;
        old = atomicCAS(address_as_i, assumed, __float_as_int(fmaxf(val, __int_as_float(assumed))));

    } while (old != assumed);

    return __int_as_float(old);
}

inline __device__ float atomicMin(float* address, float val) {
    int* address_as_i = (int*)address;
    int old = *address_as_i;
    int assumed = 0;
    do {
        assumed = old;
        old = atomicCAS(address_as_i, assumed, __float_as_int(fminf(val, __int_as_float(assumed))));

    } while (old != assumed);

    return __int_as_float(old);
}

// atomicMax/atomicMin 是在显存原值上累积, d_max/d_min 必须先初始化, 否则读到垃圾值
template <typename T>
__global__ void InitMaxMin(T* max_ptr, T* min_ptr, const int n) {
    int gid = threadIdx.x + blockDim.x * blockIdx.x;
    if (gid < n) {
        //! 这里用最大数值的负数,
        //! FLT_MIN 只是一个接近0的数值
        max_ptr[gid] = -FLT_MAX;
        min_ptr[gid] = FLT_MAX;
    }
}

template <typename T>
__global__ void ReduceMaxMinPerTensor(const T* input_ptr, const int nums, T* max_ptr, T* min_ptr,
                                      const int channel, const int HW) {
    // shared memory size is
    // blockSize * 2 * sizeof(float)
    // 单个block 处理一个channel. thread 处理HW
    extern __shared__ unsigned char smem[];
    T* max_shared = reinterpret_cast<T*>(smem);
    T* min_shared = max_shared + blockDim.x;
    int gid = threadIdx.x + blockDim.x * blockIdx.x;
    int tid = threadIdx.x;
    //! @必须初始化
    max_shared[tid] = -FLT_MAX;
    min_shared[tid] = FLT_MAX;
    // per- tensor 不需要这么遍历, 直接全搞到shared 就可以了
    // for (int i = blockIdx.x; i < channel; i += gridDim.x) {
    //     const T* start = input_ptr + i * HW;

    //     for (int j = threadIdx.x; j < HW; j += blockDim.x) {
    //         max_shared[tid] = fmax(max_shared[tid], start[j]);
    //         min_shared[tid] = fmin(min_shared[tid], start[j]);
    //     }
    //     __syncthreads();
    // }
    // 真正的per-tensor
    int total_thread_num = gridDim.x * blockDim.x;
    // 边界必须是 nums, channel*HW 在宏传参下是 nums*HW, 会越界
    for (int i = gid; i < nums; i += total_thread_num) {
        //! for 循环在以gid 在遍历, 实际赋值在tid手上
        max_shared[tid] = fmax(max_shared[tid], input_ptr[i]);
        min_shared[tid] = fmin(min_shared[tid], input_ptr[i]);
    }
    __syncthreads();

    for (int j = blockDim.x / 2; j > 0; j >>= 1) {
        if (tid < j) {
            max_shared[tid] = fmax(max_shared[tid], max_shared[tid + j]);
            min_shared[tid] = fmin(min_shared[tid], min_shared[tid + j]);
        }
        __syncthreads();
    }

    if (tid == 0) {
        atomicMax(max_ptr, max_shared[0]);
        atomicMin(min_ptr, min_shared[0]);
    }
}

template <typename T>
__global__ void ReduceMaxMinPerChannel(const T* input_ptr, const int nums, T* max_ptr, T* min_ptr,
                                       const int num_channels, const int HW) {
    int tid = threadIdx.x;
    int gid = threadIdx.x + blockDim.x * blockIdx.x;
    // 每个thread有自己的
    extern __shared__ unsigned char smem[];
    T* shared_max = reinterpret_cast<T*>(smem);
    //! 注意
    T* shared_min = reinterpret_cast<T*>(smem) + blockDim.x;
    // loop channel - 一个 block 去干一个channel的活
    for (int channel = blockIdx.x; channel < num_channels; channel += gridDim.x) {
        const T* start = input_ptr + channel * HW;
        const T* end = input_ptr + (channel + 1) * HW;
        shared_max[tid] = -FLT_MAX;
        shared_min[tid] = FLT_MAX;
        __syncthreads();
        for (int j = tid; j < HW; j += blockDim.x) {
            shared_max[tid] = fmax(shared_max[tid], start[j]);
            shared_min[tid] = fmin(shared_min[tid], start[j]);
        }
        __syncthreads();

        // thread 内归约
        for (int j = blockDim.x / 2; j > 0; j >>= 1) {
            if (tid < j) {
                shared_max[tid] = fmax(shared_max[tid], shared_max[tid + j]);
                shared_min[tid] = fmin(shared_min[tid], shared_min[tid + j]);
            }
            __syncthreads();
        }
        if (tid == 0) {
            max_ptr[channel] = shared_max[0];
            min_ptr[channel] = shared_min[0];
        }

        // !不需要加因为最后一个for循环已经加过了
        // __syncthreads();
    }
}

// round(value/ scale).clip(min, max)
template <typename T>
__global__ void QuantizePerTensorSymmetric(const T* in_ptr, const T* scale_ptr, const int nums,
                                           const double quantization_bit, T* out_ptr,
                                           const int channel, const int HW) {
    // 直接量化就写完了
    int tid = threadIdx.x;
    int gid = threadIdx.x + blockDim.x * blockIdx.x;
    T upper_bound = static_cast<T>(std::pow(2.0, quantization_bit - 1) - 1);
    T lower_bound = -upper_bound - 1;
    for (int j = gid; j < nums; j += blockDim.x * gridDim.x) {
        out_ptr[j] = nearbyint((in_ptr[j] / scale_ptr[0]));
        out_ptr[j] = out_ptr[j] > upper_bound ? upper_bound : out_ptr[j];
        out_ptr[j] = out_ptr[j] < lower_bound ? lower_bound : out_ptr[j];
    }
    return;
}

template <typename T>
__global__ void QuantizePerChannelSymmetric(const T* in_ptr, const T* scale_ptr, const int nums,
                                            const double quantization_bit, T* out_ptr,
                                            const int channel, const int HW) {
    // 按照channel去处理

    T upper_bound = static_cast<T>(std::pow(2.0, quantization_bit - 1) - 1);
    T lower_bound = -upper_bound - 1;
    for (int index = blockIdx.x; index < channel; index += gridDim.x) {
        const T* start = in_ptr + index * HW;
        T* out = out_ptr + index * HW;
        T scale = scale_ptr[index];

        for (int j = threadIdx.x; j < HW; j += blockDim.x) {
            out[j] = getnearbyint(start[j] / scale);
            out[j] = out[j] > upper_bound ? upper_bound : out[j];
            out[j] = out[j] < lower_bound ? lower_bound : out[j];
        }
    }
    return;
}

template <typename T>
__global__ void GetScaleAndZPSymmetric(const T* max_ptr, const T* min_ptr, const int max_nums,
                                       const double quantization_bit, T* scale, T* zero_point) {
    // 有时候per-tensor的时候 max 和 min 是两个数值
    // 有时候per-channel的时候 max 和 min 是两个数组
    // max_ptr_size == scale_size == max_nums
    int tid = threadIdx.x;
    int gid = threadIdx.x + blockDim.x * blockIdx.x;
    // 只是一对一的处理
    T max_int_value = static_cast<T>(std::pow(2.0, quantization_bit - 1)) - 1;
    for (int j = gid; j < max_nums; j += blockDim.x * gridDim.x) {
        T absmax = fmax(fabs(max_ptr[j]), fabs(min_ptr[j]));
        scale[j] = absmax / max_int_value;
        zero_point[j] = 0.0;
    }
    return;
}

// use macro to reduce redundant code
#define LAUNCH_GPU_KERNEL(GetMinMaxFunc, QuantFunc, scale_size, channel, HW)                      \
    cudaMalloc((void**)&d_scale, scale_size * sizeof(float));                                     \
    cudaMalloc((void**)&d_zeropoint, scale_size * sizeof(float));                                 \
    cudaMalloc((void**)&d_max, scale_size * sizeof(float));                                       \
    cudaMalloc((void**)&d_min, scale_size * sizeof(float));                                       \
    cudaEvent_t start, stop;                                                                      \
    cudaEventCreate(&start);                                                                      \
    cudaEventCreate(&stop);                                                                       \
    cudaEventRecord(start);                                                                       \
    InitMaxMin<float><<<(scale_size + 255) / 256, 256>>>(d_max, d_min, scale_size);               \
    GetMinMaxFunc<float><<<gridSize, blockSize, blockSize * 2 * sizeof(float), 0>>>(              \
        d_input, nums, d_max, d_min, channel, HW);                                                \
    GetScaleAndZPSymmetric<float>                                                                 \
        <<<1, blockSize>>>(d_max, d_min, scale_size, quantization_bit, d_scale, d_zeropoint);     \
    QuantFunc<float><<<gridSize, blockSize>>>(d_input, d_scale, nums, quantization_bit, d_output, \
                                              channel, HW);                                       \
    cudaEventRecord(stop);                                                                        \
    cudaEventSynchronize(stop);                                                                   \
    cudaEventElapsedTime(&milliseconds, start, stop);

int main() {
    //* 初始化--------
    //* 内存copy
    //* if per-tensor or per-channel
    //*     launch kernel
    //* check ressult to CPU

    //* function - order
    //* (1) CPU scale and zero point - per-tensor - symme
    //* (2) CPU scale and zero point - per-tensor - asymme
    //* (3) CPU quantize - per-tensor
    //* (4) CPU scale and zero point - per-channel - symme
    //* (5) CPU scale and zero point - per-channel - asymme
    //* (6) CPU quantize - per-channel

    //* -------------------------------
    //* (1) GPU scale and zero point - per-tensor - symme
    //* (2) GPU scale and zero point - per-tensor - asymme
    //* (3) GPU quantize - per-tensor
    //* (4) GPU scale and zero point - per-channel - symme
    //* (5) GPU scale and zero point - per-channel - asymme
    //* (6) GPU quantize - per-channel

    float milliseconds = 0;
    constexpr int nums = 400 * 20 * 10;
    constexpr int HW = 20 * 10;
    constexpr int channel = 400;
    constexpr int quantization_bit = 8;

    float* input = (float*)malloc(nums * sizeof(float));
    float cpu_min = FLT_MAX;
    float cpu_max = FLT_MIN;

    for (int i = 0; i < nums; i++) {
        // generate float input inside [-1, 1],[-3,3]
        input[i] = -3 + static_cast<float>(rand()) / (static_cast<float>(RAND_MAX / 6));
        cpu_min = std::min(input[i], cpu_min);
        cpu_max = std::max(input[i], cpu_max);
    }
    // 上来求CPU的最小 最大数值

    float* output = (float*)malloc(nums * sizeof(float));

    float *d_input, *d_output;
    cudaMalloc((void**)&d_input, sizeof(float) * nums);
    cudaMalloc((void**)&d_output, sizeof(float) * nums);
    cudaMemcpy(d_input, input, nums * sizeof(float), cudaMemcpyHostToDevice);
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, 0);
    int maxblocks = deviceProp.maxGridSize[0];
    int blockSize = 256;
    // channel, 一个 block 处理一个channel,
    int gridSize =
        std::min<int>((nums + blockSize - 1) / blockSize, std::min<int>(channel, maxblocks));
    printf("gridsize blocksize are  %d, %d\n", gridSize, blockSize);

    float *d_scale, *d_zeropoint, *d_max, *d_min;
    bool per_tensor_quantize = false;
    if (per_tensor_quantize) {
        //! 不要混用宏和函数看起来容易有问题, 都有歧义了
        LAUNCH_GPU_KERNEL(ReduceMaxMinPerTensor, QuantizePerTensorSymmetric, 1, nums, HW);
    } else {
        // switch to per channel
        LAUNCH_GPU_KERNEL(ReduceMaxMinPerChannel, QuantizePerChannelSymmetric, channel, channel,
                          HW);
    }

    // launch kernel result is in d_output
    cudaMemcpy(output, d_output, sizeof(float) * nums, cudaMemcpyDeviceToHost);
    // (per tensor) get CPU output to validate GPU result is right or not
    float* CPUOutput = (float*)malloc(sizeof(float) * nums);
    if (per_tensor_quantize) {
        float* scale = (float*)malloc(sizeof(float) * 1);
        float* zeropoint = (float*)malloc(sizeof(float) * 1);
        GenScalePerTensorSymmetricCPU<float>(input, quantization_bit, nums, scale, zeropoint);
        QuantizationPerTensorSymmetricCPU<float>(input, *scale, quantization_bit, nums, CPUOutput);
        free(scale);
        free(zeropoint);
    } else {
        float* scale = (float*)malloc(sizeof(float) * channel);
        float* zeropoint = (float*)malloc(sizeof(float) * channel);
        GenScalePerChannelSymmetricCPU<float>(input, quantization_bit, HW, channel, nums, scale,
                                              zeropoint);
        QuantizationPerChannelSymmetricCPU<float>(input, scale, quantization_bit, HW, nums,
                                                  CPUOutput);
        free(scale);
        free(zeropoint);
    }
    if (CheckResult(output, CPUOutput, nums)) {
        printf("the ans is right");
    } else {
        printf("the ans is wrong\n");
        printf("first two CPUoutput are %f, %f\n", CPUOutput[0], CPUOutput[1]);
        printf("first two output are %f, %f\n", output[0], output[1]);
    }
    printf("Quantize kernel latency = %f ms\n", milliseconds);
    free(input);
    free(output);
    free(CPUOutput);
    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_scale);
    cudaFree(d_zeropoint);
    cudaFree(d_max);
    cudaFree(d_min);
    return 0;
}