import torch

import triton
import triton.language as tl


def get_cuda_autotune_config():
    return [
        triton.Config(
            {
                "BLOCK_SIZE_M": 128,
                "BLOCK_SIZE_N": 256,
                "BLOCK_SIZE_K": 64,
                "GROUP_SIZE_M": 8,
            },
            num_stages=3,
            num_warps=8,
        ),
        triton.Config(
            {
                "BLOCK_SIZE_M": 64,
                "BLOCK_SIZE_N": 256,
                "BLOCK_SIZE_K": 32,
                "GROUP_SIZE_M": 8,
            },
            num_stages=4,
            num_warps=4,
        ),
        triton.Config(
            {
                "BLOCK_SIZE_M": 128,
                "BLOCK_SIZE_N": 128,
                "BLOCK_SIZE_K": 32,
                "GROUP_SIZE_M": 8,
            },
            num_stages=4,
            num_warps=4,
        ),
        triton.Config(
            {
                "BLOCK_SIZE_M": 128,
                "BLOCK_SIZE_N": 64,
                "BLOCK_SIZE_K": 32,
                "GROUP_SIZE_M": 8,
            },
            num_stages=4,
            num_warps=4,
        ),
        triton.Config(
            {
                "BLOCK_SIZE_M": 64,
                "BLOCK_SIZE_N": 128,
                "BLOCK_SIZE_K": 32,
                "GROUP_SIZE_M": 8,
            },
            num_stages=4,
            num_warps=4,
        ),
        triton.Config(
            {
                "BLOCK_SIZE_M": 128,
                "BLOCK_SIZE_N": 32,
                "BLOCK_SIZE_K": 32,
                "GROUP_SIZE_M": 8,
            },
            num_stages=4,
            num_warps=4,
        ),
        triton.Config(
            {
                "BLOCK_SIZE_M": 64,
                "BLOCK_SIZE_N": 32,
                "BLOCK_SIZE_K": 32,
                "GROUP_SIZE_M": 8,
            },
            num_stages=5,
            num_warps=2,
        ),
        triton.Config(
            {
                "BLOCK_SIZE_M": 32,
                "BLOCK_SIZE_N": 64,
                "BLOCK_SIZE_K": 32,
                "GROUP_SIZE_M": 8,
            },
            num_stages=5,
            num_warps=2,
        ),
        # * Good config for fp8 inputs.
        triton.Config(
            {
                "BLOCK_SIZE_M": 128,
                "BLOCK_SIZE_N": 256,
                "BLOCK_SIZE_K": 128,
                "GROUP_SIZE_M": 8,
            },
            num_stages=3,
            num_warps=8,
        ),
        triton.Config(
            {
                "BLOCK_SIZE_M": 256,
                "BLOCK_SIZE_N": 128,
                "BLOCK_SIZE_K": 128,
                "GROUP_SIZE_M": 8,
            },
            num_stages=3,
            num_warps=8,
        ),
        triton.Config(
            {
                "BLOCK_SIZE_M": 256,
                "BLOCK_SIZE_N": 64,
                "BLOCK_SIZE_K": 128,
                "GROUP_SIZE_M": 8,
            },
            num_stages=4,
            num_warps=4,
        ),
        triton.Config(
            {
                "BLOCK_SIZE_M": 64,
                "BLOCK_SIZE_N": 256,
                "BLOCK_SIZE_K": 128,
                "GROUP_SIZE_M": 8,
            },
            num_stages=4,
            num_warps=4,
        ),
        triton.Config(
            {
                "BLOCK_SIZE_M": 128,
                "BLOCK_SIZE_N": 128,
                "BLOCK_SIZE_K": 128,
                "GROUP_SIZE_M": 8,
            },
            num_stages=4,
            num_warps=4,
        ),
        triton.Config(
            {
                "BLOCK_SIZE_M": 128,
                "BLOCK_SIZE_N": 64,
                "BLOCK_SIZE_K": 64,
                "GROUP_SIZE_M": 8,
            },
            num_stages=4,
            num_warps=4,
        ),
        triton.Config(
            {
                "BLOCK_SIZE_M": 64,
                "BLOCK_SIZE_N": 128,
                "BLOCK_SIZE_K": 64,
                "GROUP_SIZE_M": 8,
            },
            num_stages=4,
            num_warps=4,
        ),
        triton.Config(
            {
                "BLOCK_SIZE_M": 128,
                "BLOCK_SIZE_N": 32,
                "BLOCK_SIZE_K": 64,
                "GROUP_SIZE_M": 8,
            },
            num_stages=4,
            num_warps=4,
        ),
    ]


# * 自动调参：矩阵乘可以使用不同的分块大小、线程数和流水线级数。
# * Triton 会实际运行这些方案，选出当前矩阵尺寸下速度最快的一种。
# * - configs：候选方案列表，例如 BLOCK_SIZE_M=64、num_warps=4。
# * - key：矩阵尺寸参数。这里 key=["M", "N", "K"]，表示尺寸变化时重新测速；
# *   尺寸不变时，直接使用上次选出的最快方案，不再重复测试。


@triton.autotune(
    configs=get_cuda_autotune_config(),
    key=["M", "N", "K"],
)
@triton.jit
def matmul_kernel(
    # * Pointers to matrices
    # * 矩阵指针
    a_ptr,
    b_ptr,
    c_ptr,
    # * Matrix dimensions
    # * 矩阵维度
    M,
    N,
    K,
    # * The stride variables represent how much to increase the ptr by when moving by 1
    # * element in a particular dimension. E.g. `stride_am` is how much to increase `a_ptr`
    # * by to get the element one row down (A has M rows).
    # * 这些步幅变量表示在特定维度移动 1 个元素时,`ptr` 应该增加多少。例如,`stride_am` 指示了为了访问下一行的元素（假设 `A` 有 `M` 行）,需要增加多少 `a_ptr`。
    stride_am,
    stride_ak,  # *
    stride_bk,
    stride_bn,  # *
    stride_cm,
    stride_cn,
    # * Meta-parameters
    # * 元参数
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,  # *
    GROUP_SIZE_M: tl.constexpr,  # *
):
    """Kernel for computing the matmul C = A x B.
    A has shape (M, K), B has shape (K, N) and C has shape (M, N)
    """
    """计算矩阵乘法 C = A x B 的核心算法。
    其中,A 的形状为 (M, K),B 的形状为 (K, N),C 的形状为 (M, N)。
    """
    # * -----------------------------------------------------------
    # * Map program ids `pid` to the block of C it should compute.
    # * 将程序 ID `pid` 映射到它应计算的 C 块。
    # * This is done in a grouped ordering to promote L2 data reuse.
    # * 这是按组顺序进行的,以促进 L2 数据重用。
    # * See above `L2 Cache Optimizations` section for details.
    # * 详细信息请参见上述的 `L2 缓存优化` 部分。

    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    # * 一个组内由多少块
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    # * 之际在那个组里面
    group_id = pid // num_pid_in_group
    # * group在全局得行数
    first_pid_m = group_id * GROUP_SIZE_M
    # * 实际该组有几行, 最后一组可能有截断
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    # * 6 % 4 = 2, 在group 内实际上是以M 方向位例子得
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    # *
    pid_n = (pid % num_pid_in_group) // group_size_m

    # * ----------------------------------------------------------
    # * Create pointers for the first blocks of A and B.
    # * 创建 A 和 B 第一个块的指针
    # * We will advance this pointer as we move in the K direction
    # * and accumulate
    # * 在沿着 K 方向移动时,我们将推进这个指针并累加
    # * `a_ptrs` is a block of [BLOCK_SIZE_M, BLOCK_SIZE_K] pointers
    # * `a_ptrs` 是一个 [BLOCK_SIZE_M, BLOCK_SIZE_K] 大小的指针块
    # * `b_ptrs` is a block of [BLOCK_SIZE_K, BLOCK_SIZE_N] pointers
    # * `b_ptrs` 是一个 [BLOCK_SIZE_K, BLOCK_SIZE_N] 大小的指针块

    # * See above `Pointer Arithmetic` section for details
    # * 详细信息请参见上述的 `指针算术` 部分。
    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    # * -----------------------------------------------------------
    # * Iterate to compute a block of the C matrix.
    # * 迭代计算 C 矩阵的一个块。
    # * We accumulate into a `[BLOCK_SIZE_M, BLOCK_SIZE_N]` block
    # * of fp32 values for higher accuracy.
    # * 我们累加到一个 `[BLOCK_SIZE_M, BLOCK_SIZE_N]` 大小的 fp32 值块,以提高精度。
    # * `accumulator` will be converted back to fp16 after the loop.
    # * `accumulator` 在循环结束后将转换回 fp16。
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        # * Load the next block of A and B, generate a mask by checking the K dimension.
        # * 加载 A 和 B 的下一个块,通过检查 K 维度生成一个掩码。
        # * If it is out of bounds, set it to 0.
        # * 如果超出边界设为 0
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)
        # * We accumulate along the K dimension.
        # * 通过着 K 维度进行累加。
        accumulator = tl.dot(a, b, accumulator)
        # * Advance the ptrs to the next K block.
        # * 指针前进到下一个 K 块。
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk
    c = accumulator.to(tl.float16)

    # * -----------------------------------------------------------
    # * Write back the block of the output matrix C with masks.
    # * 写回带有掩码的输出矩阵 C 的块。
    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)


def matmul(a, b):
    # * Check constraints.
    # * 检查约束
    assert a.shape[1] == b.shape[0], "Incompatible dimensions"
    assert a.is_contiguous(), "Matrix A must be contiguous"
    M, K = a.shape
    K, N = b.shape
    # * Allocates output.
    # * 分配输出
    c = torch.empty((M, N), device=a.device, dtype=torch.float16)
    # * 1D launch kernel where each block gets its own program.
    # * 1 维启动核心,其中每个块都有自己的程序。
    grid = lambda META: (
        triton.cdiv(M, META["BLOCK_SIZE_M"]) * triton.cdiv(N, META["BLOCK_SIZE_N"]),
    )
    matmul_kernel[grid](
        a,
        b,
        c,  # *
        M,
        N,
        K,  # *
        a.stride(0),
        a.stride(1),  # *
        b.stride(0),
        b.stride(1),  # *
        c.stride(0),
        c.stride(1),  # *
    )
    return c


ref_lib = "cuBLAS"


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["M", "N", "K"],
        x_vals=[128 * i for i in range(2, 33)],
        line_arg="provider",
        line_vals=[ref_lib.lower(), "triton"],
        line_names=[ref_lib, "Triton"],
        styles=[("green", "-"), ("blue", "-")],
        ylabel="TFLOPS",
        plot_name="matmul-performance-fp16",
    )
)
def benchmark(M, N, K, provider):
    a = torch.randn((M, K), device="cuda", dtype=torch.float16)
    b = torch.randn((K, N), device="cuda", dtype=torch.float16)
    quantiles = [0.5, 0.2, 0.8]

    if provider == ref_lib.lower():
        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: torch.matmul(a, b), quantiles=quantiles
        )
    if provider == "triton":
        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: matmul(a, b), quantiles=quantiles
        )

    perf = lambda ms: 2 * M * N * K * 1e-12 / (ms * 1e-3)
    return perf(ms), perf(max_ms), perf(min_ms)


benchmark.run(show_plots=True, print_data=True)
