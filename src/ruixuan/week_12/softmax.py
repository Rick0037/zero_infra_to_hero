import torch
import triton
import triton.language as tl
from triton.runtime import driver


DEVICE = torch.cuda.current_device()
properties = driver.active.utils.get_device_properties(DEVICE)
NUM_SM = properties["multiprocessor_count"]  # SM 数量
NUM_REGS = properties["max_num_regs"]  # 寄存器容量预算，以 32 位寄存器计
SIZE_SMEM = properties["max_shared_mem"]  # 共享内存容量预算，单位为字节
WARP_SIZE = properties["warpSize"]  # NVIDIA GPU 上为 32
kernels = {}

print(properties)

# {
#     "max_shared_mem": 101376,
#     "max_num_regs": 65536,
#     "multiprocessor_count": 114,
#     "warpSize": 32,
#     "sm_clock_rate": 2520000,
#     "mem_clock_rate": 10501000,
#     "mem_bus_width": 384,
# }


@triton.jit
def _softmax(
    input_ptr,
    output_ptr,
    input_row_stride,
    output_row_stride,
    n_rows,
    n_cols,
    BLOCK_SIZE: tl.constexpr,
    NUM_STAGE: tl.constexpr,
):
    block_id = tl.program_id(0)

    step = tl.num_programs(0)

    for row_idx in range(block_id, n_rows, step, num_stages=NUM_STAGE):
        row_start_ptr = input_ptr + row_idx * input_row_stride

        cover_col = tl.arange(0, BLOCK_SIZE)

        start_index_ptr = row_start_ptr + cover_col

        # * 不是 mask = BLOCK_SIZE < node_classification
        mask = cover_col < n_cols

        row = tl.load(start_index_ptr, mask=mask, other=-float("inf"))

        row_cut_max = row - tl.max(row, axis=0)

        exp_row = tl.exp(row_cut_max)

        sum_row = tl.sum(exp_row, axis=0)

        softmax_output = exp_row / sum_row

        output_start = output_ptr + row_idx * output_row_stride

        tl.store(output_start + cover_col, softmax_output, mask=mask)

    return


def softmax(x):
    assert x.is_cuda
    n_rows, n_cols = x.shape
    y = torch.empty_like(x)
    # * 向上取到不小于 n_cols 的最小 2 的幂
    BLOCK_SIZE = triton.next_power_of_2(n_cols)
    num_stages = 4 if SIZE_SMEM > 200000 else 2  # 流水线的级数
    kernel, num_programs = kernels.get(BLOCK_SIZE, (None, 0))

    # * 如果出现了没有kernel的情况
    if kernel is None:
        # * 一个 program 使用 8 个 warp，即 256 个 CUDA 线程
        num_warps = 8
        # warmup 只编译、不真正执行。返回一个编译好的 kernel 对象
        kernel = _softmax.warmup(
            # ---------------- 给 kernel 的运行时参数：按函数签名顺序传递 ----------------
            x,
            y,
            x.stride(0),
            y.stride(0),
            n_rows,
            n_cols,
            # ---------------- 给 kernel 的编译期常量：函数签名中的 tl.constexpr ----------------
            BLOCK_SIZE=BLOCK_SIZE,
            NUM_STAGE=num_stages,
            # ---------------- 给 Triton 的编译配置：不在 _softmax 函数签名中 ----------------
            num_stages=num_stages,
            num_warps=num_warps,
            grid=(1,),
        )

        # * 加载编译后的 kernel 并初始化句柄，之后才能查询资源占用
        kernel._init_handles()
        # *  每个线程使用的寄存器数量
        n_regs = kernel.n_regs
        # ! 每个 program 使用的 shared memory 字节数
        size_smem = kernel.metadata.shared
        # *  寄存器限制下，每个 SM 能容纳的 program 数量
        # * 分母 = 每线程寄存器数 × 每 warp 线程数 × 每 program 的 warp 数
        occupancy = NUM_REGS // (n_regs * WARP_SIZE * num_warps)
        # *  再考虑 shared memory 限制；用量为 0 时不构成限制，需避免除以 0
        if size_smem > 0:
            occupancy = min(occupancy, SIZE_SMEM // size_smem)
        # * SM 总数 × 每个 SM 的 program 数 = 全 GPU 建议启动的 program 数
        num_programs = NUM_SM * occupancy
        # * 按 BLOCK_SIZE 缓存编译结果和 program 数量，下次直接复用
        kernels[BLOCK_SIZE] = (kernel, num_programs)

    num_programs = min(num_programs, n_rows)

    # 启动 softmax kernel
    # [] 中的 grid 是给 Triton 的启动配置；x 和 y 会被隐式转换成矩阵首地址
    kernel[(num_programs, 1, 1)](
        # ---------------- 按 _softmax 的完整签名顺序传参 ----------------
        x,
        y,
        x.stride(0),  # 矩阵同列相邻元素的地址差
        y.stride(0),
        n_rows,
        n_cols,
        BLOCK_SIZE,
        num_stages,  # 对应 NUM_STAGE
    )

    return y


def naive_softmax(x):
    # * max dim 让那个方向进行变化, 让列方向变化, 压缩
    # * max 一般返回两个index
    x_max = x.max(dim=1)[0]
    # import pdb

    # pdb.set_trace()
    z = x - x_max[:, None]
    number_exp = torch.exp(z)
    sum_value = number_exp.sum(dim=1)
    res = number_exp / sum_value[:, None]
    return res


# triton 所提供的 benchmark 修饰器
@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["N"],  # 考虑不同配置的参数 N
        x_vals=[128 * i for i in range(2, 17)],  # N=256, 384, ..., 2048
        x_log=True,
        line_arg="provider",  # 用 provider 参数选择曲线对应的实现
        line_vals=["triton", "torch"],  # 两种被测实现
        line_names=["Triton", "Torch"],  # 对比曲线的名称
        styles=[("blue", "-"), ("green", "-")],  # 对比曲线的颜色
        ylabel="Effective bandwidth (GB/s)",
        plot_name="fused-softmax-performance",
        args={"M": 64},  # 固定 M 为 64
    )
)
def benchmark(M, N, provider):
    # 在同一个当前 CUDA stream 上生成输入并执行被测操作
    x = torch.rand(M, N, device=DEVICE, dtype=torch.float32)

    # 按顺序返回耗时的中位数、20% 分位数和 80% 分位数
    quantiles = [0.5, 0.2, 0.8]

    # 对比原生的 torch 实现
    if provider == "torch":
        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: torch.softmax(x, axis=-1), quantiles=quantiles
        )

    # 测试 Triton Softmax 算子
    if provider == "triton":
        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: softmax(x), quantiles=quantiles
        )

    # 以相同的有效读写字节数计算 GB/s
    gbps = lambda ms: 2 * x.numel() * x.element_size() / ms * 1e-6
    return gbps(ms), gbps(max_ms), gbps(min_ms)


if __name__ == "__main__":
    torch.manual_seed(0)
    # 同时覆盖小尺寸、非 2 的幂列数，以及同一 BLOCK_SIZE 下的不同形状
    for M, N in [(1, 1), (128, 256), (129, 255), (1823, 781)]:
        x = torch.randn(M, N, device=DEVICE, dtype=torch.float32)
        y_torch = naive_softmax(x)
        y_triton = softmax(x)
        torch.testing.assert_close(y_triton, y_torch, rtol=1e-5, atol=1e-7)
    print("Correctness checks passed.")

    # 对比 Triton 与 Torch 实现的有效带宽
    benchmark.run(
        print_data=True,
        show_plots=False,
        save_path="src/ruixuan/week_12/benchmark_results",
    )

# if __name__ == "__main__":
#     device = "cuda"
#     torch.manual_seed(42)

#     def check(name, x):
#         x = x.to(device)
#         out = native_softmax(x)
#         ref = torch.softmax(x, dim=-1)
#         ok = torch.allclose(out, ref, atol=1e-5)
#         print(f"[{'PASS' if ok else 'FAIL'}] {name}: shape={list(x.shape)}")
#         if not ok:
#             diff = (out - ref).abs().max().item()
#             print(f"  max diff={diff:.2e}")

#     # ---- 数值正确性 ----
#     check("随机 2D", torch.randn(10, 10))
#     check("随机 2D 非方阵", torch.randn(7, 13))
#     # 注意:本函数只支持 2D 按行 softmax,不测 1D/3D

#     # ---- 性质:每行和为 1 ----
#     x = torch.randn(5, 20, device=device)
#     y = native_softmax(x)
#     sums = y.sum(dim=-1)
#     assert torch.allclose(sums, torch.ones_like(sums)), f"sum!=1: {sums}"
#     print(f"[PASS] 每行和为 1: sums={sums}")

#     # ---- 性质:值域 [0, 1] ----
#     assert (y >= 0).all() and (y <= 1).all(), "值域异常"
#     print("[PASS] 值域 [0, 1]")

#     # ---- 性质:全相同输入 → 均匀分布 1/N ----
#     x = torch.ones(3, 5, device=device) * 7.0
#     y = native_softmax(x)
#     expected = 1.0 / 5
#     assert torch.allclose(y, torch.full_like(y, expected)), f"均匀分布失败"
#     print(f"[PASS] 全相同输入 → 均匀分布 1/5={expected}")

#     # ---- 性质:平移不变性 softmax(x) == softmax(x - c) ----
#     x = torch.randn(3, 8, device=device)
#     y1 = native_softmax(x)
#     y2 = native_softmax(x - 1000)
#     assert torch.allclose(y1, y2, atol=1e-4), "平移不变性失败"
#     print("[PASS] 平移不变性 softmax(x) == softmax(x-1000)")

#     # ---- 数值稳定性:极大输入不溢出 ----
#     x = torch.tensor([[1000.0, 1001.0, 999.0]], device=device)
#     y = native_softmax(x)
#     assert not torch.isnan(y).any() and not torch.isinf(y).any(), "极大值溢出"
#     assert torch.allclose(y.sum(), torch.tensor(1.0, device=device)), "极大值 sum!=1"
#     print(f"[PASS] 极大输入 [1000,1001,999]: {y}")

#     # ---- 数值稳定性:极小/极大混合 ----
#     x = torch.tensor([[-1000.0, 0.0, 1000.0]], device=device)
#     y = native_softmax(x)
#     assert y[0, 2] > 0.99, f"最大值应接近1,实际 {y[0,2]}"
#     assert y[0, 0] < 0.01, f"最小值应接近0,实际 {y[0,0]}"
#     print(f"[PASS] 极小极大混合 [-1000,0,1000]: {y}")

#     # ---- 边界:单元素行 ----
#     x = torch.tensor([[3.14], [1.0], [2.7]], device=device)
#     y = native_softmax(x)
#     assert torch.allclose(y, torch.ones_like(y)), "单元素行应为 1"
#     print(f"[PASS] 单元素行 [[3.14],[1.0],[2.7]] → 全1")

#     # ---- 边界:N=2 手算对比 ----
#     x = torch.tensor([[1.0, 2.0]], device=device)
#     y = native_softmax(x)
#     # exp(1)/(exp1+exp2) ≈ 0.2689
#     assert torch.allclose(y, torch.tensor([[0.2689, 0.7311]], device=device), atol=1e-3)
#     print(f"[PASS] N=2 [1,2] → {y}")

#     # ---- 单调性 ----
#     x = torch.tensor([[0.5, 0.1, 0.9, 0.3]], device=device)
#     y = native_softmax(x)
#     assert y[0, 2] > y[0, 0] > y[0, 3] > y[0, 1], f"单调性失败: {y}"
#     print(f"[PASS] 单调性: x=[0.5,0.1,0.9,0.3] → y={y}")

#     # ---- 多行独立 softmax ----
#     x = torch.tensor([[1.0, 0.0], [0.0, 1.0], [10.0, 0.0]], device=device)
#     y = native_softmax(x)
#     assert torch.allclose(
#         y[0], torch.tensor([0.7311, 0.2689], device=device), atol=1e-3
#     )
#     assert torch.allclose(
#         y[1], torch.tensor([0.2689, 0.7311], device=device), atol=1e-3
#     )
#     assert torch.allclose(y[2], torch.tensor([1.0, 0.0], device=device), atol=1e-3)
#     print(f"[PASS] 多行独立 softmax: {y}")


# -----------------------------
# @triton.jit
# def test_function(
#     x,
#     y,
#     M,
#     K,
#     num_stages,
#     N: tl.constexpr,
#     BLOCK_SIZE_M: tl.constexpr,
#     BLOCK_SIZE_N: tl.constexpr,
#     BLOCK_SIZE_K: tl.constexpr,
#     BLOCK_SIZE: tl.constexpr,
# ):
#     # * block idx
#     # Triton 的启动 grid 最多可以包含三个维度，因此 axis 的取值为 0、1、2
#     # 类似 cuda 中的 blockIdx.x / blockIdx.y / blockIdx.z
#     bid = tl.program_id(axis=0)

#     # * 有多少个block
#     # 类似地 axis 的取值为 0、1、2
#     # 对应 cuda 中的 gridDim.x / gridDim.y / gridDim.z
#     number_grid_dim_x = tl.num_programs(axis=0)

#     # * 获取一个block的偏移数值
#     offsets = bid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
#     mask = offsets < N
#     # *对于 mask=False 的位置，代码不会访问对应的内存位置，而是使用 other 来填充返回，
#     # * 如果使用了 mask，通常应当显式指定 other。否则，被屏蔽位置的返回值是未定义的，可能污染后续计算

#     x_val = tl.load(x + offsets, mask=mask, other=0.0)

#     tl.store(y + offsets, x_val, mask=mask)

#     # * num_stages 软件流水层级
#     for idx in tl.range(0, M, num_stages=num_stages):
#         ...

#     # * relu
#     # * output[i] = x[i] if condition[i] else y[i]，一个常见的例子是 ReLU 的实现：
#     output = tl.where(x > 0.0, x, 0.0)

#     # * ----------------
#     y = tl.exp(x)
#     y = tl.log(x)
#     y = tl.sqrt(x)
#     y = tl.abs(x)
#     y = tl.maximum(x, 0.0)
#     y = tl.dot(x, y)

#     accumulator = tl.zeros(BLOCK_SIZE_M, BLOCK_SIZE_N, dtype=tl.float32)

#     # * split-k, 每个步骤都挪动stride_ak 方向和 stride_bk
#     for k in range(0, K, BLOCK_SIZE_K):
#         a = tl.load(a_ptrs, mask=a_mask, other=0.0)
#         b = tl.load(b_ptrs, mask=b_mask, other=0.0)

#         accumulator = tl.dot(a, b, accumulator)
#         a_ptrs += BLOCK_SIZE_K * stride_ak
#         b_ptrs += BLOCK_SIZE_K * stride_bk
#     return
