import torch
import triton
import triton.language as tl


@triton.jit
def gemm_kernel(
    A,
    B,
    C,
    # * ---shape----
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    # * 一行或者一列对应得内层跨度
    stride_am: tl.constexpr,
    stride_ak: tl.constexpr,
    stride_bk: tl.constexpr,
    stride_bn: tl.constexpr,
    stride_cm: tl.constexpr,
    stride_cn: tl.constexpr,
    # * 处理得block多大
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    # * M, N 并行度
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offsets_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offsets_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offsets_k = tl.arange(0, BLOCK_K)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    """
    offsets_m = [8, 9, 10, 11]
    offsets_m[:, None] =
    [
        [8],
        [9],
        [10],
        [11],
    ]
    stride_am = A.stride(0) = K  # 每往下一行，地址前进 K 个元素, if stride_am = 4
    offsets_m[:, None] * stride_am =
    [[32],   
    [36],
    [40],
    [44]]   

    再加上 K 方向的列偏移：
    [[32],   + [0, 1, 2, 3]
    [36],
    [40],
    [44]]   

    广播后得到：
    [[32, 33, 34, 35],
    [36, 37, 38, 39],
    [40, 41, 42, 43],
    [44, 45, 46, 47]]

    这就是 A tile 中每个元素相对于 `A` 首地址的偏移。注意这里计算的是元素偏移，不是字节数
    """
    for k_idx in range(0, K, BLOCK_K):
        # * 本次地址 实际上是矩阵
        a_ptrs = (
            A
            + offsets_m[:, None] * stride_am
            + (k_idx + offsets_k[None, :]) * stride_ak
        )
        # * 本次地址 实际上是矩阵
        b_ptrs = (
            B
            + (k_idx + offsets_k[:, None]) * stride_bk
            + offsets_n[None, :] * stride_bn
        )
        a_mask = (offsets_m[:, None] < M) & (k_idx + offsets_k[None, :] < K)
        b_mask = (k_idx + offsets_k[:, None] < K) & (offsets_n[None, :] < N)
        a = tl.load(a_ptrs, mask=a_mask, other=0.0)
        b = tl.load(b_ptrs, mask=b_mask, other=0.0)
        acc += tl.dot(a, b)

    c_ptrs = C + offsets_m[:, None] * stride_cm + offsets_n[None, :] * stride_cn
    c_mask = (offsets_m[:, None] < M) & (offsets_n[None, :] < N)
    tl.store(c_ptrs, acc, mask=c_mask)

    return


def matmul(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    M, K = A.shape
    K2, N = B.shape
    assert K == K2

    C = torch.empty((M, N), device=A.device, dtype=torch.float32)

    block_m, block_n, block_k = 16, 16, 32

    grid = (triton.cdiv(M, block_m), triton.cdiv(N, block_n))

    gemm_kernel[grid](
        A,
        B,
        C,
        M,
        N,
        K,
        A.stride(0),
        A.stride(1),
        B.stride(0),
        B.stride(1),
        C.stride(0),
        C.stride(1),
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_K=block_k,
    )
    return C


def main():
    torch.manual_seed(0)
    device = "cuda"

    test_shapes = [
        (32, 32, 32),  # M/N/K 都与 block 对齐
        (37, 53, 29),  # M/N/K 都不对齐，且 K 小于 BLOCK_K
        (70, 45, 70),  # 多 block，K 尾部不对齐
        (1, 1, 17),  # 单行单列
        (128, 256, 64),  # 较大规则矩阵
    ]

    for M, N, K in test_shapes:
        A = torch.randn(M, K, device=device, dtype=torch.float32)
        B = torch.randn(K, N, device=device, dtype=torch.float32)

        C = matmul(A, B)
        ref = A @ B
        # fp32 输入默认可能使用 TF32，因此不使用 bitwise 相等判断
        ok = torch.allclose(C, ref, rtol=1e-2, atol=1e-2)
        max_diff = (C - ref).abs().max().item()
        print(
            f"[{'PASS' if ok else 'FAIL'}] M={M}, N={N}, K={K}, "
            f"max_diff={max_diff:.6f}"
        )
        if not ok:
            raise AssertionError(f"GEMM result mismatch: M={M}, N={N}, K={K}")


if __name__ == "__main__":
    main()
