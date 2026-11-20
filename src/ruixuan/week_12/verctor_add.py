import torch
import triton
import triton.language as tl


"""
    -----kernel------
    1. program_id, grid 层面到底分成了几个块, 看是几维度的, 从grid 传入的方式就可以看出来
    2. 记录offsets, 代表整个块的数据, 必须是 arange(0, BLOCK_SIZE)
    3. tl.load(point, mask, others)
    4. 计算
    5. tl.store(point, 内容, mask)
    -----调用方------
    1. 设定block size
    2. 计算grid
    3. add_kernel[grid] 调用
"""


@triton.jit
def add_kernel(x, y, z, n: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    # * block idx
    pid = tl.program_id(0)
    # * block_idx * block_size, 这里是一整个tile的 offsets, 不是一个线程的
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    # * mask为True的部分是有效元素(offset < n),load时无效位置用other=0.0填充,store时无效位置不写入
    mask = offsets < n
    x_val = tl.load(x + offsets, mask=mask, other=0.0)
    y_val = tl.load(y + offsets, mask=mask, other=0.0)
    z_val = x_val + y_val
    tl.store(z + offsets, z_val, mask=mask)  # * 只写mask为True的有效位置


def verctor_add(x, y):
    z = torch.empty_like(x)
    n = x.numel()
    block_size = 1024
    grid = (triton.cdiv(n, block_size),)  # * 这里只写了一个维度的
    add_kernel[grid](x, y, z, n, block_size)

    return z


if __name__ == "__main__":
    device = "cuda"
    torch.manual_seed(42)

    def check(name, x, y):
        x, y = x.to(device), y.to(device)
        out = verctor_add(x, y)
        ref = x + y
        ok = torch.allclose(out, ref)
        print(f"[{'PASS' if ok else 'FAIL'}] {name}: shape={list(x.shape)}")
        if not ok:
            print(f"  out: {out}")
            print(f"  ref: {ref}")

    # 1. BLOCK_SIZE 刚好整除 n
    check("n=BLOCK_SIZE 对齐", torch.randn(1024), torch.randn(1024))

    # 2. 最后一个 tile 不满(BLOCK_SIZE 不整除 n) → mask 边界条件
    check("n=BLOCK_SIZE+3 尾部残留", torch.randn(1027), torch.randn(1027))

    # 3. 比 BLOCK_SIZE 小很多(只有一个 block)
    check("n=5 远小于 BLOCK_SIZE", torch.randn(5), torch.randn(5))

    # 4. 多 block 大数组
    check("n=5*BLOCK_SIZE 多 block", torch.randn(5120), torch.randn(5120))

    # 5. 尾部刚好只剩 1 个元素
    check("n=2*BLOCK_SIZE+1 尾部单元素", torch.randn(2049), torch.randn(2049))

    # 6. 二维 tensor(flatten 后还是 vector add)
    check("2D tensor", torch.randn(3, 4), torch.randn(3, 4))

    # 7. 负数 + 正数混合
    check(
        "正负混合",
        torch.tensor([-1.0, 0.0, 1.0, 3.14]),
        torch.tensor([2.0, -0.5, 1.0, -3.14]),
    )

    # 8. 全零
    check("全零", torch.zeros(2048), torch.zeros(2048))

    # 9. 大数值(测试精度)
    check("大数值", torch.full((100,), 1e6), torch.full((100,), 2e6))

    # 10. 非 contiguous tensor(带 stride)
    x = torch.randn(100, 5)[:, ::2].contiguous().reshape(-1)
    y = torch.randn(300)
    check("stride/reshaped tensor", x, y)
