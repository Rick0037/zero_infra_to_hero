"""
用 numpy 或 PyTorch 实现 SmoothQuant 的 activation→weight 迁移，量化对比其效果。
1. 构造带明显 outlier 的激活 X (shape [B=8, S=128, C=512])：随机选 32 个通道乘以 30x 放大；
权重 W (shape [C=512, C_out=512]) 服从 N(0, 0.05)。

2. 计算 per-channel 迁移因子 s_j = max(|X_j|)^a / max(|W_j|)^(1-a),
取 a=0.5, 得到 X' = X/s、W' = diag(s)·W (保持 Y = XW 数学等价）。

3. 分别对 (X, W) 与 (X', W') 做 per-tensor 对称 INT8 量化+反量化，算最终 Y_int8 与 Y_fp32 的 MSE。
期望产出：一段可跑脚本，输出两个 MSE 数值, SmoothQuant 版本应显著低于 naive 版本（一般 5x-100x)。
测试方法, sweep aplha ∈ {0.3, 0.5, 0.7, 0.9}，画 MSE-vs-alpha 曲线；同时打印 X' 和 W' 的 |·|_max,
验证 activation 的量化难度确实被均摊到 weight 上。可选：把 INT8 量化替换成 FP8 (E4M3) 模拟量化，对比同一 a 下两者的 MSE 差异。

"""

import torch
import numpy as np


def mse(a, b):
    return ((a - b) ** 2).mean().item()


def per_tensor_quant(x: torch.Tensor, bit_size=8):
    x_max = x.abs().amax()
    int8_min = -(2 ** (bit_size - 1))  # -128
    int8_max = 2 ** (bit_size - 1) - 1  # 127
    scale = torch.clamp(x_max / int8_max, 1e-5)
    x_q = torch.round(x.div(scale)).clamp(int8_min, int8_max)
    return x_q, scale


def per_tensor_fake_quant(x: torch.Tensor, bit_size=8):
    x_q, scale = per_tensor_quant(x, bit_size)
    return x_q * scale


def fp8_fake_quant(x: torch.Tensor):
    x_max = x.abs().amax().clamp(1e-5)
    scale = x_max / 448.0
    return (x / scale).to(torch.float8_e4m3fn).float() * scale


def test_smooth_quant(alpha=0.5):
    # init x -> [8, 128, 512], init w -> [512, 512]
    torch.manual_seed(42)  # 固定种子，保证 sweep 时各 alpha 用同一组数据
    x = torch.randn(8, 128, 512, dtype=torch.float32)
    idx = torch.randperm(512)[:32]
    x[:, :, idx] *= 30
    w = torch.randn(512, 512, dtype=torch.float32) * 0.05

    # per-channel x_max, w_max
    x_max = x.abs().amax(dim=(0, 1))
    w_max = w.abs().amax(dim=1)
    s_j = (x_max**alpha) / (w_max ** (1 - alpha))

    # x_head, w_head
    x_head = x / s_j
    w_head = w * s_j.unsqueeze(1)

    # y_equiv = torch.matmul(x_head, w_head)
    # print((y_fp32 - y_equiv).abs().max())  # 应接近 0（~1e-4 浮点误差）

    x_fake_quant = per_tensor_fake_quant(x_head)
    w_fake_quant = per_tensor_fake_quant(w_head)
    y_int8_sq = torch.matmul(x_fake_quant, w_fake_quant)

    # naive: 直接量化 (X, W)
    x_q, sx = per_tensor_quant(x)
    w_q, sw = per_tensor_quant(w)
    y_int8_naive = torch.matmul(x_q * sx, w_q * sw)  # 反量化后再 matmul

    # FP8 (E4M3) 模拟量化，同一 alpha 下对比
    y_fp8_naive = torch.matmul(fp8_fake_quant(x), fp8_fake_quant(w))
    # y_fp8_sq = torch.matmul(fp8_e4m3_fake_quant(x_head), fp8_e4m3_fake_quant(w_head))

    y_fp32 = torch.matmul(x, w)
    mse_naive = mse(y_int8_naive, y_fp32)
    mse_sq = mse(y_int8_sq, y_fp32)
    mse_fp8_naive = mse(y_fp8_naive, y_fp32)
    # mse_fp8_sq = mse(y_fp8_sq, y_fp32)
    print(
        f"alpha={alpha}: INT8 naive={mse_naive:.6f}, SQ={mse_sq:.6f} | "
        f"FP8 naive={mse_fp8_naive:.6f}"
    )
    print(f"  |X'|_max={x_head.abs().amax():.4f}, |W'|_max={w_head.abs().amax():.4f}")
    return mse_naive, mse_sq


def sweep_alpha():
    import matplotlib.pyplot as plt

    alphas = [0.3, 0.5, 0.7, 0.9]
    naive_mses, sq_mses = [], []
    for a in alphas:
        m_naive, m_sq = test_smooth_quant(alpha=a)
        naive_mses.append(m_naive)
        sq_mses.append(m_sq)

    plt.figure(figsize=(6, 4))
    plt.plot(alphas, naive_mses, "o-", label="naive INT8")
    plt.plot(alphas, sq_mses, "s-", label="SmoothQuant INT8")
    plt.xlabel("alpha")
    plt.ylabel("MSE (Y_int8 vs Y_fp32)")
    plt.yscale("log")
    plt.legend()
    plt.grid(True, which="both", alpha=0.3)
    plt.title("MSE vs alpha")
    plt.tight_layout()
    plt.savefig(
        "mse_vs_alpha.png",
        dpi=120,
    )
    plt.show()


if __name__ == "__main__":
    sweep_alpha()
