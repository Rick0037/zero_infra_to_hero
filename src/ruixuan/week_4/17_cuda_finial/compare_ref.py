# 与 kernel 注释里的参考实现对齐:
#   softmax  -> torch.softmax(dim=-1), atol=1e-4
#   quantize -> numpy reference(每行 absmax -> scale=absmax/127 -> round+clip), 允许 ±1 量化级
#   gemv     -> torch.mv, rtol=1e-2
#   fused    -> torch.softmax 后接 numpy 量化 reference, 允许 ±1 量化级
#
# 用法: 先在 CUDA 里 --dump 跑一遍生成 bin, 再 python compare_ref.py
# 注意: M, N 要和 CUDA 里的宏保持一致
import numpy as np
import torch

M, N = 1024, 512


def load_bin(path, dtype, shape):
    return np.fromfile(path, dtype=dtype).reshape(shape)


def dump_bin(path, arr):
    arr.tofile(path)


# ---------- 输入数据: 与 CUDA 里的 init 公式逐一对齐 ----------
# softmax / quantize 输入: h_in[i] = ((i % 10) - 5) * (0.25 + 0.5 * (i / N % 8))
idx = np.arange(M * N)
factor = (0.25 + 0.5 * (idx // N % 8)).astype(np.float32)
x = (((idx % 10) - 5).astype(np.float32) * factor).reshape(M, N)

# gemv 输入: A[i] = (i % 7) - 3, x_vec[j] = (j % 5) - 2
# 注意 CUDA main 里是 GemvTest(N, N), 行数是 N=4096 不是 M
MG = N  # gemv 的 m
a_mat = ((np.arange(MG * N) % 7) - 3).astype(np.float16).reshape(MG, N)
a_vec = ((np.arange(N) % 5) - 2).astype(np.float16)

rows = []


def add_row(op, shape, max_err_str, passed):
    rows.append((op, shape, max_err_str, "PASS" if passed else "FAIL"))


# ---------- 1. softmax ----------
ref_softmax = torch.softmax(torch.from_numpy(x), dim=-1).numpy()

baseline_out = load_bin("softmax_baseline.bin", np.float32, (M, N))
err = np.abs(baseline_out - ref_softmax).max()
add_row("softmax baseline(3-pass)", f"[{M},{N}] fp32", f"{err:.3e}", err <= 1e-4)

online_out = load_bin("softmax_online.bin", np.float32, (M, N))
err = np.abs(online_out - ref_softmax).max()
add_row("softmax online(1-pass)", f"[{M},{N}] fp32", f"{err:.3e}", err <= 1e-4)

# ---------- 2. per-token 对称 int8 quantize ----------
absmax = np.abs(x).max(axis=1)
ref_scale = absmax / 127.0
ref_q = np.rint(x / ref_scale[:, None])
ref_q = np.clip(ref_q, -128, 127).astype(np.int8)

gpu_q = load_bin("quant_out.bin", np.int8, (M, N))
err_q = np.abs(gpu_q.astype(np.int32) - ref_q.astype(np.int32)).max()
add_row("quantize int8", f"[{M},{N}]", f"{err_q} (级)", err_q <= 1)

gpu_scale = load_bin("scale_out.bin", np.float32, (M,))
err_s = np.abs(gpu_scale - ref_scale).max()
add_row("quantize scale", f"[{M}]", f"{err_s:.3e}", err_s <= 1e-6)

# ---------- 3. gemv fp16: y = A @ x ----------
ref_y = (
    torch.mv(torch.from_numpy(a_mat), torch.from_numpy(a_vec))
    .numpy()
    .astype(np.float32)
)

gpu_y = load_bin("gemv_y.bin", np.float16, (MG,)).astype(np.float32)
diff = np.abs(gpu_y - ref_y)
# 与 C++ check 同判据: |diff| <= 1e-2 * |ref|
ok = np.all(diff <= 1e-2 * np.abs(ref_y))
# 最大相对误差只统计 ref != 0 的行
mask = ref_y != 0
rel_err = (diff[mask] / np.abs(ref_y[mask])).max()
add_row("gemv fp16", f"[{MG},{N}]x[{N}]", f"{rel_err:.3e} (相对)", ok)

# ---------- 4. 融合 kernel: softmax -> int8 一次写回 ----------
# 参考: 先 torch.softmax, 再按 per-token 对称量化(absmax = 每行最大概率 = 1/sum)
ref_p = torch.softmax(torch.from_numpy(x), dim=-1).numpy()
ref_fused_scale = ref_p.max(axis=1) / 127.0
ref_fused_q = np.rint(ref_p / ref_fused_scale[:, None])
ref_fused_q = np.clip(ref_fused_q, -128, 127).astype(np.int8)

fused_q = load_bin("fused_quant_out.bin", np.int8, (M, N))
err_fq = np.abs(fused_q.astype(np.int32) - ref_fused_q.astype(np.int32)).max()
add_row("fused softmax+quant int8", f"[{M},{N}]", f"{err_fq} (级)", err_fq <= 1)

fused_scale = load_bin("fused_scale_out.bin", np.float32, (M,))
err_fs = np.abs(fused_scale - ref_fused_scale).max()
add_row("fused softmax+quant scale", f"[{M}]", f"{err_fs:.3e}", err_fs <= 1e-6)

# ---------- 对齐结果表 ----------
print(f"{'op':<26} {'shape':<18} {'max_err':<16} {'result'}")
print("-" * 70)
for op, shape, err_str, res in rows:
    print(f"{op:<26} {shape:<18} {err_str:<16} {res}")
