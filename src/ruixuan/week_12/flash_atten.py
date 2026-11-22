import math
import torch
import triton
import triton.language as tl


@triton.jit
def _flash_atten_v1_kernel(
    Q, K, V, O,    # [B, H, N, D]
    M, L,          # [B, H, N]
    stride_qb, stride_qh, stride_qn, stride_qd,
    stride_kb, stride_kh, stride_kn, stride_kd,
    stride_vb, stride_vh, stride_vn, stride_vd,
    stride_ob, stride_oh, stride_on, stride_od,
    stride_mb, stride_mh, stride_mn,
    N,
    D: tl.constexpr,        # 注意力头维度（head dimension），即每个 token 的特征长度，用在 scale = 1/sqrt(D)
    BLOCK_M: tl.constexpr,  # Q 块行数 B_r
    BLOCK_N: tl.constexpr,  # K/V 块行数 B_c
    BLOCK_D: tl.constexpr,  # 头维度上加载/计算的块宽度，当D本身就是2的幂时，BLOCK_D==D；否则 BLOCK_D > D，多出来的位置用 mask 屏蔽掉
):
    # ========== V1 并行划分：grid = (batch, head) ==========
    # 每个 program 串行扫完整个 (batch, head) 的所有 K/V 块和 Q 块
    batch_id = tl.program_id(0)
    head_id = tl.program_id(1)

    # 张量在起始位置的偏移, 关于batch 以及 head的偏移
    offsets_d = tl.arange(0, BLOCK_D)
    scale = 1.0 / math.sqrt(D)
    q_block_start = Q + batch_id * stride_qb + head_id* stride_qh
    k_block_start = K + batch_id * stride_kb + head_id* stride_kh
    v_block_start = V + batch_id * stride_vb + head_id* stride_vh
    #* 输出以及中间遍量也需要读取进来
    o_block_start = O + batch_id * stride_ob + head_id* stride_oh
    m_block_start = M + batch_id * stride_mb + head_id * stride_mh
    L_block_start = L + batch_id * stride_mb + head_id * stride_mh

    for start_kv in range(0, N, BLOCK_N):
        offsets_kv = start_kv + tl.arange(0, BLOCK_N)
        mask_kv = (offsets_kv[:, None] < N) & (offsets_d[None, :] < D)
        k_current_offset = offsets_kv[:, None] * stride_kn + offsets_d[None, :] * stride_kd
        v_current_offset = offsets_kv[:, None] * stride_vn + offsets_d[None, :] * stride_vd
        #* ----- load-------
        k_val =  tl.load(k_block_start + k_current_offset, mask=mask_kv, other = 0.0)
        v_val =  tl.load(v_block_start + v_current_offset, mask=mask_kv, other = 0.0)

        for start_q in range(0, N, BLOCK_M): # * 这里 也是遍历N 
            offset_q = start_q + tl.arange(0, BLOCK_M)
            mask_q = (offset_q[:, None] < N) & (offsets_d[None, :] < D)
 
            #* 对应的矩阵
            q_current_offset = offset_q[:, None] * stride_qn +offsets_d[None, :] * stride_qd
            q_val = tl.load(q_block_start + q_current_offset, mask=mask_q, other=0.0 )
            o_current_offset = offset_q[:, None] * stride_on +offsets_d[None, :] * stride_od
            o_val = tl.load(o_block_start + o_current_offset, mask=mask_q, other=0.0 )
            # M/L 每个 Q token 对应一个值，形状均为 (BLOCK_M,)
            m_current_offset = offset_q * stride_mn
            mask_ml = offset_q < N #! 注意这里是N 的维度, 因为一行一个的数值, 不是D 的维度 
            m_val = tl.load(m_block_start + m_current_offset, mask=mask_ml, other=float('-inf'))
            l_val = tl.load(L_block_start + m_current_offset, mask=mask_ml, other=0.0)
            # import pdb; pdb.set_trace()

            s = tl.dot(q_val, tl.trans(k_val)) * scale
            # 屏蔽 K/V 越界列，防止假 0 污染 softmax 分母
            valid_kv = offsets_kv < N  # 只判断 K token 是否越界，形状为 (BLOCK_N,)
            s = tl.where(valid_kv[None, :], s, float('-inf'))
            #*逐元素取较大值 ，不会改变形状
            m_new = tl.maximum(m_val, tl.max(s, axis=1))
            alpha = tl.exp(m_val-m_new)
            #* 
            new_part_sum_value = tl.exp(s - m_new[:, None])
            l_new = alpha * l_val + tl.sum(new_part_sum_value, axis=1)

            #* ouput  输出修正 + 累加（与 5.1 Step 5 公式一致）
            o_new = o_val * (alpha * l_val / l_new)[:, None] + tl.dot(new_part_sum_value.to(v_val.type), v_val) / l_new[:, None] 

            tl.store(o_block_start + o_current_offset, o_new.to(O.dtype.element_ty),mask=mask_q)

            tl.store(m_block_start + m_current_offset, m_new, mask=mask_ml)
            tl.store(L_block_start + m_current_offset, l_new, mask=mask_ml)




def flash_attn_v1(Q, K, V, BLOCK_M=64, BLOCK_N=64):
    # Q, K, V: [B, H, N, D]
    B, H, N, D = Q.shape
    O = torch.zeros_like(Q)
    M = torch.full((B, H, N), float('-inf'), device=Q.device, dtype=torch.float32)
    L = torch.zeros((B, H, N), device=Q.device, dtype=torch.float32)

    BLOCK_D = triton.next_power_of_2(D)
    grid = (B, H)  # V1 并行：仅在 batch×head 维度上
    _flash_atten_v1_kernel[grid](
        Q, K, V, O, M, L,
        Q.stride(0), Q.stride(1), Q.stride(2), Q.stride(3),
        K.stride(0), K.stride(1), K.stride(2), K.stride(3),
        V.stride(0), V.stride(1), V.stride(2), V.stride(3),
        O.stride(0), O.stride(1), O.stride(2), O.stride(3),
        M.stride(0), M.stride(1), M.stride(2),
        N, D=D, BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_D=BLOCK_D,
    )
    return O


if __name__ == "__main__":
    torch.manual_seed(0)
    device = "cuda"

    def check(name, shape, block_m=32, block_n=32, input_scale=1.0):
        B, H, N, D = shape
        Q = torch.randn(shape, device=device, dtype=torch.float16) * input_scale
        K = torch.randn(shape, device=device, dtype=torch.float16) * input_scale
        V = torch.randn(shape, device=device, dtype=torch.float16)

        O = flash_attn_v1(Q, K, V, BLOCK_M=block_m, BLOCK_N=block_n)

        # 使用 FP32 PyTorch 注意力作为参考结果
        scores = (Q.float() @ K.float().transpose(-2, -1)) / math.sqrt(D)
        ref = torch.softmax(scores, dim=-1) @ V.float()
        max_diff = (O.float() - ref).abs().max().item()
        ok = (
            torch.isfinite(O).all().item()
            and torch.allclose(O.float(), ref, rtol=3e-2, atol=3e-2)
        )
        print(
            f"[{'PASS' if ok else 'FAIL'}] {name}: "
            f"B={B}, H={H}, N={N}, D={D}, max_diff={max_diff:.6f}"
        )
        if not ok:
            raise AssertionError(f"Attention result mismatch: {name}")

    # 对齐：一个 Q 块、一个 K/V 块
    check("基础对齐", (1, 1, 32, 32))

    # N 和 D 都不对齐，覆盖 Q/K/V 的二维 mask
    check("N/D 尾部不对齐", (1, 2, 37, 48), block_m=16, block_n=16)

    # 多 batch、多 head，并覆盖多个 Q/K/V 块
    check("多 batch/head 与多块", (2, 2, 50, 32), block_m=32, block_n=32)

    # 单 token：softmax 只有一个元素，输出应近似等于 V
    check("单 token", (1, 1, 1, 32), block_m=16, block_n=16)

    # 大 logits：验证 online softmax 不会产生 NaN/Inf
    check("大 logits 数值稳定性", (1, 1, 33, 32), block_m=16, block_n=16, input_scale=10.0)

    print("All attention tests passed.")


# @triton.jit
# def _flash_attn_v1_kernel(
#     Q, K, V, O,    # [B, H, N, D]
#     M, L,          # 行最大值与行指数和
#     stride_qb, stride_qh, stride_qn, stride_qd,
#     stride_kb, stride_kh, stride_kn, stride_kd,
#     stride_vb, stride_vh, stride_vn, stride_vd,
#     stride_ob, stride_oh, stride_on, stride_od,
#     stride_mb, stride_mh,
#     N,
#     D: tl.constexpr,        # 注意力头维度（head dimension），即每个 token 的特征长度，用在 scale = 1/sqrt(D)
#     BLOCK_M: tl.constexpr,  # Q 块行数 B_r
#     BLOCK_N: tl.constexpr,  # K/V 块行数 B_c
#     BLOCK_D: tl.constexpr,  # 头维度上加载/计算的块宽度，当D本身就是2的幂时，BLOCK_D==D；否则 BLOCK_D > D，多出来的位置用 mask 屏蔽掉
# ):
#     # ========== V1 并行划分：grid = (batch, head) ==========
#     # 每个 program 串行扫完整个 (batch, head) 的所有 K/V 块和 Q 块
#     batch_id = tl.program_id(0)
#     head_id = tl.program_id(1)

#     # 当前 (batch, head) 在各张量上的起始偏移
#     qkvo_bh_q = batch_id * stride_qb + head_id * stride_qh
#     qkvo_bh_k = batch_id * stride_kb + head_id * stride_kh
#     qkvo_bh_v = batch_id * stride_vb + head_id * stride_vh
#     qkvo_bh_o = batch_id * stride_ob + head_id * stride_oh
#     ml_bh = batch_id * stride_mb + head_id * stride_mh

#     d_range = tl.arange(0, BLOCK_D)
#     scale = 1.0 / math.sqrt(D)

#     # ============ 外循环：遍历 K/V 块（V1 关键设计） ============
#     for kv_start in range(0, N, BLOCK_N):
#         kv_range = kv_start + tl.arange(0, BLOCK_N)
#         kv_mask = kv_range < N

#         # K_j, V_j 一次加载，常驻寄存器/SRAM，被所有 Q 块复用
#         k = tl.load(K + qkvo_bh_k + kv_range[:, None] * stride_kn + d_range[None, :] * stride_kd,
#                     mask=kv_mask[:, None] & (d_range[None, :] < D), other=0.0)
#         v = tl.load(V + qkvo_bh_v + kv_range[:, None] * stride_vn + d_range[None, :] * stride_vd,
#                     mask=kv_mask[:, None] & (d_range[None, :] < D), other=0.0)

#         # ============ 内循环：遍历 Q 块 ============
#         for q_start in range(0, N, BLOCK_M):
#             q_range = q_start + tl.arange(0, BLOCK_M)
#             q_mask = q_range < N

#             # 加载 Q_i, O_i, m_i, l_i —— V1 中每个外循环都要从 HBM 重新读
#             q = tl.load(Q + qkvo_bh_q + q_range[:, None] * stride_qn + d_range[None, :] * stride_qd,
#                         mask=q_mask[:, None] & (d_range[None, :] < D), other=0.0)
#             o_i = tl.load(O + qkvo_bh_o + q_range[:, None] * stride_on + d_range[None, :] * stride_od,
#                           mask=q_mask[:, None] & (d_range[None, :] < D), other=0.0).to(tl.float32)
#             m_i = tl.load(M + ml_bh + q_range, mask=q_mask, other=float('-inf'))
#             l_i = tl.load(L + ml_bh + q_range, mask=q_mask, other=0.0)

#             # S_ij = Q_i @ K_j^T / sqrt(d)
#             s = tl.dot(q, tl.trans(k)) * scale
#             # 屏蔽 K/V 越界列，防止假 0 污染 softmax 分母
#             s = tl.where(kv_mask[None, :], s, float('-inf'))

#             # Online Softmax 更新
#             m_new = tl.maximum(m_i, tl.max(s, axis=1))
#             alpha = tl.exp(m_i - m_new)
#             p = tl.exp(s - m_new[:, None])
#             l_new = alpha * l_i + tl.sum(p, axis=1)

#             # 输出修正 + 累加（与 5.1 Step 5 公式一致）
#             o_new = o_i * (alpha * l_i / l_new)[:, None] \
#                     + tl.dot(p.to(v.dtype), v) / l_new[:, None]

#             # 把更新后的 O_i, m_i, l_i 写回 HBM，供下次外循环使用
#             tl.store(O + qkvo_bh_o + q_range[:, None] * stride_on + d_range[None, :] * stride_od,
#                      o_new.to(O.dtype.element_ty),
#                      mask=q_mask[:, None] & (d_range[None, :] < D))
#             tl.store(M + ml_bh + q_range, m_new, mask=q_mask)
#             tl.store(L + ml_bh + q_range, l_new, mask=q_mask)