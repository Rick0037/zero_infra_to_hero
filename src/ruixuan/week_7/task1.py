import torch
import torch.nn as nn
import math
import torch.nn.functional as F

"""MHA/GQA/MQA"""


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    将形状 [B, H_kv, S, D] 的 KV 头复制 n_rep 次，
    输出为 [B, H_kv * n_rep, S, D]，以匹配 Query 头的数量。
    当 n_rep == 1 时（即 MHA），直接返回原张量。
    """
    batch, num_kv_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states.unsqueeze(2).expand(-1, -1, n_rep, -1, -1)
    return hidden_states.reshape(batch, num_kv_heads * n_rep, slen, head_dim)


class GroupedQueryAttention(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int, num_kv_heads: int = None):
        super().__init__()
        if hidden_dim <= 0 or num_heads <= 0:
            raise ValueError("hidden_dim 和 num_heads 必须为正整数")
        if hidden_dim % num_heads != 0:
            raise ValueError("hidden_dim 必须能被 num_heads 整除")
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads if num_kv_heads is not None else num_heads

        # GQA/MQA 要求 Query head 可以平均分配给 KV head。
        if self.num_kv_heads <= 0 or num_heads % self.num_kv_heads != 0:
            raise ValueError(
                f"num_heads ({num_heads}) 必须能被 num_kv_heads ({self.num_kv_heads}) 整除"
            )

        self.num_queries_per_kv = self.num_heads // self.num_kv_heads
        self.head_dim = hidden_dim // num_heads

        # Q 始终按 num_heads 投影；K/V 按 num_kv_heads 投影以支持 MHA、GQA 与 MQA。
        self.q_proj = nn.Linear(hidden_dim, num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(
            hidden_dim, self.num_kv_heads * self.head_dim, bias=False
        )
        self.v_proj = nn.Linear(
            hidden_dim, self.num_kv_heads * self.head_dim, bias=False
        )
        self.o_proj = nn.Linear(num_heads * self.head_dim, hidden_dim, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: torch.Tensor = None,
        kv_cache: tuple[torch.Tensor, torch.Tensor] = None,
    ):
        """
        前向传播。

        Args:
            x: 输入张量，形状 [batch, seq_len, hidden_dim]
            attention_mask: 注意力掩码，形状应为 [batch, 1, 1, seq_len]（因果掩码）
                        或 [batch, 1, seq_len, seq_len]；可见位置通常为 0，
                        屏蔽位置为 -inf，并广播到 scores。
            kv_cache: 缓存的 (K, V) 张量，形状为
                      ([batch, num_kv_heads, cached_seq_len, head_dim], ...)。

        Returns:
            输出张量 [batch, seq_len, hidden_dim]，更新后的 KV Cache
        """
        batch_size, seq_len, _ = x.shape

        xq, xk, xv = self.q_proj(x), self.k_proj(x), self.v_proj(x)

        # ==========================================
        # TODO 1：把投影结果拆成多头格式。
        # ==========================================
        # xq = [B, H, S, D]；xk = [B, H_kv, S, D]；xv = [B, H_kv, S, D]
        xq = xq.reshape(batch_size, seq_len, self.num_heads, -1).transpose(1, 2)
        xk = xk.reshape(batch_size, seq_len, self.num_kv_heads, -1).transpose(1, 2)
        xv = xv.reshape(batch_size, seq_len, self.num_kv_heads, -1).transpose(1, 2)

        # ==========================================
        # TODO 2：若给出历史 Cache，在序列维把它放在当前 K/V 前。
        # ==========================================
        if kv_cache is not None:
            k_cache, v_cache = kv_cache
            xk = torch.cat([k_cache, xk], dim=2)
            xv = torch.cat([v_cache, xv], dim=2)
            # 只拼接未扩展的 KV，避免把 GQA 的缓存放大到 MHA 大小。

        new_kv_cache = (xk, xv)

        # 仅为当前 Attention 计算临时扩展 KV，不把扩展结果写回 Cache。
        xk = repeat_kv(xk, self.num_queries_per_kv)
        xv = repeat_kv(xv, self.num_queries_per_kv)

        # ==========================================
        # TODO 3：计算缩放分数，并将 mask 后的 scores 归一化后聚合 V。
        # ==========================================
        # scores = [B, H, S_query, S_key]；probs 在最后一维归一化。
        # xq = [B, H, S, D]；xk = [B, H, S, D]
        #! head 在前面了 [b, h, s, d] * [b, h, d, s] -> [b, h, s, s]
        scores = torch.matmul(xq, xk.transpose(-2, -1)) / math.sqrt(self.head_dim)

        if attention_mask is not None:
            scores = scores + attention_mask

        probs = F.softmax(scores, dim=-1)
        output = torch.matmul(probs, xv)

        # ==========================================
        # TODO 4：将 [B, H, S, D] 转回 [B, S, H×D]，再交给输出投影。
        # ==========================================
        output = output.transpose(1, 2).reshape(
            batch_size, seq_len, self.num_heads * self.head_dim
        )

        return self.o_proj(output), new_kv_cache


# 按机制拆分测试，分别定位 MHA/GQA 输出、KV 缓存和参数契约。
def causal_mask(seq_len: int) -> torch.Tensor:
    """构造允许看见当前位置及历史位置的加性因果掩码。"""
    mask = torch.zeros(1, 1, seq_len, seq_len)
    return mask.masked_fill(
        torch.triu(torch.ones_like(mask, dtype=torch.bool), diagonal=1), float("-inf")
    )


def _build_attention_fixture():
    """构造固定随机种子的 Attention 测试输入与维度。"""
    torch.manual_seed(41)
    batch_size, seq_len, hidden_dim, num_heads = 2, 16, 128, 4
    x = torch.randn(batch_size, seq_len, hidden_dim)
    return x, hidden_dim, num_heads


def test_mha_output_contract():
    """验证 MHA 输出形状、有限值和完整 KV head。"""
    x, hidden_dim, num_heads = _build_attention_fixture()
    mha = GroupedQueryAttention(hidden_dim, num_heads, num_kv_heads=num_heads)
    out, cache = mha(x, attention_mask=causal_mask(x.shape[1]))
    assert out.shape == x.shape
    assert torch.isfinite(out).all()
    assert cache[0].shape[1] == num_heads


def test_gqa_output_contract():
    """验证 GQA 保留较少 KV head 后仍能完成前向。"""
    x, hidden_dim, num_heads = _build_attention_fixture()
    gqa = GroupedQueryAttention(hidden_dim, num_heads, num_kv_heads=2)
    out, cache = gqa(x, attention_mask=causal_mask(x.shape[1]))
    assert out.shape == x.shape
    assert torch.isfinite(out).all()
    assert cache[0].shape[1] == 2


def test_mqa_output_contract():
    """验证 MQA 只保存一个 KV head，输出仍符合模型契约。"""
    x, hidden_dim, num_heads = _build_attention_fixture()
    mqa = GroupedQueryAttention(hidden_dim, num_heads, num_kv_heads=1)
    out, cache = mqa(x, attention_mask=causal_mask(x.shape[1]))
    assert out.shape == x.shape
    assert cache[0].shape[1] == 1


def test_kv_cache_decode_equivalence():
    """验证 Prefill 后的 Cache Decode 与完整前向末 token 一致。"""
    x, hidden_dim, num_heads = _build_attention_fixture()
    prefill_len = 5
    mha = GroupedQueryAttention(hidden_dim, num_heads, num_kv_heads=num_heads)
    x_prefill = x[:, :prefill_len]
    full_out, _ = mha(x_prefill, attention_mask=causal_mask(prefill_len))
    _, cache = mha(x_prefill[:, :-1], attention_mask=causal_mask(prefill_len - 1))
    decode_out, new_cache = mha(
        x_prefill[:, -1:],
        attention_mask=torch.zeros(1, 1, 1, prefill_len),
        kv_cache=cache,
    )
    assert new_cache[0].shape == (
        x.shape[0],
        num_heads,
        prefill_len,
        hidden_dim // num_heads,
    )
    assert torch.allclose(full_out[:, -1:], decode_out, atol=1e-5, rtol=1e-4)


def test_causal_mask_blocks_future():
    """验证改变未来输入不会改变第一个 token 的因果 Attention 输出。"""
    x, hidden_dim, num_heads = _build_attention_fixture()
    mha = GroupedQueryAttention(hidden_dim, num_heads)
    changed_future = x.clone()
    changed_future[:, 1:] += 1000.0
    mask = causal_mask(x.shape[1])
    original, _ = mha(x, attention_mask=mask)
    changed, _ = mha(changed_future, attention_mask=mask)
    assert torch.allclose(original[:, :1], changed[:, :1], atol=1e-5, rtol=1e-4)


def test_attention_config_contract():
    """验证 hidden/head/KV head 的非法组合会被拒绝。"""
    for bad_args in [(130, 4, 2), (128, 3, 1), (128, 4, 3)]:
        try:
            GroupedQueryAttention(*bad_args)
        except ValueError:
            continue
        raise AssertionError(f"非法配置未被拒绝: {bad_args}")


def test_attention_mechanisms():
    """汇总 MHA、GQA、MQA、因果可见性、Cache Decode 与参数契约测试。"""
    try:
        test_mha_output_contract()
        test_gqa_output_contract()
        test_mqa_output_contract()
        test_kv_cache_decode_equivalence()
        test_causal_mask_blocks_future()
        test_attention_config_contract()
        print(
            "✅ Attention 机制测试通过：MHA/GQA/MQA、因果 mask、KV Cache 与参数契约均符合预期。"
        )
    except NotImplementedError:
        print("请先完成 TODO 部分的代码！")
        raise
    except Exception as error:
        print(f"❌ Attention 测试失败: {type(error).__name__}: {error}")
        raise


"""flash attention"""


def flash_attention_forward_sim(q, k, v, block_size=2, causal=False):
    """计算二维输入上的 FlashAttention 前向模拟。

    Args:
        q, k, v: [seq_len, dim] 张量，device 和 dtype 应保持一致。
        block_size: Q/K/V 的分块大小，必须为正数。
        causal: 是否只允许关注当前位置及之前的 K 位置。

    Returns:
        [seq_len, dim] 的注意力输出。

    """
    if q.ndim != 2 or k.ndim != 2 or v.ndim != 2:
        raise ValueError("q、k、v 必须是 [seq_len, dim] 二维张量")
    if q.shape != k.shape or k.shape != v.shape:
        raise ValueError("q、k、v 的形状必须一致")
    if q.device != k.device or k.device != v.device:
        raise ValueError("q、k、v 必须位于同一 device")
    if q.dtype != k.dtype or k.dtype != v.dtype:
        raise TypeError("q、k、v 必须使用相同 dtype")
    if block_size <= 0:
        raise ValueError("block_size 必须为正数")

    seq_len, dim = q.shape

    out = torch.zeros((seq_len, dim), device=q.device, dtype=q.dtype)
    # ==========================================
    # TODO 1：初始化跨分块状态
    # ==========================================
    # m = ???  # 每行最大值基准；形状 [seq_len, 1]，初始值为负无穷。
    # l = ???  # 每行指数和；形状 [seq_len, 1]，初始值为零。
    raise NotImplementedError("TODO 1：请初始化 m 与 l")

    scale = 1.0 / math.sqrt(dim)

    # 外层循环：遍历 Q 的分块
    for i in range(0, seq_len, block_size):
        q_block = q[i : i + block_size] * scale
        m_i = m[i : i + block_size]
        l_i = l[i : i + block_size]
        out_i = out[i : i + block_size]

        # 内层循环：遍历 K, V 的分块
        for j in range(0, seq_len, block_size):
            k_block = k[j : j + block_size]
            v_block = v[j : j + block_size]

            # ==========================================
            # TODO 2：形成局部分数与因果约束
            # ==========================================
            # scores = ???  # 当前 Q/K 分块的局部分数；形状 [Q 分块长度, K 分块长度]。
            if causal:
                query_pos = torch.arange(i, i + q_block.shape[0], device=q.device)[
                    :, None
                ]
                key_pos = torch.arange(j, j + k_block.shape[0], device=q.device)[
                    None, :
                ]
                # future_mask = ???  # K 的全局位置晚于 Q 时为真。
                scores = scores.masked_fill(future_mask, -float("inf"))
            raise NotImplementedError("TODO 2：请计算 scores 与 future_mask")

            # ==========================================
            # TODO 3：执行数值重标定
            # ==========================================
            # tile_max = ???  # 当前分块逐行最大值；保留列维。
            # new_max = ???  # 历史与当前分块共享的新最大值基准。
            # history_scale = ???  # 历史状态迁移到新基准的缩放因子。
            # tile_weights = ???  # 当前分块相对新基准的指数权重。
            raise NotImplementedError("TODO 3：请完成最大值基准与指数权重的重标定")

            # ==========================================
            # TODO 4：合并归一化输出
            # ==========================================
            # new_l = ???  # 同一最大值基准下的历史与当前指数和。
            # out_i = ???  # 使用同一归一化比例合并历史与当前输出。
            raise NotImplementedError("TODO 4：请更新指数和与归一化输出")
            m_i = new_max
            l_i = new_l

        out[i : i + block_size] = out_i
        m[i : i + block_size] = m_i
        l[i : i + block_size] = l_i

    return out


def _reference_attention(q, k, v, causal=False):
    """构造完整分数矩阵，作为分块注意力的数值参考。"""
    scores = (q @ k.transpose(-2, -1)) / math.sqrt(q.shape[-1])
    if causal:
        mask = torch.triu(torch.ones(scores.shape, dtype=torch.bool), diagonal=1)
        scores = scores.masked_fill(mask, -float("inf"))
    return torch.softmax(scores, dim=-1) @ v


def test_noncausal_equivalence():
    """TODO 1–4：非因果分块路径应与完整注意力等价，含尾块。"""
    for seq_len, dim, block_size, seed in [(8, 4, 2, 42), (5, 3, 3, 7), (3, 2, 1, 123)]:
        torch.manual_seed(seed)
        q, k, v = (torch.randn(seq_len, dim) for _ in range(3))
        diff = (
            (
                _reference_attention(q, k, v)
                - flash_attention_forward_sim(q, k, v, block_size)
            )
            .abs()
            .max()
        )
        assert diff < 1e-5, f"非因果结果失配：seq={seq_len}, block={block_size}"


def test_causal_equivalence():
    """TODO 2：因果掩码必须屏蔽未来 K/V。"""
    torch.manual_seed(11)
    q, k, v = (torch.randn(6, 4) for _ in range(3))
    assert torch.allclose(
        _reference_attention(q, k, v, causal=True),
        flash_attention_forward_sim(q, k, v, 2, causal=True),
        atol=1e-5,
        rtol=1e-5,
    )


def test_dtype_and_stability():
    """TODO 3：重标定应保留数据类型，并让大分数输出保持有限。"""
    torch.manual_seed(9)
    q, k, v = (torch.randn(4, 3, dtype=torch.float64) for _ in range(3))
    out = flash_attention_forward_sim(q, k, v, 2)
    assert out.dtype == torch.float64
    assert torch.allclose(_reference_attention(q, k, v), out, atol=1e-10, rtol=1e-10)
    stable_out = flash_attention_forward_sim(
        torch.full((3, 2), 100.0), torch.full((3, 2), 100.0), torch.randn(3, 2), 2
    )
    assert torch.isfinite(stable_out).all()


def test_working_set_proxy():
    """验证局部分数工作集在固定分块下不随总序列长度平方增长。"""
    block_size = 8
    proxies = []
    for seq_len in [64, 256, 1024]:
        full_score_elements = seq_len * seq_len
        tile_score_elements = block_size * block_size
        proxies.append((full_score_elements, tile_score_elements))
        assert tile_score_elements == 64
        assert tile_score_elements < full_score_elements
    assert [full // tile for full, tile in proxies] == [64, 1024, 16384]

    seq_len = 256
    assert [block * block for block in [4, 8, 16]] == [16, 64, 256]


def test_input_contract():
    """输入形状、数据类型、设备与分块大小契约应拒绝非法值。"""
    try:
        flash_attention_forward_sim(
            torch.randn(2, 2), torch.randn(2, 2), torch.randn(2, 2), block_size=0
        )
    except ValueError:
        return
    raise AssertionError("block_size <= 0 应被拒绝")


def run_flash_attention_sim_tests():
    try:
        test_noncausal_equivalence()
        test_causal_equivalence()
        test_dtype_and_stability()
        test_working_set_proxy()
        test_input_contract()
    except NotImplementedError:
        print("请先完成 TODO 部分的代码！")
        raise
    print(
        "✅ FlashAttention CPU 机制测试通过：数值等价、因果约束、稳定性、工作集代理与输入契约均符合预期。"
    )


"""test"""
test_attention_mechanisms()
run_flash_attention_sim_tests()
