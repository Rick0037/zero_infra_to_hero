import nunpy as np  # 注意：应为 numpy（原代码有拼写错误）
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# =====================================================================
# 函数：将句子转换为模型输入的批次数据
# =====================================================================
def make_batch(sentences):
    """
    将输入的 3 个句子（源语言、目标输入、目标标签）转换为数值化的 tensor

    Args:
        sentences: [源句子, 解码器输入, 目标标签]
            例如: ['ich mochte ein bier P', 'S i want a beer', 'i want a beer E']

    Returns:
        input_batch:  编码器输入  [1, src_len]
        output_batch: 解码器输入  [1, tgt_len]
        target_batch: 目标标签    [1, tgt_len]
    """
    # sentences[0].split() -> ['ich', 'mochte', 'ein', 'bier', 'P']
    # src_vocab[n] 将每个单词映射为词表中的索引
    # 外层 [] 构成一个 batch
    input_batch = [[src_vocab[n] for n in sentences[0].split()]]
    output_batch = [[tgt_vocab[n] for n in sentences[1].split()]]
    target_batch = [[tgt_vocab[n] for n in sentences[2].split()]]
    # 转换为 LongTensor（整数张量），Embedding 层需要整数索引作为输入
    return torch.LongTensor(input_batch), torch.LongTensor(output_batch), torch.LongTensor(target_batch)


# =====================================================================
# 函数：生成 Sinusoid 位置编码表
# =====================================================================
def get_sinusoid_encoding_table(n_position, d_model):
    """
    生成 Transformer 论文中的正弦位置编码表

    数学公式:
        PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
        PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))

    Args:
        n_position: 序列最大长度（位置数量）
        d_model:    嵌入维度

    Returns:
        sinusoid_table: [n_position, d_model] 的位置编码表
    """

    def cal_angle(position, hid_idx):
        """计算 position / 10000^(2i/d_model)"""
        return position / np.power(10000, 2 * (hid_idx // 2) / d_model)

    def get_posi_angle_vec(position):
        """对某个位置，计算所有维度的角度值"""
        return [cal_angle(position, hid_j) for hid_j in range(d_model)]

    # 生成 [n_position, d_model] 的角度表
    sinusoid_table = np.array([get_posi_angle_vec(pos_i) for pos_i in range(n_position)])
    # 偶数维度（0, 2, 4...）用 sin
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
    # 奇数维度（1, 3, 5...）用 cos
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1
    return torch.FloatTensor(sinusoid_table)


# =====================================================================
# 函数：生成 Padding 掩码
# =====================================================================
def get_attn_pad_mask(seq_q, seq_k):
    """
    生成 padding 掩码：将填充位置（值为 0 的位置）标记为需要被遮蔽

    为什么需要 Padding Mask？
        输入序列长度不一致时用 0 填充，但 Attention 不应关注这些填充位置

    Args:
        seq_q: Query 序列 [batch_size, len_q]
        seq_k: Key   序列 [batch_size, len_k]

    Returns:
        mask: [batch_size, len_q, len_k]  True 表示该位置是 padding，需要遮蔽
    """
    batch_size, len_q = seq_q.size()
    batch_size, len_k = seq_k.size()
    # seq_k.data.eq(0): 找出值为 0（PAD）的位置，返回 True/False
    # .unsqueeze(1): 增加一维 -> [batch_size, 1, len_k]
    pad_attn_mask = seq_k.data.eq(0).unsqueeze(1)  # batch_size x 1 x len_k(=len_q), one is masking
    # expand 扩展到 [batch_size, len_q, len_k]
    # 含义: 对于 Query 的每个位置，都遮蔽 Key 中的 padding 位置
    return pad_attn_mask.expand(batch_size, len_q, len_k)  # batch_size x len_q x len_k


# =====================================================================
# 函数：生成 Subsequent 掩码（下三角掩码）
# =====================================================================
def get_attn_subsequent_mask(seq):
    """
    生成后续掩码：防止解码器在预测时看到未来的 token（因果掩码）

    为什么需要 Subsequent Mask？
        解码时，第 i 个位置只能看到 0~i 的位置，不能看到未来的词
        这是自回归生成的关键

    Args:
        seq: [batch_size, len_seq]

    Returns:
        subsequent_mask: [batch_size, len_seq, len_seq]
            上三角为 1（需要遮蔽），下三角和对角线为 0
    """
    # 形状: [batch_size, len_seq, len_seq]
    attn_shape = [seq.size(0), seq.size(1), seq.size(1)]
    # np.triu(k=1): 生成上三角矩阵（不含对角线），k=1 表示主对角线之上
    #   [[0, 1, 1],
    #    [0, 0, 1],
    #    [0, 0, 0]]
    subsequent_mask = np.triu(np.ones(attn_shape), k=1)
    subsequent_mask = torch.from_numpy(subsequent_mask).byte()
    return subsequent_mask


# =====================================================================
# 模块 1：缩放点积注意力（Scaled Dot-Product Attention）
# =====================================================================
class ScaledDotProductAttention(nn.Module):
    """
    缩放点积注意力

    数学公式:
        Attention(Q, K, V) = softmax(Q·K^T / √d_k) · V
    """

    def __init__(self):
        super(ScaledDotProductAttention, self).__init__()

    def forward(self, Q, K, V, attn_mask):
        """
        Args:
            Q: [batch_size, n_heads, len_q, d_k]
            K: [batch_size, n_heads, len_k, d_k]
            V: [batch_size, n_heads, len_k, d_v]
            attn_mask: [batch_size, n_heads, len_q, len_k]  True 表示需要遮蔽
        """
        # === 步骤 1: 计算注意力分数 ===
        # K.transpose(-1, -2): 交换最后两维 -> [batch, n_heads, d_k, len_k]
        # matmul 后: [batch, n_heads, len_q, len_k]
        # 除以 √d_k 进行缩放，防止点积过大导致 softmax 梯度消失
        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(d_k)  # [batch_size x n_heads x len_q x len_k]

        # === 步骤 2: 应用掩码 ===
        # 将 mask 为 True 的位置填充为 -1e9（很小的负数）
        # softmax 后这些位置的值会接近 0，相当于"看不见"
        scores.masked_fill_(attn_mask, -1e9)

        # === 步骤 3: Softmax 归一化 ===
        # 在最后一个维度（len_k）上做 softmax，得到注意力权重
        attn = nn.Softmax(dim=-1)(scores)  # [batch_size x n_heads x len_q x len_k]

        # === 步骤 4: 加权求和 ===
        # 用注意力权重对 V 加权求和
        # attn: [batch, n_heads, len_q, len_k] × V: [batch, n_heads, len_k, d_v]
        # -> context: [batch, n_heads, len_q, d_v]
        context = torch.matmul(attn, V)
        return context, attn


# =====================================================================
# 模块 2：多头注意力（Multi-Head Attention）
# =====================================================================
class MultiHeadAttention(nn.Module):
    """
    多头注意力机制

    核心思想:
        将 Q、K、V 分成多个头，每个头独立做注意力，最后拼接
        不同的头可以关注不同的信息（语法、语义等）
    """

    def __init__(self):
        super(MultiHeadAttention, self).__init__()
        # 三个线性变换层，将 d_model 映射到 d_k * n_heads（稍后会分头）
        self.W_Q = nn.Linear(d_model, d_k * n_heads)  # 生成 Q
        self.W_K = nn.Linear(d_model, d_k * n_heads)  # 生成 K
        self.W_V = nn.Linear(d_model, d_v * n_heads)  # 生成 V
        # 输出投影层：将多头拼接后的维度映射回 d_model
        self.linear = nn.Linear(n_heads * d_v, d_model)
        # LayerNorm 归一化
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, Q, K, V, attn_mask):
        """
        Args:
            Q, K, V: [batch_size, len, d_model]
            attn_mask: [batch_size, len_q, len_k]
        """
        # q: [batch_size x len_q x d_model], k: [batch_size x len_k x d_model], v: [batch_size x len_k x d_model]
        residual, batch_size = Q, Q.size(0)  # residual 用于残差连接

        # === 步骤 1: 线性投影 + 分头 ===
        # (B, S, D) -proj-> (B, S, D) -split-> (B, S, H, W) -trans-> (B, H, S, W)
        # self.W_Q(Q): [batch, len_q, d_k*n_heads]
        # .view: 重塑为 [batch, len_q, n_heads, d_k]
        # .transpose(1,2): 交换第 1、2 维 -> [batch, n_heads, len_q, d_k]
        q_s = self.W_Q(Q).view(batch_size, -1, n_heads, d_k).transpose(1, 2)  # [batch x n_heads x len_q x d_k]
        k_s = self.W_K(K).view(batch_size, -1, n_heads, d_k).transpose(1, 2)  # [batch x n_heads x len_k x d_k]
        v_s = self.W_V(V).view(batch_size, -1, n_heads, d_v).transpose(1, 2)  # [batch x n_heads x len_k x d_v]

        # === 步骤 2: 扩展掩码到多头 ===
        # attn_mask 原形状: [batch, len_q, len_k]
        # .unsqueeze(1): -> [batch, 1, len_q, len_k]
        # .repeat(1, n_heads, 1, 1): -> [batch, n_heads, len_q, len_k]
        attn_mask = attn_mask.unsqueeze(1).repeat(1, n_heads, 1, 1)  # [batch_size x n_heads x len_q x len_k]

        # === 步骤 3: 调用缩放点积注意力 ===
        # context: [batch, n_heads, len_q, d_v]
        # attn:    [batch, n_heads, len_q, len_k]
        context, attn = ScaledDotProductAttention()(q_s, k_s, v_s, attn_mask)

        # === 步骤 4: 合并多头 ===
        # .transpose(1, 2): [batch, len_q, n_heads, d_v]
        # .contiguous(): 确保内存连续（view 前需要）
        # .view: [batch, len_q, n_heads * d_v]
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, n_heads * d_v)

        # === 步骤 5: 输出投影 + 残差连接 + LayerNorm ===
        output = self.linear(context)  # [batch, len_q, d_model]
        # 残差连接: output + residual，然后 LayerNorm
        return self.layer_norm(output + residual), attn  # output: [batch_size x len_q x d_model]


# =====================================================================
# 模块 3：位置前馈神经网络（Position-wise Feed-Forward Network）
# =====================================================================
class PoswiseFeedForwardNet(nn.Module):
    """
    位置前馈网络（FFN）

    数学公式:
        FFN(x) = max(0, x·W1 + b1) · W2 + b2

    作用:
        对每个位置独立做两层线性变换 + ReLU 激活
        增加模型的非线性表达能力
    """

    def __init__(self):
        super(PoswiseFeedForwardNet, self).__init__()
        # 用 1D 卷积模拟位置 wise 的全连接
        # kernel_size=1 等价于对每个位置做线性变换
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, inputs):
        """
        Args:
            inputs: [batch_size, len_q, d_model]
        """
        residual = inputs  # 保存输入用于残差连接

        # inputs.transpose(1, 2): [batch, d_model, len_q]（Conv1d 需要通道在前）
        # self.conv1: [batch, d_ff, len_q]
        # ReLU 激活
        output = nn.ReLU()(self.conv1(inputs.transpose(1, 2)))
        # self.conv2: [batch, d_model, len_q]
        # .transpose(1, 2): 转回 [batch, len_q, d_model]
        output = self.conv2(output).transpose(1, 2)

        # 残差连接 + LayerNorm
        return self.layer_norm(output + residual)


# =====================================================================
# 模块 4：编码器层（Encoder Layer）
# =====================================================================
class EncoderLayer(nn.Module):
    """
    单个编码器层，包含:
        1. 多头自注意力（Self-Attention）
        2. 前馈神经网络（FFN）
    每个子层都有残差连接 + LayerNorm
    """

    def __init__(self):
        super(EncoderLayer, self).__init__()
        self.enc_self_attn = MultiHeadAttention()  # 编码器自注意力
        self.pos_ffn = PoswiseFeedForwardNet()  # 前馈网络

    def forward(self, enc_inputs, enc_self_attn_mask):
        """
        Args:
            enc_inputs: [batch_size, len_q, d_model]
            enc_self_attn_mask: padding 掩码
        """
        # === 子层 1: 自注意力 ===
        # Q=K=V=enc_inputs（自注意力: 自己关注自己）
        enc_outputs, attn = self.enc_self_attn(enc_inputs, enc_inputs, enc_inputs, enc_self_attn_mask)

        # === 子层 2: 前馈网络 ===
        enc_outputs = self.pos_ffn(enc_outputs)  # [batch_size x len_q x d_model]
        return enc_outputs, attn


# =====================================================================
# 模块 5：解码器层（Decoder Layer）
# =====================================================================
class DecoderLayer(nn.Module):
    """
    单个解码器层，包含:
        1. 掩码多头自注意力（Masked Self-Attention）
        2. 编码器-解码器交叉注意力（Cross-Attention）
        3. 前馈神经网络（FFN）
    """

    def __init__(self):
        super(DecoderLayer, self).__init__()
        self.dec_self_attn = MultiHeadAttention()  # 解码器自注意力
        self.dec_enc_attn = MultiHeadAttention()  # 编码器-解码器交叉注意力
        self.pos_ffn = PoswiseFeedForwardNet()  # 前馈网络

    def forward(self, dec_inputs, enc_outputs, dec_self_attn_mask, dec_enc_attn_mask):
        """
        Args:
            dec_inputs:          [batch, len_dec, d_model]  解码器输入
            enc_outputs:         [batch, len_enc, d_model]  编码器输出
            dec_self_attn_mask:  解码器自注意力掩码（padding + subsequent）
            dec_enc_attn_mask:   交叉注意力掩码（padding）
        """
        # === 子层 1: 掩码自注意力 ===
        # Q=K=V=dec_inputs（解码器关注自己，但有掩码防止看到未来）
        dec_outputs, dec_self_attn = self.dec_self_attn(dec_inputs, dec_inputs, dec_inputs,
                                                        dec_self_attn_mask)

        # === 子层 2: 交叉注意力 ===
        # Q=dec_outputs, K=V=enc_outputs
        # 解码器用当前状态去"查询"编码器的输出
        dec_outputs, dec_enc_attn = self.dec_enc_attn(dec_outputs, enc_outputs, enc_outputs,
                                                      dec_enc_attn_mask)

        # === 子层 3: 前馈网络 ===
        dec_outputs = self.pos_ffn(dec_outputs)
        return dec_outputs, dec_self_attn, dec_enc_attn


# =====================================================================
# 模块 6：编码器（Encoder）
# =====================================================================
"""
编码器
"""
class Encoder(nn.Module):
    """
    完整的编码器:
        1. 词嵌入（Word Embedding）
        2. 位置编码（Positional Encoding）
        3. N 个编码器层堆叠
    """

    def __init__(self):
        super(Encoder, self).__init__()
        # 将输入单词进行 Embedding
        # src_vocab_size: 源语言词表大小
        # d_model: 嵌入维度
        self.src_emb = nn.Embedding(src_vocab_size, d_model)
        # 添加位置编码（使用预计算的正弦表，freeze=True 表示不更新）
        self.pos_emb = nn.Embedding.from_pretrained(get_sinusoid_encoding_table(src_len + 1, d_model), freeze=True)
        # N 个编码器层
        self.layers = nn.ModuleList([EncoderLayer() for _ in range(n_layers)])

    def forward(self, enc_inputs):
        """
        Args:
            enc_inputs: [batch_size, src_len]  单词索引序列
        Returns:
            enc_outputs:     [batch_size, src_len, d_model]
            enc_self_attns:  每层的注意力权重（用于可视化）
        """
        # === 步骤 1: 词嵌入 + 位置编码 ===
        # self.src_emb(enc_inputs): [batch, src_len, d_model]
        # self.pos_emb(...): [batch, src_len, d_model]
        # 两者相加得到最终输入表示
        enc_outputs = self.src_emb(enc_inputs) + self.pos_emb(torch.LongTensor([[1, 2, 3, 4, 0]]))

        # === 步骤 2: 生成 padding 掩码 ===
        enc_self_attn_mask = get_attn_pad_mask(enc_inputs, enc_inputs)

        # === 步骤 3: 逐层通过编码器层 ===
        enc_self_attns = []
        for layer in self.layers:
            enc_outputs, enc_self_attn = layer(enc_outputs, enc_self_attn_mask)
            enc_self_attns.append(enc_self_attn)
        return enc_outputs, enc_self_attns


# =====================================================================
# 模块 7：解码器（Decoder）
# =====================================================================
class Decoder(nn.Module):
    """
    完整的解码器:
        1. 词嵌入 + 位置编码
        2. N 个解码器层堆叠
    """

    def __init__(self):
        super(Decoder, self).__init__()
        self.tgt_emb = nn.Embedding(tgt_vocab_size, d_model)
        self.pos_emb = nn.Embedding.from_pretrained(get_sinusoid_encoding_table(tgt_len + 1, d_model), freeze=True)
        self.layers = nn.ModuleList([DecoderLayer() for _ in range(n_layers)])

    def forward(self, dec_inputs, enc_inputs, enc_outputs):
        """
        Args:
            dec_inputs:  [batch_size, tgt_len]  解码器输入（目标序列）
            enc_inputs:  [batch_size, src_len]  编码器输入（用于生成 padding mask）
            enc_outputs: [batch_size, src_len, d_model]  编码器输出
        """
        # === 步骤 1: 词嵌入 + 位置编码 ===
        dec_outputs = self.tgt_emb(dec_inputs) + self.pos_emb(torch.LongTensor([[5, 1, 2, 3, 4]]))

        # === 步骤 2: 生成解码器自注意力掩码 ===
        # 2.1 padding 掩码
        dec_self_attn_pad_mask = get_attn_pad_mask(dec_inputs, dec_inputs)
        # 2.2 后续掩码（防止看到未来 token）
        dec_self_attn_subsequent_mask = get_attn_subsequent_mask(dec_inputs)
        # 2.3 合并两个掩码: 相加大于 0 的位置需要遮蔽
        dec_self_attn_mask = torch.gt((dec_self_attn_pad_mask + dec_self_attn_subsequent_mask), 0)

        # === 步骤 3: 生成交叉注意力掩码 ===
        # Query 来自解码器，Key 来自编码器，所以掩码基于 enc_inputs
        dec_enc_attn_mask = get_attn_pad_mask(dec_inputs, enc_inputs)

        # === 步骤 4: 逐层通过解码器层 ===
        dec_self_attns, dec_enc_attns = [], []
        for layer in self.layers:
            dec_outputs, dec_self_attn, dec_enc_attn = layer(dec_outputs, enc_outputs,
                                                             dec_self_attn_mask, dec_enc_attn_mask)
            dec_self_attns.append(dec_self_attn)
            dec_enc_attns.append(dec_enc_attn)
        return dec_outputs, dec_self_attns, dec_enc_attns


# =====================================================================
# 模块 8：完整的 Transformer 模型
# =====================================================================
class Transformer(nn.Module):
    """
    完整的 Transformer 模型 = Encoder + Decoder + 输出投影
    """

    def __init__(self):
        super(Transformer, self).__init__()
        # 编码器
        self.encoder = Encoder()
        # 解码器
        self.decoder = Decoder()
        # 解码器最后的分类器
        # 输入 d_model，输出 tgt_vocab_size
        # 再计算 softmax 得到每个词的概率
        self.projection = nn.Linear(d_model, tgt_vocab_size, bias=False)

    def forward(self, enc_inputs, dec_inputs):
        """
        Args:
            enc_inputs: [batch_size, src_len]  源语言输入
            dec_inputs: [batch_size, tgt_len]  目标语言输入
        Returns:
            dec_logits: [batch_size * tgt_len, tgt_vocab_size]  预测结果
        """
        # === 步骤 1: 编码器处理源序列 ===
        # enc_outputs: 源数据的特征表示
        # enc_self_attns: 编码器各层的注意力权重（用于可视化）
        enc_outputs, enc_self_attns = self.encoder(enc_inputs)

        # === 步骤 2: 解码器处理目标序列 ===
        # 输入三部分: 解码器输入、编码器输入（用于 mask）、编码器输出
        dec_outputs, dec_self_attns, dec_enc_attns = self.decoder(dec_inputs, enc_inputs, enc_outputs)

        # === 步骤 3: 投影到词表大小 ===
        # dec_logits: [batch_size, tgt_len, tgt_vocab_size]
        dec_logits = self.projection(dec_outputs)
        # 展平为 [batch_size * tgt_len, tgt_vocab_size] 以便计算交叉熵损失
        return dec_logits.view(-1, dec_logits.size(-1)), enc_self_attns, dec_self_attns, dec_enc_attns


# =====================================================================
# 函数：可视化注意力权重
# =====================================================================
def showgraph(attn):
    """绘制注意力热力图"""
    attn = attn[-1].squeeze(0)[0]  # 取最后一层第 0 个头
    attn = attn.squeeze(0).data.numpy()
    fig = plt.figure(figsize=(n_heads, n_heads))
    ax = fig.add_subplot(1, 1, 1)
    ax.matshow(attn, cmap='viridis')
    ax.set_xticklabels([''] + sentences[0].split(), fontdict={'fontsize': 14}, rotation=90)
    ax.set_yticklabels([''] + sentences[2].split(), fontdict={'fontsize': 14})
    plt.show()


# =====================================================================
# 主程序：训练和测试
# =====================================================================
if __name__ == '__main__':
    # === 1. 准备数据 ===
    """
    第一个句子 是 编码器的输入
    第二个句子 是 解码器的输入
    第三个句子 是 标签

    P 可以理解为 编码器输入结束的字符（Padding填充字符）
    S 可以理解为 Start
    E 可以理解为 End

    此外，需要注意的是，由于文本内容长度往往会不一致，因此在代码实现过程中，我们往往会设置一个最大长度max_length，
    - 大于max_length的句子，多余的部分将会被裁剪
    - 小于max_length的句子，缺少的部分将会被填充
    """
    sentences = ['ich mochte ein bier P', 'S i want a beer', 'i want a beer E']

    # === 2. 定义词表和超参数 ===
    # Transformer Parameters
    # Padding Should be Zero
    src_vocab = {'P': 0, 'ich': 1, 'mochte': 2, 'ein': 3, 'bier': 4}  # 源语言词表
    src_vocab_size = len(src_vocab)

    tgt_vocab = {'P': 0, 'i': 1, 'want': 2, 'a': 3, 'beer': 4, 'S': 5, 'E': 6}  # 目标语言词表
    number_dict = {i: w for i, w in enumerate(tgt_vocab)}  # 反向映射: 索引 -> 单词
    tgt_vocab_size = len(tgt_vocab)

    src_len = 5  # 源序列长度
    tgt_len = 5  # 目标序列长度

    d_model = 512  # Embedding 维度（模型统一维度）
    d_ff = 2048  # 前馈网络中间维度
    d_k = d_v = 64  # Q/K 和 V 的维度
    n_layers = 6  # Encoder/Decoder 层数
    n_heads = 8  # 多头注意力的头数

    # === 3. 创建模型和优化器 ===
    model = Transformer()
    criterion = nn.CrossEntropyLoss()  # 交叉熵损失函数
    optimizer = optim.Adam(model.parameters(), lr=0.001)  # Adam 优化器

    # === 4. 准备训练数据 ===
    enc_inputs, dec_inputs, target_batch = make_batch(sentences)

    # === 5. 训练循环 ===
    for epoch in range(20):
        optimizer.zero_grad()  # 梯度清零
        outputs, enc_self_attns, dec_self_attns, dec_enc_attns = model(enc_inputs, dec_inputs)
        # target_batch.contiguous().view(-1): 展平为 [batch_size * tgt_len]
        loss = criterion(outputs, target_batch.contiguous().view(-1))
        print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.6f}'.format(loss))
        loss.backward()  # 反向传播
        optimizer.step()  # 更新参数

    # === 6. 测试 ===
    predict, _, _, _ = model(enc_inputs, dec_inputs)
    # max(1): 在词表维度取最大值的索引
    predict = predict.data.max(1, keepdim=True)[1]
    print(sentences[0], '->', [number_dict[n.item()] for n in predict.squeeze()])

    # === 7. 可视化注意力 ===
    print('first head of last state enc_self_attns')
    showgraph(enc_self_attns)

    print('first head of last state dec_self_attns')
    showgraph(dec_self_attns)

    print('first head of last state dec_enc_attns')
    showgraph(dec_enc_attns)
