# Infra Learning

深度学习基础架构学习项目，主要涵盖 Attention 机制的实现与解析。

## 项目结构

```
infra_learning/
└── attention/
    ├── attention_is_all_tou_need.py   # 完整 Transformer 模型实现
    ├── self_attention.py              # Self-Attention（自注意力）实现
    └── learning.py                    # Bahdanau Attention（加性注意力）实现
```

## 模块说明

### 1. Bahdanau Attention（加性注意力）

- **文件**: `attention/learning.py`
- **公式**: $e = V^T \tanh(W [h_{dec}; h_{enc}])$
- **用途**: Seq2Seq 模型中，解码器对编码器输出的注意力分配
- **特点**: 通过小前馈网络计算相似度

### 2. Self-Attention（自注意力）

- **文件**: `attention/self_attention.py`
- **公式**: $\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right) V$
- **用途**: Transformer 的核心组件，序列内部的自相关计算
- **特点**: Q、K、V 均来自同一输入

### 3. Transformer

- **文件**: `attention/attention_is_all_tou_need.py`
- **论文**: [Attention Is All You Need](https://arxiv.org/abs/1706.03762) (Vaswani et al., 2017)
- **组件**:
  - Scaled Dot-Product Attention（缩放点积注意力）
  - Multi-Head Attention（多头注意力）
  - Positional Encoding（位置编码）
  - Feed-Forward Network（前馈网络）
  - Encoder & Decoder（编码器和解码器）
- **示例**: 德语 -> 英语的简单翻译演示

## 环境依赖

- Python 3.8+
- PyTorch
- NumPy
- Matplotlib

```bash
pip install torch numpy matplotlib
```

## 运行

```bash
# 运行 Transformer 翻译示例
python attention/attention_is_all_tou_need.py

# 运行 Self-Attention 示例
python attention/self_attention.py
```

## 学习要点

| 机制 | 类型 | 应用场景 |
|------|------|----------|
| Bahdanau Attention | 加性注意力 | Seq2Seq 机器翻译 |
| Self-Attention | 缩放点积注意力 | Transformer、BERT、GPT |
| Multi-Head Attention | 多头注意力 | 并行关注不同子空间信息 |

## 参考

- [Bahdanau et al., 2014 - Neural Machine Translation by Jointly Learning to Align and Translate](https://arxiv.org/abs/1409.0473)
- [Vaswani et al., 2017 - Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/)
