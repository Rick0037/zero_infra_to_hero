# Triton GEMM 中 `GROUP_SIZE_M` 的由浅入深解析

最关键的一句话：

> **`GROUP_SIZE_M` 不改变 GEMM 的计算，只改变输出 tile 的调度顺序。**
> “分块”决定每个 program **算什么**；“分组”决定每个 `pid` **按什么顺序去算哪一块**。

## 1. 朴素 GEMM：每次只算一个输出元素

计算：

\[
C=A\times B
]

最朴素的写法是：

```python
for m in range(M):
    for n in range(N):
        acc = 0
        for k in range(K):
            acc += A[m, k] * B[k, n]
        C[m, n] = acc
```

例如，`C[2, 3]` 的计算需要：

- A 的第 2 行
- B 的第 3 列

## 2. 分块 GEMM：每次计算一个 C tile

逐个元素计算太慢，所以把 C 切成小块：

```text
C 矩阵
┌───────┬───────┬───────┐
│ C tile│ C tile│ C tile│
├───────┼───────┼───────┤
│ C tile│ C tile│ C tile│
└───────┴───────┴───────┘
```

假设：

```python
BLOCK_SIZE_M = 128
BLOCK_SIZE_N = 128
```

那么一个 program 负责一个 `128×128` 的 C tile：

```python
for k in range(0, K, BLOCK_SIZE_K):
    a = 加载一块 A
    b = 加载一块 B
    acc += tl.dot(a, b)

保存 acc 到 C tile
```

此时需要两个坐标找到这个 tile：

```python
pid_m  # C tile 在 M 方向的编号
pid_n  # C tile 在 N 方向的编号
```

注意：**`pid_m=2` 不是矩阵第 2 行，而是第 2 个行块。**

它对应的实际矩阵起始行是：

```python
pid_m * BLOCK_SIZE_M
```

M、N 方向分别有多少个 tile：

```python
num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
```

到这里，分块已经完成。

## 3. 最简单的一维 pid 映射：按行依次计算

Triton 这里启动的是一维 grid，每个 program 只有一个线性编号：

```python
pid = tl.program_id(0)
```

必须把这个一维编号转换成二维 tile 坐标。

最简单的方法是：

```python
pid_m = pid // num_pid_n
pid_n = pid % num_pid_n
```

假设 C 有 **5 个行块、4 个列块**，普通顺序是：

```text
        N=0      N=1      N=2      N=3
M=0     pid 0    pid 1    pid 2    pid 3
M=1     pid 4    pid 5    pid 6    pid 7
M=2     pid 8    pid 9    pid 10   pid 11
M=3     pid 12   pid 13   pid 14   pid 15
M=4     pid 16   pid 17   pid 18   pid 19
```

也就是：

```text
先把 M=0 这一行块的所有列块算完
再算 M=1
再算 M=2
...
```

这叫 **row-major ordering（行主序）**。这种顺序可以正确计算，但数据复用未必理想。

## 4. `GROUP_SIZE_M` 到底在分什么？

现在设置：

```python
GROUP_SIZE_M = 2
```

它的意思是：

> **把 M 方向的 2 个 tile 行组成一个调度组。**

名字可以拆成：

```text
GROUP：分组
SIZE ：每组包含多少个 tile
M    ：沿 M 方向分组
```

它不是：

- 矩阵元素行数
- 线程数
- 一个 program 的计算量
- 多组独立矩阵的 Grouped GEMM

仍然是 5 个行块、4 个列块，分组后变成：

```text
第 0 组：M=0、1，包含 N=0、1、2、3，形状为 2×4 的长方形
第 1 组：M=2、3，包含 N=0、1、2、3，形状为 2×4 的长方形
第 2 组：M=4，包含 N=0、1、2、3，形状为 1×4（最后一组不足 2 行）
```

每组完整时包含的 program 数是：

```python
num_pid_in_group = GROUP_SIZE_M * num_pid_n
                 = 2 * 4
                 = 8
```

因为完整的一组有 2 个行块，每个行块有 4 个列块，所以共有 8 个 C tile。

## 5. 分组后的计算顺序

组内采用**列主序**：

> 先固定一个 N 列块，把组内几个 M 行块算完；然后进入下一个 N 列块。

上面的例子顺序变成：

```text
第 0 组：M=0、1，2×4 的长方形

pid 0 → (0,0)    pid 1 → (1,0)
pid 2 → (0,1)    pid 3 → (1,1)
pid 4 → (0,2)    pid 5 → (1,2)
pid 6 → (0,3)    pid 7 → (1,3)

第 1 组：M=2、3，2×4 的长方形

pid 8  → (2,0)   pid 9  → (3,0)
pid 10 → (2,1)   pid 11 → (3,1)
pid 12 → (2,2)   pid 13 → (3,2)
pid 14 → (2,3)   pid 15 → (3,3)

第 2 组：M=4，1×4 的长方形

pid 16 → (4,0)
pid 17 → (4,1)
pid 18 → (4,2)
pid 19 → (4,3)
```

用矩阵表示就是：

```text
             N=0      N=1      N=2      N=3
           ┌────────┬────────┬────────┬────────┐
M=0        │ pid 0  │ pid 2  │ pid 4  │ pid 6  │  ← 第 0 组
           ├────────┼────────┼────────┼────────┤
M=1        │ pid 1  │ pid 3  │ pid 5  │ pid 7  │
           ├────────┼────────┼────────┼────────┤
M=2        │ pid 8  │ pid 10 │ pid 12 │ pid 14 │  ← 第 1 组
           ├────────┼────────┼────────┼────────┤
M=3        │ pid 9  │ pid 11 │ pid 13 │ pid 15 │
           ├────────┼────────┼────────┼────────┤
M=4        │ pid 16 │ pid 17 │ pid 18 │ pid 19 │  ← 最后一组只有 1 行
           └────────┴────────┴────────┴────────┘
```

可以明显看出：**每 2 个 pid 换一列，列内两个 M 行块交替。**

这就是 `GROUP_SIZE_M=2` 的完整含义。

## 6. 每行代码如何得到这个顺序？

原代码是：

```python
pid = tl.program_id(0)

num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)

num_pid_in_group = GROUP_SIZE_M * num_pid_n
group_id = pid // num_pid_in_group

first_pid_m = group_id * GROUP_SIZE_M
group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)

pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
pid_n = (pid % num_pid_in_group) // group_size_m
```

### 6.1 确定在哪一组

```python
group_id = pid // num_pid_in_group
```

完整组有 8 个 program，所以：

```text
pid 0～7   → group_id=0
pid 8～15  → group_id=1
pid 16～19 → group_id=2
```

### 6.2 这一组从哪个 M 行块开始？

```python
first_pid_m = group_id * GROUP_SIZE_M
```

因此：

```text
第 0 组从 M=0 开始
第 1 组从 M=2 开始
第 2 组从 M=4 开始
```

### 6.3 这一组实际有几个 M 行块？

```python
group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
```

前两组：

```python
group_size_m = 2
```

最后一组只剩 M=4：

```python
group_size_m = min(5 - 4, 2) = 1
```

所以它能正确处理“最后一组不满”的情况。

### 6.4 计算组内编号

可以把：

```python
pid % num_pid_in_group
```

理解为 `local_pid`，即当前 program 在组内的位置。

例如 `pid=13`：

```text
group_id = 13 // 8 = 1
local_pid = 13 % 8 = 5
first_pid_m = 2
group_size_m = 2
```

于是：

```python
pid_m = 2 + (5 % 2) = 3
pid_n = 5 // 2 = 2
```

所以：

```text
pid 13 → C tile (3, 2)
```

再看最后一组的 `pid=18`：

```text
group_id = 18 // 8 = 2
local_pid = 18 % 8 = 2
first_pid_m = 4
group_size_m = 1
```

于是：

```python
pid_m = 4 + (2 % 1) = 4
pid_n = 2 // 1 = 2
```

所以：

```text
pid 18 → C tile (4, 2)
```

## 7. 为什么这样可能提高 L2 命中率？

计算一个 C tile `(pid_m, pid_n)` 需要：

```text
A 的第 pid_m 个行块
B 的第 pid_n 个列块
```

分组后，连续几个 program 是：

```text
(0,0) → (1,0) → (0,1) → (1,1)
```

前两个 program 的 `pid_n` 都是 0：

```text
(0,0) 和 (1,0)
```

它们需要**相同的 B 列块**，因此 B 刚被前一个 program 使用，后一个 program 可能直接从 L2 命中。

同时，这一组只使用 M=0、1 的 A 行块。等切换到下一个 N 时，这小块 A 数据还可能保留在 L2 中。

所以 grouped ordering 同时争取：

- 邻近 program 复用同一个 B tile
- 一组 program 复用同一小片 A 行块

Triton 官方教程的 9×9 分块示例中，前 9 个输出块按普通行主序需要加载 90 个输入块，分组顺序则只需 54 个；PyTorch 的相关优化文章也说明了这种“保留一条 A 行块区域、连续复用 B tile”的思路。

不过，**收益不是固定的**，它和矩阵尺寸、GPU 缓存大小、并发 program 数量有关，所以 `GROUP_SIZE_M` 通常交给 autotune 测速选择。

## 8. 这不是严格串行执行

上面说的“顺序”是 **pid 到 tile 的映射顺序**，不是：

```text
第 0 组全部完成
→ 同步等待
→ 第 1 组开始
```

GPU 上很多 program 会并发执行，也没有组间同步。

因此更准确的叫法是：

> **分组调度，而不是分组计算。**

每个 program 内部的 GEMM 计算完全没变，仍然是：

```python
for k in ...:
    acc += tl.dot(a, b)
```

## 9. 从朴素 GEMM 迁移过来的完整思路

```text
1. 朴素 GEMM
   每次计算 C 的一个元素

2. 分块 GEMM
   每个 program 计算 C 的一个 tile

3. 普通 pid 映射
   pid 按行依次对应各个 tile

4. grouped pid 映射
   每 GROUP_SIZE_M 个 M 行块作为一组
   组内按列主序对应 tile

5. 目的
   提高 A/B 数据在 L2 中的复用，不改变计算结果
```

一句话总结：

> **`BLOCK_SIZE_M/N` 决定“每个 program 算多大的块”；`GROUP_SIZE_M` 决定“每几个 M 行块组成一组，以及这些块按什么顺序领取 pid”。**

## 参考资料

- [Triton 官方 Matrix Multiplication 教程：L2 Cache Optimizations](https://triton-lang.org/main/getting-started/tutorials/03-matrix-multiplication.html)
- [PyTorch：Grouped Launch Ordering 与缓存复用](https://pytorch.org/blog/accelerating-moes-with-a-triton-persistent-cache-aware-grouped-gemm-kernel/)
- [FlagGems：Grouped Scheduling 与 L2 Cache 数据复用](https://flagos-ai.github.io/FlagGems/optimization/hygon/)
- [Triton Language：program\_id 与 num\_programs](https://triton-lang.org/main/python-api/triton.language.html)
