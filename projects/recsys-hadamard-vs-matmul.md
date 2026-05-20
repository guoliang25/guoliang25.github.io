<!-- zh -->
# 为什么 Hadamard 积的计算效率比矩阵乘法低

> **2026-05-20** · by guoliang

## 概述

从 FLOPs 角度看，Hadamard 积（逐元素乘法）是 $O(n^2)$，矩阵乘法是 $O(n^3)$，理论上 Hadamard 应该快得多。但在现代 GPU 上，Hadamard 积的实际计算效率反而远低于矩阵乘法。本文从**计算密度（Arithmetic Intensity）**、**硬件利用率**、**内存访问模式**等维度解释这一反直觉现象，并结合推荐系统中的特征交叉场景给出工程启示。

## 核心原因：计算密度（Arithmetic Intensity）

|  | 计算量 | 内存访问量 | 计算密度 (FLOPs/Byte) |
|--|--------|-----------|----------------------|
| **Hadamard (n×n)** | $n^2$ | $3n^2 \cdot \text{sizeof}$（读A + 读B + 写C） | $\approx 0.33$（FP16） |
| **MatMul (n×n)** | $2n^3$ | $3n^2 \cdot \text{sizeof}$ | $\approx \frac{2n}{3}$（随n增长） |

当 $n=1024$（FP16, 2 Bytes/元素）：

- **Hadamard 计算密度**: $\frac{n^2}{3n^2 \times 2} = \frac{1}{6} \approx 0.17$ FLOPs/Byte
- **MatMul 计算密度**: $\frac{2n^3}{3n^2 \times 2} = \frac{n}{3} \approx 341$ FLOPs/Byte

**差距约 2000 倍。**

### 这意味着什么？

- **Hadamard 积**：每从显存搬 1 字节数据，只做 ~0.17 次运算 → **Memory-bound（内存瓶颈）**
- **矩阵乘法**：每从显存搬 1 字节数据，做 ~341 次运算 → **Compute-bound（计算瓶颈）**

现代 GPU 的算力远远超过带宽（A100: 312 TFLOPS vs 2 TB/s，平衡点 = 156 FLOPs/Byte），MatMul 能充分喂饱 GPU，Hadamard 积让 GPU 99% 时间在等数据。

## 为什么矩阵乘法搬的数据和 Hadamard 一样多，却能做多得多的计算？

### 关键：数据复用

以 $n=4$ 的矩阵为例（FP32, 4 Bytes/元素）：

**Hadamard 积：每个元素只用一次**

```
C[0][0] = A[0][0] * B[0][0]   ← 读2个数，做1次乘法，写1个数。结束。
C[0][1] = A[0][1] * B[0][1]   ← 同上
...共 16 个元素

每个数据的一生：从显存读出 → 乘一下 → 写回显存，再也不用了
```

**矩阵乘法：每个元素被反复复用**

```
C[0][0] = A[0][0]*B[0][0] + A[0][1]*B[1][0] + A[0][2]*B[2][0] + A[0][3]*B[3][0]
C[0][1] = A[0][0]*B[0][1] + A[0][1]*B[1][1] + A[0][2]*B[2][1] + A[0][3]*B[3][1]
C[0][2] = A[0][0]*B[0][2] + A[0][1]*B[1][2] + A[0][2]*B[2][2] + A[0][3]*B[3][2]
C[0][3] = A[0][0]*B[0][3] + A[0][1]*B[1][3] + A[0][2]*B[2][3] + A[0][3]*B[3][3]
                ↑
注意：A[0][0] 在计算 C[0][0], C[0][1], C[0][2], C[0][3] 时全部用到了！
→ A 的每个元素被复用 n 次
→ B 的每个元素也被复用 n 次
```

**搬运量相同（都是 $3n^2$ 个数），但矩阵乘法从每份数据中榨取了 $n$ 倍的计算量。**

### 搬家类比

```
GPU 是一个工厂：
  - 工人（计算单元）：1000人，每秒能加工 1000 件产品
  - 仓库到车间的传送带（内存带宽）：每秒只能运 10 件原材料

Hadamard = "来料加工，不留库存"
  传送带送来 1 块 A 料
  传送带送来 1 块 B 料
  工人把 A 和 B 粘一下（1次加工）
  传送带把成品运走
  → 3 次传送带操作，只干了 1 次活
  → 1000 个工人里 999 个在摸鱼等传送带

矩阵乘法 = "批量进货，反复加工"
  传送带送来 1 块 A 料 → 留在车间
  这块 A 料要和 B 的 1024 个料分别配对 → 被使用 1024 次！
  → 进一次货，工人用它干 1024 次活
  → 1000 个工人全部满载运行
```

### 为什么 $2n^3$？

```
结果矩阵 C 有 n² 个元素
每个元素 = A的一行 · B的一列（点乘）
         = n 次乘法 + (n-1) 次加法 ≈ 2n 次运算

总计 = n² × 2n = 2n³
```

## 矩阵乘法独享的硬件加速

| 硬件优化 | MatMul | Hadamard |
|---------|--------|----------|
| **Tensor Core**（专用矩阵乘单元） | 一条指令完成 4×4×4 FMA | 无法使用 |
| **Tiling / Shared Memory** | 数据加载到片上反复复用 | 无复用需求 |
| **cuBLAS 极致优化** | 几十年工程打磨，接近峰值 | 仅普通 elementwise kernel |
| **Warp-level 协作** | 线程组协同计算子块 | 每个线程独立 |

## 内存访问模式对比

```
矩阵乘法 (Tiled):
  ┌───────────────────────────────────────────────────────┐
  │ 从 HBM 加载一个 128×128 tile 到 Shared Memory         │
  │ 在 Shared Memory 中反复复用（128次）                   │
  │ 有效带宽放大 128 倍                                    │
  └───────────────────────────────────────────────────────┘

Hadamard 积:
  ┌───────────────────────────────────────────────────────┐
  │ 从 HBM 读 A[i,j]     ← 一次显存访问                   │
  │ 从 HBM 读 B[i,j]     ← 一次显存访问                   │
  │ 做一次乘法            ← 一次计算                       │
  │ 写回 C[i,j] 到 HBM   ← 一次显存访问                   │
  │ 每个元素只用1次，无法利用 Shared Memory 复用            │
  └───────────────────────────────────────────────────────┘
```

## Kernel Fusion 视角

实际模型中 Hadamard 积很少单独出现，通常是复合操作的一部分：

```python
# GLU 门控
z = (W1 @ x) * sigmoid(W2 @ x)

# 如果不做 fusion：
step1: tmp1 = W1 @ x          → 写回 HBM（MatMul，可以很快）
step2: tmp2 = W2 @ x          → 写回 HBM
step3: tmp3 = sigmoid(tmp2)   → 写回 HBM（elementwise）
step4: z = tmp1 * tmp3        → 写回 HBM（Hadamard，elementwise）

Hadamard 打断了融合链，强制产生额外的 HBM 读写往返
```

## 时间线对比（A100 GPU, n=1024, FP16）

| 指标 | Hadamard | MatMul |
|------|----------|--------|
| 计算量 | 1M FLOPs | 2G FLOPs |
| 搬运量 | 6 MB | 6 MB |
| 计算耗时 | 0.000003 ms | 0.006 ms |
| 搬运耗时 | 0.003 ms | 0.003 ms |
| **实际耗时** | ~0.003 ms（卡在搬运） | ~0.006 ms（卡在计算） |
| **GPU 利用率** | 0.1% | ~100% |

Hadamard 单次确实更快，但 GPU 利用率极低。当模型需要同等量级的"有效计算"时，用 MatMul 路径反而更高效。

## 推荐系统中的工程启示

推荐模型特征交叉常见两种实现：

```python
# 方式1：Hadamard 积交叉
z = emb_a * emb_b              # O(d)计算, O(3d)访存, 计算密度极低

# 方式2：矩阵乘交叉（如 DCN-V2, AutoInt）
z = emb @ W @ emb.T           # O(d²)计算, 可复用, 计算密度高
```

虽然方式2 的 FLOPs 更多，但实际 wall-clock time 可能差不多甚至更快——因为它能把 GPU 喂饱。这也是 **DCN-V2、AutoInt、RankMixer** 等模型选择矩阵乘做交叉的工程原因之一。

## 总结

> **Hadamard 积慢不是因为计算多，而是因为计算太少。** 现代 GPU 是"算力过剩、带宽不足"的架构。矩阵乘法虽然 FLOPs 多，但能充分复用数据、喂饱算力单元；Hadamard 积每读一个数只做一次乘法，GPU 绝大部分时间在等内存，算力利用率不到 0.1%。**瓶颈不在计算，在搬运。**

<!-- en -->
# Why Hadamard Product Has Lower Compute Efficiency Than Matrix Multiplication on GPUs

> **2026-05-20** · by guoliang

## Overview

From a FLOPs perspective, element-wise (Hadamard) product is $O(n^2)$ while matrix multiplication is $O(n^3)$—Hadamard should theoretically be much faster. However, on modern GPUs, Hadamard product has significantly lower computational efficiency than MatMul. This article explains this counter-intuitive phenomenon from the perspectives of **Arithmetic Intensity**, **hardware utilization**, and **memory access patterns**, with practical implications for feature interaction in recommendation systems.

## Core Reason: Arithmetic Intensity

|  | Compute | Memory Access | Arithmetic Intensity (FLOPs/Byte) |
|--|---------|--------------|----------------------------------|
| **Hadamard (n×n)** | $n^2$ | $3n^2 \cdot \text{sizeof}$ (read A + read B + write C) | $\approx 0.17$ (FP16) |
| **MatMul (n×n)** | $2n^3$ | $3n^2 \cdot \text{sizeof}$ | $\approx \frac{n}{3}$ (grows with n) |

At $n=1024$ (FP16, 2 Bytes/element):

- **Hadamard**: ~0.17 FLOPs/Byte → **Memory-bound** (GPU starved for data)
- **MatMul**: ~341 FLOPs/Byte → **Compute-bound** (GPU fully utilized)

Modern GPUs have far more compute than bandwidth (A100: 312 TFLOPS vs 2 TB/s, balance point = 156 FLOPs/Byte). MatMul saturates the GPU; Hadamard leaves it 99% idle.

## The Key Insight: Data Reuse

**Hadamard**: each element is read once, used once, discarded.

```
C[i][j] = A[i][j] * B[i][j]   // Read 2 values, do 1 multiply, write 1 value. Done.
```

**MatMul**: each element is reused $n$ times.

```
C[0][0] = A[0][0]*B[0][0] + A[0][1]*B[1][0] + A[0][2]*B[2][0] + A[0][3]*B[3][0]
C[0][1] = A[0][0]*B[0][1] + A[0][1]*B[1][1] + A[0][2]*B[2][1] + A[0][3]*B[3][1]
              ↑
A[0][0] is used in computing C[0][0], C[0][1], C[0][2], C[0][3]
→ Each element of A is reused n times
```

Same data movement ($3n^2$ elements), but MatMul extracts $n\times$ more computation from each piece of data.

### Why $2n^3$?

```
C has n² elements.
Each element = dot product of a row and a column = n multiplies + (n-1) adds ≈ 2n ops.
Total = n² × 2n = 2n³
```

## Hardware Advantages Exclusive to MatMul

| Optimization | MatMul | Hadamard |
|-------------|--------|----------|
| **Tensor Cores** | Single instruction for 4×4×4 FMA | Cannot use |
| **Tiling / Shared Memory** | Data loaded once, reused 128× on-chip | No reuse opportunity |
| **cuBLAS** | Decades of engineering, near-peak perf | Simple elementwise kernel |

## Implications for Recommendation Systems

Feature interaction has two common implementations:

```python
# Hadamard-based crossing
z = emb_a * emb_b              # O(d) compute, O(3d) memory, very low intensity

# Matrix-based crossing (DCN-V2, AutoInt, RankMixer)
z = emb @ W @ emb.T           # O(d²) compute, high reuse, high intensity
```

Despite higher FLOPs, matrix-based approaches often achieve similar or better wall-clock time because they saturate GPU compute units. This is a key engineering reason why DCN-V2, AutoInt, and RankMixer use matrix multiplication for feature crossing.

## Summary

> **Hadamard is slow not because it computes too much, but because it computes too little.** Modern GPUs are "compute-rich, bandwidth-poor." MatMul reuses data extensively and keeps ALUs busy; Hadamard reads each value, does one multiply, and discards it—leaving the GPU idle 99.9% of the time. **The bottleneck is data movement, not computation.**
