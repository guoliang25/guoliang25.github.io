<!-- zh -->
# Flash Attention 原理拆解：从加速原理到 Online Softmax 到 Varlen 落地

> **2026-07-06** · by guoliang

## 概述

本文以对话的方式，从底往上把 Flash Attention 拆开讲清楚：为什么它能加速？Online Softmax 究竟怎么工作？为什么必须减 max？块怎么切？α/β 到底在校正什么？以及生产落地里的 varlen（packed）格式。核心结论：**Flash Attention 用"分块 + 在线 softmax + 重算"三件套，把 attention 从"显存带宽瓶颈"变成"算力瓶颈"，数学上完全等价。**

---

## 一、为什么 Flash Attention 更快：Attention 是 Memory-Bound

### 硬件不平衡

现代 GPU（以 A100 为例）算力和带宽严重不匹配：

- **算力**：312 TFLOPS (FP16)
- **HBM 带宽**：1.5–2 TB/s
- **SRAM 带宽**：~19 TB/s（但每 SM 只有 ~192 KB）

计算与访存比大约是 **150:1**——每读 1 个 float，理论上要做 150 次浮点运算才能"喂饱"GPU。标准 attention 远远达不到这个比例，卡在了显存带宽上。

### 标准实现的 3 次 HBM 往返

```
1. S = QK^T          → 写 N×N 到 HBM
2. P = softmax(S)    → 读 N×N，写 N×N
3. O = PV            → 读 N×N，写 N×d
```

- **HBM 访问量**：O(N² + Nd)
- **显存占用**：N=8K 时 N×N 中间矩阵就是 256 MB（单头 FP16）
- Softmax 需要**整行**才能算 max 和 sum，天然阻碍分块

### Flash Attention 三件套

| 技术 | 作用 |
|---|---|
| **Tiling + Kernel Fusion** | 把 Q/K/V 切成 SRAM 能装下的块，QK^T → softmax → PV 全流程在 SRAM 里做完，不物化 N×N |
| **Online Softmax** | 分块也能算出正确 softmax 的关键数学 trick，等价而非近似 |
| **Recomputation** | 反向不存 P，只存 O(N) 的 (m, l) 统计量，反向时重算——用 FLOPs 换 HBM |

### 复杂度对比

| 指标 | 标准 Attention | Flash Attention |
|---|---|---|
| HBM 访问 | O(N²·d) | **O(N²·d²/M)**（M = SRAM 容量）|
| 额外显存 | O(N²) | **O(N)** |
| FLOPs | 相同 | 相同（+反向重算） |

d=64、M=192KB 时，HBM 访问减少 **~9x**，wall-clock 也随之下降 2–4x（v1）、5–10x（v2/v3）。

---

## 二、Online Softmax 数值实例

### 场景

对一行 attention scores 做 softmax，然后乘 V。切成 3 块，每块 2 个元素：

```
S = [1, 3, 2, 5, 4, 0]      (长度 6)
V = [[1,0], [0,1], [2,2], [1,1], [0,3], [3,0]]    (2维)
```

### 标准 Softmax（对照答案）

- max = 5，exp(S − 5) = [0.0183, 0.1353, 0.0498, 1.0, 0.3679, 0.00674]
- sum = 1.5780
- **O = P @ V = [0.7213, 1.4820]** ← 目标

### Online 流程

维护三个状态：`m`（当前最大）、`l`（累积分母）、`Õ`（未归一化输出）。

**Block 1: S=[1, 3], V=[[1,0], [0,1]]** — 块内 max = 3

```
P̃ = [exp(1-3), exp(3-3)] = [0.1353, 1]
l_block = 1.1353
Õ_block = 0.1353·[1,0] + 1·[0,1] = [0.1353, 1]
```

初始化：`m₁ = 3, l₁ = 1.1353, Õ₁ = [0.1353, 1]`

**Block 2: S=[2, 5], V=[[2,2], [1,1]]** — 块内 max = 5

```
P̃ = [exp(2-5), exp(5-5)] = [0.0498, 1]
l_block = 1.0498
Õ_block = [1.0996, 1.0996]
```

**合并** — 发现新 max = 5，旧状态要"降权"：

```
m₂ = max(3, 5) = 5
α = exp(3 - 5) = 0.1353    ← 旧状态的降权因子
β = exp(5 - 5) = 1          ← 新块的降权因子

l₂ = α·l₁ + β·l_block = 0.1353·1.1353 + 1·1.0498 = 1.2034
Õ₂ = α·Õ₁ + β·Õ_block = 0.1353·[0.1353, 1] + [1.0996, 1.0996] = [1.1179, 1.2349]
```

验证：处理完 [1,3,2,5] 应有的 sum = exp(-4)+exp(-2)+exp(-3)+1 = 1.2034 ✓

**Block 3: S=[4, 0], V=[[0,3], [3,0]]** — 块内 max = 4

```
P̃ = [1, 0.0183]
l_block = 1.0183
Õ_block = [0.0549, 3]
```

**合并** — 全局 max 没变（还是 5），新块要降权：

```
m₃ = max(5, 4) = 5
α = 1, β = exp(4 - 5) = 0.3679

l₃ = 1.5780
Õ₃ = [1.1381, 2.3386]
```

**归一化**：`O = Õ₃ / l₃ = [0.7213, 1.4820]` ✓ 与标准 softmax 一致。

---

## 三、为什么必须减 max？——数值稳定性

数学上"不减 max"版本更简洁：

```
l = Σ exp(S_i)                (累积分母)
Õ = Σ exp(S_i) · V_i           (累积分子)
```

合并两个块甚至不需要缩放，`l_new = l_old + l_block`。**数学上完全等价**，那为什么真实实现全都减 max？

### 原因：FP16/BF16 动态范围窄

| 类型 | 最大值 | `exp(x)` 溢出阈值 |
|---|---|---|
| FP16 | 65504 | **x ≈ 11.09** |
| BF16 | ~3.4e38 | x ≈ 88.7 |
| FP32 | ~3.4e38 | x ≈ 88.7 |

Attention scores 是 `QK^T / √d`，训练中 |x| > 20 是常态。假设某个 s = 25，FP16 下 `exp(25) → +Inf`，一个元素就把整个和污染成 Inf，最后 O = Inf/Inf = NaN，训练直接崩。

**减 max 之后**，所有指数参数被压到 ≤ 0，`exp` 值锁在 (0, 1]，永远不会溢出。underflow 是安全的（贡献本来就小），overflow 是致命的。

### 结论

减 max 不是数学要求，是**数值稳定性的工程要求**。生产级实现（cuDNN、Flash Attention v1/v2/v3、xFormers）**全部都减 max**——混合精度是标配，减 max 的开销在 memory-bound kernel 里几乎不可见。

---

## 四、S 切块的原则

### 约束一：SRAM 容量

每个 SM 的 SRAM 大约 100–228 KB：

| GPU | Shared Memory / SM | 典型可用 |
|---|---|---|
| A100 | 192 KB | ~160 KB |
| H100 | 228 KB | ~200 KB |
| RTX 4090 | 100 KB | ~80 KB |

要同时装下 Q_block、K_block、V_block、O_block 以及中间量 S_block=Q@K^T：

```
2 · (2·Br·d + 2·Bc·d + Br·Bc) ≤ M
```

论文推导：`Bc = ⌈M / (4d)⌉`，`Br = min(⌈M / (4d)⌉, d)`。

### 约束二：硬件对齐

- **Warp size = 32**：Br、Bc 最好是 32 的倍数
- **Tensor Core tile**：mma 指令是 16×16×16，块大小要是 16 的倍数才能满速
- 实际实现里几乎总是 **{64, 128, 256}** 里选

### 约束三：Occupancy

一个 SM 上要能同时跑多个 thread block 才能藏访存延迟。一般希望**每个 SM 至少 2 个 block 常驻**，所以每个 block 用 ≤ SRAM/2。

### Flash Attention v2 CUDA 里的实际数字

```cpp
// 前向传播
d = 32:   Br=128, Bc=128
d = 64:   Br=128, Bc=128       (A100)
d = 96:   Br=64,  Bc=128
d = 128:  Br=64,  Bc=128       (最常见: LLaMA、GPT)
d = 160:  Br=64,  Bc=64
d = 192:  Br=64,  Bc=64
d = 256:  Br=64,  Bc=64
```

规律：**d 越大，块越小**。反向传播因需要额外中间量，块通常还要更小。

---

## 五、α 和 β 到底是什么：参考系换算因子

### 直觉

每个块内部为了防溢出，减的是**块内 max**：

```
Block 里存的 Õ_block = Σ exp(sᵢ - m_block) · Vᵢ
                              ↑
                         "参考点"
```

不同块的 `m_block` 通常不一样——像三个人分别用米、厘米、毫米量身高。要合并统计，**必须统一到同一个参考系**。全局参考系 = `m_new = max(所有见过的 m)`。

`exp` 的乘法性质让"换参考系"极其便宜：

```
Σ exp(sᵢ - m_new) · Vᵢ = exp(m_ref - m_new) · Σ exp(sᵢ - m_ref) · Vᵢ
```

- **α = exp(m_old − m_new)** = 把"旧累积状态"从 m_old 拉到 m_new
- **β = exp(m_block − m_new)** = 把"新块"从 m_block 拉到 m_new

**新参考系一定等于两者较大者**。所以 α 和 β 里恰好有一个 = 1，另一个 ≤ 1：
- max 来自旧的 → **α = 1**，β < 1
- max 来自新的 → **β = 1**，α < 1

### 让 α 和 β 各自表演一次的例子

```
S = [5, 8, 2],  V = [[1], [10], [100]]   (每元素单独一块)
```

**Block 2 合并（新块带来新 max=8）**——α 表演：

```
α = exp(5 - 8) = 0.0498   ← 旧的从 "5参考系" 换到 "8参考系"
β = exp(8 - 8) = 1        ← 新块无需换
```

物理意义：旧的 Õ₁=[1] 是"以 5 为基准算的 exp(0)"；换到以 8 为基准，就应该是 exp(-3)=0.0498。

**Block 3 合并（max 不变，仍是 8）**——β 表演：

```
α = exp(8 - 8) = 1         ← 旧的已经在 8 参考系
β = exp(2 - 8) = 0.00248   ← 新块从 "2参考系" 大幅降权
```

物理意义：块内算的 `Õ_block=[100]` 是 `exp(2-2)·V = 1·100`——它假设"以 2 为参考"时贡献是 1。但全局参考是 8，`sᵢ=2` 相对于 max=8 是 `exp(-6)=0.00248`。**β 就负责把这个"局部虚高"的量拉回全局真实权重**。

### 常见疑问：为什么不直接用老 max 算，省掉 β？

**因为算块的时候，还不知道新块里有没有更大的值**。流水线是这样的：

```
1. 加载新块 K/V 到 SRAM
2. 算 S_block = Q @ K_block^T          ← 此时才知道块内的分数
3. 算 m_block = max(S_block)           ← 现在才知道块内最大值
4. 和 m_old 比，得到 m_new
5. 决定 α、β
```

第 2 步算 exp 时你还不知道 m_new。理论上可以"先扫一遍找 m 再算 exp"，但那样块内数据要读两遍——**Flash Attention 追求每个块只在 SRAM 里过一遍**，KV 加载进来就直接消费掉。

### 换个视角：β 是"迟到的补票"

块内算的时候你**假装** `m_block` 就是全局最大（乐观估计）。等发现真实的 m_new 后：

- 乐观估计对了（m_block == m_new）→ β = 1，不用补
- 乐观估计错了（m_block < m_new）→ β < 1，补一个降权因子

**用一次乘法代替"回头重算整个块"**——这才是 online softmax 的巧思。

### 会不会加大计算量？

不会。缩放的是**已经聚合好的统计量**（Õ 是 d 维向量，l 是标量），不是块内每一个 exp：

```
朴素"重算"：128 个 score × (1 exp + d 乘法) = O(块大小·d)
Online：Õ_old 乘一个标量 α = O(d)
```

设 N=8192, d=128, B=128：
- 主体计算：~8.6 G FLOPs
- Rescale：~8 K FLOPs
- **少 6 个数量级**，完全可以忽略

而且这些乘法都在 SRAM 里，不占 HBM 带宽——本来就在等 HBM 的间隙里"顺手"做完。

### 无分支实现

CUDA 实现通常长这样，一次算两个因子：

```cpp
float m_new = max(m_old, m_block);
float alpha = __expf(m_old   - m_new);   // 旧状态校正
float beta  = __expf(m_block - m_new);   // 新块校正

l_new = alpha * l_old + beta * l_block;
O_new = alpha * O_old + beta * O_block;
```

比起 `if/else` 版本，无分支在 GPU 上更快（warp 内不会 divergence），代码也更对称。

### 一句话记忆口诀

**"谁贡献了新 max，谁的因子就是 1；另一边负责搬家。"**

---

## 六、Varlen 落地：K/V 是 [total_len, dim] 而不是 [batch, seq, dim]

现代 LLM 训练的 Flash Attention 输入不是三维的 `[batch, seq_len, dim]`，而是二维 packed 格式。这是 **`flash_attn_varlen_func`** 接口的设计目标：**避免 padding 和还原**。

### 输入格式

```python
q: [total_q, num_heads, head_dim]        # total_q = Σ seq_len_i
k: [total_kv, num_heads, head_dim]
v: [total_kv, num_heads, head_dim]

cu_seqlens_q: [batch_size + 1]  # 累积长度前缀和
cu_seqlens_k: [batch_size + 1]
max_seqlen_q: int
max_seqlen_k: int
```

例：batch 里 3 条序列，长度分别是 5、3、7：

```
cu_seqlens = [0, 5, 8, 15]
              ↑     ↑     ↑
        序列0起点 序列1起点 序列2起点(=末尾)
```

序列 i 的 token = `q[cu_seqlens[i] : cu_seqlens[i+1]]`。

### 内部不还原成三维

CUDA kernel 每个 thread block 处理一条序列的一个 (Q块, KV块) 组合，通过 `cu_seqlens` 定位边界：

```cpp
int batch_id = blockIdx.z;
int q_start = cu_seqlens_q[batch_id];
int q_end   = cu_seqlens_q[batch_id + 1];
int seq_len = q_end - q_start;

Q_ptr = q + q_start * stride_q;
K_ptr = k + cu_seqlens_k[batch_id] * stride_k;
```

**关键**：不同序列间**永远不会互相 attend**——每个 CUDA block 被绑定到一个 `batch_id`，K/V 加载也只在这条序列范围内。**不需要 attention mask 来隔离**，天然的 batch 隔离。

### 为什么用 packed 格式

真实场景，batch 长度 [512, 128, 256, 1024, 96]，标准 padding 到最长的 1024：

```
Padded:  5 × 1024 = 5120 tokens (含大量 padding)
Packed:  512+128+256+1024+96 = 2016 tokens
```

**节省 60% 计算量**——padding 位置的 attention 完全是浪费。序列长度差异越大，收益越明显。

### Grid 划分

```
grid = (num_q_blocks_max, num_heads, batch_size)
                 ↑
           按 max_seqlen_q 划分
```

短序列的多余 block 会 early return，浪费一点点 launch 开销但不做实际计算。`max_seqlen` 只用来决定 grid 大小，不参与内存分配。

### Sequence Packing 训练

现代 LLM 训练常用 **sequence packing**：把多个短样本拼成一条长序列填满 max_seqlen。这时 `cu_seqlens` 就是文档边界的前缀和，Flash Attention 保证不同文档间不 attend——即使物理上它们在同一条 packed 序列里。

---

## 总结

**Flash Attention 三件套的最简概括**：

1. **Tiling** —— 用 SRAM 容量决定块大小，通常 64/128/256
2. **Online Softmax** —— 用 α/β 做参考系换算，一次乘法代替整块重算
3. **Recomputation** —— 反向重算 P，只存 O(N) 的 (m, l)

**从"显存带宽瓶颈"变成"算力瓶颈"，数学上完全等价，代价是多算一点 FLOPs，收益是少读大量 HBM。**

工程落地上还有一层：**varlen packed 格式** 通过 `cu_seqlens` 消除 padding，短长混合 batch 里能再省 50%+ 计算。这是长上下文训练的标配。

<!-- en -->
# Flash Attention Deep Dive: From Speedup Principles to Online Softmax to Varlen

> **2026-07-06** · by guoliang

## Overview

This note walks through Flash Attention bottom-up: why does it accelerate? How does online softmax actually work? Why must we subtract max? How are blocks sized? What are α/β correcting? And the varlen (packed) format used in production. **Bottom line: Flash Attention uses the trio of "tiling + online softmax + recomputation" to convert attention from a memory-bandwidth-bound problem into a compute-bound one. It is mathematically equivalent, not approximate.**

---

## 1. Why Flash Attention Is Faster: Attention Is Memory-Bound

Modern GPUs have severely mismatched compute and bandwidth. On A100:

- **Compute**: 312 TFLOPS (FP16)
- **HBM bandwidth**: 1.5–2 TB/s
- **SRAM bandwidth**: ~19 TB/s (only ~192 KB per SM)

Compute-to-memory ratio is **~150:1**—every byte read must feed 150 FLOPs to saturate the GPU. Standard attention is nowhere near that ratio; it's stuck on HBM bandwidth.

### Standard implementation makes 3 HBM round-trips

```
1. S = QK^T          → write N×N to HBM
2. P = softmax(S)    → read N×N, write N×N
3. O = PV            → read N×N, write N×d
```

- HBM access: O(N² + Nd)
- Memory: N=8K → 256 MB N×N intermediate (single head, FP16)
- Softmax needs full row for max/sum → resists tiling

### Flash Attention's Trio

| Technique | Role |
|---|---|
| **Tiling + Kernel Fusion** | Split Q/K/V into SRAM-sized blocks; do QK^T → softmax → PV entirely in SRAM without materializing N×N |
| **Online Softmax** | Math trick that lets tiled attention produce *exact* softmax—not an approximation |
| **Recomputation** | Backward doesn't store P; only stores O(N) (m, l) stats and recomputes P—trades FLOPs for HBM |

Complexity:

| Metric | Standard | Flash Attention |
|---|---|---|
| HBM access | O(N²·d) | **O(N²·d²/M)** (M = SRAM size) |
| Extra memory | O(N²) | **O(N)** |
| FLOPs | same | same (+ backward recompute) |

At d=64, M=192KB, HBM traffic drops **~9×**, wall-clock 2–4× (v1), 5–10× (v2/v3).

---

## 2. Online Softmax by Example

Setup: one row of scores, split into 3 blocks of 2 elements each.

```
S = [1, 3, 2, 5, 4, 0]
V = [[1,0], [0,1], [2,2], [1,1], [0,3], [3,0]]
```

Reference answer (standard softmax): **O = [0.7213, 1.4820]**.

Maintain three states: `m` (running max), `l` (accumulated denominator), `Õ` (unnormalized output).

**Block 1** (m_block=3): `l₁=1.1353, Õ₁=[0.1353, 1]`

**Block 2** (m_block=5, block-local): `l_block=1.0498, Õ_block=[1.0996, 1.0996]`

Merge — new max=5, old state must be scaled down:

```
α = exp(3 − 5) = 0.1353    ← rescale old
β = exp(5 − 5) = 1          ← new block already at reference

l₂ = α·l₁ + β·l_block = 1.2034
Õ₂ = α·Õ₁ + β·Õ_block = [1.1179, 1.2349]
```

**Block 3** (m_block=4): merge with global max still 5:

```
α = 1, β = exp(4 − 5) = 0.3679

l₃ = 1.5780, Õ₃ = [1.1381, 2.3386]
```

Final: `O = Õ₃/l₃ = [0.7213, 1.4820]` ✓

---

## 3. Why Subtract Max? — Numerical Stability

Mathematically, "no-max" online softmax is even simpler—no rescaling on merge. **But it breaks in FP16/BF16.**

| Type | Max value | `exp(x)` overflow at |
|---|---|---|
| FP16 | 65504 | **x ≈ 11.09** |
| BF16 | ~3.4e38 | x ≈ 88.7 |

Attention scores routinely exceed |20| during training. One `exp(25)` overflows to +Inf, poisons the whole sum, and O becomes NaN. Subtracting max caps all exponents at ≤ 0, keeping `exp` values in (0, 1]. **Underflow is safe (that term's contribution is small anyway); overflow is fatal.**

Every production kernel (cuDNN, Flash Attention v1/v2/v3, xFormers) subtracts max. The overhead is invisible in a memory-bound kernel—we're waiting on HBM anyway.

---

## 4. How Block Size Is Chosen

### Constraint 1: SRAM capacity

Fit Q_block, K_block, V_block, O_block, and S_block=Q@K^T:

```
2 · (2·Br·d + 2·Bc·d + Br·Bc) ≤ M
```

Paper's derivation: `Bc = ⌈M / (4d)⌉`, `Br = min(⌈M / (4d)⌉, d)`.

### Constraint 2: hardware alignment

- Warp size = 32 → block sizes should be multiples of 32
- Tensor Core mma is 16×16×16 → block dims must be multiples of 16
- Real values are almost always in **{64, 128, 256}**

### Constraint 3: occupancy

Want ≥ 2 blocks resident per SM to hide latency → each block ≤ SRAM/2.

### Actual numbers in Flash Attention v2

```
d = 32:   Br=128, Bc=128
d = 64:   Br=128, Bc=128
d = 128:  Br=64,  Bc=128       (LLaMA, GPT)
d = 256:  Br=64,  Bc=64
```

**Larger d → smaller blocks**. Backward pass uses even smaller blocks.

---

## 5. What α and β Really Are: Reference-Frame Conversion Factors

Each block internally subtracts *its own* `m_block` for numerical stability. So different blocks live in different "reference frames"—like measuring in meters, cm, mm. To merge, we must convert everyone to a common frame: the global running max `m_new`.

Thanks to `exp`'s multiplicative property:

```
Σ exp(sᵢ − m_new) · Vᵢ = exp(m_ref − m_new) · Σ exp(sᵢ − m_ref) · Vᵢ
```

- **α = exp(m_old − m_new)** — pull old accumulator to m_new frame
- **β = exp(m_block − m_new)** — pull new block to m_new frame

Since `m_new = max(m_old, m_block)`, exactly one of α, β equals 1; the other is ≤ 1. **Whoever contributed the new max has factor = 1; the other side does the moving.**

### Common question: why not just use m_old for the new block?

Because at the time you compute `exp(S_block − ?)`, you don't yet know if `m_block > m_old`. You'd need to scan the block twice—once for max, once for exp—which defeats the "each block passes through SRAM once" goal. Instead, block-locally compute against `m_block` (numerically safe), then apply β once at merge time.

### Does this add computation?

No. The rescale multiplies **already-aggregated** vectors/scalars, not per-element:

- Naive re-computation: O(block_size · d)
- Online: `α · Õ_old` = O(d)

For N=8192, d=128, B=128: main compute ~8.6 GFLOPs, rescale ~8 KFLOPs. **6 orders of magnitude smaller.** And it happens in SRAM during HBM idle time.

### Branchless CUDA form

```cpp
float m_new  = max(m_old, m_block);
float alpha  = __expf(m_old   - m_new);
float beta   = __expf(m_block - m_new);

l_new = alpha * l_old + beta * l_block;
O_new = alpha * O_old + beta * O_block;
```

No if/else → no warp divergence.

---

## 6. Varlen Landing: K/V as [total_len, dim] Instead of [batch, seq, dim]

Modern LLM training uses `flash_attn_varlen_func`, whose entire design goal is **eliminating padding**.

### Input format

```python
q, k, v: [total, num_heads, head_dim]
cu_seqlens_q, cu_seqlens_k: [batch_size + 1]  # cumulative prefix sums
max_seqlen_q, max_seqlen_k: int
```

Example: 3 sequences of length 5, 3, 7 → `cu_seqlens = [0, 5, 8, 15]`.

### Internal handling

The kernel does **not** unpack to 3D. Each thread block is bound to a `batch_id` (via `blockIdx.z`), reads its slice via `cu_seqlens`, and only attends within that slice. **No attention mask needed for batch isolation—it's structural.**

### Why packed matters

Batch of lengths [512, 128, 256, 1024, 96]:

- Padded: 5 × 1024 = 5120 tokens
- Packed: 2016 tokens
- **60% compute saved.**

Grid launches over `max_seqlen_q`; short sequences early-return excess blocks with negligible cost.

### Sequence packing

Multiple short training samples concatenated into one long sequence. `cu_seqlens` marks document boundaries; Flash Attention prevents cross-document attention automatically.

---

## Summary

**Flash Attention in one paragraph**: Tile Q/K/V into SRAM-sized blocks; use online softmax with α/β to merge partial results without materializing the N×N attention matrix; recompute P during backward instead of storing it. Turns a memory-bound problem into a compute-bound one, with mathematical equivalence to the naive implementation. Trades a small number of extra FLOPs for a large reduction in HBM traffic. In production, pair with the varlen packed format to eliminate padding waste in long-context training.
