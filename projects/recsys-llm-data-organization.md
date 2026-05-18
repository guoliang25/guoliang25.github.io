<!-- zh -->
# LLM/VLM 样本组织 vs 推荐系统：从 Megatron 到特征交叉

> **2026-05-18** · by guoliang

## 概述

本文从 NVIDIA Megatron-LM 的数据组织方式出发，系统对比 LLM 预训练、VLM 和推荐系统在样本组织上的核心差异，深入分析哪些技术可以迁移到推荐场景，以及推荐系统需要哪些个性化设计。最终聚焦于一个关键问题：**Token 化统一序列建模是否具备传统精排模型（FM/DCN）的特征交叉能力？**

---

## 一、Megatron-LM 数据组织架构

### 三层数据集架构

```
┌─────────────────────────────────────────────────┐
│          BlendedDataset (混合层)                  │
│  按权重混合多个数据集 (如 Wikipedia 70% + CC 30%)  │
├─────────────────────────────────────────────────┤
│          GPTDataset / MegatronDataset (逻辑层)    │
│  管理 train/valid/test 切分，构建定长训练样本       │
├─────────────────────────────────────────────────┤
│          IndexedDataset (存储层)                  │
│  二进制 .bin + .idx 文件, memory-mapped 随机访问   │
└─────────────────────────────────────────────────┘
```

### 样本构建方式：连续无重叠切片

Megatron **不使用滑动窗口**。所有文档拼成连续 token 流，按 `seq_length` 逐段切出样本：

```
文档流: [doc1_tokens...][EOD][doc2_tokens...][EOD][doc3_tokens...]

Sample 0: tokens[0     : 2049]     ← 2048 input + 1 label
Sample 1: tokens[2048  : 4097]     ← 紧接上一个样本
Sample 2: tokens[4096  : 6145]     ← 无重叠，无浪费
```

样本之间唯一的"重叠"是 1 个 token（上一个样本的最后 token = 下一个样本的第一个 token），用于构建 next-token prediction 的 `(input, label)` 对。

### 核心设计特点

| 特性 | 说明 |
|------|------|
| Memory-Mapped I/O | .bin 和 .idx 文件都使用 mmap，避免全部载入内存 |
| 索引缓存 | 构建好的索引缓存为 .npy 文件，避免重复计算 |
| 确定性 | 相同 seed + config = 完全相同的样本顺序 |
| 跨文档拼接 | 样本可跨越多个文档，最小化 padding 浪费 |
| C++ 加速 | 性能关键的索引构建用 C++ 实现 |

---

## 二、LLM vs 推荐系统：样本组织对比

### 样本的基本单元

| 维度 | 推荐系统 (流式模型) | LLM (Megatron) |
|------|---------------------|----------------|
| **一条样本** | 一次用户行为 (PV/曝光/点击) | 一段连续的 token 序列 (如 2048 tokens) |
| **样本内容** | 结构化特征向量 | 扁平的 token ID 序列 |
| **样本独立性** | 每条完全独立 | 从连续文档流切出，相邻样本共享上下文 |
| **label** | 二值/多值 (点击/转化/时长) | 下一个 token (自回归) |
| **Batch size** | 大 (~5000) | 小 (4~32 per GPU) |
| **数据时效性** | 实时/准实时 | 离线静态语料 |

### 核心哲学差异

```
推荐系统:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
• 每条样本 = 一个独立的"预测问题"
• 样本的质量来自特征的丰富度 (特征工程 >> 数据量)
• 样本有明确的时间属性 (昨天的样本可能已"过期")
• 训练目标 = 拟合当前分布（分布在实时变化）

LLM:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
• 每条样本 = 一段上下文 + 需要预测的下一个词
• 样本的质量来自数据的多样性和规模 (数据量 >> 特征工程)
• 样本没有时效性
• 训练目标 = 学习语言的通用规律（分布相对稳定）
```

---

## 三、可迁移到推荐的通用特性

### 1. 二进制索引格式 (IndexedDataset 模式)

当推荐模型处理长序列特征（用户近 N 天行为），mmap 索引格式比 TFRecord 高效得多：

```
.bin = 用户行为序列的 token 化特征数据
.idx = 每个用户/session 的 offset + seq_length
→ 支持按需 mmap 读取，不需要全部加载
```

### 2. Packing（样本拼接减少 Padding）

用户行为序列长度差异极大（活跃用户 500+，新用户 3~5），Packing 策略可迁移：

```
[user_A 序列 120步][SEP][user_B 序列 80步][PAD]  → 总长 200+pad
[user_C 序列 200步]                               → 刚好
[user_D 50步][SEP][user_E 60步][SEP][user_F 90步] → 拼满 200
配合 attention_mask 隔离不同用户
```

### 3. 多源数据混合 (BlendedDataset 模式)

Megatron 的确定性交错采样策略（贪心最大误差法）可用于推荐的多场景混合训练：

- 多场景: 首页信息流 70% + 搜索 20% + 详情页推荐 10%
- 多行为: 曝光 60% + 点击 30% + 购买 10%
- 多天数据: 时间衰减加权

### 4. 特征 Token 化 + 统一序列建模

**最核心的迁移趋势**——所有特征统一到 Token 空间：

```
行为序列 token 化:
[BOS][user_age:25][user_city:BJ][SEP]
[item:iPhone15][cat:手机][price:高][action:click][time:14:00]
[item:小米14][cat:手机][price:中][action:expose][time:14:01]
...
[PREDICT][item:华为Mate60][cat:手机][price:高][?]

→ 形式上和 LLM 完全一致：给定前文，预测下一个 token
```

---

## 四、需要个性化设计的差异点

### 1. 样本独立性（不能跨用户拼接）

```
LLM: 文档流连续 → 可以跨文档自由拼接 → 切在哪里只影响 attention 范围
推荐: 行为独立 → 不同用户的序列绝不能拼到一起当连续序列处理
     （除非用 attention_mask 显式隔离）
```

### 2. 稀疏特征 vs 稠密 Token

```
LLM: 一张 Embedding 表 (vocab_size × hidden_dim)
推荐: 几十张不同大小的 Embedding 表 + 超大规模 ID 类特征(10亿+)
     + 稠密连续特征 + 变长序列特征
→ 不能简单用"一个 .bin 存所有 token"的方案
→ 需要分字段存储 + 统一索引
```

### 3. 时效性与增量更新

```
LLM: 语料固定，多 epoch 训练，预构建全部索引
推荐: 数据每小时/每天更新，有"过期"概念
→ 需要支持增量追加新分区、淘汰旧分区、动态调整混合比例
```

### 4. Loss 设计差异

```
LLM:  每个位置都有 loss (next-token prediction)，loss_mask 简单
推荐: 需要更复杂的策略：
      - 序列开头不计算 loss（上下文不足）
      - 时间衰减权重
      - 最终预测位置加权
      - 多目标并行 (click + duration + conversion)
```

---

## 五、核心问题：Token 化序列模型的特征交叉能力

### FM/DCN 的显式交叉

$$\text{FM}: \quad y = w_0 + \sum_i w_i x_i + \sum_i \sum_{j>i} \langle \mathbf{v}_i, \mathbf{v}_j \rangle \cdot x_i \cdot x_j$$

$$\text{DCN Cross Layer}: \quad \mathbf{x}_{l+1} = \mathbf{x}_0 \cdot \mathbf{x}_l^T \cdot \mathbf{w}_l + \mathbf{b}_l + \mathbf{x}_l$$

**关键**: 对任意两个特征强制计算乘法交叉，交叉信号**一步到位**到达 loss。

### Attention 的交互方式

$$\alpha_{ij} = \text{softmax}\left(\frac{\mathbf{q}_i \cdot \mathbf{k}_j^T}{\sqrt{d}}\right)$$

$$\text{output}_i = \sum_j \alpha_{ij} \cdot \mathbf{v}_j$$

**关键区别**: Attention 内部有乘法（Q·K^T），但这个乘法结果被**消费在路由决策**上（变成注意力权重），最终输出是 V 的**加权求和**（加法！）。

### 信息流路径对比

```
FM:
  feature_i ──┐
              ├──→ <v_i, v_j> ──→ prediction
  feature_j ──┘
  路径长度: 1步 | 信号损失: 无

Attention:
  token_i ──┐                      
            ├──→ α_ij(路由) ──→ 加权求和 ──→ FFN ──→ ... ──→ prediction
  token_j ──┘
  路径长度: 2~12步 | 信号损失: 梯度衰减 + 信息瓶颈
```

### 乘法结果用在哪里——这是本质区别

| | FM/DCN | Transformer Attention |
|---|--------|-------------|
| 乘法位置 | 特征 embedding 内积 | Q·K^T |
| 乘法结果去向 | **直接进入预测** | **变成路由权重** |
| 最终输出 | 乘法信号直达 loss | V 的加法混合 |
| 类比 | 直接告诉你答案 | 给你导航，你自己再判断 |

### Attention 的优势领域

虽然在**低阶确定性交叉**上效率不如 FM/DCN，Attention 在以下方面更强：

**1. 条件交叉（动态选择交叉谁）**

```
FM/DCN: 所有特征对强制交叉 → O(n²)，很多是噪声
Attention: 动态决定"该交叉谁" → 有选择性，高维下更高效
```

**2. 高阶交叉不爆炸**

```
第1层: A attend to B → A' (含 A×B 信息)
第2层: A' attend to C → A'' (含 A×B×C 信息)
→ 天然支持任意阶，且是条件性的（不是全组合）
```

**3. 序列位置敏感交叉**

```
"先看手机再看耳机" vs "先看耳机再看手机"
→ Attention 天然区分，FM/DCN 做不到
```

### 数据效率的分水岭

| 数据量 | 最优策略 |
|--------|----------|
| < 1亿样本 | 显式交叉必须有，Transformer 学不充分 |
| 1-10亿样本 | 混合方案最优（交叉层 + Transformer） |
| > 100亿样本 | 纯 Transformer 可能够用，但仍不如混合方案高效 |

### 综合评价

```
            低阶确定性交叉                  高阶动态交叉
                ↑                              ↑
    FM ●●●●●   │                              │  ●☆☆☆☆
   DCN ●●●●☆   │                              │  ●●☆☆☆
浅层Att ●●☆☆☆  │                              │  ●●●☆☆
深层Att ●●●☆☆  │     ← 找平衡 →               │  ●●●●●
混合架构 ●●●●●  │                              │  ●●●●●  ← 理论最优
```

---

## 六、工程解决方案

### 方案 1: Token 序列模型 + 显式交叉层并行

```
输入: [BOS][age:25][city:BJ][SEP][item:iPhone][cat:手机][price:高]
              │
              ▼
     ┌──────────────────┐
     │ Embedding Layer   │
     └────────┬─────────┘
              │
       ┌──────┴──────┐
       │             │
       ▼             ▼
┌────────────┐ ┌────────────┐
│ Cross Layer│ │ Transformer│  ← 并行双路
│ (FM/DCN)   │ │  Layers    │
└──────┬─────┘ └─────┬──────┘
       │             │
       └──────┬──────┘
              ▼
     ┌──────────────────┐
     │ Prediction Head   │
     └──────────────────┘
```

### 方案 2: 多粒度 Token 设计

```
Layer 1 (域内聚合): 
  user_repr = Aggregate([age:25], [city:BJ], [gender:男])  
  item_repr = Aggregate([item:iPhone], [cat:手机], [price:高])
    
Layer 2 (域间交叉):
  cross_signal = user_repr ⊗ item_repr   ← 显式乘法交叉
    
Layer 3 (序列建模):
  [behavior_1(user⊗item₁), behavior_2(user⊗item₂), ...] → Transformer
```

### 方案 3: 乘法注意力改造

$$\text{Standard}: \quad \text{output} = \sum \text{softmax}(QK^T) \cdot V$$

$$\text{Multiplicative}: \quad \text{output} = \sum \text{softmax}(QK^T) \cdot (V \odot Q)$$

或双线性注意力：$\alpha_{ij} = \mathbf{x}_i^T \cdot W \cdot \mathbf{x}_j$

---

## 七、结论

### 迁移总结

| 维度 | 直接可迁移 | 需要改造 |
|------|-----------|---------|
| 存储层 | mmap + 索引格式 | 分字段存储 + 增量更新 |
| 混合层 | BlendedDataset 加权交错 | 时间衰减 + 动态权重 |
| 样本层 | Packing + 预构建索引 | 用户间隔离 + 增量索引 |
| 序列层 | Token 化 + 自回归范式 | 多目标 + 时间衰减 loss |
| 模型层 | Transformer 序列建模 | 必须保留显式交叉层 |

### 核心观点

> **Token 化统一了"表示"，但没有统一"交互方式"。**
>
> LLM 带给推荐最大的迁移价值不是模型架构本身，而是"一切皆 Token + 统一序列建模"的数据哲学。但推荐的有限数据量决定了：**显式乘法交叉的归纳偏置在当前数据规模下仍不可或缺**。
>
> 最务实的做法是："LLM 式的序列建模 + 精排式的显式交叉" **双路并行**。

---

## 参考

- [NVIDIA/Megatron-LM](https://github.com/NVIDIA/Megatron-LM) — 数据组织源码
- [Meta HSTU (2024)](https://arxiv.org/abs/2402.17152) — Actions Speak Louder than Words
- [ByteDance RankMixer (2025)](https://arxiv.org/abs/2507.02980) — 特征交互 Scaling
- [ByteDance MixFormer (2026)](https://arxiv.org/abs/2602.06700) — 统一精排架构

<!-- en -->
# LLM/VLM Data Organization vs RecSys: From Megatron to Feature Crossing

> **2026-05-18** · by guoliang

## Overview

Starting from NVIDIA Megatron-LM's data organization, this article systematically compares the core differences in sample organization between LLM pre-training, VLM, and recommendation systems. It analyzes which techniques are transferable to RecSys and what requires custom design. The analysis culminates in a key question: **Can tokenized unified sequence modeling match the feature crossing capability of traditional ranking models (FM/DCN)?**

---

## I. Megatron-LM Data Architecture

### Three-Layer Dataset Architecture

```
┌───────────────────────────────────────────────────┐
│          BlendedDataset (Blending Layer)            │
│  Mix datasets by weight (e.g. Wikipedia 70% + CC 30%)│
├───────────────────────────────────────────────────┤
│          GPTDataset / MegatronDataset (Logic Layer) │
│  Manage train/valid/test splits, build fixed-len samples│
├───────────────────────────────────────────────────┤
│          IndexedDataset (Storage Layer)             │
│  Binary .bin + .idx files, memory-mapped random access │
└───────────────────────────────────────────────────┘
```

### Sample Construction: Contiguous Non-overlapping Slicing

Megatron **does not use sliding windows**. All documents are concatenated into a continuous token stream and sliced by `seq_length`:

```
Token stream: [doc1_tokens...][EOD][doc2_tokens...][EOD][doc3_tokens...]

Sample 0: tokens[0     : 2049]     ← 2048 input + 1 label
Sample 1: tokens[2048  : 4097]     ← immediately follows previous
Sample 2: tokens[4096  : 6145]     ← no overlap, no waste
```

---

## II. LLM vs RecSys: Sample Organization Comparison

| Dimension | RecSys (Streaming Model) | LLM (Megatron) |
|-----------|--------------------------|----------------|
| **One sample** | One user action (PV/impression/click) | A continuous token sequence (e.g. 2048 tokens) |
| **Content** | Structured feature vector | Flat token ID sequence |
| **Independence** | Fully independent samples | Cut from continuous document stream |
| **Label** | Binary/multi-value (click/conversion) | Next token (autoregressive) |
| **Batch size** | Large (~5000) | Small (4~32 per GPU) |
| **Data freshness** | Real-time/near-real-time | Offline static corpus |

---

## III. Transferable Techniques to RecSys

### 1. Binary Index Format (mmap + offset)
For long user behavior sequences, mmap indexed format beats sequential TFRecord reads.

### 2. Packing (Reduce Padding Waste)
Pack variable-length user sequences with attention mask isolation.

### 3. Multi-Source Blending (Deterministic Interleaving)
Megatron's greedy max-error blending for multi-scenario / multi-day data mixing.

### 4. Feature Tokenization + Unified Sequence Modeling
The most impactful trend — unifying all features into a single token space.

---

## IV. Custom Design Requirements for RecSys

1. **Sample Independence** — Cannot freely concatenate across users
2. **Sparse Features** — Multiple embedding tables, ultra-large ID spaces (1B+)
3. **Timeliness** — Incremental updates, data expiration, sliding windows
4. **Complex Loss** — Multi-objective, time-decay, position-weighted

---

## V. The Core Question: Attention vs Feature Crossing

### The Essential Difference

$$\text{FM}: \quad \langle \mathbf{v}_i, \mathbf{v}_j \rangle \xrightarrow{\text{direct}} \text{prediction}$$

$$\text{Attention}: \quad Q \cdot K^T \xrightarrow{\text{routing}} \text{softmax} \xrightarrow{\text{weighted sum of V}} \text{representation}$$

- **FM/DCN**: Multiplicative result → directly enters prediction (1-step path)
- **Attention**: Multiplicative result → consumed as routing weights → output is additive (multi-step path)

### Attention's Strengths

| Capability | FM/DCN | Attention |
|-----------|--------|-----------|
| Low-order deterministic crossing | ★★★★★ | ★★☆☆☆ |
| Conditional (dynamic) crossing | ★☆☆☆☆ | ★★★★★ |
| High-order without explosion | ★★☆☆☆ | ★★★★★ |
| Sequence-position-sensitive | ☆☆☆☆☆ | ★★★★★ |
| Data efficiency (few-shot) | ★★★★★ | ★★☆☆☆ |

### Practical Recommendation

The optimal architecture combines both:
- **Explicit cross layers** (FM/DCN) for guaranteed low-order feature interactions
- **Transformer layers** for sequential modeling and high-order dynamic interactions

> Token化统一了"表示"，但没有统一"交互方式"。最务实的做法是双路并行。
>
> Tokenization unifies "representation" but not "interaction". The most pragmatic approach is a dual-path architecture.

---

## References

- [NVIDIA/Megatron-LM](https://github.com/NVIDIA/Megatron-LM)
- [Meta HSTU (2024)](https://arxiv.org/abs/2402.17152)
- [ByteDance RankMixer (2025)](https://arxiv.org/abs/2507.02980)
- [ByteDance MixFormer (2026)](https://arxiv.org/abs/2602.06700)
