<!-- zh -->
# RankMixer 的 Token 数 T 与 Scaling Law 的斜率 / 截距

## 概述

本专题围绕字节跳动四篇工业级 Ranking 架构（**RankMixer / LONGER / OneTrans / MixFormer**）展开深入解读，聚焦三个容易被一笔带过但实际工业上非常关键的问题：

1. **RankMixer 里的 Token 数 $T$ 是不是拍的？** —— 答案是"有边界的拍"，被五条约束挤到只剩几个离散值。
2. **Per-token FFN 的参数量 $Td^2$ 到底怎么变？** —— 取决于你固定了 $E=Td$ 还是固定了 $d$，结论完全相反。
3. **Scaling Law 里的斜率和截距各代表什么？** —— 斜率是"算力兑换率"，截距是"起跑线"，两者独立、工业含义完全不同。

## 第一部分 · RankMixer 里的 Token 数 T 是不是拍的？

RankMixer 的特征 token 构造（论文 2.2 节）：

$$x_i = \text{Proj}\big(e_\text{input}[d \cdot (i-1) : d \cdot i]\big), \quad i = 1, \ldots, T$$

即把所有特征 embedding 拼成一条长向量 $e_\text{input} \in \mathbb{R}^{Td}$，**均匀切 $T$ 段**，每段做线性投影变成一个 $d$ 维 token。

### 五条约束

$T$ 是一个经验性工程超参，但它的可选空间被五个约束挤得非常窄：

**约束 1 · 总参数 / FLOPs 预算**。Dense 版本参数量公式：

$$\text{Param} \approx 2kLTd^2$$

主导项是 Per-token FFN 的 $T \cdot d^2$。

**约束 2 · Token Mixing 的整除性**。无参数 Token Mixing 的 reshape 要求 $d \bmod T = 0$，并且默认 $H=T$（head 数等于 token 数）。可选 $T$ 就被锁在 $\{8, 16, 32, 64\}$ 这种 2 的幂。

**约束 3 · Per-token FFN 的语义粒度**。$T$ 对应"把全部特征分成几个独立子空间做非线性变换"。$T$ 太小，user / item / context 被迫共享 FFN；$T$ 太大，每 token 拿到的特征太少。

**约束 4 · GPU 硬件对齐**。两个配置的 head dim 都恰好是 48：

| 配置 | $T$ | $d$ | head dim $d/T$ |
|------|-----|------|----------------|
| RankMixer-100M | 16 | 768 | **48** |
| RankMixer-1B | 32 | 1536 | **48** |

→ **head dim 锁在 48**（tensor core 友好），$T$ 和 $d$ 联动变化。

**约束 5 · Scaling Law 曲线平滑**。扫 $T$、$d$、$L$ 三条轴发现曲线几乎重合——$T$ 在合理范围内没有显著尖峰最优点，所以作者可以"比较随意"地取 16/32。

### 结论

$T$ 是**有边界的"拍"**——被五条约束挤到只剩 $\{8, 16, 32, 64\}$ 几个离散值，论文选 16/32 是"在这几个候选里挑硬件对齐最舒服、语义粒度也合理的两个"。

## 第二部分 · $Td^2$ 参数量怎么变：三种情境

**这是容易踩坑的点**：$T$ 变大时 Per-token FFN 参数量怎么变？完全取决于你固定了什么。

| 情境 | 固定的量 | $Td^2$ 随 $T$ 的走向 | 结论 |
|------|----------|----------------------|------|
| A · "切蛋糕"视角 | 总 embedding 预算 $E = Td$ | $Td^2 = E^2/T$ | **$1/T$ 下降** |
| B · "堆 FFN"视角 | 每 token 维度 $d$ | $Td^2 \propto T$ | **线性上升** |
| C · 只扩 $d$ | $T$ 固定 | $Td^2 \propto d^2$ | **二次上升** |
| D · 论文实际 | 两者同步放大 | $T{:}16{\to}32,\ d{:}768{\to}1536$ | **$Td^2$ 放大 8×** |

### 直觉对照

情境 A："一块蛋糕切成更多份，每份缩小得比份数增加快"—— $d$ 下降的 $d^2$ 效应比 $T$ 上升的线性效应更强。

情境 B："每份大小不变，份数越多总量越大"—— Per-token FFN 一套套堆上去。

情境 D："又加份数又加每份大小"—— 参数量爆炸式增长，论文 100M → 1B 就是这个路径。

### 为什么工业场景常用情境 A？

1. 原始特征 embedding 的总维度 $E$ 由业务决定（比如 500 个字段 × 16 维 = 8000），不是自由变量；
2. Feature Tokenization 的设计把 $e_\text{input} \in \mathbb{R}^{Td}$ 均匀切 $T$ 份—— $T$ 和 $d$ 在入口就是绑定的；
3. 但这个前提可以被打破——论文 1B 版本 $E$ 从 12288 增到 49152，$E$ 本身也扩了。

**两种视角都对，关键看你固定哪个变量。**工业上比较 token 切分策略时用 A（固定 $E$）；做 scaling 实验时用 D（两者同扩）。

## 第三部分 · Scaling Law 的斜率与截距

### 为什么可以画成一条直线？

工业推荐 scaling 论文通常拟合：

$$y \approx \alpha \cdot x^{\beta} + \gamma$$

其中 $x$ 是算力类变量（参数量、FLOPs、序列长度），$y$ 是性能指标（AUC、ΔAUC、LogLoss）。画成 **x 轴 log、y 轴线性** 后，曲线近似直线：

$$y \approx \beta \cdot \log x + \text{const}$$

一旦变直线，就有两个传统几何量：**斜率**和**截距**。

### 斜率 = 算力的兑换率

**定义**：$x$ 每翻倍，$y$ 涨多少。数学上 ≈ scaling law 里的指数 $\beta$。

**工业含义**：每多烧一倍算力能换回多少性能——**架构的 scaling 效率 / 远期潜力**。

- 斜率陡 → 加算力值得 → "远期潜力大"
- 斜率平 → 加算力没用 → "架构饱和了"

### 截距 = 起跑线

**定义**：拟合直线在 $\log x = 0$ 的 $y$ 值——直线在 y 轴上的抬升高度。

**工业含义**：给最小算力时，架构能到什么性能——**小模型的底子好不好**。

- 截距高 → "一开始就领先"
- 截距低 → "起点低"，但靠斜率可能追回来

### 斜率 × 截距的四种组合

| 斜率 | 截距 | 含义 | 典型案例 |
|------|------|------|----------|
| 高 | 高 | **双优**——起点高、上限也高 | Chinchilla 风格的新一代架构 |
| 高 | 低 | **后发制人**——小时候不行，越 scale 越强 | Transformer 早期 vs LSTM |
| 低 | 高 | **早期赢家**——起点高但加算力不涨 | 饱和的老架构 |
| 低 | 低 | 全面落后 | Baseline |

### 关键区分

- **"斜率更高"** = 未来更值得投资（扩上去差距会扩大）
- **"截距更高"** = 现在就比你强（但扩上去不一定继续甩开）

两者独立，语义完全不同。

## 第四部分 · 四大架构的 scaling 曲线实例

### RankMixer · 比 Wukong / HiFormer / DHEN 斜率更陡

RankMixer 在 Params 和 FLOPs 两条轴上都比 Wukong / HiFormer / DHEN 更陡；不同扩展方向（$L$、$d$、$T$）的曲线几乎重合，说明它"吃得下算力"。

**翻译**："同样把参数从 100M 扩到 1B，RankMixer 的涨幅比 Wukong / HiFormer 大——远期更值得投资。"

### OneTrans · 斜率显著 > RankMixer

OneTrans 论文把自己和 RankMixer 的 ΔUAUC-vs-log(FLOPs) 画在一张图里，OneTrans 那条线更陡。论文的解读：

> RankMixer-centric scaling lacks a unified backbone; its MoE-based expansion predominantly widens the FFN hidden dimension. OneTrans, offering a unified Transformer backbone, scales more parameter- and compute-efficiently.

**翻译**："RankMixer 扩上去主要靠增加 MoE FFN 的宽度，没有统一 backbone；OneTrans 有统一 backbone，所以 scaling 效率更高。"

### MixFormer · 斜率与 SOTA 序列模型持平，截距更高

MixFormer 在 512 → 10k 长度下始终在 SOTA 序列模型之上，两条线近似平行——斜率持平但截距领先。

**翻译**："MixFormer 在各种长度档位都比对手高一个固定幅度；再拉长也不会进一步拉开——一开始就胜，扩上去也不掉。"

### LONGER · 三轴各自独立 power-law

LONGER 论文把 sequence length / params / FLOPs 三条轴分别拟合 $y = \alpha x^\beta + \gamma$，都得到 $R^2 > 0.96$：

- 长度 300 → 5k：AUC / LogLoss 随 length 呈 power-law 改善；
- 固定 2 层、扫宽度：AUC vs params 强 power-law（$R^2 = 0.987$）；
- 固定 $d=32$、扫 layers + length：AUC vs FLOPs 强 power-law（$R^2 = 0.967$）。

可惜 LONGER 论文没有跟其它架构在同一张图里对比斜率。

## 一句话总结

1. **$T$ 不是纯拍的**：被 FLOPs 预算、Token Mixing 整除性、Per-token FFN 粒度、GPU 硬件对齐、scaling law 曲线五条约束挤到 $\{8,16,32,64\}$ 几个离散值，论文选 16/32 是 head dim 锁到 48 的硬件友好选择。
2. **$Td^2$ 怎么变取决于你固定什么**：固定 $E=Td$ → $1/T$ 下降；固定 $d$ → 线性上升；论文实际两者同扩。
3. **Scaling law 的斜率 = 兑换率**（未来潜力）；**截距 = 起跑线**（当下胜负）。
4. **四大架构定位**：RankMixer 比前代斜率陡（FI 领域内升级）；OneTrans 斜率 > RankMixer（统一 backbone 的未来价值）；MixFormer 斜率持平、截距更高（现在就比你强）；LONGER 三轴均强 power-law，但没画跨架构对比。

## 相关资源

- **完整笔记仓库**：<https://github.com/guoliang25/cc_paper>
- **四大架构深度对比**：[字节四大 Ranking 架构对比](https://github.com/guoliang25/cc_paper/blob/main/paper/%E5%AF%B9%E6%AF%94/%E5%AD%97%E8%8A%82%E5%9B%9B%E5%A4%A7Ranking%E6%9E%B6%E6%9E%84%E5%AF%B9%E6%AF%94-RankMixer-LONGER-OneTrans-MixFormer.md)
- **HTML 专题页（含论文原图）**：<https://guoliang25.github.io/cc_paper/rankmixer-T-and-scaling-slope-intercept.html>
- **原始论文**：
  - RankMixer：[arXiv:2507.15551](https://arxiv.org/abs/2507.15551)
  - LONGER：[arXiv:2505.04421](https://arxiv.org/abs/2505.04421)
  - OneTrans：[arXiv:2510.26104](https://arxiv.org/abs/2510.26104)
  - MixFormer：[arXiv:2602.14110](https://arxiv.org/abs/2602.14110)

<!-- en -->
# RankMixer's Token Count T and the Slope/Intercept of Scaling Law

## Overview

This topic provides a deep dive into four industrial Ranking architectures from ByteDance (**RankMixer / LONGER / OneTrans / MixFormer**), focusing on three frequently glossed-over but industrially critical questions:

1. **Is the token count $T$ in RankMixer just arbitrary?** — Answer: it's a "bounded arbitrary choice", squeezed into a handful of discrete values by five constraints.
2. **How exactly does Per-token FFN's parameter count $Td^2$ scale?** — Depends on whether you fix $E=Td$ or fix $d$; the conclusion is opposite.
3. **What do slope and intercept of a scaling law represent?** — Slope is the "compute exchange rate", intercept is the "starting line"; they are independent and have completely different industrial meanings.

## Part I · Is T in RankMixer Arbitrary?

RankMixer constructs feature tokens (paper Section 2.2) as:

$$x_i = \text{Proj}\big(e_\text{input}[d \cdot (i-1) : d \cdot i]\big), \quad i = 1, \ldots, T$$

i.e., concatenate all feature embeddings into one long vector $e_\text{input} \in \mathbb{R}^{Td}$, **uniformly slice it into $T$ segments**, and linearly project each segment into a $d$-dim token.

### Five Constraints

$T$ is an empirical engineering hyperparameter, but its feasible space is severely compressed by five constraints:

**Constraint 1 · Total parameter / FLOPs budget**:

$$\text{Param} \approx 2kLTd^2$$

Dominant term is Per-token FFN's $T \cdot d^2$.

**Constraint 2 · Divisibility of Token Mixing**. Parameter-free Token Mixing requires $d \bmod T = 0$, and defaults to $H=T$. Feasible $T$ is locked to powers of 2 like $\{8, 16, 32, 64\}$.

**Constraint 3 · Semantic granularity of Per-token FFN**. $T$ corresponds to "how many independent subspaces to divide all features into". Too small → user/item/context forced to share FFN; too large → each token has too few features.

**Constraint 4 · GPU hardware alignment**. Both configurations have head dim exactly 48:

| Config | $T$ | $d$ | head dim $d/T$ |
|--------|-----|------|----------------|
| RankMixer-100M | 16 | 768 | **48** |
| RankMixer-1B | 32 | 1536 | **48** |

→ **Head dim locked to 48** (tensor-core friendly), $T$ and $d$ scale together.

**Constraint 5 · Smooth scaling law curves**. Sweeping $T$, $d$, $L$ yields near-identical curves — no sharp optimum for $T$, so 16/32 can be chosen "liberally".

### Conclusion

$T$ is a **bounded arbitrary choice** — the five constraints squeeze it to just $\{8, 16, 32, 64\}$, and 16/32 are picked for the best hardware alignment and semantic granularity.

## Part II · How $Td^2$ Scales: Three Scenarios

**This is the easy trap**: how does Per-token FFN's parameter count change when $T$ grows? It depends entirely on what you hold fixed.

| Scenario | What's fixed | $Td^2$ vs $T$ trend | Result |
|----------|--------------|---------------------|--------|
| A · "Cake slicing" view | Total embedding budget $E = Td$ | $Td^2 = E^2/T$ | **$\propto 1/T$ decreasing** |
| B · "Stacked FFN" view | Per-token dim $d$ | $Td^2 \propto T$ | **Linear growth** |
| C · Widen only | $T$ fixed | $Td^2 \propto d^2$ | **Quadratic growth** |
| D · Paper's actual scaling | Both grow together | $T{:}16{\to}32, d{:}768{\to}1536$ | **$Td^2$ grows 8×** |

**Both views are correct — depending on what's fixed.** In industry, scenario A is natural when comparing tokenization strategies under a fixed embedding budget; scenario D is what papers actually do for scaling experiments.

## Part III · Scaling Law's Slope and Intercept

### Why can it be drawn as a straight line?

Industrial recsys scaling papers typically fit:

$$y \approx \alpha \cdot x^{\beta} + \gamma$$

On **x-axis log, y-axis linear** plots this becomes near-linear:

$$y \approx \beta \cdot \log x + \text{const}$$

### Slope = Compute Exchange Rate

**Definition**: how much $y$ increases when $x$ doubles. Mathematically ≈ exponent $\beta$.

**Industrial meaning**: how much performance each doubling of compute buys — **scaling efficiency / long-term potential**.

### Intercept = Starting Line

**Definition**: the $y$ value of the fit line at $\log x = 0$.

**Industrial meaning**: baseline performance at minimal compute — **how good the architecture is at small scale**.

### Four combinations

| Slope | Intercept | Meaning | Example |
|-------|-----------|---------|---------|
| High | High | **Dual-winner** | Chinchilla-style next-gen |
| High | Low | **Late bloomer** | Transformer early vs LSTM |
| Low | High | **Early winner** | Saturated legacy arch |
| Low | Low | Full loss | Baseline |

### Key distinction

- **"Higher slope"** = more worth investing in (gap widens at scale)
- **"Higher intercept"** = already better now (but may not keep pulling ahead)

## Part IV · Scaling Curves of Four Architectures

- **RankMixer**: slope steeper than Wukong / HiFormer / DHEN — "absorbs compute well".
- **OneTrans**: slope significantly > RankMixer — "unified backbone scales better".
- **MixFormer**: slope on par with SOTA sequence models, intercept higher — "already wins at any length, doesn't fall behind when scaled".
- **LONGER**: three axes each fit strong power-law ($R^2 > 0.96$), but no cross-architecture slope comparison.

## One-liner Summary

1. **$T$ is not purely arbitrary** — locked to $\{8,16,32,64\}$ by five constraints; 16/32 chosen for head-dim-48 alignment.
2. **How $Td^2$ scales depends on what you fix** — $1/T$ under fixed $E$; linear under fixed $d$.
3. **Scaling slope = exchange rate** (future potential); **intercept = starting line** (current standing).
4. **Four architectures' positions**: RankMixer steeper than predecessors; OneTrans steeper than RankMixer; MixFormer intercept higher, slope flat; LONGER strong per-axis but no cross-arch comparison.

## Resources

- **Full notes repo**: <https://github.com/guoliang25/cc_paper>
- **Four-architecture deep comparison**: [Markdown](https://github.com/guoliang25/cc_paper/blob/main/paper/%E5%AF%B9%E6%AF%94/%E5%AD%97%E8%8A%82%E5%9B%9B%E5%A4%A7Ranking%E6%9E%B6%E6%9E%84%E5%AF%B9%E6%AF%94-RankMixer-LONGER-OneTrans-MixFormer.md)
- **HTML version with paper figures**: <https://guoliang25.github.io/cc_paper/rankmixer-T-and-scaling-slope-intercept.html>
- **Original papers**:
  - RankMixer: [arXiv:2507.15551](https://arxiv.org/abs/2507.15551)
  - LONGER: [arXiv:2505.04421](https://arxiv.org/abs/2505.04421)
  - OneTrans: [arXiv:2510.26104](https://arxiv.org/abs/2510.26104)
  - MixFormer: [arXiv:2602.14110](https://arxiv.org/abs/2602.14110)
