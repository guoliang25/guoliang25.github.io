<!-- zh -->
# RMSNorm vs LayerNorm：从 LLM 到推荐系统的迁移之路

> **2026-09-20** · by guoliang

## 概述

一个观察到的行业信号：LLM 侧（LLaMA / Mistral / Qwen / DeepSeek / Gemma 等）几乎全线迁移到了 **RMSNorm**，而经典推荐模型（DIN / DIEN / SIM）以及大量在线流式推荐场景仍然停留在 **LayerNorm**。到了 2026 年，随着推荐主干 Transformer 化、生成式推荐范式落地，**推荐侧也在跟着往 RMSNorm 迁**——但不是全盘替换，而是分位置分模块地推进。

本文把两者的原理、差异、各自适合什么场景讲清楚，最后落到"什么时候该在推荐里换 RMSNorm、什么时候不该换"这个工程问题上。

---

## 一、原理

### LayerNorm (Ba et al., 2016)

对单个样本的特征维度做「去均值 + 除标准差 + 仿射」：

$$
\text{LN}(x) = \gamma \odot \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}} + \beta,\quad \mu = \frac{1}{d}\sum_{i} x_i,\ \sigma^2 = \frac{1}{d}\sum_{i} (x_i - \mu)^2
$$

两个可学习参数 $\gamma$、$\beta$，同时做 **re-centering（重中心化）** 和 **re-scaling（重缩放）**。

### RMSNorm (Zhang & Sennrich, 2019)

省掉均值那一步，只用均方根做缩放：

$$
\text{RMS}(x) = \gamma \odot \frac{x}{\sqrt{\frac{1}{d}\sum_{i} x_i^2 + \epsilon}}
$$

只保留 $\gamma$，只做 **re-scaling**，**无 re-centering、无 $\beta$**。

---

## 二、差异一览

| 维度 | LayerNorm | RMSNorm |
|---|---|---|
| 中心化 | ✅ 减均值 | ❌ 不减 |
| 参数量 | $\gamma + \beta$ | 只有 $\gamma$ |
| 计算量 | 2 次归约（均值 + 方差） | 1 次归约（平方和） |
| 数值/带宽开销 | 更贵 | 便宜 ~7–30%（长序列越明显） |
| 表达力 | 稍强 | 论文 & 实测：几乎无损 |

---

## 三、为什么 LLM 都换成 RMSNorm

三层原因叠加：

1. **等效性**：LLaMA 系列消融显示，去掉 re-centering 对语言建模 loss 基本无影响 —— 均值那一项在残差流里被后续的 Linear 吸收掉了。
2. **速度 / 带宽**：Transformer 里 norm 调用极其频繁（每层 pre-norm 两次），少一次 reduce + 少一个参数在 GPU/TPU 上是可观收益，在 megatron / FSDP 里 all-reduce 更会被放大。
3. **训练稳定性**：Pre-RMSNorm + 深层残差流表现更稳、梯度更好；这条被 LLM 训练规模化验证得极其充分。

结果就是：**GPT-NeoX、LLaMA、Mistral、Qwen、DeepSeek、Gemma** 几乎全线走了 RMSNorm 这条路。

---

## 四、为什么推荐系统曾经不迁 RMSNorm

在 DIN / DIEN / SIM 时代，推荐系统里 LayerNorm 一直是主流。原因也很实在：

1. **特征均值有强漂移**：user / item / context 特征分布随时间、流量、活动漂移得非常厉害，**re-centering 本身就有意义** —— 把当天的均值打掉，让下游 MLP 看到的是"相对偏差"，能显著缓解分布漂移。RMSNorm 少了这一步，抗漂移能力就打折扣。
2. **数值尺度差异大**：连续特征（时长、点击率）、embedding pooling 后的向量、cross 特征混在一起，量纲和均值都不齐 —— 减均值这一步实际上承担了"不同特征块对齐"的作用。
3. **网络深度浅**：DIN / DIEN / SIM / 早期 RankMixer 一般十几层封顶，norm 调用总次数比 LLM 少 1–2 个数量级，RMSNorm 省的那点算力换不来什么，反而失去 $\beta$ 的自由度。
4. **$\gamma$、$\beta$ 在稀疏场景很有用**：某些通道天然分布偏离 0，$\beta$ 允许每个通道单独重定位，对 CTR / CVR 稀疏正样本的场景更友好。
5. **历史惯性 + 稳定优先**：推荐系统上线要求平稳、可 A/B 归因，换 norm 的 ROI 不如换特征、换序列建模来得高。

---

## 五、2026 年的转折：推荐也在迁 RMSNorm

现在（2026 年）行业发生了两件事，让推荐侧开始跟进 RMSNorm：

### 5.1 迁移正在发生的信号

- **大厂序列建模主干 Transformer 化**：DIN → BST → SIM → HSTU（Meta）→ RankMixer / TWIN / LONGER 这条线，越来越像"推荐里的小 LLM"，backbone 直接照搬 LLaMA 系列的 **Pre-RMSNorm + RoPE + SwiGLU** 三件套。
- **生成式推荐（Generative Recommender）**：Meta 的 HSTU、快手的 OneRec、字节的生成式召回/排序，本质就是 decoder-only Transformer 做 next-item / next-action 预测，norm 直接沿用 LLM 那一套 RMSNorm。
- **长序列建模**：用户行为序列拉到 1k–10k 甚至更长后，attention + FFN 的 norm 调用极频繁，RMSNorm 省掉均值那次 reduce 的收益在长序列下被放大，跟 LLM 的动机彻底对齐。

### 5.2 为什么现在能迁了（之前的顾虑逐步被化解）

1. **特征漂移问题被前置解决**：现代推荐 pipeline 里 `BatchNorm on dense` / `feature-wise standardization` / `online z-score` 已经在特征入口就把均值打掉了，backbone 里的 re-centering 变成冗余，去掉 $\beta$ 也不再痛。
2. **Embedding 主导**：序列建模主干几乎全是 embedding + attention，dense 手工特征被压缩成小分支或直接干掉，量纲问题不再由 norm 层扛。
3. **深度上去了**：HSTU、生成式排序模型层数从个位数拉到 20–40+，norm 计算开销开始有感，RMSNorm 的算力/带宽收益从"不值一提"变成"可量化"。
4. **训练稳定性验证过关**：Pre-RMSNorm + residual stream 的 scaling law 在 LM 上被验证得非常充分，推荐侧直接吃到这份红利，训练更稳、更容易堆深。
5. **$\gamma$ 已够用**：实测消融显示，去掉 $\beta$ 对推荐指标（AUC / GAUC / NDCG）影响在噪声内，甚至偶有小涨（少一组参数、过拟合略降）。

---

## 六、工程落地：分位置分模块

目前的实操共识大致是：

| 位置 | 建议 |
|---|---|
| 序列 Transformer 主干（attention + FFN） | ✅ 迁 RMSNorm，收益明确 |
| 生成式召回 / 排序 decoder | ✅ 直接对齐 LLM 栈 |
| Dense 特征 MLP / 交叉网络（DCN、PPNet、MMoE tower） | ⚠️ 仍常留 LayerNorm 或 BatchNorm，$\beta$ 有用、量纲杂 |
| 特征入口 | ⚠️ 用 BN / 标准化处理漂移，不指望 norm 层扛 |

### 迁移检查清单

- **backbone 是纯 Transformer 结构** → 可以直接 pre-RMSNorm 试点，风险很低
- **仍是 embedding + 大 MLP 交叉的老架构** → 没必要强行换，收益不足以对冲风险
- **序列长度 ≥ 512 且层数 ≥ 12** → RMSNorm 的算力收益开始有感，值得上
- **有明确的量化 / 部署压力** → RMSNorm 少一组参数、少一次 reduce，对推理侧也更友好

---

## 七、总结

- **LLM**：token embedding 分布相对稳定 + 网络极深 + 算力敏感 → **RMSNorm 更划算**，几乎是既定标准。
- **老式推荐（DIN 时代）**：特征漂移严重 + 量纲杂 + 网络浅 → **LayerNorm 的 re-centering + $\beta$ 仍然值得**。
- **现代推荐（2026 年后）**：Transformer 主干化 + 生成式范式 + 特征标准化前置 → **序列建模那部分事实上已经在跟着 LLM 栈迁 RMSNorm**，dense / cross 分支才是 LayerNorm 的最后阵地。

RMSNorm 不是"更先进所以要换"的问题，而是**当推荐架构越来越像 LLM 时，norm 的选择自然收敛到 LLM 的答案**。反过来，如果你的模型还没走到那一步，LayerNorm 依然是稳妥选择。

---

## 参考文献

- [1] Ba, Kiros, Hinton. "Layer Normalization." 2016. arXiv:1607.06450
- [2] Zhang, Sennrich. "Root Mean Square Layer Normalization." NeurIPS 2019. arXiv:1910.07467
- [3] Touvron et al. "LLaMA: Open and Efficient Foundation Language Models." 2023. arXiv:2302.13971
- [4] Zhai et al. "Actions Speak Louder than Words: Trillion-Parameter Sequential Transducers for Generative Recommendations (HSTU)." ICML 2024.
- [5] Zhou et al. "Deep Interest Network for Click-Through Rate Prediction (DIN)." KDD 2018.

<!-- en -->
# RMSNorm vs LayerNorm: The Migration Path from LLM to RecSys

> **2026-09-20** · by guoliang

## Overview

An industry signal worth tracking: on the LLM side (LLaMA / Mistral / Qwen / DeepSeek / Gemma etc.) almost the entire stack has migrated to **RMSNorm**, while classical recommendation models (DIN / DIEN / SIM) and many online streaming recommendation systems still stay on **LayerNorm**. As of 2026, with recommendation backbones going Transformer-native and generative recommendation paradigms landing in production, **the recommendation side is now migrating to RMSNorm as well** — but not wholesale, rather position-by-position, module-by-module.

This note walks through the principles, differences, and per-scenario fit of both, and closes on the engineering question: "when should you switch to RMSNorm in recommendation, and when shouldn't you?"

---

## 1. Principles

### LayerNorm (Ba et al., 2016)

Per-sample "de-mean + divide-by-std + affine" along the feature dimension:

$$
\text{LN}(x) = \gamma \odot \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}} + \beta,\quad \mu = \frac{1}{d}\sum_{i} x_i,\ \sigma^2 = \frac{1}{d}\sum_{i} (x_i - \mu)^2
$$

Two learnable parameters $\gamma$ and $\beta$, doing both **re-centering** and **re-scaling**.

### RMSNorm (Zhang & Sennrich, 2019)

Drops the mean step, uses only root-mean-square for scaling:

$$
\text{RMS}(x) = \gamma \odot \frac{x}{\sqrt{\frac{1}{d}\sum_{i} x_i^2 + \epsilon}}
$$

Keeps only $\gamma$, does only **re-scaling** — **no re-centering, no $\beta$**.

---

## 2. Differences at a Glance

| Aspect | LayerNorm | RMSNorm |
|---|---|---|
| Centering | ✅ subtracts mean | ❌ no |
| Params | $\gamma + \beta$ | $\gamma$ only |
| Compute | 2 reductions (mean + variance) | 1 reduction (sum of squares) |
| Compute / bandwidth | more expensive | ~7–30% cheaper (grows with seq len) |
| Expressiveness | slightly stronger | paper + practice: essentially no loss |

---

## 3. Why LLMs All Switched to RMSNorm

Three overlapping reasons:

1. **Equivalence**: LLaMA ablations show removing re-centering has no measurable effect on LM loss — the mean term gets absorbed by the subsequent Linear in the residual stream.
2. **Speed / bandwidth**: norm is called extremely frequently in Transformers (twice per pre-norm layer). One fewer reduce + one fewer parameter is a real GPU/TPU gain, amplified in megatron / FSDP all-reduce.
3. **Training stability**: Pre-RMSNorm + deep residual streams train more stably with cleaner gradients — validated exhaustively at LM training scale.

Net result: **GPT-NeoX, LLaMA, Mistral, Qwen, DeepSeek, Gemma** — essentially every major open-weight LLM now uses RMSNorm.

---

## 4. Why RecSys Historically Didn't Migrate

Through the DIN / DIEN / SIM era, LayerNorm remained the default. The reasons were substantive:

1. **Feature means drift heavily**: user / item / context feature distributions drift with time, traffic, and campaigns. **Re-centering itself has value** — subtracting today's mean lets the downstream MLP see "relative deviation", which materially eases distribution shift. Without it, RMSNorm loses drift robustness.
2. **Numerical scale mismatch**: continuous features (dwell time, CTR), pooled embeddings, and cross features share a network with wildly different scales and means. Subtracting the mean actually acts as a per-feature-block aligner.
3. **Shallow depth**: DIN / DIEN / SIM / early RankMixer typically cap at a dozen layers. Norm call count is 1–2 orders of magnitude below LLMs — RMSNorm's compute savings buy nothing, at the cost of losing $\beta$'s degree of freedom.
4. **$\gamma$, $\beta$ are useful in sparse settings**: some channels have distributions naturally offset from 0. $\beta$ lets each channel re-locate independently — a friendly property for sparse-positive CTR/CVR scenarios.
5. **Historical inertia + stability priority**: production RecSys demands smooth launches and clean A/B attribution. Swapping norm has lower ROI than swapping features or sequence modeling.

---

## 5. The 2026 Turning Point: RecSys Is Also Migrating

Two things happened in the industry that turned this around:

### 5.1 Migration signals

- **Big-tech sequence backbones going Transformer-native**: DIN → BST → SIM → HSTU (Meta) → RankMixer / TWIN / LONGER — increasingly a "small LLM inside recommendation", with backbones directly adopting the LLaMA-family **Pre-RMSNorm + RoPE + SwiGLU** trio.
- **Generative Recommenders**: Meta's HSTU, Kuaishou's OneRec, ByteDance's generative retrieval/ranking — all essentially decoder-only Transformers doing next-item / next-action prediction, inheriting the LLM RMSNorm stack directly.
- **Long-sequence modeling**: as user behavior sequences stretch to 1k–10k or beyond, attention + FFN norm calls become dominant, and RMSNorm's saved reduce scales with sequence length — the motivation converges with LLM's.

### 5.2 Why it can happen now (prior concerns have been addressed)

1. **Feature drift is handled upstream**: modern RecSys pipelines already handle drift via `BatchNorm on dense` / `feature-wise standardization` / `online z-score` at feature-entry. Re-centering inside the backbone becomes redundant, and dropping $\beta$ no longer hurts.
2. **Embedding-dominated backbones**: sequence backbones are almost entirely embedding + attention, dense hand-crafted features get compressed into a small branch or eliminated. Norm layers no longer carry the scale-alignment burden.
3. **Depth is now real**: HSTU and generative rankers scale from single-digit layers to 20–40+. Norm compute becomes measurable, and RMSNorm's savings move from "irrelevant" to "quantifiable".
4. **Training stability proven**: Pre-RMSNorm + residual stream scaling laws are extensively validated on LM. RecSys inherits this for free — easier to train deep, easier to stabilize.
5. **$\gamma$ is enough**: ablations show dropping $\beta$ has no measurable impact on RecSys metrics (AUC / GAUC / NDCG), sometimes marginally positive (fewer params → slightly less overfit).

---

## 6. Engineering Landing: Position-by-Position

Current practical consensus:

| Position | Recommendation |
|---|---|
| Sequence Transformer backbone (attention + FFN) | ✅ Migrate to RMSNorm, clear win |
| Generative retrieval / ranking decoder | ✅ Directly align with LLM stack |
| Dense-feature MLP / cross networks (DCN, PPNet, MMoE tower) | ⚠️ Keep LayerNorm or BatchNorm — $\beta$ still useful, scales still messy |
| Feature entry | ⚠️ Use BN / standardization to handle drift; don't ask norm to handle it |

### Migration checklist

- **Backbone is pure Transformer** → safe to pilot pre-RMSNorm, low risk
- **Still embedding + heavy MLP crossing** → not worth forcing the switch, ROI doesn't justify risk
- **Seq len ≥ 512 and depth ≥ 12** → RMSNorm compute benefit becomes meaningful, worth adopting
- **Explicit quantization / deployment pressure** → fewer params + one fewer reduce also helps inference

---

## 7. Summary

- **LLM**: token embedding distributions are relatively stable + networks are very deep + compute-sensitive → **RMSNorm wins**, effectively the established standard.
- **Legacy RecSys (DIN era)**: severe feature drift + mixed scales + shallow networks → **LayerNorm's re-centering + $\beta$ still earn their keep**.
- **Modern RecSys (2026+)**: Transformer backbones + generative paradigm + upstream feature standardization → **the sequence-modeling portion is de-facto migrating to RMSNorm alongside the LLM stack**; the dense / cross branches remain LayerNorm's last stronghold.

RMSNorm isn't "newer, therefore switch". Rather, **as recommendation architectures converge on LLM structure, the norm choice naturally converges to the LLM answer**. Conversely, if your model hasn't reached that structural point, LayerNorm remains the safe default.

---

## References

- [1] Ba, Kiros, Hinton. "Layer Normalization." 2016. arXiv:1607.06450
- [2] Zhang, Sennrich. "Root Mean Square Layer Normalization." NeurIPS 2019. arXiv:1910.07467
- [3] Touvron et al. "LLaMA: Open and Efficient Foundation Language Models." 2023. arXiv:2302.13971
- [4] Zhai et al. "Actions Speak Louder than Words: Trillion-Parameter Sequential Transducers for Generative Recommendations (HSTU)." ICML 2024.
- [5] Zhou et al. "Deep Interest Network for Click-Through Rate Prediction (DIN)." KDD 2018.
