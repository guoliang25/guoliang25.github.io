<!-- zh -->
# 多模态序列建模的 Adapter 方案 · RankMixer 无痛接入

> **2026-04-29** · by guoliang

## 概述

在已有 RankMixer 基线之上，如何把"多模态用户行为序列"无痛地接入，而不破坏 RankMixer 已经跑稳的 scaling 曲线？本项目从 **Adapter（Houlsby et al. ICML'19）** 的参数高效微调（PEFT）思路出发，给出一个**冻结 RankMixer + 多模态 Cross-Attention Adapter**的落地方案。

## 核心思想

Adapter 的三个特性让它天生契合推荐系统：**任务隔离**（不同任务互不影响）、**在线增量**（新任务只加 Adapter）、**小参数开销**（~3% 新参数）。在此之上，把多模态序列建模的输出当作 Cross-Attention 的 KV，让 RankMixer 的中间特征 token 作为 Query 按需去读取多模态信号。

## 整体架构

| 组件 | 角色 | 状态 |
|------|------|------|
| **RankMixer Block（N 层）** | 特征交互主干 | Stage 2 冻结 / Stage 3 低 LR |
| **多模态序列 Encoder** | 历史行为的视觉/文本/音频编码 | 冻结（或预训练） |
| **MM-Adapter（插在每个 RankMixer Block 后）** | Cross-Attention + bottleneck | **可训练** |
| **Tower** | 任务头 | **可训练** |

## Adapter 模块公式

原始 NLP Adapter：

$$\text{Adapter}(h) = h + W_\text{up}\,\sigma(W_\text{down}\,h)$$

改造为 Cross-Attention 版本（接入多模态序列 $M$）：

$$Z = H^{(\ell)} W_\text{down},\quad Z' = \text{MHA}(Q=Z,\ K=M W_K,\ V=M W_V),\quad H^{(\ell+1)} = H^{(\ell)} + Z' W_\text{up}$$

**近零初始化**保证启动时 $Z' W_\text{up} \approx 0$，整个 Adapter 输出 $\approx H^{(\ell)}$——模型在 step 0 等价于原 RankMixer baseline，训练过程中 Adapter 逐步激活注入多模态信号。

## PyTorch 实现（约 20 行）

```python
class MultiModalAdapter(nn.Module):
    def __init__(self, d_rank, d_mm, m=64, n_heads=4):
        super().__init__()
        self.down = nn.Linear(d_rank, m)
        self.mm_attn = nn.MultiheadAttention(
            embed_dim=m, num_heads=n_heads,
            kdim=d_mm, vdim=d_mm, batch_first=True
        )
        self.up = nn.Linear(m, d_rank)
        # 近零初始化：启动时 Adapter 输出 ≈ 0，不扰动 RankMixer
        nn.init.normal_(self.down.weight, std=1e-2)
        nn.init.normal_(self.up.weight,   std=1e-2)
        nn.init.zeros_(self.down.bias)
        nn.init.zeros_(self.up.bias)

    def forward(self, h_rank, mm_seq_kv, mm_key_padding_mask=None):
        z = self.down(h_rank)
        z_out, _ = self.mm_attn(
            query=z, key=mm_seq_kv, value=mm_seq_kv,
            key_padding_mask=mm_key_padding_mask
        )
        return h_rank + self.up(z_out)
```

## 三阶段训练策略

| 阶段 | 冻结 | 训练 | LR | 目的 |
|------|------|------|----|------|
| **Stage 1** | — | RankMixer 全量 | $10^{-3}$ | 跑 baseline 指标作对照 |
| **Stage 2** | RankMixer + 多模态 encoder | MM-Adapter + Tower | $10^{-3}$ | 快速验证多模态信号的增量 |
| **Stage 3** | 多模态 encoder（可选冻结） | RankMixer（$10^{-5}$）+ MM-Adapter（$10^{-3}$）+ Tower | 差 2 个数量级 | 联合微调榨取最后几个点 |

Stage 2 是关键——只训 Adapter 显存小迭代快，能在 1 周内回答"多模态信号到底有没有增量"这个问题，风险极低且可完全回滚。

## 与其它融合方案的对比

| 维度 | Stacked | Parallel | Unified (OneTrans/MixFormer) | **Adapter (本方案)** |
|------|---------|----------|------------------------------|--------------------|
| 改动面 | 小 | 中 | 大（重写 backbone） | **最小** |
| 信息保留 | 压成向量 | 融合前保留 | Token 级全程 | Token 级 + Cross-Attn |
| 深层交互 | 无 | 仅最后一次 | 每层 | **每层** |
| 可回滚 | 中 | 中 | 难 | **完全可回滚** |
| 多模态组合 | 统一编码 | 统一编码 | 统一 token | **每模态独立 Adapter** |
| 工程风险 | 低 | 中 | 高 | **低** |

## 落地 Checklist

- 先跑 baseline，确认 RankMixer 主干指标稳定
- **Stage 2 冻主干训 Adapter**——快速、廉价回答"Adapter 值不值"
- **分长尾桶看指标**——多模态信号对冷启动/长尾 item 增益应显著大于热门
- **对比 Tower-only baseline**——Adapter 必须超过"只扩大 Tower"才算真有价值
- 监控 Adapter 权重 norm（接近 0 = 没学到；爆炸 = 初始化太大）
- 每任务 Adapter 独立存（几 MB），像配置文件一样管理
- A/B 重点看冷启动 / 新用户 / 长尾 item

## 完整解读（含 Mermaid 架构图 + 论文原始结构图）

**👉 HTML 专题页：[Adapter 迁移到推荐系统 · 多模态序列 × RankMixer 的落地方案](https://guoliang25.github.io/cc_paper/adapter-multimodal-rankmixer.html)**

## 相关资源

- 原始论文：[Houlsby et al., Parameter-Efficient Transfer Learning for NLP, ICML 2019](https://arxiv.org/abs/1902.00751)
- 论文 Markdown 笔记：[cc_paper · 1902-Adapter](https://github.com/guoliang25/cc_paper/blob/main/paper/2019/1902-Adapter-Google-%E5%8F%82%E6%95%B0%E9%AB%98%E6%95%88%E5%BE%AE%E8%B0%83.md)
- 相关专题：[字节四大 Ranking 架构对比](https://github.com/guoliang25/cc_paper/blob/main/paper/%E5%AF%B9%E6%AF%94/%E5%AD%97%E8%8A%82%E5%9B%9B%E5%A4%A7Ranking%E6%9E%B6%E6%9E%84%E5%AF%B9%E6%AF%94-RankMixer-LONGER-OneTrans-MixFormer.md)（RankMixer / LONGER / OneTrans / MixFormer）

<!-- en -->
# Multimodal Sequence Adapter · Lossless Integration with RankMixer

> **2026-04-29** · by guoliang

## Overview

Given an existing RankMixer baseline, how do we integrate multimodal user behavior sequence modeling without breaking the already-stable scaling curve? This project draws on the **Adapter (Houlsby et al. ICML'19)** PEFT idea and proposes a **frozen-RankMixer + multimodal Cross-Attention Adapter** solution.

## Core Idea

Adapter's three properties make it naturally fit recommender systems: **task isolation**, **online incrementality**, **small parameter overhead** (~3% new params). On top of that, we use the multimodal sequence encoding as Cross-Attention KV, letting RankMixer's intermediate feature tokens (as queries) selectively attend to multimodal signals.

## Architecture

| Component | Role | Status |
|-----------|------|--------|
| **RankMixer Block (N layers)** | Feature interaction backbone | Frozen in Stage 2 / low LR in Stage 3 |
| **Multimodal Sequence Encoder** | Vision/text/audio encoding of history | Frozen (or pretrained) |
| **MM-Adapter (after each RankMixer block)** | Cross-Attention + bottleneck | **Trainable** |
| **Tower** | Task head | **Trainable** |

## Adapter Formula

Original NLP Adapter:

$$\text{Adapter}(h) = h + W_\text{up}\,\sigma(W_\text{down}\,h)$$

Cross-Attention variant (taking multimodal sequence $M$):

$$Z = H^{(\ell)} W_\text{down},\quad Z' = \text{MHA}(Q=Z,\ K=M W_K,\ V=M W_V),\quad H^{(\ell+1)} = H^{(\ell)} + Z' W_\text{up}$$

**Near-zero initialization** ensures $Z' W_\text{up} \approx 0$ at step 0, so the model starts equivalent to the original RankMixer baseline; Adapter gradually activates to inject multimodal signals.

## Three-Stage Training

| Stage | Frozen | Trained | LR | Goal |
|-------|--------|---------|----|----- |
| **Stage 1** | — | Full RankMixer | $10^{-3}$ | Baseline metrics as control |
| **Stage 2** | RankMixer + MM encoder | MM-Adapter + Tower | $10^{-3}$ | Validate MM signal value quickly |
| **Stage 3** | MM encoder (optional) | RankMixer ($10^{-5}$) + MM-Adapter ($10^{-3}$) + Tower | 2-order gap | Joint fine-tune for final gains |

Stage 2 is key — training only Adapter is memory-light and iteration-fast, and can answer the "is multimodal signal worth it" question within a week with zero risk and full rollback.

## Comparison with Other Fusion Paradigms

| Dimension | Stacked | Parallel | Unified | **Adapter (ours)** |
|-----------|---------|----------|---------|-------------------|
| Change footprint | Small | Medium | Large | **Minimal** |
| Info preservation | Vector-level | Pre-fusion | Token-level | Token + Cross-Attn |
| Deep interaction | None | Once at end | Every layer | **Every layer** |
| Rollback | Medium | Medium | Hard | **Fully rollback** |
| Modality combo | Unified enc | Unified enc | Unified token | **Per-modality Adapter** |
| Risk | Low | Medium | High | **Low** |

## Checklist

- Run baseline first, confirm RankMixer backbone metrics
- **Stage 2 frozen backbone + Adapter only** — fast, cheap answer to "is it worth it"
- **Bucket by popularity** — MM signal should gain more on cold/long-tail items
- **Compare with Tower-only baseline** — Adapter must beat "just enlarge the Tower"
- Monitor Adapter weight norm (near 0 = not learning; explosion = init too large)
- Per-task Adapter stored independently (few MBs), managed like config files
- A/B focus on cold-start / new users / long-tail items

## Full Deep-Dive (with Mermaid diagrams + original paper figure)

**👉 HTML companion page: [Adapter → RecSys · Multimodal Sequence × RankMixer](https://guoliang25.github.io/cc_paper/adapter-multimodal-rankmixer.html)**

## Resources

- Original paper: [Houlsby et al., Parameter-Efficient Transfer Learning for NLP, ICML 2019](https://arxiv.org/abs/1902.00751)
- Markdown paper notes: [cc_paper · 1902-Adapter](https://github.com/guoliang25/cc_paper/blob/main/paper/2019/1902-Adapter-Google-%E5%8F%82%E6%95%B0%E9%AB%98%E6%95%88%E5%BE%AE%E8%B0%83.md)
- Related topic: [Four ByteDance Ranking Architectures Compared](https://github.com/guoliang25/cc_paper/blob/main/paper/%E5%AF%B9%E6%AF%94/%E5%AD%97%E8%8A%82%E5%9B%9B%E5%A4%A7Ranking%E6%9E%B6%E6%9E%84%E5%AF%B9%E6%AF%94-RankMixer-LONGER-OneTrans-MixFormer.md) (RankMixer / LONGER / OneTrans / MixFormer)
