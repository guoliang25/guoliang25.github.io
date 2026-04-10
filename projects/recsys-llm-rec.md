<!-- zh -->
# LLM驱动的生成式推荐系统

## 概述

本项目探索将大语言模型（LLM）深度融入推荐系统全链路，构建下一代生成式推荐架构。不同于传统的"召回-粗排-精排-重排"范式，我们利用LLM的强大语义理解能力，在用户兴趣建模、候选集重排序和推荐理由生成等环节实现突破。

## 系统架构

<!-- ![系统架构图](./images/recsys-llm-arch.png) -->

系统整体分为三个核心模块：

| 模块 | 技术方案 | 作用 |
|------|----------|------|
| 用户兴趣理解 | LLM + User Behavior Sequence | 从行为序列中提取深层语义兴趣 |
| 候选集重排序 | LLM as Re-ranker | 结合上下文信息对候选集进行语义重排 |
| 推荐理由生成 | Prompt Engineering + Fine-tuning | 生成个性化的推荐解释文本 |

## 关键技术

### 用户兴趣建模

传统推荐系统使用ID Embedding表示用户和物品，但存在冷启动和跨域迁移困难的问题。我们提出基于LLM的语义兴趣表征方法：

1. **行为序列编码**：将用户历史行为转化为自然语言描述
2. **兴趣提取**：利用LLM从行为描述中提取多维度兴趣标签
3. **兴趣融合**：将语义兴趣与传统ID特征进行深度融合

### 损失函数

模型训练采用多任务学习框架，总损失函数定义为：

$$L_{total} = \alpha \cdot L_{rec} + \beta \cdot L_{gen} + \gamma \cdot L_{align}$$

其中：
- $L_{rec}$ 为推荐任务的交叉熵损失
- $L_{gen}$ 为文本生成的语言模型损失
- $L_{align}$ 为语义对齐的对比学习损失

推荐损失具体定义为：

$$L_{rec} = -\sum_{i=1}^{N} \left[ y_i \log(\hat{y}_i) + (1-y_i) \log(1-\hat{y}_i) \right]$$

其中 $y_i \in \{0, 1\}$ 是真实标签，$\hat{y}_i$ 是模型预测的点击概率。

## 实验结果

在离线评估中，相比传统推荐基线模型：

| 指标 | 基线模型 | LLM增强模型 | 提升 |
|------|----------|-------------|------|
| AUC | 0.7234 | 0.7512 | +3.84% |
| NDCG@10 | 0.4156 | 0.4523 | +8.83% |
| 推荐解释满意度 | 62.3% | 81.7% | +31.1% |

## 后续规划

- 探索更高效的LLM推理加速方案（如量化、蒸馏）
- 研究用户隐私保护下的LLM推荐方案
- 拓展到多模态推荐场景

<!-- en -->
# LLM-Driven Generative Recommendation System

## Overview

This project explores deep integration of Large Language Models (LLMs) into the full recommendation pipeline, building a next-generation generative recommendation architecture. Unlike the traditional "retrieval-pre-ranking-ranking-re-ranking" paradigm, we leverage LLM's powerful semantic understanding for breakthroughs in user interest modeling, candidate re-ranking, and recommendation explanation generation.

## System Architecture

<!-- ![System Architecture](./images/recsys-llm-arch.png) -->

The system consists of three core modules:

| Module | Technical Approach | Purpose |
|--------|-------------------|---------|
| User Interest Understanding | LLM + User Behavior Sequence | Extract deep semantic interests from behavior sequences |
| Candidate Re-ranking | LLM as Re-ranker | Semantic re-ranking with contextual information |
| Explanation Generation | Prompt Engineering + Fine-tuning | Generate personalized recommendation explanations |

## Key Technologies

### User Interest Modeling

Traditional recommendation systems use ID Embeddings for users and items, but face cold-start and cross-domain transfer challenges. We propose an LLM-based semantic interest representation method:

1. **Behavior Sequence Encoding**: Convert user behavior history into natural language descriptions
2. **Interest Extraction**: Use LLMs to extract multi-dimensional interest tags from behavior descriptions
3. **Interest Fusion**: Deep fusion of semantic interests with traditional ID features

### Loss Function

The model is trained with a multi-task learning framework. The total loss function is defined as:

$$L_{total} = \alpha \cdot L_{rec} + \beta \cdot L_{gen} + \gamma \cdot L_{align}$$

Where:
- $L_{rec}$ is the cross-entropy loss for the recommendation task
- $L_{gen}$ is the language model loss for text generation
- $L_{align}$ is the contrastive learning loss for semantic alignment

The recommendation loss is specifically defined as:

$$L_{rec} = -\sum_{i=1}^{N} \left[ y_i \log(\hat{y}_i) + (1-y_i) \log(1-\hat{y}_i) \right]$$

Where $y_i \in \{0, 1\}$ is the ground truth label, and $\hat{y}_i$ is the predicted click probability.

## Experimental Results

In offline evaluation, compared to traditional recommendation baselines:

| Metric | Baseline | LLM-Enhanced | Improvement |
|--------|----------|--------------|-------------|
| AUC | 0.7234 | 0.7512 | +3.84% |
| NDCG@10 | 0.4156 | 0.4523 | +8.83% |
| Explanation Satisfaction | 62.3% | 81.7% | +31.1% |

## Future Work

- Explore more efficient LLM inference acceleration (quantization, distillation)
- Research privacy-preserving LLM recommendation approaches
- Extend to multimodal recommendation scenarios
