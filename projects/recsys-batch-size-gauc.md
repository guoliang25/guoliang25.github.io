<!-- zh -->
# RecSys 生产训练里 batch_size 的深度分析：从现象到 Optimizer 耦合

> **2026-08-17** · by guoliang

## 概述

一个真实的观察：在一个 MixerFormer 主干的 pCVR 模型上，把 batch_size 从 500 降到 300，离线 GAUC 提升 +0.1%。lr、epoch 数、step 数、样本组织（day-by-day 时间顺序）都没变，只动了 batch。

小 batch 效果更好这个现象几乎每个 RecSys 团队都遇到过，"教科书答案"通常是：小 batch 有隐式正则、平坦极小值、BN 敏感……但对着我们真实的训练配置逐条对照，会发现**教科书答案只解释了一部分**。本文把归因拆成四层：优化视角、RecSys 视角、归一化视角、指标视角；再叠加**Optimizer 配置耦合**这层深挖，能解释为什么这个现象在我们生产训练里被明显放大。附带把"lr 为什么这么难调"（sum→mean 换算对不齐）这个老坑一并讲透，最后给出一套按投入产出比排序的解法工具箱。

**核心 insight**：教科书讨论 batch_size 时默认的是标准 Adam(β1=0.9)，而 RecSys 生产训练里 dense 部分常用低 β1、极大 β2 的定制 optimizer + sparse embedding 独立走 Adagrad。**这类配置本质上把 batch_size 的敏感度显著放大**——这才是"batch_size 变一点效果就飘"的深层原因。

---

## 一、现象与配置

### 观察

| 变量 | 配置 A（基线） | 配置 B |
|---|---|---|
| batch_size | 500 | 300 |
| lr | 2.5e-6（不变）| 2.5e-6（不变）|
| epoch / step | 相同 | 相同 |
| 训练模式 | day-by-day 按时间顺序 | 同上 |
| 模型主体 | MixerFormer | 同上 |
| Norm 层 | 以 LayerNorm 为主，少量 BN | 同上 |
| 离线 GAUC | 基线 | **+0.1%** |

### Optimizer 配置

```yaml
gpu_optimizer_type:
  dense: "cadam"
  sparse: "adagrad_sparse"

cadam:
  learning_rate: 2.5E-6
  beta_1: 1.0E-5              # ⚠️ 极低
  beta_2: 0.9999              # ⚠️ 极高
  epsilon: 1.0E-8
  fused: true
  optimizer_version: "v2"

adagrad_sparse:
  lr: 0.005
  eps: 1.0                    # ⚠️ 极大
  initial_accumulator_value: 0.0
  foreach: true
```

三处标注 ⚠️ 的参数偏离标准值，都是团队踩过坑总结出的定制配置，也是后文归因分析的关键。

---

## 二、教科书归因：优化视角

### 1. 梯度噪声正则化 & 平坦极小值（Keskar et al., ICLR 2017）

小 batch 的梯度是真梯度的高噪声估计，这层噪声等价于隐式正则化：
- 大 batch → 收敛到**尖锐极小值**（sharp minima）→ 权重扰动敏感 → 泛化差
- 小 batch → 收敛到**平坦极小值**（flat minima）→ 对分布漂移鲁棒 → 泛化好

Keskar 用 Hessian 特征值证明了 sharp/flat 与泛化的相关性，是这个方向的奠基工作。

### 2. 有效噪声尺度 = lr / batch_size

这是**最常被忽略的耦合**：绝大多数人改 batch 时不同步调 lr，实际是"变 batch × 隐式变 lr"两件事同时发生。

- batch 500→300 → 有效噪声尺度 × 5/3 ≈ 1.67
- 如果原 lr 偏保守（我们的 2.5e-6 就是），这个隐式抬升方向恰好合适

**Linear Scaling Rule**（Goyal et al., 2017）：`batch × k → lr × k`（SGD 系）
**Square-Root Rule**（Adam 系更稳）：`batch × k → lr × √k`

必做的对照实验：`batch=500, lr × 1.67`，看 GAUC 差距是否收敛。

### 3. 优化步数 vs epoch 数

同 epoch 数下，batch=1024 vs 4096 的 optimizer 步数差 4 倍。Adam 二阶矩收敛、embedding 稀疏更新次数都依赖 step 数。**改成"同 step 数"对比时，差距往往缩小甚至反转**——但本项目已经控制了 step 相同，所以这条不适用。

---

## 三、RecSys 特有归因

### 1. Sparse Embedding 更新粒度

推荐模型 90%+ 参数在 embedding table 里，每个样本只稀疏点亮几个 embedding：

- **大 batch**：同一个 item embedding 在 batch 内被多个用户命中 → 一次 step 里被"平均更新"→ 个体信号被稀释
- **小 batch**：每次更新更"个性化"，梯度方向保留更多样本级信息
- **长尾 item**（一个 batch 里可能只出现 1~2 次）：大 batch 下和头部 item 梯度一起被优化器动量稀释

### 2. Day-by-day 训练下的分布漂移适应（本项目关键）

Day-by-day 增量训练 = 每天有一次显著的分布 shift（新广告主、新流量、新素材）：

- 大 batch 每 step 覆盖跨度更大的样本 → 梯度是"跨分布平均"→ 对当天新增/漂移信号响应迟钝
- 小 batch 每 step 更 local → 对**最新数据分布的适应更精细**
- 离线 GAUC 通常在**次日/近几天数据**上评估 → 谁对最近分布适应得好谁赢

**验证方法**：把训练模式改成"混合 shuffle 而非严格 day-by-day"，看小 batch 优势是否被削弱。

### 3. 头部样本主导 vs 长尾覆盖

User-item 分布是幂律的：
- 大 batch 内头部 item 占比高 → 梯度被头部 pattern 主导
- 小 batch 更"局部"→ 长尾 pattern 有机会主导某一步的更新
- GAUC 惩罚"用户内排序错误"，而非"整体 CTR 偏移"→ 头部主导反而伤 GAUC

---

## 四、归一化视角：BN vs LN 的 batch 敏感度差异

### BN 对 batch 敏感的三个原因

BN 训练时统计量：`μ_B = (1/B) Σ x_i`，`σ²_B = (1/B) Σ (x_i - μ_B)²`

1. **统计方差 ∝ 1/B**：小 batch 的 μ_B, σ²_B 是 noisy estimate → 每 step 的归一化基准在抖 → 隐式正则
2. **Train/Eval 不一致**：训练用 batch stats，推理用 running stats（EMA）→ 小 batch 下 running stats 收敛慢、方差大
3. **样本相关性污染统计**：batch 内不 iid（比如同用户多条曝光打包）→ 统计量有偏

### LN 完全不敏感

LN：`μ_i = (1/D) Σ_d x_{i,d}`——**每个样本独立归一化**，只依赖 feature 维度 D。batch=1 和 batch=10000 效果完全一样。

### 定量：batch 500 vs 300 下 BN 差异有多大

BN 统计量方差比：`σ²_B(300) / σ²_B(500) = 500/300 ≈ 1.67`，标准差比值 √1.67 ≈ 1.29。

某层激活真实分布 μ=0, σ=1 时：

| batch | μ_B 的抖动范围 (±1σ) | σ_B 的抖动范围 (±1σ) |
|---|---|---|
| 500 | ±0.045 | ±0.032 |
| 300 | ±0.058 | ±0.041 |

**绝对差异 ~0.013**——比 lr × 1.67 的更新方向差异小得多，远小于 dropout 0.1 的噪声。

### BN 敏感的真正阈值

| batch_size | BN 状态 |
|---|---|
| ≥ 32 | 统计量足够稳定，BN 表现基本一致 |
| 16 ~ 32 | 轻微退化 |
| 8 ~ 16 | 明显退化，建议 GroupNorm/LN |
| < 8 | BN 基本不可用 |

**500 vs 300 都在 BN 舒适区，差异可忽略。本项目主要用 LN，这条归因基本可以划掉。**

---

## 五、指标视角：GAUC 为什么对 batch_size 敏感

GAUC = 每个用户内部算 AUC 再加权平均，衡量"同一用户下正样本 pCVR 是否高于负样本"。

- BCE loss 是 **pointwise** 的，梯度并不直接优化 within-user 排序
- 小 batch 训练时，**同一用户的多条样本更可能在相近 step 里被看到**（如果 shuffle 粒度是样本级）→ 模型对"这个用户偏好什么"的记忆更新更连贯
- 大 batch 时同用户样本被稀释在一大堆其他用户里，within-user 的 pairwise 信号被"洗掉"

**GAUC 对 batch_size 的敏感度 > AUC 对 batch_size 的敏感度**——这是 RecSys 里的特殊现象，CV/NLP 讨论 batch_size 通常看整体 loss/accuracy，感知不到。

---

## 六、深层归因：Optimizer 配置如何放大 batch_size 敏感度

**这一节是本项目相比"教科书讨论"的核心增量**。

### 1. cAdam β1 = 1e-5：动量缺失让 batch 噪声直接透传

标准 Adam β1 = 0.9，一阶矩平滑窗口 ≈ 10 步：`m_t = 0.9 m_{t-1} + 0.1 g_t`。这个动量项吸收了 60~70% 的梯度噪声。

本项目 β1 = 1e-5，窗口 ≈ 1 步：`m_t ≈ g_t`。相当于把 Adam 退化成 RMSProp（只保留二阶矩自适应）。

后果：
- 每步的 update 方向完全由**当前 batch** 决定
- **小 batch 的高梯度噪声 100% 转化为 update 抖动**（在 β1=0.9 下会被动量过滤掉大部分）
- 这就是为什么 batch 500→300 这么小的变化在本项目里能看到 GAUC +0.1%

**换言之：低 β1 的 optimizer 配置放大了 batch_size 的敏感度。**

### 2. β2 = 0.9999：稳定 scale 但训练动力学被锁定

标准 β2 = 0.999，窗口 1000 步。本项目 0.9999 → 窗口 10000 步。

结合 β1 ≈ 0 一起看，这个组合的信号是：
- **一阶矩**：完全不平滑，只信当前梯度方向
- **二阶矩**：极度平滑，用长期稳定的 scale 归一化梯度

设计意图很清晰：为**稳态 online 训练**定制。不平滑方向（保持对当前样本的响应），但极度平滑 scale（防止数值不稳定）。非常合理，符合 day-by-day 增量训练的语义。

代价：`√v̂` 变化极慢，训练启动后 lr 的有效强度基本被锁死。**想通过调 lr 快速对齐效果几乎不可能**——你调完 lr，前 10000 步 `v̂` 都没稳定，看不出趋势；等稳定了，训练已经跑完。

### 3. Dense/Sparse 双 Optimizer：lr 语义分裂

Dense (cAdam) 和 Sparse (Adagrad) 走两套完全不同的更新逻辑：

| 维度 | cAdam (Dense) | Adagrad (Sparse) |
|---|---|---|
| 步长语义 | 自适应二阶矩，`lr × sign(g)` 上界 | 累积平方梯度，`lr / √G` 衰减 |
| 更新频率 | 每 step 全参数更新 | 只在被点亮时更新 |
| 长尾行为 | 无特别保护 | 天然自适应（G 小 → 步长大） |
| lr 数量级 | 2.5e-6 | 0.005 |

好处：这个组合直接解决了"稀疏 embedding 的 Adam 状态失衡"这个经典大坑。
代价：**调 dense lr 时 sparse 在按自己的节奏走**，两边耦合但你调不动整体。

### 4. Adagrad eps=1.0：天然 warmup + 长尾保护

Adagrad update: `θ ← θ - lr × g / (√G + ε)`

标准 eps = 1e-10（只防除零）。本项目 eps = **1.0**，比标准大 10 个数量级。

行为：
- 训练早期 G 很小 → update ≈ `lr × g / 1.0 = 0.005 g` → **前期退化成 SGD**
- G 累积到 ≈ 1.0 后 → Adagrad 自适应特性启动
- 相当于**天然的 warmup**（前期步长平稳）
- 对**长尾 embedding**（G 一直很小）特别友好：始终保持较大有效步长

配合 `initial_accumulator_value = 0.0`：训练启动时 update 稳定进入，避免大 g 撞小 G 的震荡。

### 小结：这套 optimizer 配置是"为 day-by-day 增量训练定制的稳态动力学"

- 低 β1 → 保持对当前样本的敏感响应
- 高 β2 → 稳定长期 scale
- Adagrad eps=1.0 → 长尾保护 + 天然 warmup
- 双 optimizer → 消解稀疏梯度不平衡

**这套配置的代价是极难微调**：任何超参改动（batch、lr、loss scale、precision）都会立刻反映在 update 上，没有动量做缓冲。**这也是"batch 500→300 GAUC +0.1%"这个信号在本项目里能被清晰观察到的技术原因。**

---

## 七、为什么 lr 这么难调：Sum→Mean 换算对不齐的深层原因

一个反复踩到的坑：loss 从 sum 改成 mean 后，怎么调优化器参数都对不齐效果。理论上 sum 和 mean 差 B 倍，`lr × B` 应该等价，但实际远不是这么简单。

### 数学上的关系

```
loss_sum  = Σ_{i=1..B} l_i
loss_mean = (1/B) Σ l_i = loss_sum / B
∇(loss_mean) = ∇(loss_sum) / B
```

梯度差 B 倍。

### SGD 下的理论换算

`lr_mean = lr_sum × B`——干净利落。

### Adam 下：`lr` 几乎不用改（但不完全等价）

Adam 更新：`θ ← θ - lr × m̂ / (√v̂ + ε)`

- `m̂` scale × 1/B
- `√v̂` scale × 1/B
- `m̂ / √v̂` **scale 不变**
- 所以 Adam 下 sum→mean 时 lr 理论上几乎不用改

**但只是"几乎"**，因为：

1. **ε 项没被 scale 掉**：`√v̂` 从 10 变到 0.01 后，ε=1e-8 相对量级不同，某些参数的更新被 ε 抑制
2. **v̂ 收敛需要时间**：前几百步 `v̂` 不准，sum vs mean 的 `v` 累积速度不一样 → 前期 update 差异大
3. **Weight Decay 有效强度漂移（AdamW 里最难发现的一条）**：
   - sum + wd=0.01：每 step wd 更新 = `-lr × 0.01 × θ`，与 loss 梯度量级匹配
   - 换 mean 后 loss 梯度小 B 倍，wd 更新没变 → **wd 有效强度变强了 B 倍**
   - 解法：`wd × (1/B)`

### RecSys 独有的坑：Sparse Embedding 的 Adam State 时间戳

- Embedding 的 Adam state 只在被点亮时才推进 t
- sum→mean 后有些 embedding 的 `m̂ / √v̂` 比值突变，行为不再一致
- 这个几乎无解，只能用 Adagrad/FTRL 绕开（本项目已经用了）

### 梯度裁剪的绝对阈值失效

`clip_grad_norm = 1.0` 在 sum 下几乎不触发，mean 下触发频率大变。解法：clip 阈值按 1/B 缩放。

### 完整的 Sum→Mean 对齐清单

```
sum → mean:
  lr:            Adam 几乎不用改，SGD 需 × B
  wd:            除以 B（或用 decoupled wd 的绝对形式）
  ε:             除以 B（可选，影响小）
  clip 阈值:      除以 B
  warmup steps:  增加（让 v̂ 有时间收敛）
```

**结论：即使做完这些，因为 sparse embedding 那条几乎无解，完全对齐是不可能的。行业共识：sum/mean 换完只求收敛到相近效果，别期望完全对齐。**

---

## 八、解法工具箱：按投入产出比排序

针对"想要小 batch 的泛化 + 大 batch 的效率"这个矛盾，业界已有一套系统性方法。按成本从低到高：

### Tier 1：零成本 —— 调优化器让大 batch 逼近小 batch

#### 1.1 加回 momentum（对本项目最直接）

- β1 从 1e-5 试 0.5 / 0.7 / 0.9
- 验证假设："β1≈0 让 batch 敏感度被放大"
- 如果 β1=0.9 + batch=500 效果 ≥ β1=1e-5 + batch=300 → 直接切换，训练效率提升 67%

#### 1.2 Linear/Sqrt Scaling Rule + Warmup

- batch × k → lr × k（SGD）或 lr × √k（Adam）
- 必须配 warmup，前 500~2000 步用小 lr 逐步升到目标
- **Facebook 2017 的 1-hour ImageNet** 就是这套（8192 batch 训 ResNet50 不掉精度）

#### 1.3 LAMB (You et al., ICLR 2020)

**核心洞察**：大 batch 下不同层的 weight/gradient norm 比例失衡是精度下降的根本原因。

**算法**：
```
每层:
  r_t = Adam(g_t)                        # 常规 Adam 方向
  r_t += λ × θ                           # decoupled weight decay
  trust_ratio = ‖θ_layer‖ / ‖r_t‖        # per-layer 缩放
  θ ← θ - lr × trust_ratio × r_t
```

**物理意义**：
- ‖θ‖/‖r‖ 让每层的**相对更新幅度**保持一致
- 实际步长 = `lr × ‖θ‖`——不再受梯度大小影响
- BERT-Large 从 3 天训到 76 分钟不掉精度

**在 RecSys 里的用法**：只对 dense 部分用 LAMB，Sparse 保持 Adagrad。落地成本 = 换 optimizer 类。
**Caveat**：小 batch 下 LAMB 没优势，只在 batch ≥ 8K 时值得用。

### Tier 2：低成本 —— 训练策略调整

#### 2.1 Gradient Accumulation（反过来用）

- 常规：多 step 累积后更新一次（在小卡上训大 batch）
- **反过来**：把大 batch 切成 N 份小 batch 分别更新 —— 数据吞吐 = 大 batch，更新语义 = 小 batch

#### 2.2 Progressive Batch Size

- 训练前期用小 batch（探索 + 强正则）
- 训练后期用大 batch（精调 + 高效）
- Facebook 2018 "Don't Decay Learning Rate, Increase Batch Size" 的思路
- **对 day-by-day 训练特别友好**：每天开头小 batch 抓分布漂移，稳定后切大 batch

#### 2.3 SWA (Stochastic Weight Averaging)

- 训练最后阶段把多个 step 的 weight 做平均
- 相当于**权重空间做集成**，天然指向平坦极小值
- 落地成本 = 加几行代码
- **对 GAUC 这种稳定性敏感指标特别有效**

### Tier 3：中成本 —— 数据/样本层面

#### 3.1 Importance Sampling / Hard Negative Mining

**本质洞察**：小 batch 效果好，是因为每 step 的样本"信息量"更集中。反过来说，如果大 batch 里能筛出"高信息量样本"，效果也能追上。

- **Hard Negative Mining**：大 batch 里选 loss 最高的 top-k 参与更新
- **Focal Loss**：给难样本更大权重（pCVR 场景天然适合）
- **Loss-based sampling**：按 loss 加权采样

对 pCVR 特别有效：正样本本来就少，让每个正样本被用足。

#### 3.2 Uncertainty-based Reweighting (Kendall et al., CVPR 2018)

**核心洞察**：不同任务的固有噪声（aleatoric uncertainty）不同 → 不确定性高的任务，loss 权重应该小。

**数学**：假设任务 i 输出 `y_i ~ N(f_i(x), σ_i²)`，σ_i² 是可学习参数：

```
L = Σ_i [ L_i / (2σ_i²) + log σ_i ]
```

- `L_i / (2σ_i²)`：σ 大 → 权重降低（这个任务学不动，别硬学）
- `log σ_i`：正则项，防止 σ → ∞ 的 trivial 解

**实现**：
```python
log_var = nn.Parameter(torch.zeros(1))
total_loss = torch.exp(-log_var) * task_loss + log_var
```

**在 RecSys 里的用法**：
- CTR + CVR + 时长 + 完播多任务，用 Kendall 加权替代手工调 w1/w2/w3
- **单任务下推广到 per-sample**：让高不确定性样本自动降权 → 软的 focal loss
- pCVR 场景特别友好：正样本稀疏、方差大，自动降权避免过拟合噪声

**Caveat**：
1. 假设 σ 与 x 无关（同方差假设），有 Data-dependent Uncertainty 变体
2. 对 loss scale 敏感，混合任务时 log σ 收敛不稳
3. 需要 log_var clip 或初始化技巧防止 trivial 解

### Tier 4：中高成本 —— 模型层面

#### 4.1 SAM (Foret et al., ICLR 2021)

**核心洞察**：显式优化"平坦极小值"目标：

```
min_θ  max_{‖ε‖≤ρ}  L(θ + ε)
```

**算法**：
```
1. ε* = ρ × ∇L(θ) / ‖∇L(θ)‖    # 找 loss 最陡上升方向
2. g_sam = ∇L(θ + ε*)            # 在扰动点算梯度
3. θ ← θ - lr × g_sam
```

**代价**：每 step 两次 forward-backward，训练时间翻倍。
**收益**：CIFAR/ImageNet +1~2pt 精度，对数据不平衡、噪声标签、OOD 泛化收益尤大。ViT 训练几乎标配。

**变体**：
- **ESAM** (2022)：成本降到 1.3x
- **GSAM** (2022, Google)：Surrogate 目标更清晰
- **ASAM** (2021)：Adaptive SAM，per-parameter 缩放

**在 RecSys 里的用法**：
- 离线评估用的高精度模型可以用 SAM
- 生产训练用 ESAM/LookSAM 这种低成本变体
- **注意**：BN 与 SAM 有微妙冲突（两次 forward 的 BN stats 不一致），LN 无此问题——**对本项目适配性其实不错**

#### 4.2 显式正则

小 batch 的核心收益是"隐式正则"→ 加显式正则也是等价路径：
- Dropout（对 Mixer 系有效）
- Weight decay（本项目目前没启用）
- Label smoothing（BCE 版本：label 从 {0,1} → {ε, 1-ε}）
- Stochastic Depth（对 Mixer/Transformer 生效）

成本最低但也最"钝"，达不到小 batch 的效果上限。

---

## 九、针对本项目的具体建议

按投入产出比排序：

### 🎯 Tier 1（一定做，几乎零成本）：

1. **试 β1 = 0.9**（batch=500）—— 验证核心假设：低 β1 是否是 batch 敏感度被放大的主因
2. **加 warmup + sqrt scaling**：`batch=500, lr=3.25e-6, warmup 1000 step`
3. **加 SWA**：训练最后 20% step 做权重平均

### 🎯 Tier 2（值得做）：

4. **Focal Loss + hard negative mining**：pCVR 正样本稀疏场景天然适合
5. **Progressive batch size**：每天开头 batch=300 跑 N step，然后切 batch=500 —— 保留 day-by-day 分布适应红利
6. **Dense 换 LAMB**：Sparse 保持 Adagrad，只改 dense

### 🎯 Tier 3（有条件）：

7. **ESAM/LookSAM**：如果 GAUC 提升的绝对价值 > 1.3x 训练时间成本
8. **Kendall 加权扩展到 per-sample**：软 focal loss

### ❌ 不建议：

- **纯粹调 lr**：踩过 sum→mean 换算的坑就知道，这个方向的收益上限低
- **改 β2**：0.9999 是为本项目训练动力学定制的
- **改 Adagrad eps**：eps=1.0 是长尾保护的关键，动了长尾会崩

---

## 十、验证实验路径

以下实验按优先级排列，每组建议 3 seed 确认稳定性。

### 主对照（回答"是不是有效 lr 变了"）

- A0: `batch=500, lr=2.5e-6`（基线）
- A1: `batch=500, lr=3.25e-6`（sqrt scaling）
- A2: `batch=500, lr=4.17e-6`（linear scaling）
- A3: `batch=300, lr=2.5e-6`（当前观察到 +0.1%）

如果 A1 或 A2 追平 A3 → 现象主因是隐式 lr 变化。

### 变体对照（回答"是不是低 β1 放大了 batch 敏感度"）

- B1: `batch=500, β1=0.9`
- B2: `batch=300, β1=0.9`

如果 B1 追平 A3 → 证实"低 β1 放大 batch 敏感度"假设。
如果 B1 和 B2 差距 << A0 和 A3 差距 → 进一步证实。

### 数据组织对照（回答"是不是 day-by-day 适应"）

- C1: `batch=500, 全量样本 shuffle`（打破时间顺序）
- C2: `batch=300, 全量样本 shuffle`

如果 C1 和 C2 差距 << A0 和 A3 差距 → day-by-day 分布适应是重要贡献。

### 分层分析

拆分 GAUC 到子人群，看 batch=300 的收益集中在哪里：
- 新广告主 vs 老广告主
- 高频用户 vs 低频用户
- 新增当天样本 vs 历史样本

集中在"新广告主"或"新增当天样本"→ 支持"分布适应"归因。
集中在"低频用户"→ 支持"sparse embedding 更新粒度"归因。

---

## 十一、总结

小 batch 效果更好这个现象在 RecSys 里几乎是每个团队都遇到的老问题，但**教科书归因只解释了一半**。本项目里 GAUC +0.1% 的深层原因至少有四层叠加：

1. **有效 lr 抬升**：lr / batch 变大 1.67x，恰好补偿了原本偏保守的 lr
2. **Day-by-day 分布适应**：小 batch 每步更 local，对最新数据分布响应更快
3. **Optimizer 配置放大**：cAdam β1=1e-5 让梯度噪声不经动量过滤直接影响 update
4. **GAUC 对 within-user 排序的敏感性**：小 batch 让同用户样本更集中，pairwise 信号更强

BN 敏感性在本项目不适用（主要用 LN）；sparse embedding 更新粒度部分被 Adagrad 消解。

**核心 insight**：RecSys 生产训练里的定制 optimizer 配置（低 β1、高 β2、双优化器、大 eps）本质上是为了稳态在线学习设计的，但**代价是训练动力学被锁定、超参极难微调、对 batch_size 敏感度被显著放大**。这解释了为什么 batch 500→300 这么小的变化在本项目里能看到 GAUC +0.1% 的明显信号。

lr 难调不是错觉，而是 optimizer 内部机制（Adam 的 gradient normalization、weight decay 耦合、二阶矩收敛延迟）+ RecSys 特有机制（sparse embedding 时间戳、双优化器 lr 语义分裂）共同作用的结果。Sum→Mean 换算永远对不齐是常态而非异常。

想真正解决 batch_size 敏感度，正确方向不是"精细调 lr"，而是：加动量、加显式正则、换现代大 batch 优化器（LAMB）、上 SAM 系方法显式指向平坦极小值。

---

## 参考文献

- [1] Keskar et al. "On Large-Batch Training for Deep Learning: Generalization Gap and Sharp Minima." ICLR 2017. arXiv:1609.04836
- [2] Goyal et al. "Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour." 2017. arXiv:1706.02677
- [3] You et al. "Large Batch Optimization for Deep Learning: Training BERT in 76 minutes." ICLR 2020. arXiv:1904.00962 (LAMB)
- [4] Foret et al. "Sharpness-Aware Minimization for Efficiently Improving Generalization." ICLR 2021. arXiv:2010.01412 (SAM)
- [5] Kendall et al. "Multi-Task Learning Using Uncertainty to Weigh Losses for Scene Geometry and Semantics." CVPR 2018. arXiv:1705.07115
- [6] Smith et al. "Don't Decay the Learning Rate, Increase the Batch Size." ICLR 2018. arXiv:1711.00489
- [7] Izmailov et al. "Averaging Weights Leads to Wider Optima and Better Generalization." UAI 2018. (SWA)
- [8] Lin et al. "Focal Loss for Dense Object Detection." ICCV 2017. arXiv:1708.02002

<!-- en -->
# batch_size in Production RecSys Training: From Phenomenon to Optimizer Coupling

> **2026-08-17** · by guoliang

## Overview

A real observation: on a MixerFormer-based pCVR model, dropping batch_size from 500 to 300 improved offline GAUC by +0.1%. lr, epoch count, step count, and data organization (day-by-day time order) were all unchanged — only batch moved.

"Small batch trains better" is a phenomenon nearly every RecSys team has hit. Textbook answers usually mention implicit regularization, flat minima, BN sensitivity — but checking these against a real training config reveals the textbook answers only explain part of the story. This note decomposes attribution into four layers (optimization view, RecSys view, normalization view, metric view), then adds a fifth layer — **optimizer configuration coupling** — that explains why the phenomenon is amplified in production. Along the way we unpack why lr is so hard to tune (the classic sum→mean loss reparameterization pitfall), and close with a toolkit of solutions ranked by ROI.

**Core insight**: Textbook batch_size discussions assume standard Adam (β1=0.9), but production RecSys often uses low-β1, ultra-high-β2 custom optimizers for dense params + independent Adagrad for sparse embeddings. **This kind of configuration significantly amplifies batch_size sensitivity** — which is the underlying reason "batch_size wiggles → GAUC wiggles" in real training.

---

## 1. Phenomenon and Configuration

### Observation

| Variable | Config A (baseline) | Config B |
|---|---|---|
| batch_size | 500 | 300 |
| lr | 2.5e-6 (unchanged) | 2.5e-6 (unchanged) |
| epoch / step | same | same |
| Training | day-by-day time-ordered | same |
| Backbone | MixerFormer | same |
| Norm | mostly LN, few BN | same |
| Offline GAUC | baseline | **+0.1%** |

### Optimizer Configuration

```yaml
gpu_optimizer_type:
  dense: "cadam"
  sparse: "adagrad_sparse"

cadam:
  learning_rate: 2.5E-6
  beta_1: 1.0E-5              # ⚠️ very low
  beta_2: 0.9999              # ⚠️ very high
  epsilon: 1.0E-8
  fused: true
  optimizer_version: "v2"

adagrad_sparse:
  lr: 0.005
  eps: 1.0                    # ⚠️ very large
  initial_accumulator_value: 0.0
  foreach: true
```

Three ⚠️-marked params deviate from standard values — these are custom tunings the team accumulated from past incidents, and they are central to the attribution below.

---

## 2. Textbook Attribution: Optimization View

### 2.1 Gradient noise regularization & flat minima (Keskar et al., ICLR 2017)

Small-batch gradients are noisy estimates of the true gradient — this noise acts as implicit regularization:
- Large batch → sharp minima → sensitive to weight perturbation → poor generalization
- Small batch → flat minima → robust to distribution shift → better generalization

Keskar's Hessian eigenvalue analysis is the foundational reference.

### 2.2 Effective noise scale = lr / batch_size

**The most commonly missed coupling**: changing batch without adjusting lr is really "changing batch × implicit changing lr" simultaneously.

- batch 500→300 → effective noise scale × 1.67
- If original lr was conservative (our 2.5e-6 is), this implicit lift lands in a better spot

**Linear Scaling Rule** (Goyal et al., 2017): `batch × k → lr × k` (SGD)
**Square-Root Rule** (better for Adam): `batch × k → lr × √k`

Required control experiment: `batch=500, lr × 1.67`, check whether GAUC gap collapses.

### 2.3 Optimizer step count vs epoch count

Under same epochs, batch=1024 vs 4096 differ 4x in optimizer steps. Adam second-moment convergence and embedding sparse updates all depend on step count. **Same-step-count comparisons often narrow or reverse the gap** — but this project already controlled steps, so this doesn't apply.

---

## 3. RecSys-Specific Attribution

### 3.1 Sparse embedding update granularity

90%+ of RecSys model params live in embedding tables; each sample lights up only a few embeddings:

- **Large batch**: same item embedding hit by multiple users in one batch → "averaged" update → individual signal diluted
- **Small batch**: each update is more "personalized", direction retains more per-sample info
- **Long-tail items** (may appear only 1~2 times per batch): under large batch, their gradients get diluted with head items in optimizer momentum

### 3.2 Distribution shift adaptation in day-by-day training (key for this project)

Day-by-day incremental training = one significant distribution shift per day (new advertisers, new traffic, new creatives):

- Large batch per step covers wider sample distribution → gradient is "cross-distribution average" → slower response to new/shifted signals
- Small batch per step is more local → finer adaptation to most recent data distribution
- Offline GAUC is usually evaluated on next-day / recent-day data → whoever adapts faster wins

**Validation**: switch to shuffled (non-day-ordered) training and see if small-batch advantage shrinks.

### 3.3 Head dominance vs tail coverage

User-item distribution is power-law:
- Large batch has higher head-item ratio → gradient dominated by head patterns
- Small batch more local → tail patterns get to dominate some steps
- GAUC penalizes within-user ranking errors, not overall CTR bias → head dominance actually hurts GAUC

---

## 4. Normalization View: BN vs LN

### Why BN is batch-sensitive

BN training stats: `μ_B = (1/B) Σ x_i`, `σ²_B = (1/B) Σ (x_i - μ_B)²`

1. **Stat variance ∝ 1/B**: small batch → noisy μ_B, σ²_B → normalizing baseline wobbles → implicit regularization
2. **Train/eval mismatch**: training uses batch stats, inference uses running stats (EMA) → small batch → slow, noisy running stats
3. **Sample correlation pollutes stats**: non-iid batches (e.g., same user's multiple impressions packed together) → biased stats

### LN is completely insensitive

LN: `μ_i = (1/D) Σ_d x_{i,d}` — normalized per-sample, only depends on feature dim D. batch=1 and batch=10000 give identical results.

### Quantifying BN diff at batch 500 vs 300

BN stat variance ratio: `σ²_B(300) / σ²_B(500) = 500/300 ≈ 1.67`, std ratio ≈ 1.29.

For a layer with true μ=0, σ=1:

| batch | μ_B jitter (±1σ) | σ_B jitter (±1σ) |
|---|---|---|
| 500 | ±0.045 | ±0.032 |
| 300 | ±0.058 | ±0.041 |

**Absolute diff ~0.013** — much smaller than lr × 1.67 update-direction changes, well below dropout 0.1 noise.

### Real BN sensitivity threshold

| batch_size | BN status |
|---|---|
| ≥ 32 | stats stable, BN fine |
| 16 ~ 32 | slight degradation |
| 8 ~ 16 | significant degradation, prefer GroupNorm/LN |
| < 8 | BN unusable |

**500 vs 300 are both in BN comfort zone. This project uses LN anyway, so this attribution is essentially ruled out.**

---

## 5. Metric View: Why GAUC Is Sensitive to batch_size

GAUC = per-user AUC weighted average, measures "within a user, is positive pCVR > negative pCVR".

- BCE loss is **pointwise** — its gradient doesn't directly optimize within-user ranking
- Under small batch, **multiple samples from the same user are more likely to appear in nearby steps** (if shuffling is sample-level) → model's memory of "what this user likes" updates more coherently
- Under large batch, same-user samples get diluted among many other users → within-user pairwise signal gets "washed out"

**GAUC's sensitivity to batch_size > AUC's** — a RecSys-specific phenomenon; CV/NLP batch_size discussions focus on overall loss/accuracy and miss this.

---

## 6. Deep Attribution: How Optimizer Config Amplifies batch_size Sensitivity

**This section is where this note goes beyond textbook batch_size discussions.**

### 6.1 cAdam β1 = 1e-5: no momentum → batch noise passes through directly

Standard Adam β1 = 0.9, first-moment window ≈ 10 steps: `m_t = 0.9 m_{t-1} + 0.1 g_t`. This momentum absorbs 60~70% of gradient noise.

This project β1 = 1e-5, window ≈ 1 step: `m_t ≈ g_t`. Essentially reduces Adam to RMSProp (only second-moment adaptivity).

Consequences:
- Each step's update direction is fully determined by **current batch**
- **Small-batch high gradient noise is 100% transmitted to update jitter** (under β1=0.9 most would be filtered)
- This is why batch 500→300 gives a visible +0.1% GAUC signal

**In other words: low β1 optimizer configuration amplifies batch_size sensitivity.**

### 6.2 β2 = 0.9999: stable scale but training dynamics locked in

Standard β2 = 0.999, window 1000 steps. This project 0.9999 → window 10000 steps.

Combined with β1 ≈ 0, the design signal is:
- **First moment**: no smoothing, trust only current gradient direction
- **Second moment**: heavy smoothing, use long-term stable scale for normalization

Clear design intent: **custom-tailored for steady-state online training**. Don't smooth direction (stay responsive to current samples), but heavily smooth scale (numerical stability). Very reasonable, matches day-by-day incremental training semantics.

Cost: `√v̂` moves extremely slowly, so lr's effective magnitude gets locked shortly after training starts. **Trying to align effect by tuning lr is nearly impossible** — you tune lr, first 10000 steps `v̂` hasn't stabilized, trend invisible; by the time it stabilizes, training is done.

### 6.3 Dense/Sparse dual optimizer: lr semantics split

Dense (cAdam) and Sparse (Adagrad) follow two completely different update logics:

| Aspect | cAdam (Dense) | Adagrad (Sparse) |
|---|---|---|
| Step semantics | Adaptive second moment, `lr × sign(g)` upper bound | Cumulative squared grad, `lr / √G` decay |
| Update frequency | All params every step | Only when lit up |
| Long-tail behavior | No special protection | Natural adaptivity (small G → large step) |
| lr magnitude | 2.5e-6 | 0.005 |

Benefit: this combination directly solves the classic "sparse embedding + Adam state imbalance" trap.
Cost: **tuning dense lr while sparse follows its own rhythm** — the two sides are coupled but you can't tune globally.

### 6.4 Adagrad eps=1.0: natural warmup + long-tail protection

Adagrad update: `θ ← θ - lr × g / (√G + ε)`

Standard eps = 1e-10 (just prevents divide-by-zero). This project eps = **1.0**, 10 orders of magnitude larger than standard.

Behavior:
- Early training: G small → update ≈ `lr × g / 1.0 = 0.005 g` → **degenerates to SGD initially**
- Once G accumulates to ≈ 1.0 → Adagrad adaptivity kicks in
- Effectively a **natural warmup** (early steps are stable)
- Especially friendly to **long-tail embeddings** (G stays small) — maintains larger effective step

Combined with `initial_accumulator_value = 0.0`: startup update enters smoothly, avoids large-g-vs-small-G oscillation.

### Summary: this optimizer config is a steady-state dynamic tailored for day-by-day incremental training

- Low β1 → maintains responsiveness to current samples
- High β2 → stabilizes long-term scale
- Adagrad eps=1.0 → long-tail protection + natural warmup
- Dual optimizer → resolves sparse gradient imbalance

**The cost: extreme difficulty in fine-tuning**. Any hyperparameter change (batch, lr, loss scale, precision) immediately reflects in updates, no momentum buffer. **This is exactly the technical reason "batch 500→300 → GAUC +0.1%" is such a clean signal in this project.**

---

## 7. Why lr Is So Hard to Tune: The sum→mean Reparameterization Trap

A recurring pain: after switching loss from sum to mean, no amount of optimizer tuning aligns effects. Theoretically sum and mean differ by B, so `lr × B` should be equivalent, but reality is far more complex.

### Math

```
loss_sum  = Σ_{i=1..B} l_i
loss_mean = (1/B) Σ l_i = loss_sum / B
∇(loss_mean) = ∇(loss_sum) / B
```

Gradients differ by B.

### Under SGD

`lr_mean = lr_sum × B` — clean.

### Under Adam: lr barely needs to change (but not fully equivalent)

Adam update: `θ ← θ - lr × m̂ / (√v̂ + ε)`

- `m̂` scales × 1/B
- `√v̂` scales × 1/B
- `m̂ / √v̂` **scale unchanged**
- So Adam sum→mean theoretically doesn't need lr change

**But only "barely"**, because:

1. **ε doesn't scale**: `√v̂` going from 10 to 0.01 changes ε=1e-8's relative magnitude → some params' updates get ε-suppressed
2. **v̂ needs time to converge**: first few hundred steps have inaccurate `v̂`, sum vs mean accumulate `v` at different rates → large early-update difference
3. **Weight Decay effective strength drifts (the hardest AdamW pitfall to spot)**:
   - sum + wd=0.01: per-step wd update = `-lr × 0.01 × θ`, matches gradient magnitude
   - After switching to mean, gradient smaller by B, wd update unchanged → **wd effective strength B× stronger**
   - Fix: `wd × (1/B)`

### RecSys-specific: sparse embedding Adam state timestamps

- Embedding Adam state only advances t when the vector is touched
- After sum→mean, some embeddings' `m̂ / √v̂` ratios jump discontinuously
- Nearly unfixable — only Adagrad/FTRL sidesteps it (this project already does)

### Gradient clipping thresholds break

`clip_grad_norm = 1.0` rarely triggers under sum, triggers frequently under mean. Fix: scale threshold by 1/B.

### Full sum→mean alignment checklist

```
sum → mean:
  lr:            Adam barely changes, SGD × B
  wd:            divide by B (or use absolute decoupled form)
  ε:             divide by B (optional, small effect)
  clip threshold: divide by B
  warmup steps:  increase (give v̂ time to converge)
```

**Bottom line: even after all these adjustments, sparse embedding makes full alignment impossible. Industry consensus: after sum/mean switch, aim only for "similar convergence", don't expect full alignment.**

---

## 8. Solution Toolkit: Ranked by ROI

To resolve "want small-batch generalization + large-batch efficiency", the industry has a systematic set of methods. Sorted by cost:

### Tier 1: Zero cost — tune optimizer to close the gap

#### 1.1 Restore momentum (most direct for this project)

- Try β1 ∈ {0.5, 0.7, 0.9}
- Validates hypothesis: "β1≈0 amplifies batch sensitivity"
- If β1=0.9 + batch=500 ≥ β1=1e-5 + batch=300 → switch, 67% training efficiency gain

#### 1.2 Linear/Sqrt Scaling Rule + Warmup

- batch × k → lr × k (SGD) or lr × √k (Adam)
- Must pair with warmup, first 500~2000 steps ramp up
- **Facebook 2017 1-hour ImageNet** used this (8192 batch on ResNet50 with no accuracy loss)

#### 1.3 LAMB (You et al., ICLR 2020)

**Core insight**: under large batch, imbalance in weight/gradient norm ratios across layers is the root cause of accuracy loss.

**Algorithm**:
```
For each layer:
  r_t = Adam(g_t)                        # standard Adam direction
  r_t += λ × θ                           # decoupled weight decay
  trust_ratio = ‖θ_layer‖ / ‖r_t‖        # per-layer scaling
  θ ← θ - lr × trust_ratio × r_t
```

**Physical meaning**:
- ‖θ‖/‖r‖ keeps **relative update magnitude** consistent across layers
- Actual step = `lr × ‖θ‖` — no longer affected by gradient scale
- BERT-Large: 3 days → 76 minutes with no accuracy loss

**In RecSys**: use LAMB only on dense; keep sparse on Adagrad. Landing cost = swap optimizer class.
**Caveat**: LAMB has no edge at small batch, only worth it at batch ≥ 8K.

### Tier 2: Low cost — training strategy

#### 2.1 Gradient Accumulation (inverse use)

- Standard: accumulate multiple steps then update once (train large batch on small GPU)
- **Inverse**: split large batch into N small-batch updates — throughput = large batch, semantics = small batch

#### 2.2 Progressive Batch Size

- Early: small batch (exploration + strong regularization)
- Late: large batch (fine-tune + efficient)
- Facebook 2018 "Don't Decay the Learning Rate, Increase the Batch Size" thesis
- **Especially fit for day-by-day training**: small batch at day start (catch distribution shift), switch to large after stabilizing

#### 2.3 SWA (Stochastic Weight Averaging)

- Average weights across last-N training steps
- Equivalent to **ensembling in weight space**, natural flat-minima seeker
- Landing cost = a few lines of code
- **Especially effective for GAUC-like stability-sensitive metrics**

### Tier 3: Medium cost — data/sample level

#### 3.1 Importance Sampling / Hard Negative Mining

**Core insight**: small batch works because each step's samples are "information-dense". Reverse: if large batches can filter high-info samples, they can catch up.

- **Hard Negative Mining**: pick top-k loss samples for update
- **Focal Loss**: weight hard samples higher (natural fit for pCVR)
- **Loss-based sampling**: sample-probability proportional to loss

Especially effective for pCVR: positives are already sparse, make each one count.

#### 3.2 Uncertainty-based Reweighting (Kendall et al., CVPR 2018)

**Core insight**: different tasks have different intrinsic noise (aleatoric uncertainty) → high-uncertainty tasks should get lower loss weight.

**Math**: assume task i output `y_i ~ N(f_i(x), σ_i²)`, with σ_i² a learnable parameter:

```
L = Σ_i [ L_i / (2σ_i²) + log σ_i ]
```

- `L_i / (2σ_i²)`: large σ → low weight (this task is unlearnable, don't force it)
- `log σ_i`: regularizer, prevents trivial σ → ∞ solution

**Implementation**:
```python
log_var = nn.Parameter(torch.zeros(1))
total_loss = torch.exp(-log_var) * task_loss + log_var
```

**In RecSys**:
- Multi-task (CTR + CVR + duration + completion) — replace hand-tuned weights with Kendall
- **Single-task extension to per-sample**: down-weight high-uncertainty samples → soft focal loss
- Great for pCVR: sparse positives with high variance auto-downweighted to avoid overfitting noise

**Caveats**:
1. Assumes σ independent of x (homoscedastic); Data-dependent Uncertainty variants exist
2. Sensitive to loss scale, log σ unstable under mixed-scale tasks
3. Needs log_var clip or init tricks to avoid trivial solutions

### Tier 4: Medium-high cost — model level

#### 4.1 SAM (Foret et al., ICLR 2021)

**Core insight**: explicitly optimize for flat minima:

```
min_θ  max_{‖ε‖≤ρ}  L(θ + ε)
```

**Algorithm**:
```
1. ε* = ρ × ∇L(θ) / ‖∇L(θ)‖    # find steepest ascent direction
2. g_sam = ∇L(θ + ε*)            # compute gradient at perturbed point
3. θ ← θ - lr × g_sam
```

**Cost**: 2× forward-backward per step, doubles training time.
**Benefit**: CIFAR/ImageNet +1~2pt accuracy; especially strong for imbalanced data, label noise, OOD generalization. Nearly standard for ViT training.

**Variants**:
- **ESAM** (2022): cost down to 1.3×
- **GSAM** (2022, Google): clearer surrogate objective
- **ASAM** (2021): per-parameter adaptive perturbation

**In RecSys**:
- High-precision offline eval models can use SAM
- Production: use ESAM/LookSAM variants
- **Note**: BN and SAM have subtle conflicts (BN stats differ across two forwards), LN doesn't — **actually fits this project well**

#### 4.2 Explicit regularization

If small-batch's core benefit is implicit regularization → adding explicit regularization is equivalent:
- Dropout (effective for Mixer variants)
- Weight decay (not currently enabled in this project)
- Label smoothing (BCE version: label {0,1} → {ε, 1-ε})
- Stochastic Depth (effective for Mixer/Transformer)

Cheapest but bluntest; ceiling below small-batch effect.

---

## 9. Concrete Recommendations for This Project

Ranked by ROI:

### 🎯 Tier 1 (must do, essentially zero cost):

1. **Try β1 = 0.9** (batch=500) — validate the core hypothesis: is low β1 the main driver of amplified batch sensitivity?
2. **Warmup + sqrt scaling**: `batch=500, lr=3.25e-6, warmup 1000 steps`
3. **Add SWA**: average weights over last 20% of training steps

### 🎯 Tier 2 (worth doing):

4. **Focal Loss + hard negative mining**: natural fit for sparse-positive pCVR
5. **Progressive batch size**: batch=300 for first N steps each day, then switch to batch=500 — preserve day-by-day adaptation while gaining efficiency
6. **LAMB for dense**: keep sparse on Adagrad

### 🎯 Tier 3 (conditional):

7. **ESAM/LookSAM**: if GAUC gain is worth 1.3× training time
8. **Extend Kendall reweighting to per-sample**: soft focal loss

### ❌ Not recommended:

- **Tuning lr alone**: sum→mean experience taught us the ceiling here is low
- **Change β2**: 0.9999 is custom-tailored to this project's training dynamics
- **Change Adagrad eps**: eps=1.0 is critical for long-tail protection; changing it will crash the tail

---

## 10. Verification Experiment Path

Ordered by priority; run each with 3 seeds for stability check.

### Main control (answers "is it effective lr change")

- A0: `batch=500, lr=2.5e-6` (baseline)
- A1: `batch=500, lr=3.25e-6` (sqrt scaling)
- A2: `batch=500, lr=4.17e-6` (linear scaling)
- A3: `batch=300, lr=2.5e-6` (observed +0.1%)

If A1 or A2 matches A3 → primary cause is implicit lr change.

### Variant control (answers "does low β1 amplify batch sensitivity")

- B1: `batch=500, β1=0.9`
- B2: `batch=300, β1=0.9`

If B1 matches A3 → confirms low-β1 amplification.
If B1 vs B2 gap << A0 vs A3 gap → further confirms.

### Data organization control (answers "is it day-by-day adaptation")

- C1: `batch=500, fully shuffled` (break time order)
- C2: `batch=300, fully shuffled`

If C1 vs C2 gap << A0 vs A3 → day-by-day adaptation contributes significantly.

### Stratified analysis

Split GAUC across sub-populations to locate where batch=300 wins:
- New advertisers vs old
- High-frequency users vs low-frequency
- Same-day new samples vs historical

Concentrated in "new advertisers" or "same-day new" → supports distribution-adaptation attribution.
Concentrated in "low-frequency users" → supports sparse-embedding-granularity attribution.

---

## 11. Summary

Small batch trains better is an old problem every RecSys team hits, but **textbook attribution only explains half of it**. The +0.1% GAUC in this project layers at least four factors:

1. **Effective lr lift**: lr/batch × 1.67, coincidentally compensates for a conservatively-set lr
2. **Day-by-day distribution adaptation**: small batch more local per step, responds faster to latest data
3. **Optimizer config amplification**: cAdam β1=1e-5 transmits gradient noise to updates without momentum filtering
4. **GAUC sensitivity to within-user ranking**: small batch keeps same-user samples closer together, stronger pairwise signal

BN sensitivity doesn't apply (this project uses LN); sparse embedding granularity is partially mitigated by Adagrad.

**Core insight**: production RecSys custom optimizer configs (low β1, high β2, dual optimizer, large eps) are engineered for steady-state online learning, but **the price is locked training dynamics, difficult hyperparameter fine-tuning, and significantly amplified batch_size sensitivity**. This explains why such a small change as batch 500→300 gives a clean GAUC +0.1% signal in this project.

lr is genuinely hard to tune — not an illusion — because of optimizer internals (Adam's gradient normalization, weight decay coupling, second-moment convergence delay) combined with RecSys-specific mechanisms (sparse embedding timestamps, dual optimizer lr semantics split). Sum→mean never fully aligning is the norm, not the exception.

The right direction to solve batch_size sensitivity isn't "finer lr tuning" — it's adding momentum, adding explicit regularization, switching to modern large-batch optimizers (LAMB), or applying SAM-family methods that explicitly seek flat minima.

---

## References

- [1] Keskar et al. "On Large-Batch Training for Deep Learning: Generalization Gap and Sharp Minima." ICLR 2017. arXiv:1609.04836
- [2] Goyal et al. "Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour." 2017. arXiv:1706.02677
- [3] You et al. "Large Batch Optimization for Deep Learning: Training BERT in 76 minutes." ICLR 2020. arXiv:1904.00962 (LAMB)
- [4] Foret et al. "Sharpness-Aware Minimization for Efficiently Improving Generalization." ICLR 2021. arXiv:2010.01412 (SAM)
- [5] Kendall et al. "Multi-Task Learning Using Uncertainty to Weigh Losses for Scene Geometry and Semantics." CVPR 2018. arXiv:1705.07115
- [6] Smith et al. "Don't Decay the Learning Rate, Increase the Batch Size." ICLR 2018. arXiv:1711.00489
- [7] Izmailov et al. "Averaging Weights Leads to Wider Optima and Better Generalization." UAI 2018. (SWA)
- [8] Lin et al. "Focal Loss for Dense Object Detection." ICCV 2017. arXiv:1708.02002
