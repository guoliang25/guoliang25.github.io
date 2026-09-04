<!-- zh -->
# 梯度拆分 + 梯度聚合：Burn→Online 切换抖动的两把钥匙

> **2026-09-04** · by guoliang

## 概述

一个上线场景里反复遇到的现象：Burn 阶段（大 batch 冷启/周期重训）离线 AUC 漂亮，checkpoint 完整继承给 Online（小 batch 实时训练），切换的瞬间——**权重完全接得上，pCTR 却跳一下、AUC 先掉后爬，恢复期几小时**。

问题不在权重，也不在样本，而在**optimizer 内部的历史状态**。这篇文章把这个"看不见的断层"拆开讲清楚，并引出对应的两个孪生工具：**梯度拆分（gradient split）**和**梯度聚合（gradient aggregation）**。

**核心 insight**：Burn 和 Online 的对齐不是一个维度的事——**梯度拆分对齐"更新步数"，梯度聚合对齐"每步方差"，两个维度都要修好，optimizer state 的演化路径才算真正打通**。这是一对孪生手段，缺一不可。

本文与之前 [batch_size 深度分析](recsys-batch-size-gauc.md) 一脉相承：都是围绕"自适应优化器如何被 batch_size 相关的量所影响"这个主题。上一篇讲的是单阶段训练里 batch_size 敏感度被优化器配置放大；本文讲的是双阶段训练里 batch_size 差异如何在 optimizer state 上留下断层，以及如何修补。

---

## 一、场景与现象

### Burn 和 Online 差的不只是 batch size

推荐模型上线走两个阶段：

- **Burn 阶段**：冷启或周期性重训，多机多卡 + 大 batch，一步吃掉海量样本，把模型训到可上线状态
- **Online 阶段**：接过 Burn 的 checkpoint，切换到小 batch 实时训练，跟随线上分布

第一直觉是"大 batch vs 小 batch"，但真正的差异更深。假设 Burn 的一次大 batch 恰好装得下 N 个 Online 小 batch 的样本量：

| 训练方式 | 同样 N 个小 batch 数据 | Optimizer 感受到的更新次数 |
|---|---|---|
| Burn 大 batch 一次 apply | 合成 1 次大梯度 | **1 次** |
| Online 小 batch 连续训练 | 逐个更新 | **N 次** |

模型看过的样本量一样，但 optimizer 走的步数一个是 1、一个是 N。切换到 Online 时，optimizer 会突然从"大步低频"节奏切到"小步高频"节奏——**权重可以无缝迁移，但历史累加的二阶量、动量、稀疏化状态这些"记忆"未必能无缝迁移**。

### 两个对不上的维度

即便强行把 Burn 也拆成 N 步 apply(G/N)，你会撞上第二个问题：**每一步的 G/N 是"N 个梯度求平均"的低方差版本，而真实 Online 的每一步 g_i 是单 batch 的高方差版本**。同样是 N 步，每步的"梯度尺度"完全不是一回事。

所以 Burn↔Online 的断层其实有**两个维度**：

- **更新次数不齐**：Burn 1 次 vs Online N 次
- **每步方差不齐**：Burn 每步方差 Var(g)/N vs Online 每步方差 Var(g)

两个维度都对不上，optimizer 状态就走不上同一条演化路径。

要理解为什么这两个维度重要，得先打开 optimizer 内部看一眼。

---

## 二、打开自适应优化器：以 FTRL 的 n 和 z 为例

自适应优化器（FTRL、FTML、Adam）之所以能给每个参数一个个性化学习率，是因为它们**内部维护了梯度的历史信息**。

FTRL 每个参数各自维护两个累加器：

| 累加器 | 累加的是什么 | 作用 | 特点 |
|---|---|---|---|
| **n** | Σ g² （梯度的**平方**） | 决定**学习率大小** | 只增不减、**无衰减** |
| **z** | ≈ Σ g （梯度**本身**） | 决定**权重方向** | 正负会抵消 |

一个直观类比：

- **n 是里程表**——只增不减，记录"这台车总共开了多远"
- **z 是油门累积**——踩下、抬起、再踩下，反映"想往哪个方向走、多用力"
- **权重更新幅度 ≈ f(油门, 里程表)**——新车（n 小）轻踩就飙起来，老车（n 大）同样的油门只是慢慢挪

### FTRL 单步更新流程

假设第 t 步，参数 w_t，累加器 n_{t-1} 和 z_{t-1}，新梯度 g_t：

1. **算学习率变化量**：`σ_t = (√n_t - √n_{t-1}) / α`
2. **更新方向累加器**：`z_t = z_{t-1} + g_t - σ_t · w_t`
3. **更新尺度累加器**：`n_t = n_{t-1} + g_t²`
4. **恢复权重**：

```
w_{t+1} = -(z_t - sign(z_t)·λ₁) / [(β + √n_t)/α + λ₂]
          ─────────────┬──────    ────────┬────────
              来自 z（方向）              来自 n（缩放）
```

**n 在整个流程里的角色可以用一句话概括**：它不参与"算梯度"，但它参与"决定梯度如何转化为权重更新"——**n 是那个"音量旋钮"，历史越长，旋钮拧得越紧**。

### 用数值感受 n 的记忆效应

某参数 w_t = 0.5、z_{t-1} = 2.0，来了同一个新梯度 g_t = 0.3。对比两种情况（α = 0.1, β = 1.0, λ₁ = 0.01）：

| 场景 | n_{t-1} | 更新后 w_{t+1} |
|---|---|---|
| "老手"参数 | 100 | **-0.021**（变化温和） |
| "新兵"参数 | 0.1 | **-0.118**（变化剧烈） |

**同样的输入梯度，n 不同，参数更新幅度差 5 倍以上**。这就是 n 的记忆效应——它给每个参数一个个性化的学习率。

### 关键结论：n（和 Adam 的 v）才是断层的震中

- **FTRL 的 n**：纯累加、**无衰减**，一旦偏差就**永久留存**
- **FTML 的 n**：同上
- **Adam 的 v**（二阶动量，EMA 衰减）：能自愈，但 β₂=0.999 时半衰期 ~693 步，完全恢复约 3000 步
- **z、m 这类方向量**：几十步就自然刷新，不是关键问题

**所以"Burn→Online 对齐"的核心，本质是对齐 n（FTRL/FTML）和 v（Adam）——这些"梯度尺度类"的二阶量。** 后面讲的所有事，都围绕这个中心。

---

## 三、修补工具一：梯度拆分——对齐"更新次数"

### 定义

思路很朴素——**Burn 的一次 apply(G) 拆成 N 次 apply(G/N)**：

```
原本：  apply(G)                              （1 次大更新）
改为：  apply(G/N), apply(G/N), ..., apply(G/N)   （N 次小更新）
```

其中 G 是当前大 batch 聚合出的梯度，N = gradient_split。梯度总量不变、样本不变、前向不变，**只让 optimizer 状态多推进 N 步**。

### 三个容易混淆的边界

- **不是梯度累积**：累积是"多次 backward 合成一次 apply"，方向相反
- **不是精确重放 Online**：真实 Online 每步的 g_i 各不相同，拆分用的是 **G/N 重复 N 次**
- **只动 optimizer，不动前向**：改变的只是"梯度被 optimizer 消费的方式"

### 为什么对 FTRL 是必需的

对比 FTRL 中 n 的增量：

| | Burn 不拆分（1 次 apply G） | Burn 拆分（N 次 apply G/N） |
|---|---|---|
| n 的单步增量 | +G² | +(G/N)² |
| N 次累积 | G² | **N·(G/N)² = G²/N** |
| 有效学习率 α/(β+√n) | √n 涨得快 → lr 衰减快 | √n 涨得慢 N 倍 → lr 衰减慢 |
| L1 稀疏化判断 | 只判 1 次 |z| < λ₁ | 判 N 次，中间可能提前触发置零 |

拆分之后 n 慢了 N 倍。但由于 **n 不可自愈**，只要 Burn 不做拆分，n 就永远偏小 → Online 接手后所有 feature 的有效学习率被系统性拉高 → 切换后剧烈抖动。

### Adam 的情况

对比 Adam 中 v 的增量：

- 不拆分：Δv = (1-β₂)·G²
- 拆分 N 次（等比求和后）：Δv = (G²/N²)·(1-β₂^N)

当 β₂=0.999、N=32 时，1-β₂^N ≈ 0.0315，**拆分后 Δv 远小于不拆分**。虽然 Adam 的 v 带 EMA 衰减、理论上可以自愈，但 β₂ 通常 0.999，恢复期常需 ~3000 步——这段时间直接影响 Online 初期的分数校准和学习速度。

**结论**：对 FTRL/FTML 拆分是必需的；对 Adam 拆分能显著缩短恢复期。

---

## 四、但只做拆分还不够——隐蔽的方差差距

这里是全文最关键的一个转折。

假设 N 个小 batch 梯度 g_1...g_N 独立同分布，均值 μ、方差 Var(g)。看两侧每步 apply 用的梯度：

- **Burn 拆分**每步用 G/N（N 个梯度的**平均**）：方差 = Var(g)/N
- **真实 Online** 每步用 g_i（单 batch）：方差 = Var(g)

每步 g² 的期望 = Var + μ²，两侧 N 步累积的 n 增量：

| | 每步 g² 期望 | N 步累积 |
|---|---|---|
| Burn 拆分 | Var(g)/N + μ² | **Var(g) + N·μ²** |
| 真实 Online | Var(g) + μ² | **N·Var(g) + N·μ²** |
| 差值 | — | **(N-1)·Var(g)** |

**即便做了拆分，Burn 累积的 n 仍然系统性小于 Online**，差在方差项 (N-1)·Var(g) 上。

**在推荐场景**，梯度方差往往远大于均值平方（稀疏 embedding、长尾样本、类别不均衡都放大这一点）。此时 Var(g) ≫ μ²，两侧 n 累积比值 ≈ N——

> **拆分做完，n 累积仍然比 Online 少约 N 倍。**

这就是"只拆分不够"的证据。

---

## 五、修补工具二：梯度聚合——对齐"每步方差"

### 思路

既然让 Burn 变得更"噪声"不现实（那要重跑 N 次 forward），就反过来：**让 Online 也聚合 M 个 batch 求平均、再 apply 一次**。

聚合后 Online 每步方差 = Var(g)/M。

### 对齐单元的核心概念

**关键点**：只要在 Burn 和 Online 之间选定同一个**"对齐单元"** = M 个 Online batch 的平均，让两侧每次 apply 都是这个单元——

- **Burn 侧**：把大 batch 视为若干个"对齐单元"，拆分成对应次数 apply
- **Online 侧**：每 M 个原始 batch 聚合成 1 次 apply

**两侧每次进入 optimizer 的梯度，统计分布完全一致。**

---

## 六、用数值把一切串起来

场景设定：

- Online batch = 1024，Burn batch = 32768（N = 32）
- 单个 Online batch 梯度：μ = 0.1，Var(g) = 1.0（推荐场景常见"方差远大于均值平方"）
- 优化器 FTRL，看某参数 n 的累积

对比同样处理 32 个 Online batch 数据量后，四种做法下 n 增加多少：

| 做法 | apply 步数 | 每步 g² 期望 | n 累积 |
|---|---|---|---|
| A: Burn 不拆分 | 1 | ≈0.041 | **0.041** |
| B: Burn 只拆分 | 32 | ≈0.00004 | **0.0013** |
| C: 真实 Online（目标） | 32 | ≈1.01 | **32.3** |
| D: Burn 拆分 + Online 聚合到统一对齐单元 | 匹配 | 匹配 | **匹配** |

- **A vs C 差 ~800 倍** → 步数、方差都对不上
- **B vs C 差 ~24000 倍** → 只做拆分，方差反而拉得更开
- **D 是唯一能真正对齐的方案**

### 一张示意图收束

Burn batch = 32 × Online batch，选 M = 8：

```
真实 Online 32 步:  g₁ | g₂ | g₃ | ... | g₃₂     32 次 apply，每步方差 Var(g)

只拆分 Burn:        G/32 | G/32 | ... | G/32    32 次 apply，方差 Var(g)/32²
                    ↑ 步数对齐了，但方差差了 32 倍

拆分 + 聚合 (M=8):
  Burn 侧:  ḡ₁₋₈ | ḡ₉₋₁₆ | ḡ₁₇₋₂₄ | ḡ₂₅₋₃₂    4 次 apply，每步方差 Var(g)/8
  Online侧: ḡ₁₋₈ | ḡ₉₋₁₆ | ḡ₁₇₋₂₄ | ḡ₂₅₋₃₂    4 次 apply，每步方差 Var(g)/8
                    ↑ 步数、方差都对齐
```

**核心：两侧的每次 apply 都必须是"M 个真实小 batch 的平均"这个统一单元**，才能同时对齐步数和方差。

---

## 七、统一的心智模型

一句话总结：

> **梯度拆分 = Burn 向 Online 对齐"步数"；梯度聚合 = Online 向 Burn 对齐"方差"。**

对称的图景：

| 维度 | Burn 大 apply | 真实 Online | 修补手段 |
|---|---|---|---|
| **更新次数** | 1 步 | N 步 | **梯度拆分**：Burn 拆到 N 步 |
| **每步方差** | 极小（大 batch 平均） | Var(g) | **梯度聚合**：Online 每 M 步聚合成 1 步 |

两者的连接点是 M——**对齐单元的大小**。Burn 拆分的每一份，必须与 Online 聚合出的每一份，在统计意义上等价：

> **在 Burn 和 Online 之间画一条对齐单元线，两侧都以"M 个 Online batch 的平均"作为最小 apply 单元。Burn 侧多个单元拆开、Online 侧多个原始 batch 合成——目的都是让每次进入 optimizer 的梯度，服从同一个分布。**

### 各优化器的敏感度矩阵

| 状态量 | 优化器 | 关键性 | 衰减机制 | 自愈能力 | 拆分定位 | 聚合定位 |
|---|---|---|---|---|---|---|
| n（累计尺度） | FTRL | 关键 | 无衰减、纯累加 | **永不** | **必需** | **强烈建议** |
| n（累计尺度） | FTML | 关键 | 同 FTRL n | **永不** | **必需** | **强烈建议** |
| v（累计尺度） | Adam | 关键 | EMA (β₂=0.999) | 慢（~3000 步） | 值得做 | 值得做 |
| L1 稀疏化中间判断 | FTRL | 重要 | — | 依赖 n 和后续梯度 | 建议 | — |
| v 的 EMA 累积 | FTML | 次要 | EMA (β₂=0.999) | 可逐步恢复 | 建议 | — |
| z / σ_t 路径差异 | FTRL | 次级 | — | 受 n 影响 | — | — |
| m 的路径差异 | Adam | 非关键 | EMA (β₁=0.9) | 十几步即刷新 | — | — |

---

## 八、什么时候必须做，什么时候可以省

### 适用/不适用矩阵

| 场景 | 拆分 | 聚合 |
|---|---|---|
| Burn/Online batch 比很大 + FTRL/FTML | **必需** | **强烈建议** |
| Burn/Online batch 比很大 + Adam | 建议 | 建议（尤其在意校准） |
| Burn/Online batch 接近 | 可省 | 可省 |
| Burn 只导权重、不继承 optimizer state | 无意义 | 视情况 |
| plain SGD、无自适应状态 | 通常不需要 | 视噪声情况 |
| 业务对突发响应极敏感 | 正常做 | 谨慎（M 不宜大） |

### 聚合的代价必须明确

Online 从每 batch apply 变成 M batch 一次，**跟随线上分布的速度慢 M 倍**。所以工程实践里聚合更常见的用法是**切换窗口期策略**：

- 刚切换的头几分钟/几小时，用较大的 M 让 optimizer state 平稳过渡
- 状态稳定后，逐步把 M 降到 1，恢复 Online 的高频响应特性

### N 和 M 的选取原则

- 核心原则：**N × Online batch = Burn batch**，M 与 N 保持一致（同一个对齐单元）
- N 较大（如 > 32）时，Python 侧循环 apply 的 kernel launch 开销显著上升，优先走 C++ kernel 内拆分路径
- N 较小（如 2~4）时，Python 侧拆分开销可接受

---

## 九、工程实现路径

### 两条路径：Python 侧 vs Kernel 内

**Python 侧拆分**（v1 / 默认）：

- 通用路径，理论上支持所有常见优化器
- 在 apply_gradients 里做两件事：
  1. 梯度缩放：`grad = grad / gradient_split`（sparse 走 `IndexedSlices(values/N, indices, shape)`）
  2. 循环 apply：用 `tf.while_loop` 循环 N 次，每次调用对应 optimizer 的 apply_gradients
- 缺点：图节点和 kernel launch 开销随 N 增大

伪代码：
```python
def cond(gradient_i):
    return tf.less(gradient_i, self.gradient_split)

def body(gradient_i):
    # 循环 gradient_split 次，每次使用缩放后的梯度 grad/N 更新参数
    self.dense_optim.apply_gradients(...)
    self.sparse_optim[...].apply_gradients(...)
    return (gradient_i + 1,)

tf.while_loop(cond, body, [gradient_i])
```

**C++ kernel 内拆分**（v2 / 性能优化）：

- 仅支持部分 optimizer（如 ftml_dense、ftrl_sparse）
- 把 N 次拆分逻辑下沉到 optimizer kernel 内部
- 语义上仍是拆分更新，但只有一次 kernel launch

触发条件示例：
- `dense_optimizer_type == 'ftml'` 且 `version == 'v2'`，或开启 `filter_grad`
- `sparse_optimizer_type == 'ftrl_sparse'` 且 `version == 'v2'`

对这些 optimizer，构造时把 `grad_split_num = gradient_split` 写进 config 并打开对应标志位（`dense_split_update` / `sparse_optim_split_update[type] = True`），Python 层将不再做 `grad / N` 缩放和 `tf.while_loop`，底层 kernel 通过 `grad_split_num` 在 C++ 内部完成 N 次拆分。

### 不支持 kernel 内拆分时的兼容处理

对不支持 v2 kernel 内拆分的 optimizer，需要在配置里 pop 掉 `grad_split_num`，避免传给底层不认识的参数导致报错——统一回退到 Python 侧 `grad / N + while_loop`。

如果既没有 kernel 支持，也没有 Python 侧显式拆分，就**不能认为该 optimizer 已经正确执行了梯度拆分**。

---

## 十、常见工程坑

- **梯度尺度误配**：G 已经是 mean gradient 时再除以 N，**梯度尺度被压小两次**，会导致更新过小。上线前一定要先校准梯度尺度语义
- **训练耗时上涨**：Python 侧拆分会 N 倍增加图节点和 kernel launch，N 大时优先走 C++ kernel 内拆分
- **optimizer 覆盖不完整**：kernel 内拆分只支持部分 optimizer，其他 optimizer 需 Python 兜底或显式 pop 配置
- **与 lr schedule 交互**：step counter 推进 N 倍后，若 lr schedule 依赖 global step，实际学习率曲线可能与预期不符
- **状态偏差极难排查**：**表面 loss/AUC 通常正常，但 n / v 已经偏离**——梯度拆分/聚合的配套 monitoring 比它本身更重要

---

## 十一、如何验证你的模型需不需要

核心原则：**梯度拆分和聚合都不是为了提升 Burn 离线 AUC，而是为了减少 Burn→Online 切换时 optimizer state 不连续带来的恢复期**。所以不能只看最终指标或离线 AUC。

### 三组对照实验

| 实验组 | 做法 | 用途 |
|---|---|---|
| Online 小 batch baseline | Online batch 从同一 checkpoint 训 | 理想状态轨迹参考 |
| Burn 不拆分/不聚合 | 大 batch 直接 apply，切 Online | 观察偏差有多大 |
| Burn 拆分 + Online 聚合 | 完整对齐，切 Online | 验证是否更接近 baseline |

### 关键技术指标

| 指标 | 观察什么 | 异常信号 |
|---|---|---|
| **参数 change ratio** | ‖W_t - W_burn‖ / ‖W_burn‖，按参数组/特征层分层 | 切换后明显尖刺或系统性低于 Online baseline |
| **Optimizer state 分布** | FTRL n、Adam v、有效学习率 α/(β+√n) 的均值和分位数 | 不拆分组 n/v 明显偏大或偏小，有效学习率系统性偏移 |
| **Sparse 非零率** | embedding 参数激活比例 | 切换前后非零率突变（FTRL L1 稀疏化路径异常）|
| **分数校准** | pCTR/pCVR 均值、分桶 calibration ratio | 切换瞬间均值突变或分桶系统性偏移 |
| **业务指标** | 切换窗口期 AUC/GAUC、新样本学习速度 | 短期明显回撤或学习持续变慢 |

### 三级证据判断法

| 证据强度 | 现象 | 结论 |
|---|---|---|
| **强** | n 或 v 明显不一致 + 切换期 change ratio 分数校准异常 | 需要梯度拆分/聚合 |
| **中** | 离线 AUC 差异不明显，但切换期参数更新速度或 sparse 非零率异常 | 建议做实验验证 |
| **弱** | 只有最终指标轻微波动，无状态量和切换期证据 | 暂不能证明需要，优先补 state 观测 |

**简明判断**：如果 n 或 v 与 Online baseline 明显不一致、且切换窗口期分数校准异常，就是强证据——需要对齐。

---

## 十二、更本质的心智模型

回过头看，这些工具的存在其实揭示了一件更本源的事——

> **在自适应优化器主导的推荐系统里，"看过多少样本"不是唯一的训练量维度。"optimizer 感受到了多少次更新、每次更新的梯度尺度和方差是多少"是同等重要的另一个维度。**

理解了这一点，batch size、gradient split、gradient aggregation、learning rate schedule、warmup 就不再是各自独立的 trick——它们都在回答同一个问题：

**如何让 optimizer state 的演化路径匹配业务的训练节奏。**

Burn↔Online 的断层，只是这个问题在两阶段训练场景下的一个具体表现。**拆分修的是步数、聚合修的是方差，两个维度都修好，optimizer state 的路径才算真正打通。**

---

## 与相关话题的联系

- **[batch_size 深度分析](recsys-batch-size-gauc.md)**：单阶段训练里 batch_size 敏感度如何被优化器配置放大。同一主线的"上一集"——都在讲自适应优化器的二阶量如何被 batch 相关的信号影响。
- **learning rate schedule / warmup**：本质上也是"控制 optimizer state 演化路径"的手段，只是在时间轴上做调整、不在梯度形状上做调整。
- **Gradient Accumulation**：方向与梯度拆分**相反**——多次 backward 合成一次 apply。适用场景（想在小卡上训大 batch）也完全不同。

## 参考

- FTRL: McMahan et al. "Ad Click Prediction: a View from the Trenches." KDD 2013.
- FTML: Zheng & Kwok. "Follow the Moving Leader in Deep Learning." ICML 2017.
- Adam: Kingma & Ba. "Adam: A Method for Stochastic Optimization." ICLR 2015.

<!-- en -->
# Gradient Split + Gradient Aggregation: Two Keys to Burn→Online Switch Jitter

> **2026-09-04** · by guoliang

## Overview

A recurring on-launch phenomenon: Burn phase (large-batch cold-start / periodic retraining) shows a beautiful offline AUC, checkpoint transfers cleanly to Online (small-batch real-time training), and the moment you switch — **weights connect perfectly, yet pCTR jumps, AUC drops-then-recovers, and the recovery period lasts hours.**

The problem isn't in weights, isn't in samples — it's in **optimizer internal history state**. This note dissects that invisible fault line and introduces the two twin tools that patch it: **gradient split** and **gradient aggregation**.

**Core insight**: Aligning Burn↔Online isn't a one-dimensional job — **gradient split aligns "update step count", gradient aggregation aligns "per-step variance". Both dimensions must be fixed for optimizer state evolution paths to truly connect.** They are a twin pair, neither alone is enough.

This note is a direct sequel to the earlier [batch_size deep-dive](recsys-batch-size-gauc.md): both revolve around "how batch_size-related signals affect adaptive optimizers". The previous note covered how batch_size sensitivity is amplified by optimizer configuration in single-phase training; this one covers how batch_size differences leave a fault line in optimizer state across two-phase training, and how to fix it.

---

## 1. Scenario and Phenomenon

### Burn vs Online: not just batch size

Recommendation model launch typically has two phases:

- **Burn phase**: cold-start or periodic retraining, multi-node multi-GPU + large batch, one step consumes huge sample volumes, gets the model to a launchable state
- **Online phase**: inherits Burn's checkpoint, switches to small-batch real-time training, tracks live distribution

The first instinct is "large batch vs small batch", but the real difference runs deeper. Suppose Burn's one big batch equals N Online small batches in sample volume:

| Training mode | Same N small-batch data | Optimizer step count |
|---|---|---|
| Burn: one apply on big batch | Aggregated into 1 large gradient | **1 step** |
| Online: N consecutive small-batch updates | Individual updates | **N steps** |

Same samples seen by the model, but optimizer walks 1 step vs N steps. Switching to Online abruptly changes optimizer rhythm from "big step, low frequency" to "small step, high frequency" — **weights transfer seamlessly, but accumulated second-moment quantities, momentum, sparsification state do not**.

### Two misaligned dimensions

Even if you split Burn into N apply(G/N) steps, you hit a second problem: **each G/N is the low-variance version of "N gradients averaged", while real Online each step uses a single-batch high-variance g_i**. Same N steps, entirely different per-step gradient scales.

So the Burn↔Online fault line actually spans **two dimensions**:

- **Update count mismatch**: Burn 1 step vs Online N steps
- **Per-step variance mismatch**: Burn per-step variance Var(g)/N vs Online per-step variance Var(g)

Any single dimension misaligned means optimizer state walks the wrong path.

To understand why these two dimensions matter, we need to open up the optimizer.

---

## 2. Inside Adaptive Optimizers: FTRL's n and z

Adaptive optimizers (FTRL, FTML, Adam) give each parameter a personalized learning rate by **maintaining gradient history internally**.

FTRL maintains two accumulators per parameter:

| Accumulator | Accumulates | Role | Property |
|---|---|---|---|
| **n** | Σ g² (**squared**) | Determines **learning rate scale** | Monotonically increases, **no decay** |
| **z** | ≈ Σ g (raw) | Determines **weight direction** | Positive/negative cancel |

An intuitive analogy:

- **n is the odometer** — only increases, records "how far this car has driven"
- **z is cumulative throttle** — pressed and released, reflects "which direction, how hard"
- **Weight update magnitude ≈ f(throttle, odometer)** — new car (small n) revs up with a light touch, old car (large n) barely moves with the same push

### FTRL single-step update

At step t, parameter w_t, accumulators n_{t-1} and z_{t-1}, new gradient g_t:

1. **Compute learning rate change**: `σ_t = (√n_t - √n_{t-1}) / α`
2. **Update direction accumulator**: `z_t = z_{t-1} + g_t - σ_t · w_t`
3. **Update scale accumulator**: `n_t = n_{t-1} + g_t²`
4. **Restore weight**:

```
w_{t+1} = -(z_t - sign(z_t)·λ₁) / [(β + √n_t)/α + λ₂]
          ─────────────┬──────    ────────┬────────
             from z (direction)         from n (scaling)
```

**One-liner**: n doesn't participate in "computing the gradient" — it participates in "deciding how the gradient translates to weight update". **n is that "volume knob" — the longer the history, the tighter the knob**.

### Numerical intuition for n's memory effect

Some parameter w_t = 0.5, z_{t-1} = 2.0, same new gradient g_t = 0.3 (α = 0.1, β = 1.0, λ₁ = 0.01):

| Scenario | n_{t-1} | Updated w_{t+1} |
|---|---|---|
| "Veteran" param | 100 | **-0.021** (mild change) |
| "Rookie" param | 0.1 | **-0.118** (dramatic change) |

**Same input gradient, different n, weight update magnitude differs by 5×**. This is n's memory effect — it gives each parameter a personalized learning rate.

### Key conclusion: n (and Adam's v) is the epicenter

- **FTRL's n**: pure accumulation, **no decay**, bias is **permanent**
- **FTML's n**: same
- **Adam's v** (second moment, EMA-decayed): self-heals, but with β₂=0.999 half-life ~693 steps, full recovery ~3000 steps
- **z, m (direction quantities)**: refresh within a few dozen steps, not critical

**So the essence of "Burn→Online alignment" is aligning n (FTRL/FTML) and v (Adam) — these "gradient-scale-class" second-moment quantities.** Everything else revolves around this.

---

## 3. Patching Tool 1: Gradient Split — Aligning "Update Count"

### Definition

The idea is simple — **split Burn's one apply(G) into N apply(G/N)** calls:

```
Before:  apply(G)                                (1 large update)
After:   apply(G/N), apply(G/N), ..., apply(G/N)   (N small updates)
```

G is the aggregated gradient from the current big batch, N = gradient_split. Total gradient unchanged, samples unchanged, forward unchanged — **only optimizer state moves N more steps**.

### Three easily confused boundaries

- **Not gradient accumulation**: accumulation is "multiple backwards synthesizing one apply", opposite direction
- **Not exact Online replay**: real Online's g_i differ each step; split uses **G/N repeated N times**
- **Only touches optimizer, not forward**: only changes "how the gradient is consumed by optimizer"

### Why it's mandatory for FTRL

Compare FTRL n increment:

| | Burn no split (1× apply G) | Burn split (N× apply G/N) |
|---|---|---|
| n single-step increment | +G² | +(G/N)² |
| N-step accumulation | G² | **N·(G/N)² = G²/N** |
| Effective lr α/(β+√n) | √n grows fast → lr decays fast | √n grows N× slower → lr decays slower |
| L1 sparsification check | Only 1× |z| < λ₁ check | N× checks, intermediate steps may trigger early zeroing |

Post-split n grows N× slower. But since **n cannot self-heal**, without split n stays permanently small → all features' effective lr gets systematically inflated after Online takeover → severe post-switch jitter.

### Adam case

Compare Adam v increment:

- No split: Δv = (1-β₂)·G²
- N-split (geometric sum): Δv = (G²/N²)·(1-β₂^N)

At β₂=0.999, N=32: 1-β₂^N ≈ 0.0315, so **split Δv far smaller than no-split**. Although Adam's v has EMA decay and can theoretically self-heal, with β₂ typically 0.999, recovery often takes ~3000 steps — directly impacting early Online score calibration and learning speed.

**Conclusion**: split is mandatory for FTRL/FTML; for Adam split significantly shortens the recovery window.

---

## 4. But Split Alone Isn't Enough — The Hidden Variance Gap

This is the most critical turning point.

Suppose N small-batch gradients g_1...g_N are i.i.d. with mean μ and variance Var(g). Compare per-step apply gradients:

- **Burn split** each step uses G/N (average of N gradients): variance = Var(g)/N
- **Real Online** each step uses g_i (single batch): variance = Var(g)

Per-step g² expectation = Var + μ². N-step accumulated n increment:

| | Per-step g² expectation | N-step accumulation |
|---|---|---|
| Burn split | Var(g)/N + μ² | **Var(g) + N·μ²** |
| Real Online | Var(g) + μ² | **N·Var(g) + N·μ²** |
| Diff | — | **(N-1)·Var(g)** |

**Even with split, Burn's accumulated n remains systematically smaller than Online**, differing by the variance term (N-1)·Var(g).

**In RecSys scenarios**, gradient variance is often far larger than squared mean (sparse embeddings, long-tail samples, class imbalance all amplify this). When Var(g) ≫ μ², the two-side ratio ≈ N —

> **After split, n accumulation is still ~N× smaller than Online.**

This is the evidence for "split alone isn't enough".

---

## 5. Patching Tool 2: Gradient Aggregation — Aligning "Per-Step Variance"

### The idea

Since making Burn noisier isn't practical (would require N re-runs of forward), do the reverse: **let Online also aggregate M batches, average, then apply once**.

Post-aggregation Online per-step variance = Var(g)/M.

### The "alignment unit" concept

**Key**: pick a single **"alignment unit"** = average of M Online batches, and have both sides use this unit as the atomic apply:

- **Burn side**: view the big batch as several "alignment units", split into that many applies
- **Online side**: aggregate every M raw batches into 1 apply

**Every gradient entering the optimizer has the exact same statistical distribution on both sides.**

---

## 6. Tying It All Together with Numbers

Setup:

- Online batch = 1024, Burn batch = 32768 (N = 32)
- Per Online-batch gradient: μ = 0.1, Var(g) = 1.0 (common "variance far larger than mean squared" in RecSys)
- FTRL, tracking some parameter's n accumulation

Comparing four approaches after the same 32 Online-batch worth of data:

| Approach | Apply steps | Per-step g² expectation | n accumulation |
|---|---|---|---|
| A: Burn no split | 1 | ≈0.041 | **0.041** |
| B: Burn split only | 32 | ≈0.00004 | **0.0013** |
| C: Real Online (target) | 32 | ≈1.01 | **32.3** |
| D: Burn split + Online aggregate to unified alignment unit | Matched | Matched | **Matched** |

- **A vs C differs ~800×** — neither steps nor variance aligned
- **B vs C differs ~24000×** — split alone actually widens the variance gap
- **D is the only approach that truly aligns**

### One-shot diagram

Burn batch = 32 × Online batch, pick M = 8:

```
Real Online 32 steps:  g₁ | g₂ | g₃ | ... | g₃₂        32 applies, each variance Var(g)

Burn split only:       G/32 | G/32 | ... | G/32       32 applies, variance Var(g)/32²
                       ↑ Step count aligned, variance off by 32×

Split + aggregate (M=8):
  Burn side:  ḡ₁₋₈ | ḡ₉₋₁₆ | ḡ₁₇₋₂₄ | ḡ₂₅₋₃₂        4 applies, each variance Var(g)/8
  Online side: ḡ₁₋₈ | ḡ₉₋₁₆ | ḡ₁₇₋₂₄ | ḡ₂₅₋₃₂        4 applies, each variance Var(g)/8
                       ↑ Both step count and variance aligned
```

**Core: every apply on both sides must be "average of M real small batches" — this unified unit is what aligns both step count and variance simultaneously.**

---

## 7. Unified Mental Model

One-line summary:

> **Gradient split = Burn aligning "step count" to Online. Gradient aggregation = Online aligning "variance" to Burn.**

Symmetric picture:

| Dimension | Burn large apply | Real Online | Fix |
|---|---|---|---|
| **Update count** | 1 step | N steps | **Split**: Burn → N steps |
| **Per-step variance** | Extremely small (large-batch avg) | Var(g) | **Aggregate**: Online every M steps → 1 |

The connecting piece is M — **the alignment unit size**. Each Burn split part must be statistically equivalent to each Online aggregated part:

> **Draw an alignment-unit line between Burn and Online. Both sides use "average of M Online batches" as the minimum apply unit. Burn side decomposes multiple units, Online side composes multiple raw batches — the goal is that every gradient reaching the optimizer follows the same distribution.**

### Optimizer sensitivity matrix

| State quantity | Optimizer | Criticality | Decay | Self-heal | Split needed | Aggregate needed |
|---|---|---|---|---|---|---|
| n (cumulative scale) | FTRL | Critical | None, pure accumulation | **Never** | **Mandatory** | **Strongly recommended** |
| n (cumulative scale) | FTML | Critical | Same as FTRL's n | **Never** | **Mandatory** | **Strongly recommended** |
| v (cumulative scale) | Adam | Critical | EMA (β₂=0.999) | Slow (~3000 steps) | Worth doing | Worth doing |
| L1 sparsification intermediate check | FTRL | Important | — | Depends on n and future grads | Recommended | — |
| v EMA accumulation | FTML | Secondary | EMA (β₂=0.999) | Gradual recovery | Recommended | — |
| z / σ_t path diff | FTRL | Minor | — | Affected by n | — | — |
| m path diff | Adam | Non-critical | EMA (β₁=0.9) | Refreshes in dozens of steps | — | — |

---

## 8. When to Do It, When to Skip

### Applicability matrix

| Scenario | Split | Aggregate |
|---|---|---|
| Large Burn/Online batch ratio + FTRL/FTML | **Mandatory** | **Strongly recommended** |
| Large Burn/Online batch ratio + Adam | Recommended | Recommended (esp. calibration-sensitive) |
| Similar Burn/Online batch sizes | Skippable | Skippable |
| Burn only exports weights, no state inheritance | No point | Depends |
| Plain SGD, no adaptive state | Usually unneeded | Depends on noise |
| Business highly sensitive to burst response | Do normally | Careful (M shouldn't be large) |

### Aggregation's cost must be understood

Online goes from apply-per-batch to apply-per-M-batches, **making it M× slower to track live distribution**. In practice, aggregation is more commonly used as a **switch-window strategy**:

- First few minutes/hours after switch: larger M for smooth optimizer state transition
- Once state stabilizes: gradually reduce M to 1, restoring Online's high-frequency responsiveness

### Choosing N and M

- Core principle: **N × Online batch = Burn batch**, M consistent with N (same alignment unit)
- When N is large (e.g. > 32), Python-side loop apply's kernel launch overhead becomes significant → prefer C++ in-kernel split path
- When N is small (e.g. 2~4), Python-side split overhead is acceptable

---

## 9. Engineering Implementation Paths

### Two paths: Python-side vs In-kernel

**Python-side split** (v1 / default):

- Universal path, theoretically supports all common optimizers
- Two things happen in apply_gradients:
  1. Gradient scaling: `grad = grad / gradient_split` (sparse becomes `IndexedSlices(values/N, indices, shape)`)
  2. Loop apply: `tf.while_loop` runs N times, each call invokes the optimizer's apply_gradients
- Downside: graph nodes and kernel launches grow with N

Pseudo-code:
```python
def cond(gradient_i):
    return tf.less(gradient_i, self.gradient_split)

def body(gradient_i):
    # Loop gradient_split times, each iteration uses scaled grad/N to update params
    self.dense_optim.apply_gradients(...)
    self.sparse_optim[...].apply_gradients(...)
    return (gradient_i + 1,)

tf.while_loop(cond, body, [gradient_i])
```

**C++ in-kernel split** (v2 / performance path):

- Only supports partial optimizers (e.g. ftml_dense, ftrl_sparse)
- Sinks N-step split logic into the optimizer kernel's C++ body
- Semantically still N-step split, but one kernel launch

Trigger conditions example:
- `dense_optimizer_type == 'ftml'` && `version == 'v2'`, or `filter_grad` enabled
- `sparse_optimizer_type == 'ftrl_sparse'` && `version == 'v2'`

For these optimizers, when constructing, write `grad_split_num = gradient_split` into config and open the corresponding flag (`dense_split_update` / `sparse_optim_split_update[type] = True`). Python-side then skips `grad / N` scaling and `tf.while_loop`; the underlying kernel handles N-step split internally via `grad_split_num`.

### Compatibility handling when kernel-in split isn't supported

For optimizers without v2 kernel-in split support, pop `grad_split_num` from config before passing to the low-level op (otherwise unrecognized param causes errors). Fall back uniformly to Python-side `grad / N + while_loop`.

If neither kernel-side nor Python-side explicit split is done, **you cannot claim gradient split has been correctly executed for that optimizer**.

---

## 10. Common Engineering Pitfalls

- **Gradient scale mismatch**: If G is already mean gradient, dividing by N again **compresses the scale twice**, causing updates to be too small. Always calibrate gradient scale semantics before launch
- **Training time inflation**: Python-side split multiplies graph nodes and kernel launches by N; prefer C++ in-kernel split when N is large
- **Incomplete optimizer coverage**: In-kernel split only supports partial optimizers; others need Python fallback or explicit config pop
- **Interaction with lr schedule**: After step counter advances N×, if lr schedule depends on global step, actual lr curve may deviate from expectation
- **State drift extremely hard to debug**: **Surface-level loss/AUC usually looks fine, but n / v have already drifted** — monitoring is more important than the technique itself

---

## 11. How to Verify Your Model Needs This

Core principle: **Gradient split and aggregation are not for boosting Burn offline AUC — they're for reducing Burn→Online switch-window recovery caused by optimizer state discontinuity**. So don't rely solely on final metrics or offline AUC.

### Three-group comparison

| Group | Setup | Purpose |
|---|---|---|
| Online small-batch baseline | Online batch training from same checkpoint | Ideal trajectory reference |
| Burn no split/aggregate | Big batch direct apply, switch to Online | Observe magnitude of deviation |
| Burn split + Online aggregate | Full alignment, switch to Online | Verify closeness to baseline |

### Key technical metrics

| Metric | What to observe | Anomaly signal |
|---|---|---|
| **Parameter change ratio** | ‖W_t - W_burn‖ / ‖W_burn‖, layered by param group / feature | Post-switch spike or systematic drop below Online baseline |
| **Optimizer state distribution** | FTRL n, Adam v, effective lr α/(β+√n) mean and quantiles | No-split group n/v clearly biased, effective lr systematically shifted |
| **Sparse activation ratio** | Embedding param non-zero ratio | Ratio jumps around switch (FTRL L1 sparsification path abnormal) |
| **Score calibration** | pCTR/pCVR mean, bucketed calibration ratio | Instant mean shift or systematic bucket bias at switch |
| **Business metrics** | Switch-window AUC/GAUC, new-sample learning speed | Short-term regression or sustained slower learning |

### Three-tier evidence framework

| Evidence strength | Phenomenon | Conclusion |
|---|---|---|
| **Strong** | n or v clearly inconsistent + switch-window change ratio + calibration anomaly | Split/aggregation needed |
| **Medium** | Offline AUC diff not obvious, but switch-window param update speed or sparse ratio abnormal | Recommend experimental verification |
| **Weak** | Only minor final metric fluctuation, no state or switch-window evidence | Cannot confirm need, prioritize adding state observability |

**Simple rule**: If n or v clearly deviates from Online baseline AND switch-window score calibration is abnormal — strong evidence, alignment needed.

---

## 12. A More Essential Mental Model

Stepping back, the existence of these tools reveals a more fundamental point:

> **In adaptive-optimizer-driven RecSys, "how many samples seen" isn't the only training-volume dimension. "How many updates the optimizer felt, and what scale/variance each of those gradients had" is an equally important second dimension.**

Once this clicks, batch size, gradient split, gradient aggregation, learning rate schedule, warmup are no longer disconnected tricks — they all answer the same question:

**How to make optimizer state evolution paths match the business's training rhythm.**

Burn↔Online fault line is just one instantiation of this question under two-phase training. **Split fixes step count, aggregation fixes variance — with both dimensions fixed, optimizer state paths are truly connected.**

---

## Related Topics

- **[batch_size deep-dive](recsys-batch-size-gauc.md)**: How batch_size sensitivity is amplified by optimizer configuration in single-phase training. Same main thread's "previous episode" — both about how second-moment adaptive-optimizer quantities are influenced by batch-related signals.
- **learning rate schedule / warmup**: Essentially another means of "controlling optimizer state evolution path" — adjusting along the time axis rather than gradient shape.
- **Gradient Accumulation**: Direction **opposite** to gradient split — multiple backwards synthesizing one apply. Applicable scenario (train large batch on small GPU) is also entirely different.

## References

- FTRL: McMahan et al. "Ad Click Prediction: a View from the Trenches." KDD 2013.
- FTML: Zheng & Kwok. "Follow the Moving Leader in Deep Learning." ICML 2017.
- Adam: Kingma & Ba. "Adam: A Method for Stochastic Optimization." ICLR 2015.
