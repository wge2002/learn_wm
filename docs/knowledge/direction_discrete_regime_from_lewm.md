# 方向定案：从 LeWM 理论出发的动态离散 regime

> 本 MD = **定案主方向**（2026-06-24，替代并删除原 `direction_discrete_anticompounding.md`）。
> 出发点不是"再试一种离散"，而是**从 LeWM 自己的训练理论推出离散应该长什么样、放哪里**，再给一条可执行的多步实验脉络。
> 诊断线见 [diagnosis_world_model_drift.md](diagnosis_world_model_drift.md)；已证伪的离散尝试合并归档见 [discrete_attempts_falsified_archive.md](discrete_attempts_falsified_archive.md)；B 线候选池见 [idea_brainstorm_unchosen.md](../idea_brainstorm_unchosen.md)。

## 用户硬约束

1. 有理论支持 + 认可度。
2. 最终结论简单、少超参。
3. 离散保留，但**不是全离散**：连续骨干保精度 + 稀疏/动态离散 regime 当锚点。
4. **不强行对齐，自然涌现**；离散尽量从 LeWM 自身 loss / 已训好的模型里长出来，不堆新 loss。
5. 产出 = **对 LeWM 的理论分析 → 一个 solid 方法**，不是调参堆。

## 一句话

> LeWM 的 loss = 预测 + SIGReg。SIGReg 把**边缘** latent 正则成各向同性高斯 ⇒ 表征里没有可离散的结构（这条 theorem 顺手解释了你全部表征级离散负结果）。但 SIGReg **不约束转移** `p(z'|z,a)`；而接触物理是分段的 ⇒ 离散 regime 天然活在**动力学 `f`** 里。所以我们的离散 = 从已训好的 `f` 里**自然读出**的分段 regime，当稀疏锚点；连续 `z` 继续保精度。

## Part 1 — 理论地基：为什么离散只能活在 transition

LeWM 训练目标（已从 `scripts/train/lewm.py` + `config/lewm.yaml` 确认）：

```text
loss = pred_loss + λ·sigreg_loss
pred_loss   = ‖predict(z_{≤t}, a_{≤t}) − z_{t+1}‖²   # latent MSE，单步 teacher-forced（num_preds=1）
sigreg_loss = SIGReg(emb)                            # 把 emb 的边缘分布推成各向同性高斯
```

SIGReg（Balestriero & LeCun 2025，**你们自家实验室**）：用随机 1D 方向的正态性检验，把 embedding 的**边缘分布**逼成各向同性高斯（每维去相关、独立、高斯）。

**Theorem 级推论：**

```text
SIGReg ⇒ 边缘 p(z) ≈ 各向同性高斯 ⇒ 没有簇 / 没有模 / 没有偏好方向
⇒ 任何"在表征 z 上找离散"（VQ、聚类、snap-back、码本）by construction 注定失败：
   在一个被显式正则成"无结构"的分布里找簇，找不到是必然，不是没调好。
```

**这条一次性解释了你之前所有离散负结果**（见 archive）：Phase 6/7 测到的"随机 latent 近正交 cosine 0.009、无窄锥、effective dim ~76、drift 是各向同性扩散、事后投影只降 ~4% 无用"——**全是 SIGReg 各向同性的直接后果**。负结果不是失败，是定理的证据。

**关键转折：SIGReg 只约束边缘 `p(z)`，不约束转移 `p(z'|z,a)`。**

```text
点"在哪"（marginal）：被正则成无结构 → 离散没法活在这。
点"怎么动"（transition f）：没被约束。而接触物理是 piecewise 的
  （自由移动 / 接触 / 推 / 脱离，hybrid dynamical systems）
  ⇒ f 本身是分段的 ⇒ 离散 regime 天然藏在动力学里。
```

## Part 2 — 我们的离散，从这个理论重述

```text
连续 z      = 精度（SIGReg 各向同性，保持不动，别动它）
离散 regime = "我在 piecewise f 的哪一段"，从已训好的 f 里读出来，稀疏、动态
锚点        = 进入一个 regime 段就 commit，连续在段内滚；regime 边界（事件）处再决定
```

- **自然涌现、不堆 loss**：regime 从训好的 `f` 的局部行为里读（Jacobian / 下一步分叉），无监督；要训也只 structure `f`（piecewise predictor），**绝不在 `z` 上加聚类 loss**（那跟 SIGReg 对着干）。
- **两个收益是副产品**：锚到稳定 regime ⇒ 连续部分只需段内预测，自由扩散被压（抗漂）；接触/岔路处 regime 分开 ⇒ 不糊（多模态 commit）。"验证/监控"也是副产品：regime 边界 = 该不该 re-ground 的信号。
- **对齐自然涌现 + 量，不强行**：跨轨迹 regime 是否同义，用 purity 量（你 Stage 1d 那套）；需要复用时才付对齐的账。

## 什么已死（现在由理论解释，不再走）

| 死路 | 为什么（理论 + 数据） |
| --- | --- |
| 全 latent 离散 | 精度地板；且 effect 输连续（TD-MPC2 > DreamerV3） |
| 表征级 VQ / 聚类 / snap-back | **SIGReg ⇒ 边缘无簇，by construction 找不到** |
| test-time 锚点修正器 | on-manifold 扩散 + 边缘无结构可 snap；min-cost CEM 自带 commit |

## Part 3 — 实验脉络（多步，可执行）

> 原则：分析在前、训练在后、便宜在前、贵在后；每步一个 go/kill 判据。

### Step A — 存在性（纯分析，不训练，最"自然"）

- **目的**：离散 regime 到底在不在训好的 LeWM 动力学里？
- **怎么测**：拿 `quentinll/lewm-pusht`，沿 PushT 专家轨迹算 predictor 的局部 Jacobian / 下一步残差方向，聚类；和 Phase 4 已有的接触事件对齐；**对照**：直接聚类 `z`（应无结构）vs 聚类 `f` 的局部行为（应现 regime）。
- **判据**：`f` 落成几簇 + 对上接触，而 `z` 不成簇 → **存在性成立（本身就是一条 paper 级分析）**，绿灯。`f` 也糊成一团 → 加最小分段先验，或诚实收掉。

### Step B — 自然涌现 / readout（最小训练，SIGReg 兼容）

- **目的**：把 regime 变成可用的离散变量，且证明它让动力学更好。
- **怎么测**：piecewise / mixture-of-experts 门控 predictor `f = Σ g_k(z,a)·f_k`，用**原 LeWM loss + 多步 unroll**训（不加 z 聚类 loss），对打单体 predictor。
- **判据**：regime-conditioned 把 **mse@k 斜率压平**（抗漂）+ regime 无监督地对上接触（purity↑）→ 方法与"自然涌现"同时落地。压不平 → 方法死。

### Step C — 锚点收益（控制层）

- **目的**：把 regime 当稀疏锚点，是否真帮 rollout / planning。
- **怎么测**：进段 commit regime、段内连续滚、regime 边界再决定（动态锚点）。(i) open-loop drift / planner cost-rank（复用 Phase 4/5 机器）；(ii) 真多模态处 commit（`lewm_twogoal` / TwoRoom，between-ness）；(iii) regime 边界触发自适应 re-ground，比固定间隔 re-ground（等预算）。
- **判据**：regime-anchored 在 planner 相关指标上胜单体连续 → 收益兑现。

### Step D — 对齐 / 复用（可选，要组合才做）

- **目的**：regime 是不是一套跨 episode 通用词表。
- **怎么测**：跨 episode purity；若高 → 可拼接 / 可搜索的计划词表 → toy 组合 / 搜索验证。
- **判据**：purity 高且能跨 goal 复用 → 解锁 (b) 计划词表 avenue。

### Step E — scale / 泛化（回答"测得准"）

- **目的**：出 toy / PushT 后还成立吗？
- **怎么测**：随机性轴（DMC / Craftax）+ FoV shift（visual / geometry）下重测 A–C；regime 抓的是动力学不是外观，理应更 shift-robust。
- **判据**：跨环境 + shift 下仍成立 → solid；只在 PushT → scale-limited（诚实记）。

## 总证伪判据

```text
A 死：f 也无 regime 结构 → 离散在 LeWM 动力学里也没有，方向收缩。
B 死：regime-conditioned 不压平 mse@k → 自然涌现不带来抗漂。
C 死：锚点对 planner 无收益 → 回到诊断已知杠杆（re-ground / uncertainty）。
E 死：不过 scale / shift → scale-limited，诚实写。
```

## 理论 & 文献锚点

- **LeJEPA / SIGReg**（Balestriero & LeCun 2025，你们实验室；边缘各向同性高斯）：https://github.com/rbalestr-lab/lejepa
- **Hybrid / piecewise dynamical systems**：接触动力学分段 = `f` 有 regime 的理论依据。
- **Dreamer RSSM**（连续 h + 离散 z 混合）：https://arxiv.org/pdf/2010.02193 ；**Director**（稀疏离散子目标 + 连续 worker）：https://arxiv.org/abs/2206.04114
- **DCWM (ICLR 2025)** + **TD-MPC2**（全离散 vs 连续、精度地板背景）：https://arxiv.org/abs/2503.00653 ，https://arxiv.org/abs/2310.16828

## Markdown 地图

```text
docs/knowledge/diagnosis_world_model_drift.md          诊断线（病：on-manifold 各向同性扩散）
docs/knowledge/discrete_attempts_falsified_archive.md  已结案合并：失败的离散尝试 + 一览
docs/knowledge/direction_discrete_regime_from_lewm.md  ← 本文：定案主方向（理论→方法→脉络）
docs/idea_brainstorm_unchosen.md                       B 线候选池
```
