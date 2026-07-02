# 创新性迭代路线图:从 stop-gradient trick 到 8 分理论/方法对象

> 目的:回应当前最核心的问题:如果"encoder 单步、predictor 多步"只是一个低创新性的
> stop-gradient 修补,那怎样循环迭代理论和方法,把方向提高到严格审稿人愿意给 8/10 的程度?
>
> 本文只讨论**理论对象、方法对象、实现可行性、创新性评分**。不讨论代码实现。
> 相关背景稿:
> [theory_predictive_sufficiency_dynamics.md](theory_predictive_sufficiency_dynamics.md),
> [theory_predictive_sufficiency_acceptance_audit.md](theory_predictive_sufficiency_acceptance_audit.md)。
> 若要看"低创新评分如何循环迭代提分"的操作协议,见
> [theory_lewm_innovation_iteration_protocol.md](theory_lewm_innovation_iteration_protocol.md)。

---

## 0. 当前方案为什么创新性低

当前最小方案:

```text
one-step LeWM shapes encoder
+ stop-gradient multi-step rollout trains dynamics only
```

它的优点是干净、可证伪、容易做。但作为论文主方法,创新性偏低:

1. **stop-gradient 分离 dynamics/representation 不是新对象。**
   DreamerV3 类 world model 已经显式使用 dynamics loss / representation loss 的 stop-gradient 分离,
   并用 free bits 防止 dynamics 学到 trivial predictable state
   ([DreamerV3/Nature](https://www.nature.com/articles/s41586-025-08744-2))。
2. **multi-horizon/prefix supervision 已经在 LeWM 语境里出现。**
   Fast-LeWorldModel 用 action-prefix prediction 替代 autoregressive rollout,
   用 dense prefix-level supervision 降低长程 latent error
   ([Fast-LeWM](https://arxiv.org/abs/2606.26217),
   [project page](https://fast-lewm.github.io/))。
3. **LeWM 的主张本来就是少 loss、SIGReg、防 collapse。**
   如果我们只加 stop-grad 多步项,审稿人会问:这是 LeWM 的常规 ablation,还是一个新理论?
   LeWM 本身的核心是 next-embedding prediction + Gaussian regularizer
   ([LeWM](https://arxiv.org/abs/2603.19312), [project page](https://le-wm.github.io/))。

因此 related-work 定位必须非常明确:

| 相关工作 | 它已经覆盖什么 | 我们不能再 claim 什么 | 我们仍可 claim 的空位 |
| --- | --- | --- | --- |
| DreamerV3 | dynamics/representation stop-gradient,free bits 防 trivial dynamics | stop-gradient separation 本身新 | reconstruction-free Gaussian JEPA 的 control-sufficiency erosion |
| LeWM | two-term end-to-end JEPA,SIGReg 防 collapse | Gaussian regularizer/LeWM baseline 新 | SIGReg 防 collapse 但不防 predictive-control gap |
| Fast-LeWM | action-prefix prediction,dense multi-horizon supervision,降低 rollout error | multi-horizon supervision 新 | 低 rollout/self-drift 仍可能不等于 control sufficiency |

所以新意必须落在:

```text
not "how to reduce rollout error",
but "why reducing self-referential rollout error can erase control-sufficient variables,
and how to detect/protect those variables."
```

所以当前方案更像:

```text
Score 4/10:
good diagnostic and sensible baseline, but not a paper-level method object.
```

要提分,必须把主创新从"stop-grad 放哪里"升级为:

```text
发现并形式化一个新 failure mode:
  predictive coordinate selection conflicts with control sufficiency.

提出一个新可测对象:
  sufficiency erosion / predictive-control gap / task-critical conditional entropy.

提出一个新方法原则:
  not merely train predictor longer, but preserve high-value high-uncertainty variables.
```

---

## 1. 评分阶梯

| 分数 | 方向形态 | 为什么是这个分数 |
| --- | --- | --- |
| 4/10 | stop-grad multi-step 修 LeWM | 合理但像已有技巧;理论和方法都不够新 |
| 6/10 | 证明 LeWM 的 predictive-coordinate selection bias + PushT 证据 | 有理论洞见,但可能只是诊断 paper |
| 6/10 | 提出 sufficiency probe/ranking regularizer | 解决对症,但可能像 auxiliary loss |
| 8/10 | 定义"预测-控制充分性冲突"的新理论对象 + erosion certificate + 方法闭环 | 有新 failure mode、新指标、新训练原则、实验证明 |
| 8/10+ | 通用到多 JEPA/world model,并在 contact/hybrid + toy theorem 中验证 | 从 LeWM-specific patch 变成 latent WM 原理 |

核心判断:

```text
8 分不是靠把 stop-grad 结果跑好。
8 分来自把"低 self-drift 但坏 planning"解释成一个可定义、可测、可预测、可修复的新现象。
```

---

## 2. 候选高创新 reformulation

### A. Sufficiency Erosion Certificate

**一句话.**
多步 self-consistency 不是单纯降低 error,而是在改变 encoder 的坐标选择。
我们定义并预测哪类任务变量会被 erosion。

**理论对象.**

```text
Predictive explained variance at horizon k:
  V_k(y) = Var(E[y_{t+k} | C_t, A_{t:t+k-1}])

Task value / control sensitivity:
  W(y) = expected sensitivity of terminal task cost/ranking to y

Erosion risk:
  R_k(y) = W(y) * (1 - V_k(y) / Var(y))
```

直觉:

```text
high W(y), low V_k(y)  => 任务重要但难预测 => 最容易被 MSE+Gaussian 牺牲。
```

**方法对象.**
不是直接提出复杂模型,而是先提出一个 certificate:

```text
给定一个 trained/训练中的 WM,估计每个物理变量或 probe direction 的 R_k。
R_k 高的变量必须被 sufficiency anchor 保护。
```

**评分预估.**

```text
6/10 alone:
  作为诊断理论很强,但方法贡献还不够。

8/10 if paired with method:
  如果 R_k 能预测 angle loss、planning collapse,并指导 loss 修复,就变成新原则。
```

**需要的证明.**

1. 线性高斯 theorem:多步 MSE 选择 top eigenspace of horizon-weighted explained variance。
2. 任务变量被丢的条件:存在替代变量 $u$ 使 $V_k(u)>V_k(y)$ 但 $W(u)<W(y)$。
3. PushT:angle 的 estimated erosion risk 高于 position/background factors。

---

### B. Predictive-Control Gap (PCG)

**一句话.**
LeWM 优化的是 predictability,planning 需要 action-ranking control sufficiency。
二者之间的 gap 应先作为 Pareto 诊断,再谨慎转成标量。

**理论对象.**

```text
Predictability:
  P(φ) = - E Tr Var(Z_{t+1}|C_t)

Control sufficiency:
  C(φ) = pairwise rank agreement between latent terminal cost and true task cost

Predictive-Control Gap:
  Pareto-PCG opens when P increases but C decreases.
```

现象:

```text
multi-step co-training: P ↑, C ↓  => PCG opens.
one-step LeWM:          P lower, C high.
desired method:         P ↑, C high.
```

**方法对象.**

```text
PCG-constrained training:
  minimize predictive loss
  subject to C(φ) >= C_min

or Lagrangian:
  L = L_pred + λ SIGReg + β L_rollout - γ C(φ)
```

这里 $C(φ)$ 可以从强到弱有多个版本:

1. ranking sufficiency:latent terminal distance preserves true terminal task ranking。
2. Control-Fisher / controllability-gated sufficiency:保护 high sensitivity × high action-controllability 方向。
3. state sufficiency:probe block pose/angle,只作为机制干预,不是主定义。

**评分预估.**

```text
6/10:
  如果只是加 supervised probe,像 auxiliary loss。

8/10:
  如果 ranking sufficiency / Pareto-PCG / controllability-gated SER
  是新 certificate + theorem + 实证闭环。
```

**需要的证明.**

- ranking sufficiency / Pareto-PCG 比 self-drift 更能预测 planning。
- Pareto-PCG 在 multi-step collapse 中显著恶化。
- 方法关闭 Pareto-PCG,ranking/planning 恢复。

---

### C. Predictability-Sufficiency Decomposition (PSD)

**一句话.**
不要强迫所有 latent 变量都同样可预测。把 latent 分成:

```text
z_p = predictable dynamics coordinates
z_s = sufficient but hard-to-predict control coordinates
```

LeWM/SIGReg 的问题不是 latent 不高斯,而是把所有坐标都放进同一个 isotropic MSE 预测目标。

**理论对象.**

```text
z = [z_p, z_s]

z_p:
  optimized for low conditional variance and long-horizon rollout stability.

z_s:
  optimized for control sufficiency / goal ranking / state decodability.
  dynamics may be stochastic/uncertain, not forced into deterministic low-MSE rollout.
```

这比 stop-grad 更像新方法,因为它承认:

```text
任务关键变量可能本质上高条件熵;
正确做法不是让它消失,而是在 planner 中保留它的不确定性。
```

**方法对象.**

```text
PSD-JEPA:
  SIGReg over full z or separately over z_p,z_s.
  deterministic MSE rollout on z_p.
  Gaussian/uncertainty-aware transition on z_s.
  control sufficiency anchor on z_s.
  planner cost uses both mean and uncertainty / risk.
```

**评分预估.**

```text
8/10 if executed well:
  新 latent ontology + 新 loss + 新 planning implication。

风险:
  复杂度上升,可能被认为和 stochastic state-space/Dreamer 太接近。
```

**需要区分已有工作.**

Dreamer 有 stochastic latent,但它不是从 "Gaussian JEPA predictive-coordinate erosion"
推出的,也不针对 reconstruction-free latent L2 planning 的 control sufficiency gap。
Fast-LeWM 做 prefix multi-horizon prediction,但仍可能优化 self-consistency;
PSD 的目标是保留 hard-to-predict but control-critical variables。

---

### D. Control-Fisher / Goal-Fisher Sufficiency

**一句话.**
任务关键性不应该靠人工说 angle 重要,而应该由 goal cost 对状态变量的敏感性定义。

**理论对象.**

```text
Control Fisher / Goal Fisher:
  F_y = E[ ∇_y c(S,G) ∇_y c(S,G)^T ]

Prediction uncertainty:
  Σ_y|C = Var(Y_{t+k}|C,A)

High-risk direction:
  directions with high F_y and high Σ_y|C.
```

这把"难预测但任务关键"写成一个动力学-控制乘积:

```text
risk = Tr(F_y Σ_y|C)
```

**方法对象.**

```text
Fisher-sufficient latent training:
  preserve directions with high Tr(F Σ)
  by weighting sufficiency/ranking loss along those directions,
  while letting low-F high-uncertainty nuisance factors be compressed.
```

**评分预估.**

```text
8/10 potential:
  更理论、更控制味,不只是 probe loss。

实现性:
  PushT 有 task cost/pose,容易估计 F_y;
  泛化到别的 goal-conditioned env 也合理。
```

**风险.**
需要真实 state 或 task cost gradients;如果只能在 simulator 中用,会被问是否仍是 self-supervised JEPA。
可以定位为 "decision-aware JEPA for control",不是纯自监督。

---

### E. Counterfactual Ranking Sufficiency

**一句话.**
Planning 的本质不是重建状态,而是给 action candidates 排序。
所以 sufficiency 应直接定义在 counterfactual action ranking 上。

**理论对象.**

```text
For candidates a,b:
  true ranking:    c(S^a_H,G) < c(S^b_H,G)
  latent ranking:  ||z^a_H-z_g||^2 < ||z^b_H-z_g||^2

Control sufficiency = probability of ranking agreement.
```

**方法对象.**

```text
Ranking-sufficient rollout loss:
  keep LeWM/SIGReg for representation stability,
  add pairwise ranking loss over candidate futures,
  optionally only on high-disagreement/high-contact samples.
```

**评分预估.**

```text
6/10:
  direct and effective, but could look like decision-aware auxiliary loss.

8/10:
  if tied to theorem showing MSE self-drift is not order-preserving and ranking sufficiency fixes exactly that.
```

**实现性.**
最贴近最终 planning,但最贵,因为要 candidate rollouts / true state costs。

---

## 3. 推荐主线:先做 A+B,再决定是否升级 C/D

我建议下一轮不要直接跳到复杂 PSD。先把 paper 的新对象立起来:

```text
Sufficiency Erosion in Gaussian Joint-Embedding World Models
```

核心 paper spine:

1. **Theory.**
   SIGReg/whitened MSE induces predictive-coordinate selection.
   Multi-step MSE changes selection toward horizon-weighted predictable directions.
2. **Diagnostic object.**
   Define Predictive-Control Gap / Erosion Risk.
3. **Empirical discovery.**
   In PushT, multi-step LeWM lowers self-drift but opens PCG:
   angle/control ranking erodes.
4. **Method.**
   Erosion-aware training:
   preserve high-risk directions via a small sufficiency constraint;
   train dynamics long-horizon only inside this protected coordinate system.
5. **Validation.**
   PCG predicts planning better than self-drift;
   reducing PCG recovers planning.

This can score:

```text
4 -> 6:
  if only theory + diagnostic.

6 -> 8:
  if PCG/erosion risk predicts failure and guides a method that improves the Pareto frontier.
```

---

## 4. 具体迭代循环

### Iteration 0:当前状态

```text
Score: 4/10
Main object: stop-grad multi-step
Problem: known trick, untested, history confound, no novelty wall crossed.
```

### Iteration 1:把 claim 改成 "sufficiency erosion"

改理论稿标题/核心:

```text
From Low Self-Drift to Sufficiency Erosion:
Gaussian JEPA World Models Select Predictable Coordinates
```

新增定义:

```text
Predictive-Control Gap
Erosion Risk
Control Sufficiency
```

目标分:

```text
Score: 4 -> 6 only if action-ranking definition and certificate are sharp.
```

### Iteration 2:做 controlled toy theorem + toy experiment

设计一个线性高斯/hybrid toy:

```text
state = [x_easy, y_hard]
x_easy: highly predictable, low task value
y_hard: low predictability / regime switching, high task value
SIGReg/whitened MSE with D=1 chooses x_easy.
Adding multi-step increases pressure against y_hard.
Sufficiency constraint recovers y_hard.
```

这个 toy 是创新性关键,因为它把"选择可预测坐标"从故事变成可证明现象。

目标分:

```text
Score: 6/10 -> 7/10
```

### Iteration 3:选择方法对象

不要急着写 "stop-grad is method"。三个选择:

| 方法 | 创新性 | 实现性 | 推荐 |
| --- | --- | --- | --- |
| dynamics-only rollout | 低 | 高 | baseline/ablation |
| PCG-constrained LeWM | 中 | 中 | 主方法 v1 |
| PSD latent split | 高 | 中低 | 主方法 v2,若 v1 不够新 |
| Control-Fisher sufficiency | 高 | 中 | 理论味最强,适合 PushT |

当前最推荐:

```text
PCG-constrained LeWM + Control-Fisher weighting
```

因为它比 simple probe loss 更理论,比 PSD 更容易落地。

### Iteration 4:再让 reviewer 打分

每轮让独立 reviewer 只回答三件事:

```text
1. Is the main object new, or just a known trick?
2. Does the theorem actually imply the method?
3. Would the evidence, if positive, raise the paper to 8?
```

如果 reviewer 仍说 "auxiliary loss",就升级到 PSD;
如果 reviewer 说 "good object, need results",就开始实验。

---

## 5. 推荐的下一版方法草案

下一版已单独展开为:
[theory_control_sufficient_gjepa.md](theory_control_sufficient_gjepa.md)。

### 名字

```text
Erosion-Aware LeWM (EA-LeWM)
```

或更理论:

```text
Control-Sufficient Gaussian JEPA (CS-GJEPA)
```

### Loss

不要只写:

```text
L = L_one_step + β L_rollout_sg
```

写成:

```text
L = L_Gaussian-JEPA
  + β L_dynamics-only-rollout
  + γ L_control-sufficiency
```

其中:

```text
L_control-sufficiency =
  E[ weight(y) * error(q(z), y_task) ]

weight(y) ∝ task sensitivity * rollout uncertainty
          ≈ Control-Fisher * conditional variance
```

这让方法从"加 probe"变成:

```text
protect high-risk task-critical uncertainty directions.
```

### 关键预测

1. high erosion-risk variables 是 multi-step co-training 中最先掉的变量。
2. self-drift 和 planning 可以反向,但 PCG 与 planning 同向。
3. EA-LeWM 不一定最低 self-drift,但在 drift/PCG/planning Pareto 上最好。

---

## 6. 当前结论

下一轮不该问:

```text
stop-grad multi-step 能不能 work?
```

应该问:

```text
我们能不能定义并测出 sufficiency erosion,
证明它比 self-drift 更解释 planning,
再用 erosion-aware objective 修它?
```

如果能,这才像 8 分方向。stop-grad 可以保留,但位置应该是:

```text
baseline / minimal ablation / anti-erosion component
```

而不是主创新。
