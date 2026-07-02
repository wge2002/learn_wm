# LeWM 理论创新迭代协议:从低创新 loss 到 8 分方法对象

> 目的:把当前方向从"多步 loss 怎么写"转成"创新性如何持续提分"。
> 本文不讨论代码实现,只讨论理论对象、方法对象、可实现性和审稿评分循环。
>
> 相关草稿:
> [theory_predictive_sufficiency_dynamics.md](theory_predictive_sufficiency_dynamics.md),
> [theory_predictive_sufficiency_novelty_loop.md](theory_predictive_sufficiency_novelty_loop.md),
> [theory_control_sufficient_gjepa.md](theory_control_sufficient_gjepa.md)。

---

## 0. 先把问题说准

如果当前主张仍是:

```text
LeWM + multi-step rollout + stop-gradient
```

那么创新性大概率只有 4/10。它像一个合理训练技巧,不太像一个新理论或新方法。
要提分,不能继续微调 loss 权重或梯度路径,而要改变 paper 的中心对象。

更强的中心对象应当是:

```text
Gaussian joint-embedding world models can reduce self-referential prediction error
by selecting predictable but less control-sufficient coordinates.
```

也就是说,真正要研究的不是:

```text
怎样让 latent rollout MSE 更低?
```

而是:

```text
什么时候 latent rollout MSE 变低反而说明 representation 正在丢控制充分性?
这个风险能否被定义、预测、验证、再用来设计方法?
```

这才是从低创新 loss 走向 8 分方向的核心。

---

## 1. 评分不是单轴,而是最短板

把内部评分拆成四个轴:

| 轴 | 2 分 | 4 分 | 6 分 | 8 分 |
| --- | --- | --- | --- | --- |
| 现象 | 常见训练不稳定 | 一个任务上的失败 | 可复现 failure mode | 新的、可迁移的 failure mode |
| 理论 | 直觉解释 | toy 推导 | 定义清楚且有反例/定理 | 定理直接推出方法原则 |
| 方法 | 换 loss 权重 | 已有 trick 组合 | 由诊断指标指导 | 新目标解决新 failure mode |
| 证据 | 单 seed 曲线 | 常规 ablation | 机制证据闭环 | 新指标预测失败且指导修复 |

一个保守评分可以写成:

$$
S_{\mathrm{paper}}
=
\min
\left\{
S_{\mathrm{phenomenon}},
S_{\mathrm{theory}},
S_{\mathrm{method}},
S_{\mathrm{evidence}}
\right\}
+ B_{\mathrm{coupling}},
$$

其中 $B_{\mathrm{coupling}}$ 只在四件事互相咬合时才存在:

```text
theory predicts diagnostic,
diagnostic predicts failure,
method optimizes diagnostic,
evidence shows planning improves through that diagnostic.
```

所以如果方法还是普通 stop-gradient,即使结果变好,主分也会被
$S_{\mathrm{method}}$ 卡住。反过来,如果理论很漂亮但方法只是 state auxiliary loss,
也会被 $S_{\mathrm{method}}$ 和 $S_{\mathrm{evidence}}$ 卡住。

---

## 2. 当前方向的真实起点

当前方向的最低可接受诚实定位:

```text
Score 4/10:
  stop-gradient multi-step is a useful diagnostic baseline,
  but it is not the main contribution.
```

它的价值是暴露一个现象:

```text
lower self-drift may not imply better planning.
```

但这句话本身还不够。审稿人会问:

1. 这是不是训练细节或 history mismatch?
2. 这是不是所有 world model 都知道的 compounding error?
3. 这是不是加一个 state decoder 就能解决的普通表示问题?
4. 这个方法和 Dreamer/Fast-LeWM 的区别在哪里?

因此下一步不是把 stop-gradient 写得更像主方法,而是把它降级为:

```text
anti-erosion baseline
```

主线改成:

```text
sufficiency erosion in Gaussian JEPA world models
```

### 独立审稿 round 1

两个独立 agent 已按 NeurIPS/ICLR 口径审查当前文档,共同结论是:

```text
current score: 4/10
not 6/10 yet
```

主要原因:

1. 当前 theorem 仍是 fixed-context linear toy,真实 LeWM 的 context 依赖 encoder。
2. stop-gradient dynamics separation 和 multi-horizon prediction 都不是新贡献。
3. state-SER 容易被看成 "发现 angle 重要,然后加 angle loss"。
4. scalar PCG 的 normalization 任意,不应作为主指标。
5. control sufficiency 必须定义为 candidate action ranking,而不是 state decodability。
6. SER 需要 controllability/action-relevance,否则会保护不可控噪声。
7. 必须证明 SER/PCG 比 self-drift、probe R²、inverse dynamics、uniform uncertainty 更能预测 failure。

因此下一轮的最低目标不是"把 stop-gradient 说得更好",而是:

```text
把 passive variable-selection story
升级成 action-conditioned ranking-sufficiency theorem and certificate.
```

---

## 3. 提分循环

每一轮只做一件事:找到当前最低分轴,然后改变理论或方法对象,而不是修辞。

### Iteration 1:从 trick 到 phenomenon

当前审稿意见:

```text
This is just a stop-gradient training trick.
```

要做的改变:

```text
把主 claim 从 "stop-gradient improves rollout" 改成
"self-prediction can conflict with control sufficiency"。
```

新增定义:

$$
P_k(\phi)
=
-\mathbb{E}\,
\mathrm{Tr}\,
\mathrm{Var}
\left(
Z_{t+k}
\mid
C_t,A_{t:t+k-1}
\right),
$$

$$
S_H(\phi)
=
\mathbb{E}_{a,b}
\left[
\mathbf{1}
\left(
\operatorname{sign}(J_z(a)-J_z(b))
=
\operatorname{sign}(J_y(a)-J_y(b))
\right)
\right],
$$

其中 $J_z$ 是 latent terminal cost,$J_y$ 是真实任务 cost。

定义 predictive-control gap:

$$
\Pi_{k,H}(\phi)
=
\left(
P_k(\phi),
S_H(\phi)
\right).
$$

主张不是某个任意标量差值,而是 Pareto gap:

$$
\Delta P_k>0,
\qquad
\Delta S_H<0.
$$

这一轮目标:

```text
Score 4 -> 6:
  先把问题变成一个可定义的 gap,
  而不是一个 loss engineering choice。
```

### Iteration 2:从 phenomenon 到 theorem

当前审稿意见:

```text
The gap is plausible but not theoretically inevitable.
```

要做的改变:

```text
构造一个最小反例 theorem:
存在一个线性高斯控制问题,
MSE+whitening 的最优 predictive representation
严格不同于 control-sufficient representation。
```

一个最小版本:

设 $Y=(y_1,\ldots,y_m)$ 已白化,latent 维度 $D<m$。
每个方向的预测可解释方差为:

$$
\rho_i
=
\frac{
\mathrm{Var}
\left(
\mathbb{E}
\left[
y_i
\mid
C
\right]
\right)
}{
\mathrm{Var}(y_i)
}.
$$

任务敏感性和 action controllability 为:

$$
\omega_i
=
\mathbb{E}
\left[
\left(
\frac{\partial c(Y,G)}{\partial y_i}
\right)^2
\right].
$$

$$
\kappa_i
=
\mathbb{E}
\,
\mathrm{Var}_{a\sim Q}
\left(
\mathbb{E}
\left[
y_i^{t+H}
\mid
C_t,a
\right]
\right).
$$

若存在 $u,v$ 使:

$$
\rho_u > \rho_v,
\qquad
\omega_u\kappa_u < \omega_v\kappa_v,
$$

则 predictive objective 会偏向 $u$,而 control objective 需要 $v$。
这时 $v$ 的 erosion risk 是:

$$
\mathrm{SER}(v)
=
\omega_v\kappa_v
\left(
1-\rho_v
\right).
$$

这一轮目标:

```text
Score 6 -> 7:
  证明这不是观察到的个例,
  而是 predictive coordinate selection 与 control sufficiency 的结构性冲突。
```

### Iteration 3:从 theorem 到 certificate

当前审稿意见:

```text
The theorem is toy; how do I know it explains the real failure?
```

要做的改变:

```text
提出可测 certificate,让它预测真实模型中哪些变量会掉、哪些 checkpoint 会坏。
```

对方向 $v$ 定义:

$$
\sigma^2_{k}(v)
=
\mathbb{E}
\,
\mathrm{Var}
\left(
v^\top Y_{t+k}
\mid
C_t,A_{t:t+k-1}
\right),
$$

$$
\omega_H(v)
=
\mathbb{E}
\left[
\left(
\frac{\partial c(Y_{t+H},G)}
{\partial v^\top Y_{t+H}}
\right)^2
\right],
$$

$$
\mathrm{SER}_{k,H}(v)
=
\omega_H(v)\,
\sigma^2_k(v).
$$

certificate 必须回答四个预测问题:

1. 高 $\mathrm{SER}^{ctrl}$ 变量是不是最先在 multi-step co-training 中掉 probe?
2. ranking sufficiency / Pareto-PCG 是否比 self-drift 更能预测 planning success?
3. 高 $\mathrm{SER}^{ctrl}$ 是否集中在 contact/hybrid windows,例如 PushT 的 block angle?
4. 它是否比 probe R²、inverse dynamics accuracy、uniform uncertainty 更早预测 failure?

这一轮目标:

```text
Score 7:
  paper 开始有新诊断对象。
```

### Iteration 4:从 certificate 到 method

当前审稿意见:

```text
This is an interesting diagnosis, but the method is just an auxiliary loss.
```

要做的改变:

```text
方法必须由 SER/PCG 推出,而不是随手加 state prediction。
```

弱版本是 SER-weighted state intervention:

$$
\mathcal{L}_{\mathrm{SER\text{-}state}}
=
\min_q
\mathbb{E}
\left[
\sum_i
\alpha_i
\left(
q_i(Z_t)-Y_{t,i}
\right)^2
\right],
$$

其中:

$$
\alpha_i
\propto
\widehat{\omega}_i
\,
\widehat{\kappa}_i
\,
\widehat{\sigma}^2_i.
$$

强版本是 ranking sufficiency,应作为主方法候选:

$$
\mathcal{L}_{\mathrm{SER\text{-}rank}}
=
\mathbb{E}_{a,b}
\,
\widehat{\alpha}(a,b)
\,
\max
\left(
0,
m+J_z(a)-J_z(b)
\right),
$$

当真实任务 cost 满足 $J_y(a)<J_y(b)$ 时使用。

关键区别:

```text
普通 auxiliary loss:
  preserve all state variables equally.

SER-aware method:
  preserve only directions that are task-sensitive, action-controllable,
  and prediction-uncertain.

Ranking-SER method:
  preserve action ordering, not raw state reconstruction.
```

这一轮目标:

```text
Score 7 -> 8:
  方法不再是 "加状态监督",
  而是 "用 erosion certificate 选择要保护的控制充分信息"。
```

### Iteration 5:如果仍被认为不够新,升级 latent ontology

当前审稿意见:

```text
Still an auxiliary constraint on top of LeWM.
```

要做的改变:

```text
从一个 loss 升级成一个 latent ontology:
predictable dynamics coordinates + control-sufficient uncertain coordinates。
```

设:

$$
Z_t
=
\left[
Z^p_t,
Z^s_t
\right],
$$

其中 $Z^p_t$ 用于可预测 dynamics,$Z^s_t$ 用于控制充分性。

目标变成:

$$
\mathcal{L}
=
\mathcal{L}_{\mathrm{pred}}
\left(
Z^p
\right)
+
\lambda
\mathcal{R}_{\mathrm{SIGReg}}
\left(
Z^p,Z^s
\right)
+
\gamma
\mathcal{L}_{\mathrm{control}}
\left(
Z^s
\right).
$$

这一版的主张更强:

```text
not every control-sufficient variable should be forced into deterministic low-MSE rollout.
some variables should be represented as sufficient but uncertain.
```

这是 8 分以上路线,但复杂度更高,只有在 SER-aware 方法被认为不够新时再升级。

---

## 4. 可实现性分层

这里的可实现性不是代码难度,而是研究闭环能否落地。

| 层级 | 需要什么信号 | 可实现性 | 创新性 | 角色 |
| --- | --- | --- | --- | --- |
| State-SER | task state labels | 高 | 中 | 机制干预 |
| Fisher-SER | task cost gradient or finite difference | 中 | 高 | 理论主线 |
| Ranking-SER | candidate action ranking | 中 | 高 | 主方法候选 |
| PSD split | latent factor split and uncertainty planning | 中低 | 高 | 备选升级 |
| fully unsupervised SER | no task signal | 低 | 很高 | 不建议第一版押注 |

推荐路径:

```text
不要一开始追 fully unsupervised。
先用 State-SER/Fisher-SER 证明机制,
再用 Ranking-SER + placebo anchors 把方法从 "state auxiliary"
推到 "decision-aware ranking sufficiency"。
```

理由:

```text
8 分论文不要求一开始完全无监督。
它要求新 failure mode、理论对象、诊断指标和方法闭环成立。
```

---

## 5. 审稿意见到方法修改的映射

| reviewer attack | 说明 | 下一步理论修改 | 下一步方法修改 |
| --- | --- | --- | --- |
| known trick | stop-gradient 不新 | 把对象改成 PCG/SER | stop-gradient 降级为 baseline |
| toy only | theorem 太窄 | 写最小反例 theorem 和适用边界 | 用 certificate 连接真实模型 |
| auxiliary loss | 像状态监督 | 证明 high $\mathrm{SER}^{ctrl}$ 才需要保护 | 从 uniform state loss 改为 ranking-SER,并加入 placebo anchors |
| task-specific | 只适合 PushT | 用 control sensitivity 定义任务关键性 | 在多个 goal/cost 上估计 Fisher 或 ranking |
| not self-supervised | 用了 state label | 定位为 decision-aware world model | 主方法换成 ranking/cost supervision |
| not predictive | 只看 planning | 保留 Gaussian JEPA predictive core | 方法只约束高风险方向,不重建全部状态 |
| confounded evidence | 训练/评测不公平 | 重写 claim 为 hypothesis | 设计能排除 confound 的实验 gate |

这个表是迭代提分的操作手册:

```text
每收到一个低分理由,
不要先 defend。
先问它打在四个轴的哪一个短板上,
然后改变中心对象或方法对象。
```

---

## 6. 最推荐的 paper spine

如果要把方向推到 8 分,我建议主线写成:

```text
Sufficiency Erosion in Gaussian Joint-Embedding World Models
```

paper spine:

1. **Observation.**
   Lower latent self-drift can coincide with worse planning.
2. **Theory.**
   Gaussian predictive objectives select predictable coordinates under finite capacity.
3. **Gap.**
   Predictability and action-ranking control sufficiency can conflict.
4. **Certificate.**
   ranking sufficiency, Pareto-PCG, and controllability-gated SER predict which variables/checkpoints fail.
5. **Method.**
   Ranking-SER / Control-Fisher-SER protects high sensitivity, high controllability, high uncertainty directions.
6. **Evidence.**
   The method improves planning by closing PCG, not necessarily by minimizing self-drift.

一句话创新:

```text
We do not propose another rollout loss;
we identify and correct a sufficiency erosion mechanism in Gaussian JEPA world models.
```

---

## 7. 什么时候算真正提到 8 分

内部 gate:

```text
Gate 1:
  strict reviewer can no longer summarize the paper as "stop-gradient multi-step"。

Gate 2:
  theorem creates a counterexample where predictive optimality and control optimality differ。

Gate 3:
  ranking sufficiency / Pareto-PCG / controllability-gated SER predict failures
  that self-drift, probe R², inverse dynamics, and uniform uncertainty cannot predict。

Gate 4:
  method uses SER/PCG to decide which action-ranking information to preserve。

Gate 5:
  ablation shows uniform state sufficiency, wrong-variable anchors, and inverse-dynamics auxiliaries
  are weaker than Ranking-SER / Control-Fisher-SER。
```

如果五个 gate 都过,才可以诚实预期:

```text
8/10:
  new failure mode,
  formal mechanism,
  diagnostic certificate,
  method derived from the certificate,
  evidence that the certificate explains and fixes planning.
```

如果只过前两个:

```text
6/10:
  interesting theory/diagnosis, method still weak.
```

如果只跑出 stop-gradient 改善:

```text
4/10:
  useful engineering result, not enough innovation.
```

---

## 8. 下一轮最该做的思想实验

在任何实验或实现之前,先写一个 reviewer-proof toy theorem:

```text
There exists an action-conditioned whitened Gaussian control problem with latent dimension D,
where the representation minimizing Gaussian predictive MSE is not action-ranking sufficient,
and the missing direction is exactly the one with high controllability-gated SER.
```

它需要证明三件事:

1. predictive MSE 选择高 $\rho_i$ 方向。
2. planning ranking 需要高 $\omega_i\kappa_i$ 方向。
3. 当 $\rho_u>\rho_v$ 且 $\omega_u\kappa_u<\omega_v\kappa_v$ 时,
   Pareto-PCG / $\mathrm{SER}^{ctrl}$ 能提前指出 $v$ 会被牺牲。

这个 theorem 是创新性的地基。没有它,方法容易被说成调 loss。
有了它,方法才像是从一个新 failure mode 推出来的。
