# 方法草案 v2:Control-Sufficient Gaussian JEPA

> 目的:把当前低创新性的 "stop-grad multi-step" 提升成一个更有论文对象感的方向。
> 核心不是再设计一个 rollout trick,而是定义并解决 **Gaussian JEPA 的 predictive-control gap**:
> 预测目标偏好可预测坐标,而 planning 需要控制充分坐标。
>
> 相关文档:
> [theory_predictive_sufficiency_dynamics.md](theory_predictive_sufficiency_dynamics.md),
> [theory_predictive_sufficiency_novelty_loop.md](theory_predictive_sufficiency_novelty_loop.md),
> [theory_predictive_sufficiency_acceptance_audit.md](theory_predictive_sufficiency_acceptance_audit.md)。

---

## 0. 当前定位

当前 stop-gradient multi-step 方案的角色应降级为:

```text
anti-erosion component / baseline
```

而不是主创新。主创新改成:

```text
Control-Sufficient Gaussian JEPA:
  在 Gaussian/SIGReg joint embedding world model 中,
  显式识别并保护 "高任务价值 + 高预测不确定性" 的因素,
  使 long-horizon dynamics training 不再通过改写 encoder 坐标来刷低 self-drift。
```

这比 "只训 predictor 多步" 多了三个新对象:

1. **Predictive-Control Gap (PCG)**: self-predictability 与 control sufficiency 的分离度。
2. **Sufficiency Erosion Risk (SER)**: 哪些变量/方向会被 predictive MSE 牺牲。
3. **Control-Fisher weighting**: 用任务敏感性决定哪些高不确定方向必须被保护。

---

## 1. 基本定义

设任务相关状态或可解释变量为 $Y_t$。在 PushT 中可以取:

$$
Y_t = (x^{agent}_t, x^{block}_t, \theta^{block}_t).
$$

LeWM latent:

$$
Z_t = \phi(O_{\le t}), \qquad Z_t \sim \text{approximately Gaussian by SIGReg}.
$$

Predictor 上下文:

$$
C_t=(Z_{t-h+1:t}, A_{t-h+1:t}).
$$

### 1.1 Predictability score

这里避免把低条件方差叫成真正的 statistical sufficiency。预测目标真正奖励的是
**predictability**:

$$
P_k(\phi)
=
-\mathbb{E}\,\mathrm{Tr}\,\mathrm{Var}(Z_{t+k}\mid C_t,A_{t:t+k-1}).
$$

它越大,latent 越容易被当前上下文预测。它不是控制充分性,也不保证 latent terminal
distance 能给候选动作正确排序。

### 1.2 Control sufficiency as action ranking

planning 首先需要的是 action-ranking preservation,而不是 state reconstruction。
给定同一个当前状态/goal 下的候选动作序列 $a,b$,定义真实任务 cost 和 latent cost:

$$
J_y(a)=c(S^a_{t+H},G),
\qquad
J_z(a)=\lVert \hat Z^a_{t+H}-Z_g\rVert^2.
$$

主指标应是 pairwise ranking agreement:

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

或使用 Kendall-$\tau$ / pairwise AUC。这个定义直接对应 CEM/MPC 选择动作的需求。

state decodability 只能作为便宜代理/机制诊断:

$$
S_{state}(\phi)
=
-\min_q \mathbb{E}\lVert q(Z_t)-Y_t\rVert^2.
$$

它不是充分条件:一个 latent 可以线性解码 block angle,但欧氏 goal distance 仍然给出错误的候选动作排序。

### 1.3 Predictive-Control Gap

标量差值容易因为 normalization 变得任意。因此主文更稳的写法是 Pareto 诊断:

$$
\Pi_{k,H}(\phi)
=
\left(
P_k(\phi),S_H(\phi)
\right).
$$

所谓 predictive-control gap opening 是:

$$
\Delta P_k
>
0,
\qquad
\Delta S_H
<
0,
$$

即 self-predictability 变好,action-ranking control sufficiency 变差。
若必须报告单个数,只能使用预先固定的标准化:

$$
\mathrm{PCG}_{k,H}(\phi)
=
\frac{P_k(\phi)-\mu_P}{\sigma_P}
-
\frac{S_H(\phi)-\mu_S}{\sigma_S},
$$

其中 $\mu_P,\sigma_P,\mu_S,\sigma_S$ 必须在训练前由 baseline/checkpoint 集合固定,
并报告 confidence interval。否则 PCG 只能作图,不能当作独立 claim。

这给了一个比 self-drift 更接近 planning 的诊断对象。

---

## 2. Sufficiency Erosion Risk

PCG 是 model-level 指标。我们还需要 variable/direction-level 诊断:

```text
哪些因素会被 Gaussian predictive objective 牺牲?
```

对任务变量 $Y_t$ 的某个方向 $v$ 定义:

$$
\sigma^2_{pred,k}(v)
=
\mathbb{E}\,\mathrm{Var}(v^\top Y_{t+k}\mid C_t,A_{t:t+k-1}).
$$

定义任务敏感性:

$$
\omega_H(v)
=
\mathbb{E}\left[
\left(
\frac{\partial c(S_{t+H},G)}{\partial (v^\top Y_{t+H})}
\right)^2
\right].
$$

则 sufficiency erosion risk:

$$
\mathrm{SER}_{k,H}(v)
=
\omega_H(v)\,\sigma^2_{pred,k}(v).
$$

但这还不够。审稿人会指出:某个变量可能 task-sensitive 但本质上不可控或纯外生噪声,
保护它不一定改善 planning。为避免把不可控随机性误判成 sufficiency,主方法应使用
controllability-gated SER:

$$
\kappa_H(v)
=
\mathbb{E}_{C_t,G}
\mathrm{Var}_{a\sim Q(\cdot\mid C_t,G)}
\left(
\mathbb{E}
\left[
v^\top Y_{t+H}
\mid
C_t,a
\right]
\right),
$$

$$
\mathrm{SER}^{ctrl}_{k,H}(v)
=
\omega_H(v)\,\kappa_H(v)\,\sigma^2_{pred,k}(v).
$$

其中 $Q$ 是 planning candidate distribution。$\kappa_H(v)$ 低说明候选动作几乎不能改变该方向,
即使它难预测也不该被主方法强保护。

解释:

```text
高 σ²_pred:
  这个方向难预测,预测 MSE 有压力把它弱化/替换。

高 ω:
  这个方向对 goal/control 重要,不能丢。

高 κ:
  这个方向会被候选动作实质改变,影响 action ranking。

高 SER^ctrl:
  正是 "难预测、可控、且任务关键" 的危险方向。
```

更准确地说,高 $\mathrm{SER}^{ctrl}$ 表示:

```text
该方向既影响目标排序,又被候选动作实质改变,同时对 predictive MSE 显得昂贵。
```

PushT 的预测:

```text
block angle direction has higher SER than block xy in contact windows,
and much higher SER than visual/background nuisance directions.
```

这比"角度重要"更强,因为它给出了可测量的风险量:

```text
risk = task sensitivity × action controllability × prediction uncertainty
```

---

## 3. Toy theorem:为什么 SER 会被 predictive MSE 牺牲

考虑已白化变量 $Y=(y_1,\ldots,y_m)$,latent 维度 $D<m$。
对任意方向 $v_i$,预测 explained variance 为:

$$
\rho_i
=
\frac{\mathrm{Var}(\mathbb{E}[y_i\mid C])}{\mathrm{Var}(y_i)}
=
1-\frac{\mathbb{E}\mathrm{Var}(y_i\mid C)}{\mathrm{Var}(y_i)}.
$$

在 fixed-context linear toy setting 中,MSE+whitening 会选择最大 $\rho_i$ 的方向。
任务最优 representation 则应选择高 $\omega_i\kappa_i$ 的方向。

因此存在冲突条件:

$$
\rho_u > \rho_v
\quad\text{but}\quad
\omega_u\kappa_u < \omega_v\kappa_v.
$$

此时 predictive representation 选择 $u$,control-sufficient representation 需要 $v$。
如果 $v$ 同时有高 $\omega_v\kappa_v(1-\rho_v)$,则 $v$ 是高 $\mathrm{SER}^{ctrl}$ 方向。

更贴近 planning 的 action-ranking 版本如下:

```text
There exists an action-conditioned whitened Gaussian control problem with D=1
where the predictive-MSE optimum is z=y_u, but every representation that preserves
the optimal action ranking must contain y_v.
```

构造方式:

1. $y_u$ 是高可预测、低控制价值变量,满足 $\rho_u>\rho_v$。
2. $y_v$ 是低可预测、高控制价值变量,终端 cost 只依赖 $y_v$ 与 goal 的距离。
3. 候选动作 $a,b$ 主要改变 $y_v$ 的终态,使 $J_y(a)<J_y(b)$ 当且仅当
   $y_v^a$ 更接近 goal。
4. 若 $D=1$ 且 representation 选择 $z=y_u$,则存在动作对 $a,b$ 在真实 cost 上可分,
   但 $J_z(a)=J_z(b)$ 或排序错误。

因此,预测最优和 ranking-sufficient 不能同时满足。这个 theorem 比 passive variable-selection
更强,因为它直接连接到 CEM/MPC 的候选动作排序。

**结论.**
Gaussian predictive objective 的偏置不是"丢随机信息",而是:

```text
在有限 latent/capacity 下,它优先保留高 explained variance 方向,
即使低 explained variance 方向对 control 更重要。
```

Multi-step training 把 $\rho_i$ 替换为 horizon-weighted explained variance:

$$
\bar \rho_i
=
\sum_{k=1}^K w_k \rho_{i,k}.
$$

contact/hybrid 变量通常随 $k$ 增大更难预测,所以多步训练会扩大冲突。

---

## 4. 方法:SER-aware Gaussian JEPA

目标:

```text
降低 self-drift 不能以牺牲高 SER 方向为代价。
```

提出:

$$
\mathcal{L}_{CS\text{-}GJEPA}
=
\mathcal{L}_{1step}
+\lambda \mathcal{R}_{SIGReg}
+\beta \mathcal{L}_{dyn}
+\gamma \mathcal{L}_{SER}.
$$

其中:

### 4.1 Dynamics-only rollout component

这部分只是 anti-erosion component,不是主创新:

$$
\mathcal{L}_{dyn}
=
\sum_{k=1}^{K}
w_k
\mathbb{E}
\left\lVert
f^{(k)}(\mathrm{sg}[Z_{t-h+1:t}], A_{t:t+k-1})
-
\mathrm{sg}[Z_{t+k}]
\right\rVert^2.
$$

它负责让 dynamics 在固定坐标系里滚稳,不让 multi-step gradient 改写 encoder。

### 4.2 SER sufficiency component

主方法候选应该优先是 ranking-SER,而不是 state reconstruction。给定真实任务 cost
满足 $J_y(a)<J_y(b)$ 的候选对:

$$
\mathcal{L}_{rank}
=
\mathbb{E}_{a,b}
\,
\widehat{\alpha}(a,b)
\max
\left(
0,
m+
J_z(a)-J_z(b)
\right).
$$

权重 $\widehat{\alpha}(a,b)$ 来自预先估计的 $\mathrm{SER}^{ctrl}$,更关注:

```text
high task sensitivity,
high action controllability,
high prediction uncertainty,
high ranking disagreement.
```

弱监督 state-SER 只能作为 mechanism intervention:

$$
\mathcal{L}_{SER\text{-}state}
=
\min_q
\mathbb{E}
\left[
\sum_i
\alpha_i
\left(q_i(Z_t)-Y_{t,i}\right)^2
\right],
$$

其中:

$$
\alpha_i
\propto
\widehat{\omega}_i\,
\widehat{\kappa}_i\,
\widehat{\sigma}^2_{pred,i}.
$$

state-SER 的作用是证明机制因果性:保护 high-SER 方向是否能关闭 PCG。
它不能单独作为 8 分主方法,因为审稿人会把它看成 supervised state auxiliary loss。

### 4.3 为什么不是普通 auxiliary loss

普通 state probe:

```text
保留所有 state dimensions equally。
```

SER-aware loss:

```text
只强保护 "high task sensitivity × high prediction uncertainty" 的方向。
```

它的理论含义是:

```text
不是让 representation 记住更多,
而是阻止 predictive objective 丢掉最危险的少数控制关键不确定方向。
```

---

## 5. 实验闭环

### 5.1 Toy Gaussian / hybrid environment

构造:

```text
x_easy:
  high predictability, low task sensitivity

y_hard:
  low predictability due to switching/noise, high task sensitivity

latent dimension D=1
```

预测:

| 方法 | 选择方向 | pred loss | control success |
| --- | --- | --- | --- |
| MSE+whitening | $x_easy$ | low | bad |
| MSE+SER | $y_hard$ or mixed | higher | good |
| multi-step MSE | more strongly $x_easy$ | lowest self-drift | worst |

这一步是理论可信度的核心。

### 5.2 PushT diagnosis

必须测:

1. 在看 planning success 之前预先估计 $\mathrm{SER}^{ctrl}$,并预测哪些变量/窗口会 erode。
2. $\mathrm{SER}^{ctrl}(\text{angle}) > \mathrm{SER}^{ctrl}(\text{position})$ in contact windows。
3. Multi-step co-training 主要降低 high-SER variable decodability 和 action-ranking agreement。
4. ranking sufficiency / Pareto-PCG 比 self-drift、probe R²、inverse-dynamics accuracy 更能预测 planning。
5. matched planner history 后,现象仍成立;否则旧的 82% vs 22% 只能作为 confounded seed。

### 5.3 PushT method

比较:

```text
one-step LeWM
co-trained multi-step
dynamics-only rollout
uniform state-sufficiency loss
wrong-variable / low-SER placebo anchor
inverse-dynamics or controllability auxiliary
SER-aware sufficiency loss
Ranking-SER / Control-Fisher-SER
Ranking-SER + dynamics-only rollout
```

8 分级别结果不是简单 "success rate up",而是:

```text
self-drift may not be minimum,
but action-ranking sufficiency improves,
Pareto-PCG closes,
planning recovers.
```

关键反事实:

```text
uniform state loss 或 low-SER placebo 不能同等修复 planning;
否则主方法只是普通 auxiliary supervision。
```

---

## 6. Reviewer-facing novelty claim

推荐摘要句:

```text
We identify sufficiency erosion, a failure mode of Gaussian joint-embedding
world models where lower self-referential rollout error is achieved by selecting
more predictable but less control-sufficient coordinates. We derive this as a
predictive-coordinate selection pressure under whitened Gaussian embeddings,
introduce predictive-control gap and sufficiency erosion risk as diagnostics,
and train decision-aware Gaussian JEPAs that protect directions with high task
sensitivity, high action controllability, and high prediction uncertainty.
```

比 stop-grad 方案强的地方:

| 维度 | stop-grad multi-step | CS-GJEPA / SER-aware |
| --- | --- | --- |
| 理论对象 | gradient routing | predictive-control gap |
| 方法对象 | stop-grad rollout | erosion-aware sufficiency protection |
| 新指标 | 无 | ranking sufficiency, Pareto-PCG, controllability-gated SER |
| 预测能力 | 弱 | 预测哪些变量/窗口/候选排序会失败 |
| 与相关工作的差异 | 容易被 Dreamer/Fast-LeWM 覆盖 | 聚焦 control sufficiency erosion |

---

## 7. 当前预期评分

若只写 stop-grad:

```text
4/10
```

若只补 toy theorem + PushT diagnosis,但 PCG/SER 没有预注册预测力:

```text
4-6/10
```

若 action-conditioned theorem 成立,ranking-SER/Control-Fisher-SER 能预测 failure,
且通过 matched baseline 与 placebo 排除 auxiliary-loss 解释:

```text
8/10
```

关键是证明:

```text
the new object explains something self-drift/probes/inverse dynamics cannot,
and the method uses that object to choose which action-ranking information to preserve.
```

---

## 8. 下一轮审稿问题

下一次给独立 reviewer 时,不要问 "stop-grad 方法几分"。
应该问:

```text
Given the CS-GJEPA / SER formulation:
1. Is sufficiency erosion a novel and well-defined failure mode?
2. Is controllability-gated SER = task sensitivity × action controllability × prediction uncertainty theoretically justified?
3. Would an action-conditioned ranking theorem + pre-registered Pareto-PCG/SER evidence + Ranking-SER method be enough for 8/10?
4. Does this collapse to ordinary supervised state loss or decision-aware auxiliary loss?
5. What related work would kill the novelty?
```

如果 reviewer 仍给 6,下一步才考虑升级为 PSD latent split。

---

## 9. 独立审稿 round 1 后的收敛结论

两个独立视角的共同评分都是:

```text
current score: 4/10
```

共同理由:

1. stop-gradient / dynamics-only rollout 不是新方法对象。
2. passive predictive-PCA theorem 太窄,必须升级到 action-conditioned ranking theorem。
3. scalar PCG 过于任意,应以 ranking sufficiency + Pareto-PCG 为主。
4. state-SER 容易退化成 supervised state auxiliary loss。
5. SER 必须加入 controllability / action-relevance,否则会保护不可控噪声。
6. 8 分需要 SER/PCG 预先预测 failure,并通过 placebo/baseline 证明 causal repair。

因此下一版 paper 的主对象应改成:

```text
Sufficiency Erosion in Reconstruction-Free Gaussian JEPA World Models
```

主方法应改成:

```text
Ranking-SER / Control-Fisher-SER decision-aware Gaussian JEPA
```

而不是:

```text
stop-gradient multi-step LeWM
```

与相关工作边界:

1. DreamerV3:已有 representation/dynamics stop-gradient 和 free bits。
2. Fast-LeWM:已有 LeWM 语境下 multi-horizon/action-prefix prediction。
3. LeWM:已有 next-embedding prediction + Gaussian regularization。
4. inverse-dynamics / controllability representation:已有"保留可控信息"大方向。
5. decision-focused learning / value-equivalent modeling:已有"为决策训练模型"大方向。

所以 novelty wall 必须是:

```text
reconstruction-free Gaussian JEPA 的 self-prediction objective
会产生可测的 control-sufficiency erosion;
我们给出 action-ranking certificate,并用它指导要保护的 latent 信息。
```

---

## 10. 自审:如何避免退化成普通 supervised state loss

最容易被审稿人攻击的点:

```text
你只是发现 angle 重要,然后加 angle prediction loss。
```

所以方法必须分三档写清楚,并把主贡献放在前两档之间的理论对象上。

### Tier 0:diagnostic-only SER

不训练任何新模型,只测:

```text
SER_i = task sensitivity_i × prediction uncertainty_i
PCG = predictive score - control sufficiency score
```

如果 SER/PCG 能预测:

1. 哪些变量在 multi-step co-training 中掉 probe。
2. 哪些 checkpoint planning 会崩。
3. 哪些 contact windows 最危险。

那么即使不加新 loss,也已经比普通 state probe 更强:

```text
它不是说 "state matters",
而是说 "predictive Gaussian objective will selectively erode these state factors"。
```

### Tier 1:weakly supervised SER anchor

使用环境 state,但只保护 high-SER directions:

```text
not all-state reconstruction,
only task-sensitive × prediction-uncertain directions.
```

这仍然有监督,创新性有限,但可以作为 proof-of-mechanism。

审稿定位:

```text
This is not our claim of fully self-supervised learning;
it is a controlled intervention showing that protecting predicted high-risk variables
closes the predictive-control gap.
```

### Tier 2:counterfactual/ranking SER

更强的主方法应尽量不依赖人工指定 angle:

```text
用 action candidates 的 true/estimated task ranking 来定义 control sufficiency。
用 disagreement windows 和 high uncertainty windows 自适应加权。
```

这里的 supervision 来自 task/planning objective,不是 state labels:

$$
\mathcal{L}_{SER\text{-}rank}
=
\mathbb{E}_{a,b}
\mathrm{stopgrad}(\widehat{\mathrm{SER}}(a,b))
\max(0,m+J_z(a)-J_z(b)).
$$

当 $c(a)<c(b)$ 时约束 latent ranking。这样主张变成:

```text
Gaussian JEPA needs decision-aware sufficiency where self-prediction conflicts with planning.
```

这比 state probe 更像 8 分方法,但实现更贵。

### 推荐写法

paper 中把三个层次排序为:

1. **SER/PCG diagnostics**:证明新 failure mode。
2. **State-SER intervention**:证明机制因果性。
3. **Ranking-SER method**:主方法,减少人工状态指定。

如果时间不够,至少要完成 1+2,并诚实把方法定位为:

```text
mechanism-driven intervention
```

而不是吹成 fully self-supervised universal WM。
