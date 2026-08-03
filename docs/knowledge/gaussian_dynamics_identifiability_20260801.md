# Gaussian marginal 到底约束了什么：controlled dynamics 的不可辨识性分析

日期：2026-08-01  
配套脚本：
[`scripts/plan/gaussian_dynamics_identifiability_toy.py`](../../scripts/plan/gaussian_dynamics_identifiability_toy.py)  
原始结果：
[`gaussian_dynamics_identifiability_20260801/results.json`](gaussian_dynamics_identifiability_20260801/results.json)

## 0. 判决

用户原始直觉基本正确，但需要把“弹性”拆成三个不同问题：

1. **marginal non-identifiability**：`z_t ~ N(0,I)` 不决定
   `P(z_{t+1}|z_t,a_t)`；同一 Gaussian marginal 可以承载静止、旋转、
   非线性 twist、OU 和独立刷新等完全不同的 dynamics；
2. **Gaussian gauge freedom**：即使精确满足 full-distribution Gaussian，
   `d>=2` 时仍有无限 Gaussian-measure-preserving 非线性坐标变换；
3. **finite SIGReg slack**：实际 SIGReg 只在 finite batch、finite random
   projections、finite knots 上做检验；低维 folded sheet 或有限 codebook 可以
   与真正满维 Gaussian 得到几乎相同的 statistic。

还必须再区分第四件事：

4. **quotient / content selection**：encoder 不只是给同一状态换坐标，还会决定
   observation 中哪些因素被保留。The Obsessed Encoder 的主要失败发生在这一层：
   模型选中了容易预测的 episode key，而不是 control-sufficient state。

因此，更准确的研究命题不是：

> Gaussian manifold 中哪一个 manifold 更好？

而是：

> 在 Gaussian measure constraint 下，predictive objective 先选择了哪个
> controlled quotient，再在该 quotient 上选择了什么 gauge、metric 和
> transition semigroup？

一句话版本：

> **Gaussianity specifies latent volume, not latent dynamics. Prediction selects
> a quotient and a geometry; without control-sufficiency constraints, it can
> select a stable but semantically empty dynamics.**

---

## 1. 四个对象不能混在一起

令 observation/state 为 `X_t`，action 为 `A_t`，encoder 为 `h`：

```text
Z_t = h(X_t).
```

需要区分：

| 层 | 数学对象 | 它回答的问题 |
| --- | --- | --- |
| marginal measure | `mu_Z = Law(Z_t)` | latent 总体如何分布、是否 collapse |
| quotient/content | `sigma(h)` 或 `X / ~_h` | observation 的哪些差异被 encoder 保留 |
| controlled transition | `{P_a(dz'|z)}` | action 下 latent 如何演化 |
| metric | `g_z` 或 Euclidean latent norm | 哪些误差/位移在 loss 与 planner 中昂贵 |

SIGReg 主要观察第一层；The Obsessed Encoder 暴露第二层；JEPA prediction loss
约束第三层的一部分；latent-L2 planning 又把第四层当成决策距离。

把这四层混成“一个 Gaussian latent”会造成两个错误推论：

```text
Gaussian marginal  =>  linear-Gaussian transition       (错误)
低 prediction loss =>  control-sufficient representation (错误)
```

---

## 2. Gaussian marginal 为什么不识别 dynamics

### 2.1 transition 是 coupling，不是 marginal

假设简化为平稳 marginal：

```text
Z_t ~ gamma,     gamma = N(0,I).
```

transition kernel `P` 对应一个两时刻 coupling：

```text
Pi(dz,dz') = gamma(dz) P(dz'|z).
```

SIGReg 只检查 `Pi` 的一个 marginal `gamma`。即使再要求另一个 marginal 也是
`gamma`，所有允许 coupling 仍组成巨大集合 `Gamma(gamma,gamma)`。

在 controlled setting，行为数据只要求类似：

```text
integral gamma(dz) pi(da|z) P_a(B|z) = gamma(B).
```

这是对整族 `{P_a}` 的一个聚合约束。它不识别每个 action 的 counterfactual
transition。即便考虑更强、更简单的情形——action 外生，且每个 `P_a` 都单独保持
`gamma`——下面的不可辨识性仍然存在。

### 2.2 五个完全不同但同 marginal 的例子

对 `Z_t ~ N(0,I_2)`：

```text
identity:       Z_{t+1} = Z_t
rotation:       Z_{t+1} = Q_a Z_t
radial twist:   (r,theta)_{t+1} = (r, theta + a*c*r^2)
OU:             Z_{t+1} = rho Z_t + sqrt(1-rho^2) eps_t
refresh:        Z_{t+1} = eps_t
```

其中 `eps_t ~ N(0,I)`。五者的下一时刻 marginal 都是标准 Gaussian，但它们分别
具有：零变化、可控线性变化、强非线性可控变化、随机收缩和完全不可预测变化。

所以 `N(0,I)` 不决定：

- temporal correlation；
- deterministic/stochastic；
- action sensitivity；
- Markov kernel 或 generator；
- local Jacobian gain；
- controllability；
- off-policy transition。

### 2.3 squared prediction loss 只识别 conditional mean

encoder 固定时，one-step squared loss 的 Bayes predictor 是：

```text
f*(z,a) = E[Z_{t+1} | Z_t=z, A_t=a].
```

因此 point predictor 最多识别 observed support 上的 conditional mean；它不恢复
conditional variance、multimodality 或完整 kernel。若 behavior policy 没有
persistent excitation，未覆盖 action 的 transition 仍任意。

encoder 与 predictor 联合学习时自由度更大，因为 encoder 还能主动丢掉使未来难以
预测的变量，让 induced conditional mean 变简单。

---

## 3. 精确 Gaussian 下仍有 gauge freedom

定义标准 Gaussian 的 automorphism group：

```text
Aut(gamma) = {T invertible : T_# gamma = gamma}.
```

在 `d>=2` 时它不只包含正交矩阵。例如二维 radial twist：

```text
T(r,theta) = (r, theta + alpha(r))
```

保持半径与条件角分布，因而保持标准 Gaussian，但局部导数一般不是正交矩阵。

若真实系统确定：

```text
x' = F_a(x),
```

且 predictor 能表达任意共轭 dynamics，则：

```text
h_T = T o h
f_T = T o f o T^{-1}
```

对所有 rollout horizon 都保持零 prediction error，同时保持同一 Gaussian marginal。

这条结论有一个重要边界：**非零 residual 下，Euclidean MSE 一般不在非线性 `T`
下不变。** 因而真实 LeWM 中的 finite predictor、compression、partial
observability、soft SIGReg、optimization bias 和 recursive horizon，才会从 gauge
族中产生选择压力。

换言之：

```text
无限容量 + 确定世界 + 零误差  => dynamics 不选 gauge
有限容量/非零误差/递归闭包缺陷 => dynamics loss 开始选 gauge
```

---

## 4. 数值构造 A：finite SIGReg 能看见什么

### 4.1 protocol

配套脚本逐项复刻当前仓库 `stable_worldmodel/wm/loss.py::SIGReg` 的 statistic：

```text
latent D       = 192
batch B        = 256
random proj    = 1024
Epps-Pulley knots = 17
repeat         = 6
```

比较四种 distribution：

1. 真正 `N(0,I_192)`；
2. 3D linear sheet 随机嵌入 192D，并匹配总 variance；
3. 3D torus 经 192 个高频 Fourier feature 平滑映入 192D；
4. 12-bit key 查 4096 个 whitened Gaussian-like codeword。

第三种支持集的内在维数至多为 3；第四种只有 4096 个原子。二者都不可能是满维
Gaussian density。

### 4.2 结果

| distribution | SIGReg mean ± std | covariance participation rank | support dimension upper bound |
| --- | ---: | ---: | ---: |
| true Gaussian | `1.047 ± 0.044` | `187.6` | `192` |
| linear 3D sheet | `19.819 ± 1.068` | `3.0` | `3` |
| folded Fourier 3D sheet | `1.067 ± 0.075` | `187.5` | `3` |
| 12-bit codebook | `1.073 ± 0.062` | `187.6` | `0`（4096 atoms） |

判读：

- SIGReg 能清楚抓住显式低秩 linear collapse；
- 但 nonlinear folded sheet 与 12-bit codebook 的 statistic 几乎和真实 Gaussian
  相同；
- covariance effective rank 同样把 3D folded sheet、12-bit codebook 和真实
  192D Gaussian 都读成约 188 维；
- 因而 global covariance rank 与 finite random-projection normality 都不能替代
  local intrinsic dimension 或 information-sufficiency 检查。

这不是声称训练中的 LeWM 一定学到脚本里的 Fourier map。它是一个 existence
counterexample：**只要 encoder 有能力把低维变量非线性铺散，当前 finite SIGReg
观测本身不能排除它。**

它精确复现了 [The Obsessed Encoder](https://www.enigma.inc/posts/obsessed-encoder)
所说的 “low-dimensional sheet folded until it looks Gaussian” 的几何机制。

---

## 5. 数值构造 B：同一精确 Gaussian，不同 dynamics

### 5.1 protocol

从 100,000 个 `N(0,I_2)` 样本出发，action 独立均匀取 `{-1,+1}`。五个 transition
均由构造保证下一时刻仍精确为 `N(0,I_2)`。表中：

- `copy MSE`：用 `z_{t+1}=z_t` 的 shortcut predictor；
- `bilinear R2`：用 features `[z, a*z]` 做线性回归；
- `oracle MSE`：知道正确 conditional mean 后的 Bayes residual；
- `gain p95`：conditional-mean map 的局部最大奇异值 95% 分位。

### 5.2 结果

| dynamics | SIGReg | copy MSE / coord | action-bilinear R2 | oracle MSE / coord | gain p95 |
| --- | ---: | ---: | ---: | ---: | ---: |
| identity | `0.973` | `0.000` | `1.000` | `0.000` | `1.00` |
| controlled rotation | `1.026` | `0.996` | `1.000` | `0.000` | `1.00` |
| controlled radial twist | `1.034` | `2.196` | `0.020` | `0.000` | `14.95` |
| OU, `rho=0.85` | `1.031` | `0.299` | `0.722` | `0.278` | `0.85` |
| independent refresh | `0.896` | `1.994` | `0.000` | `1.000` | `0.00` |

所有模型的 empirical covariance relative error 都小于 `0.009`，marginal SIGReg
也在同一 finite-sample 刻度；dynamics 指标却覆盖了整个范围。

最关键的是 radial twist：

- 对 action-aware nonlinear oracle，它完全确定，Bayes MSE 为零；
- 对 `[z,a*z]` bilinear predictor，它几乎不可拟合，`R2≈0.02`；
- 它仍精确保持 Gaussian；
- 局部 gain 的 p95 约 `15`、p99 约 `23`。

所以“dynamics 好不好预测”不能脱离 predictor family 与坐标 gauge 来定义。同一个
underlying controlled flow 经 Gaussian-preserving 非线性整形后，可以从线性、稳定、
易拟合变成局部高增益、对有限 predictor 很难拟合，而 marginal 完全不变。

---

## 6. 解析构造 C：horizon 何时会放大 shortcut obsession

### 6.1 exact two-factor Gaussian model

令 shortcut `s_t` 与 task factor `x_t` 是独立的 stationary Gaussian AR(1)：

```text
s_{t+1} = rho_s s_t + sqrt(1-rho_s^2) eta_t
x_{t+1} = rho_x x_t + sqrt(1-rho_x^2) eps_t.
```

对任意 `theta`：

```text
z_theta = cos(theta) s + sin(theta) x ~ N(0,1).
```

也就是说，**这里不存在 finite SIGReg loophole；每个 mixture 都精确 Gaussian。**
Gaussian constraint 无法决定 encoder 应把一维预算给 shortcut 还是 task factor。

一个纯 factor 在第 `k` 步的 Bayes MSE 为：

```text
R_k(rho) = 1 - rho^(2k).
```

平均 horizon loss：

```text
R_K(rho) = (1/K) sum_{k=1..K} [1-rho^(2k)].
```

定义：

```text
shortcut advantage = R_K(rho_task) - R_K(rho_shortcut).
```

正值表示 prediction objective 更偏好 shortcut。

### 6.2 `K × persistence` phase slice

固定 task factor `rho_x=0.85`：

| shortcut rho | K=1 | K=2 | K=3 | K=5 | K=10 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| `0.50` | `-0.472` | `-0.466` | `-0.431` | `-0.352` | `-0.217` |
| `0.85` | `0.000` | `0.000` | `0.000` | `0.000` | `0.000` |
| `0.90` | `0.088` | `0.111` | `0.125` | `0.137` | `0.124` |
| `0.99` | `0.258` | `0.348` | `0.420` | `0.524` | `0.647` |
| `1.00` | `0.278` | `0.378` | `0.459` | `0.582` | `0.750` |

结论不是“任何 shortcut 都被大 K 放大”，而是更精确的：

- shortcut 比 task dynamics 更短命时，prediction loss 偏好 task factor；
- persistence 只略高时，优势会先增后饱和；
- episode-constant shortcut (`rho=1`) 的优势随 `K` 强烈单调增加；
- 因而 horizon 是一个 **timescale selector**，不是天然的 task-sufficiency selector。

这给 The Obsessed Encoder 的 episode-square 一个直接预测：若没有额外 sufficiency
anchor，单纯增加 open-loop horizon 很可能让 episode key 更有吸引力，而不是自动
恢复 block/contact/action dynamics。

边界同样明确：若 task state 在给定 action 后完全确定，且 predictor class 可实现，
它的 Bayes risk 也是零，shortcut 不再具有严格 risk 优势。真实 LeWM 中产生差距的是
partial observability、compression、finite predictor 和 recursive closure defect。

---

## 7. The Obsessed Encoder 与当前方向各自证明什么

[The Obsessed Encoder](https://www.enigma.inc/posts/obsessed-encoder) 的 LeWM 实验给出：

```text
episode-constant 5x5 square / RandGoal
    -> 更低 prediction loss
    -> SIGReg statistic 仍健康
    -> latent similarity 按 key 而不是 content 组织
    -> planning 接近失败
```

它主要证明：

> predictive objective 会选择错误的 content quotient，anti-collapse 不保证
> task/control sufficiency。

当前 horizon-induced geometry 证据主要证明：

> 在给定训练分布与保留内容下，recursive horizon 会改变 encoder metric、局部
> Jacobian product 和 error transport。

二者并不矛盾，但必须联合起来。一个 encoder 可以同时：

```text
传播非常稳定 + 几乎不看 action + 只编码 episode key + planning 失败。
```

因此 `rate(K)`、低 Jacobian-product gain 或低 rollout drift 只能是 stability
certificate，不能单独当作 representation sufficiency certificate。

---

## 8. “更合适的 manifold”应改写成 controlled metric-measure quotient

建议对象不是单独的 Riemannian manifold，而是：

```text
(M, gamma, {P_a}, g, O_task)
```

其中：

- `M = h(X)`：encoder 选择的 quotient/support；
- `gamma`：reference measure / latent volume；
- `{P_a}`：controlled Markov kernels 或 Koopman family；
- `g`：prediction/planning 使用的 local metric；
- `O_task`：必须可由 latent 恢复的 task observables、goal cost 或 outcome。

一个“合适”的表示至少需要同时满足：

1. **approximate Markov sufficiency**：给定 `(Z_t,A_t)` 后，history 对 relevant
   future 的额外信息很小；
2. **task sufficiency**：state/goal/contact/outcome 等决策变量能从 `Z` 恢复；
3. **action identifiability**：data 对 action 有 persistent excitation，且不同
   `P_a` 在 task-relevant directions 可分；
4. **recursive regularity**：有限时域 semigroup products 不产生灾难性 error gain；
5. **non-collapse measure constraint**：`h_#mu` 保持足够 spread，但不把这一项误当成
   content 或 dynamics certificate。

可以把局部 metric 选择写成一个规范问题。令 `W_r` 表示 recursive residual/error
Gramian，`W_u` 表示 action-induced displacement Gramian：

```text
min_{G positive definite} tr(G W_r)
s.t. tr(G W_u) >= kappa,
     log det G = c,
     task-sufficiency constraints.
```

这里 `log det G=c` 类似固定局部 volume budget；`W_r` 要求压低误差；`W_u` 与 task
约束防止得到“稳定但不受控”的 shortcut metric。

坐标变换下，单个 Euclidean Jacobian norm 会变化；但 generalized pencil：

```text
W_u v = lambda W_r v
```

的 generalized eigenvalues 在可逆 congruence 下保持不变，因此比单独报告某个
坐标中的 Jacobian norm 更接近 controlled-geometry invariant。

在全局 operator 层，对 Gaussian-preserving `T`，controlled Koopman family 发生
unitary conjugacy。其谱等 conjugacy invariants 描述“同一 dynamics”；Euclidean
latent metric 和 local gain 则描述选定 gauge 对 planning 的影响。两者需要分开。

---

## 9. 下一步真实 LeWM 实验：`K × shortcut persistence`

> **2026-08-03 更正：**下面的完整网格（`K∈{1,2,3,5,10}` × 4 档 persistence ×
> `seed>=3` = 60 次训练）买不起，且这一步是**条件性的**——只在 G2 判决通过后才做。
> 削减后的角点设计（`K∈{1,5}` × `persistence∈{frame,episode}` × 3 seed = 12 次）
> 与它在整体顺序中的位置见
> [执行计划](controlled_metric_execution_plan_20260803.md)。

本文件已经完成的是机制级 existence test 与解析 phase slice，**不是**真实 LeWM
训练结果。下一步最小因果实验应复用公开 square protocol：

```text
K in {1, 2, 3, 5, 10}
persistence in {frame, 2 frames, 5 frames, episode}
seed >= 3
```

固定 renderer、5x5 大小、颜色分布、训练数据和 eval tag mode，只改变 key 的时间
持续长度。每个 checkpoint 统一报告：

### content / quotient

- `S_content / S_key / S_null` factor-swap cosine；
- block pose、agent pose、goal pose、contact/state probes；
- local intrinsic dimension；
- codebook/key decoding accuracy。

### marginal

- train directions 与 fresh directions 的 SIGReg；
- covariance participation rank；
- 二维 joint projection test；
- duplicate/nearest-neighbor structure。

### controlled dynamics

- one-step fidelity；
- paired-action effect 与 `||partial f/partial a||`；
- recursive Jacobian-product rate；
- state-error transport vs action-signal transport；
- observed-support 与 counterfactual-action probes。

### decision

- goal 25/40/60 candidate rank inversion；
- oracle terminal-cost correlation；
- CEM planning success。

预注册判据：

| 结果 | 判决 |
| --- | --- |
| K 越大，episode obsession 越强 | horizon 是 persistence selector；方法必须加入显式 sufficiency anchor |
| K 越大，obsession 减弱且 action/state probes 上升 | recursive closure 提供了额外 controlled-dynamics 选择力 |
| gain 下降但 probes/planning 不升 | 只得到稳定性，不得到 sufficiency |
| SIGReg/effrank 健康但 local ID/key dominance 异常 | 再次确认 finite marginal diagnostics 不足 |

最重要的 kill criterion：

> 如果 K-step 模型在 episode-square arm 上取得更低 error-product gain，却同时更强地
> 编码 key、丢失 state/action 并失败于 planning，那么“horizon 自动选择更合适的
> dynamics manifold”这一无条件说法被证伪。幸存版本必须写成：horizon 在给定
> sufficiency anchor 后选择更适合 recursive controlled composition 的 metric。

---

## 10. 复现

从仓库根目录运行：

```bash
/Users/wge/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3 \
  scripts/plan/gaussian_dynamics_identifiability_toy.py
```

脚本只依赖 NumPy，默认写出：

```text
docs/knowledge/gaussian_dynamics_identifiability_20260801/results.json
```

当前结果使用 seed `20260801`。所有表格数字都来自该 JSON；没有手工改写实验值。

## 11. 最终 research claim 候选

最稳、也最接近现有证据的版本是：

> **Marginal Gaussian regularization leaves both content allocation and
> controlled transition geometry underdetermined. Recursive prediction acts
> as a horizon-dependent selector over this solution class: it can improve
> error transport within a control-sufficient quotient, but it can also favor
> temporally persistent shortcuts. A useful latent world model therefore
> requires joint certificates of measure spread, task sufficiency, action
> identifiability, and finite-horizon dynamical regularity.**
