# 哪一种 Gaussian 几何适合动力学？

日期：2026-08-01  
状态：正向理论主线 + exact-Gaussian 数值构造  
配套脚本：
[`scripts/plan/dynamics_compatible_gaussian_geometry_toy.py`](../../scripts/plan/dynamics_compatible_gaussian_geometry_toy.py)  
原始结果：
[`dynamics_compatible_gaussian_geometry_20260801/results.json`](dynamics_compatible_gaussian_geometry_20260801/results.json)

## 0. 纠正后的问题

这里真正要研究的不是“Gaussian 约束有多弱”，而是：

> 在所有满足相同 Gaussian latent marginal 的表示中，哪一种局部几何使
> action-conditioned dynamics 更简单、更平滑、更容易递归组合，并且不牺牲
> action 与 task-relevant directions？

答案不是某个固定曲率的“Gaussian manifold”。更准确的对象是一个带 Gaussian
测度的 state-dependent pullback metric。Gaussian constraint 主要固定它的
**volume form**；dynamics 应该选择剩余的**方向、anisotropy 与沿轨迹的波动**。

本文得到的核心条件是：

```text
A_a(x)^T G(x') A_a(x) ~= rho_a(x)^2 G(x),     x' = F_a(x).       (1)
```

其中：

- `G(x) = Dh(x)^T Dh(x)` 是 encoder 的 pullback metric；
- `A_a(x) = D_x F_a(x)` 是真实 dynamics 的局部 Jacobian；
- `rho_a(x)` 允许真实的局部收缩/膨胀率。

若 `(1)` 精确成立，则 latent transition Jacobian 是 scaled orthogonal：

```text
J_f(z,a)^T J_f(z,a) = rho_a(x)^2 I.
```

这意味着 dynamics 没有额外 shear、方向条件数或 non-normal transient gain；多步
composition 只累计真实的 scalar contraction/expansion，而不累计坐标造成的伪放大。

因此一个第一原则答案是：

> **合适的 Gaussian 几何，是让 controlled flow 尽量 conformal，并在误差方向上
> 稳定、在 action/task 方向上保持可分辨的几何。**

---

## 1. Gaussian constraint 固定 volume，dynamics 选择 shape

### 1.1 pullback metric

先考虑 state intrinsic dimension 与 latent dimension 相同，encoder `h` 局部可逆：

```text
z = h(x),
G_h(x) = Dh(x)^T Dh(x).
```

物理状态的微小扰动 `dx` 在 latent 中的平方长度是：

```text
||dz||^2 = dx^T G_h(x) dx.
```

所以 encoder 不只是“抽 feature”；它为原状态空间选择了一个局部 metric。

### 1.2 exact Gaussianization 的 volume equation

设状态 density 为 `p_X(x)`，标准 Gaussian density 为 `gamma(z)`。若：

```text
h_# p_X = gamma,
```

change of variables 给出：

```text
p_X(x) = gamma(h(x)) |det Dh(x)|,

sqrt(det G_h(x)) = |det Dh(x)|
                 = p_X(x) / gamma(h(x)).                       (2)
```

令：

```text
omega(x) = p_X(x) / gamma(h(x)).
```

则可以把 metric 写成：

```text
G_h(x) = omega(x)^(2/d) S(x),
det S(x) = 1.                                                   (3)
```

`omega` 是局部 volume scale，`S(x)` 是 unit-determinant anisotropic shape。

Gaussian marginal 通过 `(2)` 强约束 volume，但没有单独固定：

- `S(x)` 的 eigenvectors；
- 各方向的 relative stretching；
- shear；
- `S(x)` 沿 trajectory 如何变化。

这就是“Gaussian 流形可以弹性塑形”的精确版本：**自由度主要住在 metric 的
traceless / unit-determinant shape field `S(x)` 中。**

### 1.3 一个重要的 integrability 边界

不能把任意 SPD field 都当作可实现 encoder：

- 当 `d_latent=d_state` 且 `h` 是 diffeomorphism，`G=Dh^T Dh` 必须是由 Euclidean
  坐标拉回的可积、内在平坦 metric；
- 当 `d_latent>d_state`，induced manifold 可以有内在/外在曲率，但 deterministic
  smooth low-dimensional support 不可能精确等于 full-dimensional Gaussian；
- LeWM 的 `D=192` 和 soft/sketched SIGReg 属于第二种近似情形。

因此理论优化最好直接写在 encoder `h` 或可积 gauge map `T` 上，而不是无约束地
优化任意 `G(x)`。

---

## 2. dynamics 如何规定 metric 的波动

### 2.1 latent Jacobian 与 metric transport

真实 controlled transition：

```text
x' = F_a(x),
A_a(x) = D_x F_a(x).
```

对应的 latent transition：

```text
f_a = h o F_a o h^{-1}.
```

其 Jacobian 为：

```text
J_f(z,a) = Dh(x') A_a(x) Dh(x)^{-1}.                            (4)
```

定义 relative metric transport tensor：

```text
C_a(x;G)
  = G(x)^(-1/2) A_a(x)^T G(x') A_a(x) G(x)^(-1/2).              (5)
```

`C_a` 的 eigenvalues 正好是 latent Jacobian 的 squared singular values。

这给出三个不同读数：

```text
volume rate : (1/2d) log det C_a
shear       : dev(log C_a)
worst gain  : (1/2) lambda_max(log C_a).
```

其中：

```text
dev(M) = M - tr(M)/d * I.
```

### 2.2 ideal geometry：scaled isometry / conformal dynamics

如果：

```text
C_a(x;G) = rho_a(x)^2 I,
```

等价于 `(1)`。此时：

- `dev(log C_a)=0`：没有 directional shear；
- `cond(J_f)=1`：所有 tangent directions 同比例变化；
- 任意多步 product 都仍是 scalar × orthogonal；
- 不会产生由坐标 shear 引起的 non-normal transient amplification。

注意目标不是强行令 `rho=1`。真实系统可能收缩、膨胀或耗散；metric 应去掉的是
**坐标诱导的额外 anisotropy**，不是抹掉真实 Lyapunov rate。

### 2.3 “几何波动适合动力学”的连续时间方程

连续 controlled flow：

```text
dx/dt = b_a(x),
J_a(x) = D b_a(x).
```

metric 沿 flow 的 Lie derivative 为：

```text
L_{b_a} G
  = (b_a · grad) G + J_a^T G + G J_a.                           (6)
```

conformal condition 变成 PDE：

```text
L_{b_a} G = 2 lambda_a(x) G.                                   (7)
```

这正面回答“metric 应该怎样波动”：

> `G` 沿 trajectory 的变化项 `(b·grad)G`，应当抵消 dynamics Jacobian 的
> traceless shear，只留下真实的 scalar expansion `lambda`。

因此 metric 不是越平越好，也不是波动越大越好。合适的波动满足：

```text
dev[G^{-1} L_{b_a} G] ~= 0.                                    (8)
```

在 contact/hybrid system 中，不同 regime 的 `J_a` 可能无法被同一个 smooth `G`
同时 conformalize。此时 `(8)` 的不可消除 residual 反而是 regime boundary 或真实
branching 的几何证据；不应通过过度收缩把它抹掉。

### 2.4 finite-horizon condition

令 action sequence 的 tangent product：

```text
Phi_{k:0} = A_{a_{k-1}}(x_{k-1}) ... A_{a_0}(x_0).
```

第 `k` 步 relative transport：

```text
C_k
  = G(x_0)^(-1/2) Phi_{k:0}^T G(x_k) Phi_{k:0} G(x_0)^(-1/2).   (9)
```

真正对应 LeWM recursive horizon 的 geometry objective 应直接控制：

```text
L_strain,H
  = E sum_{k=1..H} w_k ||dev(log(C_k + eps I))||_F^2,           (10)

L_gain,H
  = E sum_{k=1..H} w_k [log lambda_max(C_k) - tau_k]_+^2.      (11)
```

`(10)` 去 shear，`(11)` 只惩罚超过允许真实 rate 的放大。二者比无条件
`||J_f||` 收缩更安全，因为不会奖励所有 dynamics collapse 到常数。

---

## 3. 只有 conformal 还不够：noise、action、task 三个约束

### 3.1 stochastic/error-compatible metric

设物理坐标中的一步 innovation 或 predictor residual covariance 为 `Q(x,a)`。小时间
尺度上，其 latent MSE 为：

```text
E ||dz_error||^2 ~= tr(G(x') Q(x,a)).                           (12)
```

如果局部只固定 volume：

```text
det G = c,
```

则：

```text
min_G tr(G Q),  det G=c

=> G_error* = (c det Q)^(1/d) Q^{-1}.                           (13)
```

所以单纯 prediction-optimal geometry 会：

- 压缩高 residual/noise directions；
- 展开低 residual、容易预测的 directions；
- 把 residual 在 latent metric 下近似 isotropic 化。

这能让 dynamics 更好预测，但也正是它可能偏爱慢 feature/shortcut 的原因。

PushT 近确定性条件下，`Q` 主要不是物理 process noise，而是 finite predictor 的
approximation/closure residual。因此 `(13)` 得到的是**相对于 predictor family 的
geometry**，不是脱离模型架构的绝对物理几何。

### 3.2 finite-horizon error Gramian

把每步 residual 传播到 horizon endpoint：

```text
W_r,H
  = sum_j Phi_{H:j+1} Q_j Phi_{H:j+1}^T.                        (14)
```

endpoint latent error energy：

```text
E ||e_H||_G^2 = tr(G(x_H) W_r,H).                               (15)
```

无 action/task 约束时，dynamics-optimal endpoint metric 是：

```text
G_H* proportional to W_r,H^{-1}.                               (16)
```

因此 horizon 改变 geometry 的一个精确机制是：它把 one-step `Q` 换成累计、被
operator products 加权的 `W_r,H`。

### 3.3 action-compatible metric

令局部 action Jacobian：

```text
B(x,a) = partial F_a(x) / partial a.
```

action displacement 在 metric 下的能量：

```text
||dz_action||^2 ~= da^T B^T G(x') B da.                         (17)
```

有限时域 reachability Gramian：

```text
W_u,H
  = sum_j Phi_{H:j+1} B_j R_a B_j^T Phi_{H:j+1}^T.              (18)
```

如果只最小化 `(15)`，metric 可能连 action directions 一起压扁。更合适的局部规范
问题是：

```text
min_{G positive definite} tr(G W_r,H)
s.t. det G = c,
     tr(G W_u,H) >= kappa.                                      (19)
```

若 action constraint active，KKT 形式为：

```text
G* = alpha (W_r,H - eta W_u,H)^{-1},                            (20)
```

其中 `eta` 取到括号仍正定且 action floor 满足。

更稳定的分析对象是 generalized eigenproblem：

```text
W_u,H v_i = lambda_i W_r,H v_i.                                (21)
```

大 `lambda_i` 表示 action-induced signal 相对 recursive error 更强。合适的 metric
应优先保留/展开这些方向，而不是仅按 predictability 分配 latent length。

### 3.4 task-compatible metric

planning 最终还需要 task observability。若 task outcome/cost 对 state 的局部梯度或
observability Gramian 为 `W_task`，还应要求：

```text
tr(G W_task) >= kappa_task.                                    (22)
```

所以最终目标不是单一的 contraction metric，而是：

```text
low dynamical shear
+ low recursive error gain
+ non-degenerate action reachability
+ non-degenerate task observability
+ Gaussian volume constraint.                                  (23)
```

可以把它叫做 **balanced controlled Gaussian geometry**。

---

## 4. exact-Gaussian 构造：哪一个 gauge 真的让 dynamics 更好

### 4.1 构造目的

需要排除一个混淆：geometry 改善是否只是因为模型违反/花掉了 Gaussian constraint？

这里构造一个例子，所有 candidate encoder 都**精确保持标准 Gaussian**，只有
dynamics geometry 不同。

### 4.2 simple controlled dynamics

基础状态：

```text
y_t ~ N(0,I_2).
```

action `a in {-1,+1}` 选择两个 reflection axis：

```text
y_{t+1} = rho Q_a y_t + sqrt(1-rho^2) eps_t,
rho = 0.85,
Q_a^T Q_a = I.
```

所以每个 action 下都保持 `N(0,I)`，且正确坐标中的 conditional mean 是简单
bilinear map `rho Q_a y`，Bayes MSE/coordinate 为：

```text
1-rho^2 = 0.2775.
```

### 4.3 用 Gaussian-preserving twist 扭坏 observation geometry

定义 radial twist：

```text
T_c(r,theta) = (r, theta + c r^2).
```

它保持半径、Gaussian measure 和 Jacobian determinant 1。观察坐标为：

```text
x = T_1.25(y).
```

因此 `x` 仍精确 `N(0,I)`，但 controlled dynamics 在 `x` 中被非线性扭曲。

候选 encoder：

```text
z_beta = T_beta(x) = T_(beta+1.25)(y).
```

每一个 `beta` 都精确保持 `N(0,I)`。解析最优值是：

```text
beta* = -1.25,
```

因为它恰好取消 observation twist，恢复 `z=y`。

### 4.4 扫描结果

表中：

- `gauge=beta+1.25`；
- `metric-k95` 是 pullback metric `G` 的 log condition p95；
- `strain95` 是 one-step log singular-ratio p95；
- `gain1-95` 是 one-step max gain 除以真实 `rho`；
- `gainH-95` 是 H=5 product gain 除以真实 `rho^5`。

| beta | effective gauge | bilinear MSE | R2 | metric-k95 | strain95 | gain1-95 | gainH-95 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `-2.50` | `-1.25` | `1.0044` | `0.001` | `10.77` | `6.47` | `25.36` | `17.65` |
| `-2.00` | `-0.75` | `0.9963` | `0.009` | `8.76` | `5.45` | `15.26` | `10.65` |
| `-1.50` | `-0.25` | `0.7953` | `0.209` | `4.71` | `3.32` | `5.25` | `3.78` |
| **`-1.25`** | **`0.00`** | **`0.2791`** | **`0.722`** | **`0.00`** | **`0.00`** | **`1.00`** | **`1.00`** |
| `-1.00` | `0.25` | `0.7964` | `0.208` | `4.71` | `3.32` | `5.25` | `3.78` |
| `-0.50` | `0.75` | `0.9967` | `0.008` | `8.76` | `5.45` | `15.26` | `10.65` |
| `0.00` | `1.25` | `1.0044` | `0.001` | `10.77` | `6.47` | `25.36` | `17.65` |

三个完全一致的 selector 都选到解析真值 `beta=-1.25`：

```text
minimum finite-predictor MSE
minimum one-step conformal strain
minimum H=5 product gain.
```

同时：

- 所有 candidate 的 `||z||=||y||` sample-wise 完全相同；
- 所有 candidate 都精确 Gaussian-measure-preserving；
- empirical covariance error 全部 `<0.009`；
- geometry 最差时 bilinear predictor 几乎什么都学不到；
- 正确 gauge 恢复 Bayes limit `0.2775` 附近。

这个构造正面证明：

> **不是所有 Gaussian latent geometry 对 dynamics 等价。即使 marginal 完全相同，
> dynamics-compatible gauge 可以同时消除 metric fluctuation、Jacobian shear、
> finite-predictor approximation error 与多步伪放大。**

---

## 5. “什么 Gaussian 流形最好”的三个层次

### 5.1 deterministic geometry：conformal Gaussian gauge

目标：让 transition 易组合、少剪切。

```text
min_h E ||dev log C_H(h)||^2
s.t. h_# mu = gamma.                                           (24)
```

适合：近确定性环境、finite predictor、rollout stability。

风险：若单独使用，可能找到稳定但 task-empty 的 quotient。

### 5.2 prediction geometry：inverse-error Gaussian metric

目标：在固定 volume 下压低 recursive residual：

```text
G proportional to W_r,H^{-1}.                                 (25)
```

适合：异方差噪声、partial observability、model approximation error。

风险：会主动压缩难预测但 control-relevant 的 directions。

### 5.3 control geometry：balanced controlled Gaussian metric

目标：同时考虑 error、action reachability 与 task observability：

```text
min_G tr(G W_r,H) + beta * strain_H(G)
s.t. Gaussian volume,
     action floor,
     task floor,
     condition/smoothness bounds.                              (26)
```

这是 LeWM planning 最应追求的层次。它不是“越线性越好”或“越收缩越好”，而是：

> error channel 接近临界稳定，action/task channel 保持可分辨，局部 dynamics 尽量
> conformal，所有这些都在同一个 Gaussian volume budget 内实现。

---

## 6. 对 LeWM：如何实际让 dynamics 更好

### 6.1 最干净的科学实验：只学习 Gaussian-preserving gauge

先不要改 SIGReg，也不要让 encoder 随意重编码。对一个已训好的 baseline encoder
`h_0`，增加：

```text
z_0 = h_0(o),
z   = T_theta(z_0),
T_theta # N(0,I) = N(0,I) by construction.                     (27)
```

只优化 `T_theta` 与重新拟合的 predictor，使：

```text
L = L_1step
  + gamma L_H,recursive
  + beta L_strain,H
  + eta L_gain,H
  + action/task floors.                                        (28)
```

如果在 marginal 不变、content 固定的情况下，prediction/product gain/planning 同时
改善，就直接证明“Gaussian gauge fixing 能改善 dynamics”。这比让整个 encoder
联合训练更能隔离几何机制。

### 6.2 `T_theta` 的可实现参数化

最低风险的第一版可以用 radius-conditioned orthogonal flow：

```text
T_theta(z) = Q_theta(||z||) z,
Q_theta(r)^T Q_theta(r) = I.                                   (29)
```

它逐样本保持 `||z||`，并保持每个 radius shell 上的 uniform angular measure；适当
构造下保持标准 Gaussian。可以将 192 维分成 2D/4D blocks，用小网络产生每个 block
的 skew-symmetric generator，再指数映射成 rotation。

更一般的 Gaussian-preserving flow 可由满足：

```text
div v(z) - z^T v(z) = 0                                       (30)
```

的 vector field 积分得到，但第一版没有必要上这么强的参数化。

### 6.3 不显式求 full Jacobian 的 strain estimator

192D 下直接形成 `J_f` 太贵。可以用 JVP/VJP 估计：

```text
g_i = ||J_f v_i|| / ||v_i||,   v_i ~ random unit direction.
```

近似 objective：

```text
L_iso  = Var_i(log g_i),
L_tail = [max_i log g_i - tau]_+^2.                            (31)
```

但随机方向在高维中看不到 `sigma_max`，因此还要加入 2–4 步 power iteration / greedy
renormalized echo，估计 worst-direction gain。对 horizon product 直接通过 recursive
JVP 传播 perturbation。

### 6.4 action floor 不能省

同时测/约束：

```text
g_action = ||f(z,a+da)-f(z,a-da)|| / ||da||,
```

或者 action Gramian 的低分位 generalized SNR。任何 strain/gain 改善若伴随
`g_action`、state probe、candidate ranking 下降，都不算 dynamics 变好。

### 6.5 一个可执行的三阶段 Gate

> **2026-08-03 更正：**下面的 Gate A 是**筛子，不是判决**。它读的是 Euclidean 谱，
> 而 gauge 文档 G2 的 kill criterion 明确说*只有* Euclidean 变化不构成改善。
> Gate A 唯一需要回答的是「shear 下降与 uniform contraction 是否分离」；真正的
> 判决在坐标不变的 G2。合并后的执行顺序见
> [执行计划](controlled_metric_execution_plan_20260803.md)。

#### Gate A：post-hoc metric readout

对 K1/K5 checkpoint 测：

- one-step/H-step `log singular spread`；
- product max gain；
- action/error generalized spectrum；
- 按 free/contact regime 分层的 strain residual。

目的：验证 K5 是否真的更接近 conformal/balanced geometry，而不只是整体收缩。

#### Gate B：frozen-encoder Gaussian gauge fitting

固定 `h_0`，只训练 `T_theta + predictor`：

- `T=identity`；
- Gaussian-preserving `T_theta` + one-step；
- Gaussian-preserving `T_theta` + H-step/strain；
- unconstrained flow control。

目的：隔离 exact gauge selection 是否足以复现 K5 的 gain 与 planning 改善。

#### Gate C：end-to-end dose

若 Gate B 成立，再把 gauge layer 与 encoder 小剂量联合训练，保留：

```text
full one-step anchor
+ recursive horizon dose
+ action/task floors
+ SIGReg.
```

目的：允许 content 与 geometry 小幅共适应，但不让方法退化成任意多 loss 堆叠。

---

## 7. 一个更规范的总优化问题

令 predictor family 为 `F`，horizon 为 `H`，planner-query action distribution 为
`nu`。适合当前问题的定义可以写成：

```text
(h*,f*) = argmin_{h,f in F}
    E_{tau,a~nu} [
        L_prediction,H(h,f)
      + beta  L_conformal_strain,H(h,f)
      + eta   L_positive_gain,H(h,f)
    ]

s.t.
    D(h_# mu, N(0,I)) <= eps_gaussian,
    ActionSNR_H(h,f) >= kappa_action,
    TaskSufficiency(h) >= kappa_task.                           (32)
```

这里“最好”不是绝对的，而是相对于：

- predictor architecture/capacity；
- rollout horizon；
- behavior/planner action distribution；
- task observable；
- 允许的 Gaussian slack。

若 predictor 是无限容量、环境完全确定且 encoder 可逆，所有 gauge 都能达到零
prediction error；此时必须靠 strain、complexity、robustness 或 planning metric
定义“更好”。所以一个 dynamics-compatible manifold 永远隐含着 model class 与
finite-horizon use case。

---

## 8. 对现有 horizon-induced 方向的修正

原来的表述：

> horizon 在 Gaussian freedom 中选择更稳定的 metric。

现在可以改得更具体：

> horizon reweights the pullback metric toward a finite-horizon solution of
> the controlled conformal/balancing problem: it suppresses recursive shear
> and residual amplification while preserving action- and task-observable
> directions under a Gaussian volume constraint.

这产生四个可以区分的 empirical predictions：

1. K 增大时，不只是 `sigma_max(P_H)` 降；`dev log(P_H^T P_H)` 也应下降；
2. 若只是 uniform contraction，action Gramian 会同步下降；真正 balanced geometry
   应让 error gain 降得远多于 action gain；
3. dynamics-compatible gauge 应降低固定 predictor family 的 refit error，而不是
   只改变 encoder probe；
4. contact boundary 附近可能保留较高 irreducible strain；若所有 regime 都被压平，
   可能是 control information collapse。

---

## 9. 判决与主张候选

最简洁的理论判决：

> Gaussianization fixes a latent measure but leaves a state-dependent shape
> field. Dynamics-compatible representation learning should choose this field
> so that controlled transitions are approximately conformal, recursive
> residuals are whitened, and action/task directions remain observable.

更贴近 LeWM 的主张：

> **A good Gaussian latent is not the one with the smallest prediction error
> alone. It is the Gaussian gauge whose pullback metric minimizes finite-
> horizon dynamical shear and error amplification subject to control and task
> sufficiency.**

本文的 exact-Gaussian toy 已经证明该选择问题非空：在 marginal 完全不变时，正确
gauge 同时把 bilinear prediction MSE 从约 `1.00` 降到 Bayes limit `0.279`，把
one-step normalized p95 gain 从 `25.36` 降到 `1`，把 H=5 product excess gain 从
`17.65` 降到 `1`。

这才是“分析什么 Gaussian 流形更适合动力学”的正向版本。
