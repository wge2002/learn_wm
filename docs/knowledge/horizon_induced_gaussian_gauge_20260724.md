# Gaussian Measure, Controlled Metric

## Horizon-Induced Gauge Fixing in JEPA World Models（方向收敛稿，2026-07-24）

> **判决：继续这条线，但把对象从“可塑的 Gaussian latent 流形”收紧为
> “近 Gaussian 表示解集中的 horizon-induced metric / gauge selection”。**
>
> 最短主张：
>
> **SIGReg 主要固定 latent 的边缘测度；带自复合梯度的有限时域预测，在剩余自由度中
> 选择 encoder 所诱导的局部度量，使模型残差的多步放大变便宜，同时尽量保留动作响应
> 和任务状态。**

这是一条可以同时容纳现有理论、实验和负结果的方向。它不是“再加一个稳定谱正则”，
也不是“Gaussian latent 天然就是好 manifold”。更准确的论文对象是：

```text
population measure constraint
        × latent/physical dimension mismatch
        × finite predictor class
        × recursive horizon
        × action excitation
        ↓
horizon-selected controlled metric
```

方法层暂时不发明复杂新模型。主方法仍是已经在跑的
`full one-step anchor + dose-weighted recursive open-loop loss + SIGReg`；
新的理论负责解释它到底在选择什么，新的证书负责判定这种选择是实质改进还是坐标游戏。

---

## 1. 先把原始直觉校准到不会被一个定理推翻

### 1.1 “Gaussian manifold”混合了两个不同对象

需要区分：

1. **latent support**：`z = h(x)` 在 `R^d` 中实际落在哪个集合上；
2. **representation solution set**：所有能把训练 loss 做低的 encoder/predictor 对
   `(h,f)` 组成的函数空间集合。

用户直觉中的“很多、可以弹性塑形”主要是第二个对象。本文称它为
**near-Gaussian gauge tube**：

```text
G_eps = { h : D_SIG(h#mu, N(0,I_d)) <= eps }.
```

一般不能未经证明就把 `G_eps` 叫光滑流形；它可能分层、奇异、含不同信息量的解。
只有在等维、可逆、精确 Gaussianization 的理想情形，才有一个干净的 gauge orbit。

### 1.2 精确 LeJEPA/OU 边界里，并不存在我们想象的无限最优解

[When Does LeJEPA Learn a World Model?](https://arxiv.org/abs/2605.26379)
证明：在匹配维度、Gaussian latent、平稳独立 additive-noise/OU 正对、精确
Gaussian regularization 的 population optimum 下，alignment 会把所有非线性
Gaussian-preserving 变换排除，最优表示只剩正交变换。

所以不能把论文主张写成：

> “LeJEPA 的精确 loss 天生有无限多个同样好的非线性 Gaussian 解。”

正确说法是：

> **单独的 marginal Gaussian constraint 留有很大自由度；特定 OU alignment
> 可以把它钉死。但真实 LeWM 处在 controlled、非 OU、维度不匹配、有限容量、
> soft/sketched regularization 的区间，因此刚性定理的等号条件不成立。**

### 1.3 当前 PushT 恰好远离刚性边界

当前主配置是 `D=192`，三帧输入来自确定性模拟器，预处理只有 resize 和
ImageNet normalization，没有随机图像增强；物理状态/动作历史的维度远小于 192。
同时 SIGReg 权重有限，且每个 batch 用 1,024 个随机一维投影的 Epps–Pulley
统计量近似全分布匹配。

Cramér–Wold 定理意味着：若 population 上**所有**一维投影都精确是
`N(0,1)`，联合分布就必须是 `N(0,I)`；这里没有一个可钻的数学漏洞。实际自由度
来自目标不可达、有限投影/样本和有限权重，而不是“每个投影 Gaussian 但联合分布
可以任意”。

因此当前 setting 至少同时具备：

- ambient latent 过完备；
- population 上的精确 `N(0,I_192)` 通常不可实现；
- regularizer 是 soft、finite-sample、Monte-Carlo sketched；
- predictor 是有限 Transformer 类；
- controlled transition 非 OU，且训练动作分布不是各向同性干预；
- K-step 目标使用同一个 predictor 自复合。

这才是“弹性”的真实来源。

---

## 2. 第一组边界命题：自由度到底从哪里来

### 命题 A：过完备 deterministic encoder 不可能精确产生满维 Gaussian

设数据测度 `mu` 支持在 `m` 维光滑流形 `M` 上，encoder
`h : M -> R^d` 局部 Lipschitz，且 `m < d`。则：

```text
dim_H h(M) <= m < d,
```

所以 `h#mu` 支持在一个 Lebesgue 零测集上，和具有满维密度的
`N(0,I_d)` 互相奇异。故：

```text
h#mu != N(0,I_d).
```

这条命题的意义不是否定 SIGReg。SIGReg 仍然是有效的 anti-collapse 压力；它说明
在过完备 deterministic world 中，SIGReg 的 population target 是软预算，不是一个
能被严格满足的几何等式。

维度关系应分三种：

| regime | 表示问题 |
| --- | --- |
| `d = m`，可逆 | Gaussian-preserving gauge；研究坐标/metric 选择 |
| `d < m`，压缩 | 不同 encoder fiber 丢掉不同信息；研究 predictive/control sufficiency |
| `d > m`，过完备 | 精确满维 Gaussian 不可达；研究 soft regularizer 下的 embedding/metric allocation |

当前 `D=192` 主要属于第三种；`D=8` 更接近第二种。二者不应被同一个
“纯 gauge”故事硬解释。

### 命题 B：等维精确 Gaussianization 的 gauge group

设 `m=d`，`h_1,h_2` 都是微分同胚，且

```text
(h_1)#mu = (h_2)#mu = gamma_d = N(0,I_d).
```

则

```text
T = h_2 o h_1^{-1}
```

满足 `T#gamma_d = gamma_d`。所有精确 Gaussianizer 由

```text
Aut(gamma_d) = { T : T#gamma_d = gamma_d }
```

作用连接。

在一维，双射连续映射必单调；标准 Gaussian 的递增 automorphism 是恒等，递减情形
只剩符号翻转。维度 `d>=2` 时则有无限非线性自由度。例如在二维：

```text
T(r, theta) = (r, theta + alpha(r))
```

保持半径和角度均匀性，因此保持标准 Gaussian。取平滑非恒定
`alpha(r)=c r^2` 即得非线性可逆例子。

Gaussian-preserving 变换满足 change-of-variables 恒等式：

```text
gamma(T(z)) |det DT(z)| = gamma(z),

log |det DT(z)|
    = (||T(z)||^2 - ||z||^2) / 2.
```

它只约束局部体积因子，不固定

```text
M_T(z) = DT(z)^T DT(z)
```

的全部特征值和剪切。径向 twist 甚至满足 `||Tz||=||z||`、`det DT=1`，
仍可产生非平凡局部 metric。

这给原始直觉一个严格版本：

> **Gaussian marginal 固定的是测度/加权体积，不是唯一的局部 Riemannian metric。**

### 命题 C：纯确定性、无限容量时，dynamics 本身也不会选 gauge

若真实转移确定：

```text
x' = F(x,a),
```

且 predictor 类可实现任意共轭转移，那么对任一可逆 encoder `h`：

```text
f_h(z,a) = h(F(h^{-1}(z),a))
```

在任意 rollout horizon 上都有零预测误差。对任意
`T in Aut(gamma_d)`：

```text
h_T = T o h,
f_T(z,a) = T(f_h(T^{-1}(z),a))
```

仍然是零误差、同一 Gaussian marginal。

所以：

> **“dynamics constraint 选择了更好的 gauge”只有在过程噪声、部分可观测、
> predictor class 不闭合、压缩、soft regularization 或优化偏置至少一项存在时
> 才有内容。**

PushT 的严格 conditional-variance gate 已表明不可约随机性只有 latent scale 的
约 `1.3%–1.7%`。因此当前主要选择力不是 stochastic Bayes risk，而是
**finite-class recursive closure defect + soft Gaussian budget**。

---

## 3. 第二组命题：horizon 实际优化的是什么

### 3.1 精确的 conditional-risk 分解

令

```text
Z_t = h(X_t),
C_k = (Z_t, A_t, ..., A_{t+k-1}),
m_k(C_k) = E[Z_{t+k} | C_k].
```

对任意第 `k` 步预测器 `q_k(C_k)`，条件期望的正交投影给出：

```text
E ||Z_{t+k} - q_k(C_k)||^2
  = N_k(h) + E ||m_k(C_k) - q_k(C_k)||^2,

N_k(h) = E tr Cov(Z_{t+k} | C_k).
```

LeWM 的 `q_k` 不是独立 head，而是同一个 `f` 的递归组合 `f^(k)`。因此：

```text
L_H(h,f)
  = sum_k w_k N_k(h)
  + sum_k w_k E ||m_k(C_k) - f^(k)(Z_t,A_t:t+k-1)||^2.
```

对固定 predictor class `F` 取下确界：

```text
R_H(h;F) = N_H(h) + A_H^rec(h;F),
```

其中：

- `N_H`：表示后的不可约条件方差/部分可观测代价；
- `A_H^rec`：一个共享有限 predictor 必须闭合于自复合时的 approximation defect。

这给“horizon 选 gauge”一个精确定义：

```text
h*_H in argmin_h [
    N_H(h)
  + A_H^rec(h;F)
  + lambda D_SIG(h#mu, gamma_d)
].
```

在当前近确定 PushT 上，主要变化应来自 `A_H^rec`，而不是 `N_H`。

### 3.2 动作信号的精确三分解

若 `Z_{t+k}` 已中心化且 `Cov(Z_{t+k})=I_d`，定义：

```text
P_k = E || E[Z_{t+k}|Z_t] ||^2

U_k = E || E[Z_{t+k}|Z_t,A_t:t+k-1]
             - E[Z_{t+k}|Z_t] ||^2

N_k = E || Z_{t+k}
             - E[Z_{t+k}|Z_t,A_t:t+k-1] ||^2.
```

由嵌套条件期望的 Pythagorean identity：

```text
d = P_k + U_k + N_k.
```

并且：

```text
best MSE without actions = U_k + N_k,
best MSE with actions    = N_k,
difference               = U_k.
```

因此 `U_k` 是行为动作分布下被动作解释的未来 latent 能量。

这同时给出一个重要警告：action-conditioned prediction **不自动保证**
`U_k` 大。如果行为策略中动作几乎由状态决定，则加入动作前后的条件
sigma-algebra 几乎相同，`U_k` 可以接近零；encoder 仍可能把总方差预算分给慢但
不可控的特征。[LeJEPA identifiability paper](https://arxiv.org/abs/2605.26379)
也明确把 action-conditioned transition identifiability 留给 persistent excitation。

所以“更好的 controlled manifold”至少需要：

```text
small recursive residual amplification
+ non-trivial action-explained energy U_k
+ task/state sufficiency
```

而不是只追求 contraction。

### 3.3 线性 Gaussian 边界：为什么真正对象必须是局部非线性

现有命题保留：

```text
x' = A x + B a + eps,
Sigma = A Sigma A^T + Q,
z = W x,  W Sigma W^T = I,
F = W A W^{-1}.
```

则：

```text
F F^T = I - W Q W^T <= I,
sigma_max(F) <= 1.
```

精确线性白化后只剩正交 gauge，奇异值对正交共轭不变；`K` 没有东西可修。
因此 PushT 的 `rate(K)`、`sigma_max(P_8)` 和 K1 超临界现象必然来自：

- 非线性局部 Jacobian 场；
- 维度/信息选择；
- soft Gaussian budget；
- finite predictor approximation。

完整标量风险推导和 Prop 1–3 已在
[theory_minmodel_notes.md](theory_minmodel_notes.md) 中成立，不在这里重复。

---

## 4. 把“弹性塑形”写成一个受控 metric 变分问题

### 4.1 Encoder 选择的是 physical-state 上的 pullback metric

在局部可辨识的物理状态流形上，把 encoder 写成 `h(x)`。对小预测误差
`delta x`：

```text
||h(x + delta x) - h(x)||^2
  = delta x^T G_h(x) delta x + O(||delta x||^3),

G_h(x) = Dh(x)^T Dh(x).
```

`G_h` 是 encoder 从 latent Euclidean metric 拉回到物理状态上的
Riemannian metric；压缩时是半正定 pseudo-metric。

所以 latent prediction MSE 不是中性的：它在联合训练中同时学预测器和
“什么物理误差值得在 latent 欧氏距离里变贵”。

等维可逆情形下：

```text
rho_mu(x) = gamma_d(h(x)) |det Dh(x)|,
sqrt(det G_h(x)) = |det Dh(x)|.
```

Gaussian constraint 钉住的是与 `gamma_d(h(x))` 配对的体积密度，仍未固定
`G_h` 的方向性。于是可把核心 slogan 严格化为：

> **Gaussian fixes measure; recursive horizon selects metric.**

在过完备 PushT 中这句话是局部/近似版本，不把 soft SIGReg 冒充成硬等式。

### 4.2 有限时域 residual Gramian 与 action Gramian

沿一条基准轨迹做一阶展开：

```text
e_{j+1} = A_j e_j + r_j,
delta x_{j+1} = A_j delta x_j + B_j delta a_j.
```

令：

```text
Phi_{k,j+1} = A_{k-1} ... A_{j+1}.
```

若 `r_j` 和动作扰动零均值、跨步不相关，终点的两类协方差为：

```text
W_r^(k)
  = sum_{j<k} Phi_{k,j+1} Q_j Phi_{k,j+1}^T,

W_u^(k)
  = sum_{j<k} Phi_{k,j+1} B_j Sigma_a,j B_j^T Phi_{k,j+1}^T.
```

- `W_r`：模型残差/状态扰动经 finite horizon 放大的能量；
- `W_u`：动作扰动经同一动力学传播后产生的可控能量。

两者共享同一组 `Phi`。现有 CI-GWM veto 已看到动作切向与 top amplification
span 的 overlap 为 `0.60`，随机基线仅 `0.052`；因此它们不是两个可正交切开的
子空间，而是**同方向上的 signal/error trade-off**。

### 4.3 局部受控 metric 命题

先冻结某一终点的 `W_r > 0`、`W_u >= 0`，把 Gaussian volume 的局部影子写成
`det G = c`。定义“更好的 controlled metric”为：

```text
minimize_G    tr(G W_r)
subject to    G > 0,
              det G = c,
              tr(G W_u) >= kappa.
```

若 action-retention constraint 不活跃，AM–GM 给出唯一解：

```text
G* = alpha W_r^{-1},
alpha = (c det W_r)^(1/m).
```

也就是：高 residual-amplification 方向被压缩，低 residual 方向被扩张，同时保持
局部体积。

若动作约束活跃，KKT 条件给出：

```text
G* = alpha (W_r - eta W_u)^(-1),
eta >= 0,
```

其中 `eta` 由 `tr(G W_u)=kappa` 决定，并要求括号内正定。

这条式子是当前最像“理论主公式”的对象：

> **dynamics 不是选一个最收缩的 manifold，而是在 residual/action 的
> generalized eigen-directions 上分配 metric budget。**

它自然处理“动作与误差共享方向”，不要求 CI-GWM 那种错误的正交分解。

边界必须写清：

- 这是局部 surrogate，不保证任意 `G*(x)` 都能积分成一个全局 encoder；
- LeWM 原 loss 没有显式 `kappa`，action retention 只在充分 excitation 和
  target/state 信息压力下隐式产生；
- task relevance 还可增加一个 `tr(G W_task)>=kappa_task` 约束；
- 不同 horizon 和 action distribution 会得到不同 `G*`，不存在脱离
  `(H, pi, F, task)` 的“宇宙唯一最佳 manifold”。

### 4.4 一个不受坐标游戏影响的证书

对局部可逆坐标变换 `y=T(x)`，两类 Gramian 在同一终点按 congruence 变换：

```text
W_r -> J W_r J^T,
W_u -> J W_u J^T.
```

因此 generalized eigenvalues

```text
W_u v = lambda W_r v
```

不变。若需要数值正则，可选一个同样按 congruence 变换的 reference covariance
`W_0`，使用：

```text
CP_H
  = log det(W_u + eps W_0)
  - log det(W_r + eps W_0).
```

两个 log-determinant 中的 `2 log|det J|` 相消，`CP_H` 仍不变。

这给出关键判据：

- Euclidean `sigma_max`/trace 变化、但 controlled-predictability pencil 不变：
  主要是坐标整形；
- generalized action/residual spectrum 也改善：
  finite predictor、信息选择或真实闭合性发生了实质变化；
- D192 可以更像 gauge selection，D8 则预期出现明显 semantic selection。

在非线性轨迹上应逐终点算 generalized spectrum 后聚合标量，不能先把不同终点的
矩阵生硬平均，否则一般不再有同一个 congruence。

---

## 5. 现有证据为什么已经足以支撑这条线

以下事实来自
[Gaussian dynamics 主账本](lewm_gaussian_dynamics_direction.md)、
[最小理论笔记](theory_minmodel_notes.md) 和
[planning 状态总账](lewm_planning_status_20260721.md)。

| 证据 | 读法 |
| --- | --- |
| `rate(K)` 两训练种子均严格单调：约 `1.29 -> 1.21 -> 1.14 -> 0.99 -> 0.81` | horizon 确实连续改变 local error propagation |
| D192/D32/D8 的 `sigma_max(P8)` 均从 K1 的 `16–26` 降到 K5 的 `1.45–1.97` | 不是单一容量偶然 |
| K1↔K5：D192 Procrustes residual `0.506`，linear R² `0.857`，MLP R² `0.941`；D8 residual `0.873`，linear R² `0.365`，MLP R² `0.812` | 大容量更像非正交可逆整形，小容量更像信息选择 |
| 状态最坏误差乘积 K1→K10 降约 `37x`，动作 echo 只降约 `1.7x`，即时动作增益反升 | 不是均匀 contraction，而是 error/action 不对称 |
| `K1 / TF-K5 / sg-K5 / full-K5` 的 2×2 | 只有 self-composition gradient 到达 encoder 且 encoder/predictor 共适应时，才得到完整驯化与好 planning |
| 大 K 的 covariance trace/全局 effective rank 明显改变 | 训练确实在花 soft SIGReg budget；注意 covariance rank 不是 intrinsic dimension |
| 条件方差严格 gate 只有 latent scale 的 `1.3%–1.7%` | PushT 主机制是 deterministic finite-class closure，不是 stochastic transition |
| OGBench Cube：K5 的 H10 MSE 低 `22.5%`、H20 约低 `40%`，但 closed-loop success 持平 | 表示/rollout 改善不能越级宣称 planner 改善 |

这组证据已经排除了几个简单解释：

- 不是只把 predictor 训强；
- 不是只多看几个 teacher-forced target；
- 不是简单旋转；
- 不是均匀收缩；
- 不是随机 transition 建模；
- 不是一个标量 `rate` 就能决定 planning。

剩下最一致的解释正是：

```text
soft Gaussian solution set
    + recursive finite-class closure pressure
    + encoder/predictor co-adaptation
    -> horizon-dependent local metric and information allocation.
```

---

## 6. 方法形态：不要急着再造一个 regularizer

### 6.1 当前最合理的 drop-in objective

```text
L
  = L_1step
  + gamma * sum_{k=2..H} w_k L_openloop,k
  + lambda * L_SIGReg.
```

其中：

- `L_1step` 是 fidelity anchor，避免 sg-K5 那种“付一步代价却没买到闭合性”；
- `L_openloop` 必须把预测喂回自身，梯度必须经过 Jacobian product；
- 梯度必须到 encoder 和 predictor 两边，避免 TF/sg 两个单翼失败；
- `gamma` 是 metric-selection dose，和整数 `H` 分开；
- SIGReg 保留 anti-collapse/measure constraint，不改成 BA-GWM 对角缩放；
- 不直接正则 `sigma_max` 或 action echo，避免 EchoReg 的 lookup/flat-neighborhood
  shortcut。

论文中可把这称为 **anchor-and-dose gauge fixing**，但无需包装成新架构。

### 6.2 暂不把 Gramian 公式直接做成训练 loss

`W_r/W_u` pencil 现阶段优先作为理论对象和证书，而不是马上加一项
`logdet` loss。原因：

1. CI-GWM 已证明 action/error 共享方向，错误的显式分离会加坏 inductive bias；
2. Jacobian/Gramian estimator 很容易被局部 flatness、尺度或 lookup shortcut 欺骗；
3. 当前最强因果证据来自原生 self-composition gradient，不需要复杂辅助项；
4. 先证明原 loss 自然朝 controlled metric optimum 移动，再决定是否需要显式化。

---

## 7. 最小、可证伪的实验闭环

### Gate G0：先证明当前不是精确满维 Gaussian 刚性区

不训练新模型，对已有 D192 K1/K3/K5/K10：

1. 用 simulator physical state/history 做 local intrinsic-dimension audit；
2. 用 fresh、未参与训练的随机方向重测 Epps–Pulley，并加二维 projection/joint test；
3. 报告 local ID、global covariance effective rank、SIGReg statistic，三者不混用；
4. 明确 deterministic rendering/history 的理论维度上界。

**证伪条件**：若有效输入本身有接近 192 维的独立随机因素，且 held-out joint test
支持满维 Gaussian，则“过完备 soft tube”不是主要来源；退回等维近似分析。

### Gate G1：区分 gauge-like reshaping 与信息选择

对 matched physical states 的 K1/K5 latent：

1. 补 bidirectional held-out map，而不是只报单向 MLP R²；
2. D192 拟合受控可逆 map/flow，比较 orthogonal、linear、invertible nonlinear；
3. D8 同协议，预计可逆性明显更差；
4. 若可通过 simulator finite difference，直接估计 physical pullback metric
   `G_h = Dh^T Dh` 的特征谱、主方向和 contact/free 条件差异。

判读：

- 双向近可逆 + 非正交：gauge/metric selection；
- 单向可预测但逆向差：信息丢失/quotient selection；
- 两者都差：不能把 K1/K5 叫同一个 gauge orbit。

### Gate G2：测 controlled-predictability pencil

锁定：

- action covariance：来自相同标准化 planner perturbation；
- residual covariance：真实 one-step latent residual，而不是任意 isotropic noise；
- reference covariance `W_0`：同一 physical perturbation pushforward；
- 每条轨迹、每个终点单独计算 generalized eigenvalues/`CP_H`。

主要预测：

1. K 增大时，state-residual amplification 下降得远快于 action Gramian；
2. `CP_H` 或 generalized `action/residual` spectrum 在 K1→K3/K5 改善；
3. 改善集中在 CI veto 已发现的 shared directions，而不是正交 complement；
4. D192 的 raw Euclidean 谱变化可能大于 invariant pencil 变化；
5. D8 若发生真实 information selection，invariant pencil 也应明显改变。

**核心 kill**：若 K 只改变 Euclidean trace/singular value，而 generalized pencil
在 matched residual/action 定义下不变，则“更好的 controlled geometry”必须降级为
predictor-class-relative 的坐标整形，不能写成内在动力学改善。

### Gate G3：用已有 2×2 做因果闭环

同一证书补齐四臂：

| arm | product gradient | encoder receives it |
| --- | --- | --- |
| K1 | no | no |
| TF-K5 | no | yes |
| sg-K5 | yes | no |
| full-K5 | yes | yes |

预注册：

- 只有 full-K5 同时改善 recursive closure、保状态信息并提高受控 pencil；
- TF 主要发生无收益的信息重分配；
- sg 主要发生 predictor distortion；
- 若 TF 或 sg 和 full 一样改善 invariant pencil，则“必须 co-adapt”被证伪。

### Gate G4：方法 claim 与科学 claim 分开

科学 claim 需要：

- PushT 两训练种子；
- OGBench Cube 或另一受控 nonlinear environment 上复现 metric/pencil 定律；
- 不要求 closed-loop success 必然提高。

方法 claim 额外需要：

- anchor+dose 的补种子；
- matched-compute `K1+curvature`（Temporal Straightening）；
- matched-compute `K1+bisim`；
- far-goal/low-capacity 区域有稳定优势。

若 head-to-head 不赢，方法 claim 关闭，但理论/分析方向仍可保留；若第二环境连
metric law 都不复现，则整条方向降为 PushT case study。

---

## 8. 最新文献碰撞后的生死线

| 邻近工作 | 已经占据什么 | 本方向只能声称什么 |
| --- | --- | --- |
| [LeJEPA identifiability](https://arxiv.org/abs/2605.26379) | 匹配维度 Gaussian/OU 下的正交可辨识性；明确把维度不匹配和 controlled transition 留作开放问题 | 研究它的非理想边界：overcomplete、controlled、finite-class、recursive horizon |
| [UR-JEPA](https://arxiv.org/abs/2606.01443) | 已明确提出 SIGReg 与低维 manifold hypothesis 的张力，并用 uniform rectifiability 替换 regularizer | 不能把“Gaussian 与 manifold 冲突”当新意；我们固定 SIGReg，研究 dynamics 如何选择 metric |
| [A Control Theory of Predictability](https://arxiv.org/abs/2607.10362) | 固定 whitened encoder/线性控制 premise 下的 content–predictability frontier、non-normal tax、off-manifold reachability 和 planning gap | 不能声称首次用 operator/Gramian 解释 planning；差异是 end-to-end horizon 如何改变 encoder metric 本身 |
| [Koopman Dreamer](https://arxiv.org/abs/2607.19719) | 谱约束 latent core、bilinear action、multi-step error bound和稳定 imagination | 不能把“谱稳定 world model”当创新；我们不硬约束 predictor 谱，研究 loss-induced gauge/metric selection |
| [Temporal Straightening](https://arxiv.org/abs/2603.12231) / [GRWM](https://arxiv.org/abs/2510.26782) | on-trajectory curvature/邻近几何正则改善 latent planning 或 rollout | 必须用 head-to-head 证明 recursive closure pressure 不等价于 tangent straightening |
| [VAMP](https://arxiv.org/abs/1707.04659)、balanced realization、control contraction metrics | 最优 operator features、Gramian balancing、coordinate-invariant contraction metric 都是经典对象 | generalized pencil 不是“首次发明”；新意是它在 Gaussian JEPA gauge、self-composition gradient 和实测 2×2 中的作用 |
| [World Models as Group Actions](https://arxiv.org/abs/2605.24578) / [Delta-JEPA](https://arxiv.org/abs/2606.31232) | action composition algebra或显式 action-sensitive displacement | 我们不占 action algebra；只研究 finite-horizon action/residual 共享方向上的 metric allocation |
| [Predictive objectives discard control-relevant features](https://arxiv.org/abs/2606.30068) | reward-free predictability不保证 control/task sufficiency | 必须保留 `U_k`、task/state probe 和 persistent-excitation 边界，不能把低 residual 当充分条件 |

因此最安全的 novelty 句子是：

> **We study how recursive-horizon training fixes the encoder metric inside the
> soft, dimension-mismatched Gaussian solution set of an action-conditioned
> JEPA, and show that the selected geometry is governed by a same-direction
> action-versus-residual trade-off rather than by uniform contraction.**

不要写：

- first stable latent world model；
- first spectral/Koopman view of JEPA；
- first manifold-aware JEPA；
- lower rollout error implies better planning；
- Gaussian marginal alone leaves infinite optimal LeJEPA solutions；
- action and error occupy orthogonal subspaces。

---

## 9. 可成文的定理—实验结构

### 题目候选

```text
Gaussian Measure, Controlled Metric:
Horizon-Induced Gauge Fixing in JEPA World Models
```

或更保守：

```text
What Prediction Horizon Selects
inside Gaussian JEPA Representations
```

### 理论贡献栈

1. **Dimension-mismatch boundary**：`m<d` 的 deterministic encoder 不可能精确
   push forward 成满维 Gaussian；
2. **Gauge boundary**：等维 Gaussianizer 由 `Aut(gamma)` 连接，Gaussian 只固定
   加权 volume；给出 radial twist；
3. **No-free-selection boundary**：deterministic universal predictor 下所有 gauge
   零损失，选择力必须来自 noise/finite class/compression/soft constraint；
4. **Recursive-risk decomposition**：`N_H + A_H^rec`；
5. **Controlled ANOVA**：`d=P_k+U_k+N_k`，并指出 excitation 条件；
6. **Linear-Gaussian rigidity**：精确白化下 `sigma_max(F)<=1`、K 无效；
7. **Local controlled metric proposition**：
   `G*=alpha(W_r-eta W_u)^(-1)`；
8. **Coordinate-invariant certificate**：generalized action/residual spectrum。

其中 1–6 可写成正式命题或短证明；7 是清楚标注假设的局部变分命题；8 是线性代数
不变量。不要把尚未完成的全局 metric integrability 或非线性 identifiability
包装成 theorem。

### 实验贡献栈

1. K law：horizon 单调改变 error propagation；
2. dimension sweep：D192 gauge-like、D8 information-selective；
3. action/error asymmetry 且共享方向；
4. full/TF/sg/K1 的 2×2 因果归因；
5. invariant pencil 验证“不是纯坐标游戏”；
6. anchor+dose 对 curvature/bisim 的 matched-compute 结果；
7. 第二环境只承诺 representation/rollout law，不偷换成 planning winner。

---

## 10. 最终方向判决

这条线现在可以定为：

> **Horizon-Induced Gaussian Gauge Fixing / Controlled Metric Selection**

宏观上它回答一个清楚的问题：

> **当 anti-collapse prior 只规定 latent 应该“占据怎样的测度”，而不规定局部
> 距离应该怎样解释物理误差时，controlled recursive prediction 会选择什么 metric？**

理论上它有四个硬边界保护：

- exact OU 刚性边界；
- dimension-mismatch 不可达边界；
- deterministic universal-model gauge degeneracy；
- linear-Gaussian whitening 无 K 效应。

实证上它已经有大效应、跨种子、因果 2×2 和负结果约束；缺口不再是继续想一个
故事，而是补两个判决量：

1. **local pullback metric / bidirectional gauge map**；
2. **coordinate-invariant action/residual pencil**。

执行顺序：

> **2026-08-03 更正：下面第 1 条已作废。**D192 的 K1/K3/K5/K10 checkpoint 已经
> 不存在，任何读数之前都必须先重训最小 checkpoint 集。合并后的可执行顺序（含
> Gate A 与 G2 的从属关系、`K × persistence` 的削减网格、CritWM 处置）见
> [受控度量方向：合并后的执行计划](controlled_metric_execution_plan_20260803.md)。
> 本节以下内容保留作 gate 定义，不再作为执行顺序的依据。

1. ~~先用现有 checkpoints 完成 G0–G3，不训练新模型~~（checkpoint 已不存在）；
2. 并行等待 anchor+dose vs curvature/bisim 的已启动结果；
3. G2 通过后才把“controlled metric improvement”升为主 claim；
4. head-to-head 通过后才保留方法论文形态；
5. 任一层失败就按本文 kill rule 降级，不再发明一个新 regularizer 去救叙事。

这比“找一个更合适的 Gaussian manifold”更精确：**我们不是寻找一个静态、普适的
manifold，而是在给定 horizon、动作分布、predictor class 和任务充分性后，定义并
识别一个 dynamics-selected metric。**
