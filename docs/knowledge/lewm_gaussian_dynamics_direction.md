# LeWM 线性高斯与动力学表示:文献补充和新方向(2026-07-06)

> 目的:把最近 LeWM/JEPA-WM 相关工作和我们已有实验对齐,重新提出一个比
> "delta loss / 多步 rollout / 收缩正则"更底层的问题:
> **SIGReg 只把 latent 边缘分布钉成 Gaussian;多步预测到底把这个 Gaussian
> latent 的动力学坐标系改成了什么?**
>
> 本文分两部分:
> 1. 补充文献:这些工作分别占掉了哪些局部改法,以及它们没有分析什么。
> 2. 当前思考和方法:从我们已有结果出发,把问题重构成 controlled Gaussian JEPA
> 的动力学表示理论与证书。

---

## Part I. 补充文献:它们占了什么,没占什么

### 1. LeWorldModel / LeJEPA 理论:Gaussian marginal 是基础,但不是 dynamics theory

**LeWorldModel** ([arXiv:2603.19312](https://arxiv.org/abs/2603.19312))
给出当前基线:端到端 JEPA world model,用 next-embedding MSE + SIGReg,不依赖
reconstruction / pretrained frozen encoder / EMA teacher。它的核心优点是简单稳定:
encoder 直接学习 planning latent,predictor 学 latent dynamics,CEM 用 latent L2
做 goal cost。

它没有回答的问题是:当训练目标从 `K=1` 改成 `K=5` 时,encoder 为什么会变?
论文主要把 latent 当作模型接口,没有把 "Gaussian latent 的动力学坐标选择"当作
理论对象。

**When Does LeJEPA Learn a World Model?**
([arXiv:2605.26379](https://arxiv.org/abs/2605.26379)) 是最接近理论根的工作。
它证明 Gaussian marginal 在 LeJEPA 里有特殊地位:在合适条件下,Gaussian 是支持
线性可辨识性和 latent planning 的关键分布选择。直觉上,SIGReg 不只是防 collapse,
还给 representation 一个可分析的线性 gauge。

但它主要讨论 state-side / one-step / 受限动力学设定。对 LeWM 最关键的几个问题
仍然空着:

- action-conditioned transition `P_a` 怎么进入理论;
- 多步目标优化的是 transition product,不是单步 transition;
- encoder 与 predictor 联合训练时,`K` 如何选择 Gaussian latent 的动力学坐标;
- `z_t ~ N(0,I)` 不等于 `z_{t+1}=Az_t+Ba_t+eps` 是线性高斯动力系统。

因此 LeJEPA theory 是我们的地基,但不是终点。

### 2. Delta-JEPA / sensorimotor inverse dynamics:占掉一阶 action sensitivity

**Delta-JEPA** ([arXiv:2606.31232](https://arxiv.org/abs/2606.31232))
把 latent difference `z_{t+1}-z_t` 拿来解码 action,从而迫使 latent transition
携带 action 信息。它直接回应 LeWM 的一个弱点:只预测 future latent 可能学到
不够 action-sensitive 的表征。

这条线的本质是一阶局部约束:

```text
delta z_t = z_{t+1} - z_t  should reveal a_t
```

它有工程价值,但没有分析:

- `K=5` 为什么主要改变 encoder 而不是 predictor;
- action-conditioned Jacobian product 的谱/增益如何变;
- 同样 Gaussian marginal 下,不同 horizon 选择了什么动力学坐标;
- 为什么单步 MSE 几乎不变,但 open-loop composition 差 2 倍。

类似地,**Sensorimotor World Models** 一类 inverse dynamics regularization 工作
也在强调 "perception for action"。这些工作和 Delta-JEPA 一起说明:
**再做一个 action inverse / delta decoder 已经不是足够新的主线。**

### 3. Fast-LeWM / multi-horizon / prefix prediction:占掉接口和效率改法

**Fast-LeWorldModel** ([arXiv:2606.26217](https://arxiv.org/abs/2606.26217))
用 action-prefix prediction 替代 autoregressive latent rollout。它不是一步一步
滚 `z_{t+1}, z_{t+2}, ...`,而是并行预测 action prefix 执行后的 future latent。
这同时减少 rollout latency 和 compounding error,还可以加 self-consistency。

这占掉了最自然的工程路线:

```text
multi-horizon supervision
non-autoregressive rollout
faster CEM
lower long-horizon latent error
```

但它仍然把问题当成 "更好/更快地预测 future latent"。它没有问:
多步目标为什么会重写 Gaussian latent 坐标系,也没有把 encoder-side 的动力学几何
作为分析对象。

### 4. Sub-JEPA:占掉 Gaussian regularizer 的局部修改

**Sub-JEPA** ([arXiv:2605.09241](https://arxiv.org/abs/2605.09241))
认为全空间 isotropic Gaussian 约束可能过强,改成随机低维子空间上的 Gaussian
约束。这是在 SIGReg / marginal regularization 层面的改造。

它提醒我们:Gaussian regularizer 本身也有 bias-variance 问题。但它没有分析
action-conditioned dynamics,也没有解释 horizon loss 如何在 Gaussian gauge 的自由度中
选择可复合坐标。

所以我们不应把主线写成 "换一个 Gaussian regularizer"。更关键的是:

```text
给定 Gaussian marginal,不同 prediction horizon 如何选择 dynamics geometry?
```

### 5. Temporal Straightening:接近 dynamics,但不在 LeWM/SIGReg 问题上

**Temporal Straightening** ([arXiv:2603.12231](https://arxiv.org/abs/2603.12231))
提出让 latent trajectory 更直,并分析 linear dynamics 下 planning Hessian 的性质。
这比 delta loss 更接近 "动力学几何"。

但它和我们的目标仍不同:

- 它不是 reconstruction-free Gaussian JEPA 的 encoder-predictor co-training;
- 它没有解释 SIGReg marginal 与 finite-horizon transition geometry 的关系;
- 它没有做冻结 encoder + refit predictor 的归因;
- 它关注 trajectory straightness,而我们关心 horizon objective 如何选择 Gaussian
  latent 的 operator/Jacobian geometry。

这篇适合放 related work,但不会覆盖我们的核心问题。

### 6. RC-aux / TRM:占掉 planner-facing metric 修复

**RC-aux: Predictive but Not Plannable**
([arXiv:2605.07278](https://arxiv.org/abs/2605.07278)) 和
**TRM: Beyond Euclidean Proximity**
([arXiv:2605.22164](https://arxiv.org/abs/2605.22164)) 都在指出:
latent L2 不等于 planner 真正需要的 reachability / trajectory metric。

这条线很重要,但它的对象是 planner-facing cost/metric:

```text
given latent, how should planner measure distance/reachability?
```

我们的对象更底层:

```text
given SIGReg Gaussian latent, how does horizon prediction choose the latent dynamics itself?
```

换句话说,RC-aux/TRM 是 "修 planner 读 latent 的方式";我们想分析的是
"训练目标如何改变 latent 本身"。

### 7. Predictive Objectives Discard Exogenous Control-Relevant Features

**Predictive Objectives Discard Exogenous Control-Relevant Features**
([arXiv:2606.30068](https://arxiv.org/abs/2606.30068)) 证明 predictive objective
会丢掉不可预测但控制相关的外生变量,少量 reward/task label 可以救回来。

这和我们早期 "sufficiency erosion" 直觉接近,但它关注的是 feature retention:

```text
predictive objective may discard control-relevant variables
```

我们现在看到的现象更细:

- `D=8 K=5` 的 angle linear probe 崩了,planning 却更好;
- K-step 的收益主要在 encoder-side composition geometry;
- 不是简单 "丢了控制信息",而是 "线性可解码性/单步可预测性/可规划性"三者解耦。

因此这篇是强 related work,但没有覆盖 multi-step Gaussian dynamics selection。

### 8. AdaJEPA / test-time adaptation:占掉被动在线更新

**AdaJEPA** ([arXiv:2606.32026](https://arxiv.org/abs/2606.32026)) 在 MPC 过程中
用新观测 transition 做 test-time self-supervised update。它代表了 "world model
部署时继续适应" 这条路线。

但它仍然假设 self-supervised prediction update 是有益信号。我们的问题更前置:

```text
prediction loss 到底在 latent geometry 上施加什么选择压力?
```

如果这个问题没有回答,test-time prediction update 也可能只是在部署时继续重写
一个未被理解的 Gaussian dynamics coordinate。

### 9. ScratchWorld / WorldModelGym / phase-transition 评测趋势

**ScratchWorld**
([arXiv:2606.31689](https://arxiv.org/abs/2606.31689)) 和
**WorldModelGym**
([Reka blog](https://www.reka.ai/news/worldmodelgym)) 说明评测趋势正在从
visual fidelity / full-state overlap 转向 decision fidelity / executable consequence。

**World-Model Collapse as a Phase Transition**
([arXiv:2606.31399](https://arxiv.org/abs/2606.31399)) 则把 world model failure
描述成 horizon / state load 附近的临界崩塌。

这些趋势和我们的相图资产契合,但它们仍是评测/现象层。我们的潜在贡献可以是:
解释为什么某些 Gaussian JEPA latent 在容量和 horizon 压力下会进入不同的
dynamics coordinate regime。

---

## Part II. 当前思考和方法:Controlled Gaussian JEPA 的动力学表示理论

### 1. 从已有实验抽出的真实现象

当前最稳的实验事实不是 "K=5 drift 更低" 这么简单,而是下面四件事咬在一起:

1. **K-step 收益主要在 encoder,不是 predictor。**
   冻结 `phi_K1` / `phi_K5` 后从零 refit 同容量 predictor,`phi_K5` 仍然显著更低
   drift。说明多步共训买到的是 latent coordinate,不是单纯更会 rollout 的 `f`。

2. **单步可预测性几乎不变,复合性大变。**
   `phi_K1` 与 `phi_K5` 的 teacher-forced one-step MSE 接近,但 open-loop composition
   差约 2 倍。这排除了 "只是下一步预测更准" 的解释。

3. **SIGReg Gaussian 没有固定 dynamics gauge。**
   两个 encoder 都可以让 marginal latent 接近 Gaussian,但它们的 perturbation growth /
   refit-D* / planning 完全不同。也就是说,`z_t ~ N(0,I)` 只是边缘几何,不是动力学几何。

4. **容量压缩下,K-step 的保护作用单调放大。**
   `D=8 K=5` 甚至丢掉 angle 的线性可解码性,planning 仍显著强于 `D=8 K=1`。
   这说明 "保留可线性解码状态" 不是正确目标;更关键的是 finite-horizon composition
   geometry。

一句话:

> 多步训练不是把 LeWM 的 predictor 训练得更强,而是在 SIGReg Gaussian 的自由度里,
> 选择了一个更适合 action-conditioned finite-horizon composition 的 latent 坐标系。

### 2. 核心重构:Gaussian marginal 与 linear-Gaussian dynamics 不是一回事

LeWM/SIGReg 约束的是边缘分布:

```text
z_t = phi(o_t),        z_t marginally close to N(0, I)
```

但这不推出:

```text
z_{t+1} = A z_t + B a_t + eps,      eps ~ Gaussian
```

也不推出:

```text
same Gaussian marginal => same rollout geometry
```

真正发生的是:

```text
phi chooses a Gaussianized coordinate system;
prediction loss chooses which Gaussianized coordinate is useful for dynamics.
```

因此 "线性高斯" 在 LeWM 里应该拆成两层:

1. **Marginal Gaussianity**:SIGReg/LeJEPA 保证 `z_t` 的边缘近似标准高斯,防 collapse,
   给 linear identifiability 一个地基。
2. **Dynamical Gaussian gauge**:在所有满足 marginal Gaussian 的表示中,不同 horizon
   loss 选择不同的 action-conditioned transition geometry。

我们的实验说明,K-step loss 选择的不是更好的 one-step coordinate,而是更低
finite-horizon gain / 更可复合的 coordinate。

### 3. 理论对象:从单个 transition operator 到 action-conditioned operator product

LeJEPA theory 可以被理解为分析某种 transition operator 的 predictable modes。
在简单 OU / additive-noise 设定中,慢变量或低阶模式更容易被 one-step alignment 选出来。

但 LeWM 是 controlled dynamics。应该写成一族 action-conditioned operators:

```text
P_a h(s) = E[h(s_{t+1}) | s_t=s, a_t=a]
```

行为数据下的单步训练近似看到的是混合 operator:

```text
P_pi = E_{a ~ pi(.|s)} P_a
```

而多步 open-loop training 看到的是 operator product:

```text
P_{a_{t+K-1}} ... P_{a_t}
```

这就是关键差异。`K=1` 关心 one-step predictable modes;`K>1` 关心 product
在 finite horizon 下的增益、非正规放大、误差复合、action sequence sensitivity。

对应到局部微分形式,predictor rollout 的误差递推近似是:

```text
delta_{t+k+1} ~= J_f(z_{t+k}, a_{t+k}) delta_{t+k} + epsilon_{t+k}
```

所以 K-step objective 显式关心:

```text
J_{K:1} = J_f(z_{t+K-1},a_{t+K-1}) ... J_f(z_t,a_t)
```

而单步 objective 基本只看每一步的 local fit。多步训练会给 encoder 梯度,让它重写
coordinate,使这些 Jacobian products 的 gain 变小或更少非正规放大。

这就是我们想要的理论命题:

> **Horizon-weighted spectral filtering:** 在 Gaussian marginal constraint 下,
> K-step prediction 不是简单增加监督步数,而是对 action-conditioned operator product
> 的谱/奇异值/非正规增益施加选择压力,从而重写 latent coordinate。

### 4. "K-step 后 Gaussian 变成什么了?"

更准确的回答:

> 它仍是 marginal Gaussian,但不再应该被理解成 "线性可辨识状态坐标";
> 它变成了一个 **finite-horizon dynamical normal form**:
> 在边缘上保持 isotropic Gaussian,在动力学上把高增益/难复合方向重新分配、
> 混合或压扁,使 open-loop rollout 的误差传播更稳定。

这解释了几个看似矛盾的结果:

- angle R² 可以下降,因为线性 probe 不是目标;
- planning 可以上升,因为 finite-horizon ranking 更稳定;
- self-drift 跨模型不可比,因为不同 encoder 选择了不同 dynamics gauge;
- refit-D* 有预测力,因为它固定了 geometry 后测 "这个 Gaussian coordinate 本身
  是否适合复合预测"。

也就是说,K-step LeWM 学到的不是普通 linear Gaussian state space model,而是:

```text
marginally Gaussian,
action-conditioned,
finite-horizon-composable,
encoder-selected dynamical coordinate.
```

### 5. 可以做的证书:不要只看 drift,要看 dynamics gauge

已有 `D*` / perturbation growth 是第一层。为了把论文写到更深,需要补下面几个
dynamics certificates。

#### C1. Jacobian product spectrum / finite-horizon gain

测:

```text
sigma_max(J_{K:1}),  trace(J_{K:1}^T J_{K:1}),  log det(I + J_{K:1}^T J_{K:1})
```

按 contact / free / capacity / K 分组。预期:

- `K=5` 不是单步 MSE 更低,而是 product gain 系统性更低;
- `D=8` 下 K=1 product gain 爆,K=5 把它压住;
- 这个指标比 self-drift 更能预测 planning。

#### C2. Non-normal amplification

很多系统 eigenvalue 不大,但短时奇异值会放大。测:

```text
Henrici departure from normality: ||J^T J - J J^T||
transient growth: max_t ||J^t|| even when spectral radius < 1
```

假设:K-step training 主要压的是 transient amplification,不是简单压 spectral radius。
这会比 "contractive regularization"更有内容,因为它解释 finite-horizon 而非 asymptotic
stability。

#### C3. Operator-mode filtering

用数据估计 action-conditioned Koopman / linearized latent operators:

```text
z_{t+1} ~= A_a z_t + b_a
```

可以先离散化 action cluster 或用 local linear regression。比较 `K=1` 和 `K=5` 的:

- singular value distribution;
- controllability Gramian;
- observability / goal-alignment of modes;
- mode lifetime vs planning contribution。

目标不是证明真实 dynamics 线性,而是把 learned Gaussian coordinate 的局部线性影子
可视化。

#### C4. Gaussian gauge distance

两个 encoder 的 marginal 都近似 `N(0,I)`,但它们的 coordinate 可能不是简单正交旋转。
测:

```text
best orthogonal Procrustes alignment error between z_K1 and z_K5
CCA / SVCCA
normalizing-flow map complexity from one latent to the other
Jacobian condition number of cross-map
```

如果 `phi_K5` 只是旋转,那 dynamics 差异应可被 `f` 吃掉;如果 cross-map 强非线性,
说明 horizon loss 真的改变了 Gaussian gauge。

#### C5. Free-vs-composed D* gap

现在已有 refit-D*。更深一步要拆:

```text
D*_free(k): direct k-step head lower bound
D*_comp(k): one-step predictor self-composed lower bound
gap = D*_comp - D*_free
```

这个 gap 是 latent Markov/composability certificate。若 `K=5` 主要关闭 gap,
就能更精确地说:多步训练选择的是 "可复合 Gaussian coordinate",不是 "信息更多的 coordinate"。

### 6. 方法不是先发明新 loss,而是先证明选择压力

当前更像 analysis/theory-first paper,不应急着堆方法。推荐顺序:

1. **现象定稿**:
   K-step changes encoder-side Gaussian dynamics gauge,not predictor capacity。

2. **理论最小模型**:
   构造一个 controlled linear/non-normal toy:

   ```text
   x_{t+1} = A(a_t) x_t + noise
   z = phi(x),  z marginal N(0,I)
   ```

   允许 `phi` 在 Gaussian-preserving / whitening family 里选坐标。
   证明 one-step objective 对某些 transient growth 盲,K-step objective 选择低
   product-gain coordinate。

3. **证书验证**:
   在 PushT/相图模型上测 C1-C5,证明它们解释 `D*` 和 planning。

4. **再决定方法**:
   若证书显示问题是 non-normal amplification,方法可以不是普通 Lipschitz,而是:

   ```text
   finite-horizon transient-growth regularization
   action-conditioned product-gain shaping
   gauge-aware encoder regularization
   ```

   但方法必须从证书推出,不能先拍一个 loss。

### 7. 与现有工作的差异句

可以这样定位:

> Delta-JEPA shows that one-step latent displacement should encode action.
> Fast-LeWM improves the interface for predicting future latents.
> RC-aux/TRM repair the planner-facing metric.
> LeJEPA theory explains why Gaussian marginals support identifiability.
>
> We ask a different question:
> **under the same Gaussian marginal constraint, what dynamical coordinate does
> multi-step prediction select?**

中文版本:

> 现有工作大多在修 LeWM 的接口:让 latent 更 action-sensitive、让 rollout 更快、
> 让 planner metric 更合理、让 test-time 能自适应。
> 我们分析的是更底层的表示选择问题:
> **SIGReg 把 latent 边缘钉成 Gaussian 之后,多步预测在剩下的 gauge 自由度里选择了什么
> action-conditioned dynamics geometry?**

### 8. 当前最值得做的最小实验

不需要先训练新模型。直接用已有 `D=192/32/8 × K=1/5` 六格:

1. 对每格采样同一批 rollout windows。
2. 用 autograd 或 finite difference 测 `J_f` 和 `J_{K:1}` 的 singular values。
3. 按 contact strength / free 分组。
4. 和 `planning / D*_multi / growth@8 / self-drift / probe R²` 做相关。
5. 做 `phi_K1` vs `phi_K5` 的 Procrustes/CCA/cross-map complexity。

如果看到:

```text
K=5 primarily reduces finite-horizon product gain / transient amplification,
and this explains planning better than one-step MSE or action delta,
```

那就有一个真正不像 trick 的 paper seed。

### 9. 暂定论文形态

**题目候选**

```text
Gaussian Marginals, Non-Gaussian Dynamics:
What Multi-Step Prediction Does to LeWorldModel Representations
```

或:

```text
Horizon-Weighted Spectral Filtering in Gaussian JEPA World Models
```

**主张**

> Multi-step LeWM does not merely improve rollout prediction. It changes the
> Gaussian latent gauge selected by the encoder, suppressing finite-horizon
> operator-product amplification while leaving one-step predictability largely
> unchanged.

**贡献**

1. 现象:冻结-refit 证明收益在 encoder-side dynamics gauge。
2. 理论:Gaussian marginal constraint 下,K-step objective 是 horizon-weighted
   operator-product / transient-gain selection pressure。
3. 证书:Jacobian product spectrum、non-normal amplification、D* gap、Gaussian gauge
   distance。
4. 对账:解释 Delta-JEPA/Fast-LeWM/TRM 等局部修法为什么有效但不触及底层问题。

**风险**

- 如果 Jacobian product spectrum 与 planning 不相关,说明收缩/增益故事仍太粗;
  需要转向 action-rank 或 reachability geometry。
- 如果 `phi_K1` 与 `phi_K5` 只是简单正交变换,那 encoder 改变的 claim 会弱;
  但已有 D* refit 结果暗示不太可能。
- 理论要避免声称 "首次 latent contraction";收缩和 Lipschitz 已有大量文献。
  新意应钉在 **Gaussian JEPA 的 horizon-induced dynamics gauge selection**。

---

## Part III. 实测结果(2026-07-06,`outputs/gauge/`,零训练,6 格现有模型)

§8 的最小实验已执行(C1/C2/C4 + 步间对齐直测)。每格 256 窗,J_f 沿真轨迹
autograd 精确计算(对上下文末帧;历史通道近似,主通道)。

### III.1 算子乘积谱(C1/C2)

| cell | σ₁(P₈) med | ρ(P₈) med | 单步 σ₁ mean | 瞬态曲线 σ₁(P_k) |
| --- | ---: | ---: | ---: | --- |
| D192 K1 | 16.45 | 1.69 | 3.26 | 2.8→16.5 超线性 |
| D192 K5 | **1.55** | 0.28 | 2.00 | **1.8 全程平坦** |
| D32 K1 | 16.67 | 5.01 | 2.06 | 1.8→16.7 |
| D32 K5 | **1.45** | 0.51 | 1.42 | **~1.5 平坦** |
| D8 K1 | 26.18 | **18.04(真谱不稳定)** | 1.77 | 1.6→26.2 |
| D8 K5 | **1.97** | 1.06 | 1.27 | **~1.8 平坦** |

K=1 的病理随容量恶化升级:瞬态放大(D192,ρ=1.7)→ 真谱不稳定(D8,ρ=18)。
K=5 在所有容量下把 8 步最坏放大压到 ~1.5-2.0 且不随 k 增长。
跨 6 格与 planning 的 Spearman:σ₁(P₈) 0.60(跨容量弱;**同容量 K 对比 3/3 全中**)
——谱驯化是必要非充分,容量自身设信息上限;refit-D\*(0.94)仍是最佳单指标。

### III.2 步间对齐直测(证伪"错位相消"假设)

| cell | handoff | rt | rr | **excess=rt−rr** | 朴素上限 Πσ₁ | 相消因子 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| D192 K1 | 0.321 | 0.395 | 0.203 | +0.192 | 10382 | 1.8e-3 |
| D192 K5 | 0.343 | 0.488 | 0.186 | **+0.301** | **211** | 7.6e-3 |
| D32 K1 | 0.572 | 0.690 | 0.460 | +0.230 | 253 | 7.9e-2 |
| D32 K5 | 0.556 | 0.712 | 0.420 | +0.292 | **15.1** | 1.1e-1 |
| D8 K1 | 0.631 | 0.849 | 0.758 | +0.090 | 80.5 | 3.7e-1 |
| D8 K5 | 0.639 | 0.854 | 0.739 | +0.115 | **5.6** | 3.9e-1 |

**"步间扩张子空间错位相消"假设被证伪**:excess 全部为正且 K=5 略更高
(对齐结构不变甚至更强);相消因子 K=5 反而更弱。差异全部来自"朴素上限"列:
每步最坏增益 σ₁ 被驯化 1.4-1.6×,几何复合放大为 14-49× 的乘积差。

### III.3 精确机制:有效增长率被推到临界值 ≈1

有效每步增长率 = 每步最坏增益 × 传输率:

```text
K=1: 3.26 × 0.40 ≈ 1.30 / 步   (超临界 → 1.3^7×初值 ≈ 16.5 ✓)
K=5: 2.00 × 0.49 ≈ 0.98 / 步   (临界 → 平坦曲线 ✓)
```

> **多步训练的净效果:每步最坏方向放大倍数驯化 ~1.5×(方向交接结构不动),
> 恰好把误差传播的有效增长率从 ~1.3 压到 ≈1.0——边际稳定线上,不多压。**
> K 步目标只需要误差在 K 步内不爆,没有动力压得更低——
> 可检验推论:**有效增长率是 K 的函数,rate(K) 应随 K 单调趋近/低于 1**。

### III.4 Gauge 距离(C4)

| pair | 线性 R² | Procrustes 残差 | CCA>0.9 占比 | MLP R² | 非线性增益 |
| --- | ---: | ---: | ---: | ---: | ---: |
| D192: φ_K1↔φ_K5 | 0.857 | 0.506 | 37% | 0.941 | +0.085 |
| D8: φ_K1↔φ_K5 | 0.365 | 0.873 | 0% | 0.812 | +0.447 |

旋转假设被拒(残差 0.51/0.87)。大容量下 gauge 变化 ≈ **非正交线性整形**
(拉伸/剪切——正是改变共轭 Jacobian 范数的手段);小容量下是真正不同的
非线性信息选择(与角度丢弃/位置增强一致)。

### III.5 汇总:"K 步之后的 Gaussian latent 是什么"(全部直接测量,无推断)

```text
边缘层   : N(0,I) 不变(SIGReg)
单步层   : 可预测性不变;单步算子仍扩张(σ₁ ≈ 1.3-2.0 > 1)
乘积层   : σ₁(J_{k:1}) 对 k 平坦 —— 通过每步最坏增益驯化 ~1.5× 的几何复合,
           有效增长率被推到临界 ≈1.0;对齐/相消结构不是训练变量
gauge 层 : 大容量 = 对单步坐标系的非正交线性整形;
           小容量 = 不同的非线性信息选择
失败对照 : 单步坐标系随容量压缩从瞬态放大恶化为真谱不稳定(ρ 1.7→18)
```

下一步(§6 顺序更新):A) rate(K) 定律实验——补 K∈{2,3,10} 训练,检验 III.3 的
临界推论,把观察升级为定量律;B) 理论最小模型以"K 步最优解落在边际稳定点"
为目标命题;C) Fast-LeWM 对照(prefix 目标是否也驯化 σ₁——它推理时不自复合,
可能绕开,这是与它的机制级分界线)。

---

## Part IV. rate(K) 定律实验(2026-07-06 启动,进行中)

### 动机与预注册

III.3 的核心观察:有效每步增长率(每步最坏增益 × 传输率)K=1 ≈ 1.30、K=5 ≈ 0.98
——K 步目标似乎把误差动力学恰好推到"K 步内不爆"的**边际稳定点**,不多压。
两个点不成律,补 K ∈ {2, 3, 10} 三个模型(D=192,协议与相图完全一致:
lewm_multistep,`wm.unroll=K`,`num_steps=3+K`,30 epochs,2 卡,batch 128)。

**预注册预测(跑之前锁定)**:
- P-rate-1:有效增长率 rate(K) 随 K 单调下降,K=1 的 1.30 → K=10 应 ≤ 1.0;
- P-rate-2:瞬态曲线 σ₁(P_k) 的平坦段随训练 K 延长(K=2 模型在 k>2 后恢复增长,
  K=10 模型平坦到 k=10);
- P-rate-3:planning(matched history)随 K 先升后平/降(K=10 的 1-step 精度代价
  开始显现时),复现 What-Drives 的"rollout 长度有甜点"经验律并给出机制坐标;
- 若 K=2 就把 rate 压到 ≈1.0:结论改写为"见过一次自身误差即足够",同样成立。

**理论对照目标(推导中)**:线性高斯 + 白化约束的最小模型中,证明 K 步目标的
最优 encoder 使共轭误差动力学落在边际稳定点(损失平稳性 ⇒ rate=1 不动点),
且单步目标对该量梯度盲(Thm B 的定量化)。实验曲线 rate(K) 是该定理的直接对照。

### 训练测试全链(自动,`outputs/ratek/run_ratek_v2.sh`)

```text
训练 pd_d192_k{2,3,10}(守护进程,脱离 CC 会话)
→ make_eval_dir(model 子树 config + 最新权重软链)
→ matched-history planning(+plan_config.history_len=3,3 评测种子 × 50ep)
→ probe 导出(regime_stepB_eval_data,4000 窗 × 13 帧,含 states,seed 2025)
→ 算子谱(outputs/gauge/jac_spectrum.py:σ₁/ρ/ν/瞬态曲线,J 沿真轨迹 autograd)
→ 对齐测量(outputs/gauge/alignment.py:handoff/传输率/随机基线/相消因子)
```

汇总时与已有 K=1/K=5(及 D=32/8 格)并表,得五点 rate(K) 曲线 + 证书种群 n=9。

### 运维教训(共享机守则,已生效)

- /dev/shm 全机共享:多训练**同时启动**的 DataLoader 预取峰会打爆它并殃及同事
  (2026-07-06 事故);规则 = 错峰 ≥90s 启动、loader 用 workers≤6 + prefetch 2、
  启动前查 shm 使用率;
- 本机所有人共用 jovyan 账号,进程归属按**项目路径**判断(我们=code/wge/learn_wm);
  发射前守卫:目标卡有任何非本项目进程即中止;GPU 分配听当期口头约定
  (当前:0-5 我们,6-7 同事,临时性分配);
- kill 一律按 PID;pgrep/pkill 的模式若出现在自身命令行会自杀(踩过两次)。

### Part IV 结果(2026-07-08,链路在 07-08 00:41 自动完成)

| K | 有效增长率 | σ_max(P₈) | ρ(P₈) | 单步σ₁ | planning(3种子) |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 1.286 | 16.45 | 1.685 | 3.26 | 83.3 (86/86/78) |
| 2 | 1.216 | 8.54 | 1.185 | 2.55 | 81.3 (76/86/82) |
| 3 | 1.131 | 5.22 | 0.834 | 2.29 | 87.3 (86/88/88) |
| 5 | **0.977 ≈ 临界** | 1.55 | 0.278 | 2.00 | **88.0 (90/90/84)** |
| 10 | 0.816(亚临界) | 0.44 | 0.077 | 1.68 | 84.0 (88/82/82) |

**判决:**
- **P-rate-1 ✓**:rate(K) 严格单调下降,在 K=3→5 间穿过 1.0;且每格瞬态曲线是
  干净几何级数——(σ₈/σ₁)^{1/7} 与 eff_rate 五格全部吻合(1.21/1.14/0.98/0.84),
  **每个模型的误差传播行为被一个数完全刻画**。
- **P-rate-2 ✗→修正**:不是"平坦段随 K 延伸",是整条曲线的每步增益被 K 均匀调低
  (K=10 全程衰减)。"压到临界即停"的恒温器猜想被 K=10 证伪(0.816<1):
  K 是连续旋钮,不是恒温器。
- **P-rate-3 ✓ 且为本轮主发现**:planning 甜点形(83→81→87→**88**→84),
  **峰值恰落在 rate(K) 穿过 1.0 处**。机制读法:超临界=误差复利爆炸,CEM 在噪声里
  排序;亚临界=过度收缩把候选动作间的轨迹分离一并抹平,cost 地形变钝;
  临界=信号(候选差异)传到 horizon 而噪声不爆,信噪比最大。

**一句话定律(论文主张候选):**

> Prediction horizon 通过设定 latent 误差动力学的每步有效增益来起作用:
> rate(K) 单调下降;planning 在 rate ≈ 1(edge of stability)处最大化。

对账:给 What-Drives"rollout 长度有甜点"以机制坐标;与 RNN/reservoir 的
edge-of-chaos 传统呼应,但"horizon 为旋钮 × 规划为判据 × JEPA WM"组合未见先例
(待 novelty 审稿轮验证)。

**诚实边界与下一步:**
- rate 单调性 = 大效应,稳;planning 甜点 = 5 分小效应且每 K 仅 1 训练种子
  ——冲论文需 K∈{3,5,10} 各补 1-2 种子;
- "亚临界抹平 cost 地形"目前是解释非测量——下一个便宜实验:候选动作对在
  horizon 处的 latent 分离度 vs K(预期 K=10 显著缩小,K=5 保持);
- 理论目标定稿:最小模型证明 (a) K 步目标最优解的 rate(K) 单调性,
  (b) 规划信噪比在 rate=1 处最大 ⇒ 最优 horizon 条件。

---

## Part V. 分离度证伪 + 理论对象重钉(2026-07-08)

三线并进的当天结果:补种子训练已发射;分离度实验做完并**证伪了 Part IV 的
"亚临界抹平"读法**;理论最小模型拿到一个负结果定理和一个复现现象的非线性 toy。

### V.0 种子补充(进行中)

发现 Gate 时期已有 K=1/K=5 的第二训练种子(`iter2_baseline_s1` /
`iter2_multistep_s1`,seed=1,协议完全一致,h3 planning 已测:
K=1 → 84/82/82(均 82.7),K=5 → 86/84/88(均 86.0)——两格的 K=5>K=1
方向在第二种子上成立)。已发射 `pd_d192_k{2,3,10}_s1`(seed=1,协议同
Part IV;`outputs/ratek/run_ratek_s1.sh`,含 GPU 归属守卫 + /dev/shm 守卫,
训练 ~26h,链路自动完成 planning+probe+谱+对齐)。s1 已训模型的证书链
(`run_s1_certs.sh`)同日补测。注:`cfg.seed` 控制数据划分与 shuffle 顺序,
模型初始化由库级全局 seed 固定——与 s1 先例机制一致,是"数据种子"。

**s1 证书结果(当日完成):rate 律端点跨种子精确复现。**

| cell | rate(seed 1) | rate(seed 3072) | smax8 s1/3072 |
| --- | ---: | ---: | ---: |
| K=1 | 1.310(超临界,几何增长) | 1.286 | 19.4 / 16.5 |
| K=5 | 0.994(临界,曲线平坦) | 0.977 | 1.66 / 1.55 |

种子间 rate 差 ~0.02:**"每个模型的误差传播被一个数刻画"这个刻画本身是
训练种子稳定的**。K∈{2,3,10} 的 s1 点等训练波完成。

### V.1 候选动作分离度:预注册假设被证伪(`outputs/gauge/separation.py`)

设计:同一 context,两条候选动作序列开环 8 步,测终点分离 sep_k、
自漂移 drift_k、SNR=sep/drift。三种候选对:shuffle(他窗动作,粗尺度)、
first(只扰第一个未来动作,δ=0.25)、all(扰全部未来动作)。

**粗尺度(shuffle):分离度饱和,所有 K 都不是瓶颈。**

| K | sep₈ | drift₈ | SNR₈ |
| ---: | ---: | ---: | ---: |
| 1 | 19.41 | 6.12 | 3.03 |
| 2 | 18.94 | 4.33 | 4.02 |
| 3 | 18.77 | 3.80 | 4.63 |
| 5 | 18.54 | 4.69 | 3.76 |
| 10 | 17.73 | 3.46 | **4.96(最高)** |

sep₈ ≈ √2·|z|(两个独立随机 latent 的距离)= 饱和;K=10 只比 K=5 低 4%,
SNR 反而最高。**P-sep-1/3 被证伪:亚临界没有抹平粗尺度 cost 地形。**

**细尺度(first,动作→horizon 增益 = 回声):**

| K | gain₁(即时) | gain₈(回声) | 回声每步增长率 | drift₁ | drift₈ |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.77 | 3.02 | 1.216 | 1.05 | 6.12 |
| 2 | 0.79 | 2.73 | 1.194 | 0.95 | 4.33 |
| 3 | 0.86 | 2.55 | 1.168 | 1.05 | 3.80 |
| 5 | 1.03 | 1.97 | 1.097 | 1.60 | 4.69 |
| 10 | 1.28 | 1.76 | 1.047 | 1.65 | 3.46 |

三个新事实:
1. **多步训练重新分配动作影响的时间结构,而非削弱它**:即时动作增益随 K
   上升(0.77→1.28),回声增长率被驯化(1.22→1.05,随 K 单调),horizon-8
   总动作增益只降 1.7×——而同区间最坏状态误差乘积降 **37×**(16.45→0.44)。
   动作回声不服从状态误差的 rate 律(K=10/K=5 = 0.89,不是 rate^7 预测的 ~0.3)。
   表征保住了可控性信号,杀掉的是误差放大——这本身是"好表征"的定义级性质。
2. **一步 drift 随 K 上升**(1.05→1.65,K≥5 付 ~55% 一步精度代价)——
   Part IV "K=10 的 1-step 精度代价"由此坐实为测量。
3. K=10 掉分(84 vs 88)的机制**不是**候选分离塌缩;剩余候选解释 =
   一步拟合代价 或 纯种子噪声(补种子波将判决)。

### V.2 各向同性 slack:K 步训练花掉的 SIGReg 自由度(直接测量)

probe latent 协方差(D=192):

| K | trace | effective rank | off-diag 能量占比 |
| ---: | ---: | ---: | ---: |
| 1 | 185.5 | 117.7 | 0.652 |
| 2 | 181.8 | 118.5 | 0.644 |
| 3 | 180.2 | 116.2 | 0.651 |
| 5 | 167.4 | **79.5** | 0.780 |
| 10 | 167.2 | **91.3** | 0.740 |

K∈{1,2,3} vs K∈{5,10} 干净分裂:大 K 模型把 SIGReg 当软预算花,
用边缘各向同性换低瞬态增益(与 III.4 的"非正交线性整形"一致)。

### V.3 理论:负结果定理把对象钉死在"局部非线性增益"

**定理(线性情形,一行证明):** 线性世界 `x'=Ax+Ba+ε` 的平稳协方差满足
`Σ = AΣAᵀ + Q, Q≻0`;任何精确白化线性 encoder 的共轭单步算子
`F = WAW⁻¹`(W 白化)满足 `FFᵀ = I − Σ^{-1/2}QΣ^{-1/2} ⪯ I`,即
**σ₁(F) ≤ 1 恒成立,与 K 无关**。数值 toy(`outputs/theory/minmodel_v1.py`)
验证:所有 K 的 σ₁(F)≈1.0,K 无效应。

**推论:** PushT 观察到的 K=1 病理(SIGReg 下单步 σ₁=1.3-3.3、乘积 16×)
**本质是非线性/局部现象**:边缘白化只约束全局二阶矩,管不住局部 Jacobian 场;
LeWM 目标里唯一看得见局部 Jacobian 乘积的项就是 K 步开环 loss。
理论对象从"Gaussian gauge 选择"精化为:
**局部增益场 J(z,a) 的乘积统计 vs 全局边缘约束之间的 slack,由 horizon 分配。**

**非线性最小模型复现(`outputs/theory/minmodel_v2.py`):**
两 regime 世界(contact 高增益 σ₁≈3 / free 收缩 σ₁=0.9,半空间切换,
即 PushT contact/free 的卡通)+ MLP encoder/predictor + 软 SIGReg +
K 步开环 loss。1 种子 smoke 已复现四个签名:每步局部增益 3.24→1.85 随 K
驯化;一步 MSE 0.122→0.198 随 K 上升;即时动作增益 1.40→1.50 微升;
动作回声驯化远弱于误差乘积驯化。(3 种子正式跑进行中。)

### V.4 修正后的机制图景与理论目标

```text
边缘层   : SIGReg 钉住全局二阶矩;线性情形下这已足够(σ₁≤1 定理)
局部层   : 非线性 encoder 的局部 Jacobian 场不受边缘约束;K=1 在高增益
           regime(contact)留下超临界局部增益,复合成 16× 放大
horizon 层: K 步 loss 是唯一对局部乘积有梯度的项;K 单调压 rate,
           代价 = 一步拟合误差 + 各向同性 slack(均已测)
可控性层 : 动作→latent 增益基本被保护,且即时增益随 K 增强
           (信号通道与误差通道被不对称处理——理论必须解释这个不对称)
```

理论目标更新(替代旧 (b)"SNR 在 rate=1 最大"——已被 V.1 削弱):
- (a) 不变:最小模型证明 rate(K) 单调性(v2 已数值复现,待解析化);
- (b') 解释**误差/信号不对称**:为什么 K 步最优解驯化误差传播算子而保护
  (甚至增强)动作输入通道——猜想:拟合数据要求区分不同动作的未来
  (动作通道有监督信号),而误差方向无监督约束,是纯 slack,K 步 loss
  恰好只在误差通道上施压;
- (c') 甜点下降沿的候选机制改为**一步拟合代价 vs 误差复合的 trade-off**:
  drift_H(K) ≈ ε₁(K)·Σⱼ rate(K)ʲ,ε₁ 随 K 升、rate 随 K 降 ⇒ 固定评测
  horizon 下有内部最优 K;待补种子波判决 K=10 掉分是否真实后再定稿。

### V.5 下一步

1. 等补种子波(~07-09 下午)→ rate(K)/planning 加误差棒,判决甜点真伪;
2. minmodel_v2 3 种子跑完 → 若签名稳,把 (a)(b') 写成命题 + 证明骨架;
3. Fast-LeWM 对照(Part III 遗留)仍排在理论之后。

---

## Part VI. 补种子判决:rate 律确立,甜点证伪(2026-07-10)

`pd_d192_k{2,3,10}_s1` 训练波 07-10 01:09 完成,链路 02:02 全部落地。
与已有 seed 3072 及 iter2_*_s1(K=1/5)并表,每格 n=2 训练种子 × 3 评测种子。

### VI.1 rate(K) 定律:满分复现

| K | rate(seed 3072) | rate(seed 1) | Δ | σ_max(P₈) s1/3072 | 单步σ₁ s1/3072 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 1.286 | 1.310 | .024 | 19.4 / 16.5 | 3.42 / 3.26 |
| 2 | 1.216 | 1.207 | .009 | 8.45 / 8.54 | 2.55 / 2.55 |
| 3 | 1.131 | 1.151 | .020 | 5.36 / 5.22 | 2.26 / 2.29 |
| 5 | 0.977 | 0.994 | .017 | 1.66 / 1.55 | 1.89 / 2.00 |
| 10 | 0.816 | 0.803 | .013 | 0.32 / 0.44 | 1.64 / 1.68 |

**rate(K) 单调、跨种子最大差 0.024,K 间距远大于种子噪声;单步增益几乎
逐位复现;对齐 excess 全正(+0.27/+0.29/+0.30)。定律层面:n=2 确立。**

### VI.2 planning 甜点:被证伪

| K | 6 评测均值 | std | sem | 两训练种子各自均值 |
| ---: | ---: | ---: | ---: | --- |
| 1 | 83.0 | 3.0 | 1.2 | 83.3 / 82.7 |
| 2 | 84.0 | 6.4 | 2.6 | 81.3 / 86.7 |
| 3 | 87.3 | 4.8 | 2.0 | 87.3 / 87.3 |
| 5 | 87.0 | 2.8 | 1.1 | 88.0 / 86.0 |
| 10 | 85.7 | 5.3 | 2.2 | 84.0 / **87.3** |

- **K=10 下降沿不复现**(种子 1 给 87.3):Part IV 的"峰值恰在 rate=1"
  是单种子噪声;差异 1.5 分 vs sem ~2,无法区分。
- 稳健的形状:**K=1 最差(83.0),K≥3 平台(86-87)**。
  50 倍的最坏放大差(16.5→0.32)只映射为 ≤4 分 planning 差。
- **与 V.1 分离度实验汇合**:细尺度动作区分信号在 K=10 仍保留(gain 只降
  11%),所以亚临界不掉分——分离度实验两天前的预测被补种子判决证实。
  两条独立证据线指向同一结论:**误差/信号不对称是主故事,甜点不是**。

### VI.3 论文主张修订

废弃:"planning 在 edge of stability 处最大化"(单种子伪影)。

现在有证据的主张:

> Prediction horizon K sets the effective per-step gain of latent error
> dynamics: rate(K) is monotone, seed-stable, and crosses 1 between K=3 and 5
> (a clean, one-number law). Meanwhile action-discrimination signal is
> preserved across all K (asymmetric taming). Consequently planning is
> remarkably robust: 50x differences in worst-case error amplification map to
> <=4 points; only the strongly supercritical K=1 pays a consistent cost.

机制链:K 驯化误差通道(无监督 slack)、保护动作通道(有监督信号)⇒
规划的信号侧不随 K 变坏,噪声侧超临界时才显著变坏 ⇒ K=1 掉分、K≥3 平台。
"最优 K"问题从"临界点"弱化为"过临界即可,深亚临界无害(本任务/horizon)"。

### VI.4 更新后的下一步

1. 理论目标随判决收缩:(c') 内部最优 K 命题降级;(b') 不对称命题升为主定理
   目标(误差通道 = 纯 slack、动作通道 = 有监督,K 步梯度只压前者);
2. Fast-LeWM 对照回到队首(prefix 目标不自复合,预测:不驯化 σ₁ 但 planning
   不差——若成立,是"驯化是自复合训练的副产物而非规划必需"的第二证据);
3. 标量 toy 全分布罚(检验刚性预测)仍在理论待办。

---

## Part VII. Fast-LeWM 机制对照:2×2 因子设计(2026-07-10 发射,进行中)

### 设计

发现仓库已有 `iter2_sgmulti_b1/b2`(unroll_sg=5:多步 loss 经过乘积但只训
predictor、encoder 停梯度),h3 planning 已测 = **56.0**(b1,远差于所有臂)。
于是 Fast-LeWM 对照升级为 2×2 因子:

| 臂 | loss 有乘积梯度? | encoder 被多步塑形? | 状态 |
| --- | --- | --- | --- |
| K=1 | 否 | 否 | 已完成(rate 1.29/1.31,plan 83.0)|
| K=5 开环 | 是 | 是 | 已完成(rate 0.98/0.99,plan 87.0)|
| sg-K5 | 是(仅 predictor)| 否 | 已训好,证书链已发(plan 56.0!)|
| **TF-K5** | **否** | 是(多步监督、无自复合)| **训练中(2 种子)** |

TF-K5 = teacher-forced 多 horizon:与 lewm_multistep 完全同窗口、同监督位置,
唯一差别是 context 全用真 embedding——**梯度不经过 Jacobian 乘积**。
这是 Fast-LeWM"prefix 不自复合"的机制级代理,且不引入架构混淆
(实现:`lewm.py` 的 `unroll_tf` 分支 + `lewm_tfmulti.yaml`,
链:`outputs/ratek/run_tf_control.sh`,4 卡训练 + 1 卡 sgmulti 证书)。

### 预注册预测(2026-07-10 发射前锁定)

- **P-tf-1**:TF-K5 rate 停在超临界 ~1.3(驯化需要乘积梯度;多步监督本身无效);
- **P-tf-2**:TF-K5 planning ≈ K=1(~83)(评测是自回归 CEM,超临界照付代价);
- **P-tf-3**:sg-K5 rate 也不驯化(驯化住在 encoder gauge——refit-D* 已证
  收益在 encoder;predictor-only 的乘积梯度到不了 encoder);
- **P-tf-4**(事后解释,弱):sg-K5 planning 56 的塌坏若与超临界 rate + predictor
  过拟合多步项并存,则 predictor-only 多步是净伤害;
- **证伪线**:若 TF-K5 驯化了 σ₁ ⇒ "乘积梯度是驯化的必要成分"错,
  多步监督(数据层)本身足够 ⇒ 理论 §6 的风险分解需要重写。

### 资源

4 卡 ×~30h(TF 两种子)+ 1 卡 ~1h(sgmulti 证书,当天出)+ 链路 1h;
判决预计 07-12 早。

### sg-K5 证书结果(2026-07-10 晚,2×2 第三格判决)

| 臂 | rate | smax8 | planning |
| --- | ---: | ---: | ---: |
| K=1(乘积✗ enc✗)| 1.286/1.310 | 16.5/19.4 | 83.0 |
| sg-K5 b1(乘积✓ 仅 predictor)| **1.166** | 5.26 | **56.0** |
| sg-K5 b2(β=2)| **1.165** | 4.94 | **53.3** |
| K=5 开环(乘积✓ enc✓)| 0.977/0.994 | 1.55/1.66 | 87.0 |

**P-tf-3 定性确认、定量细化**:predictor-only 的乘积梯度买到 ~35% 的
log-rate 降幅(1.29→1.17),但**到不了临界**(0.98 需要 encoder 参与)
——驯化的大头确实住在 encoder gauge(与 refit-D* 一致)。

**意外的强事实:rate 不决定 planning。** sg-K5 的 rate 比 K=1 更好
(1.17 < 1.29),planning 却差 27 分(56 vs 83)。对着冻结 gauge 硬压
predictor 收缩会摧毁规划——**共适应(encoder-predictor 一起改)是
驯化有益的前提**。这与 VI 的结论同向:rate 是机制证书,不是 planning
预测器;它再次支持"发论文的主张要钉在不对称驯化,不是 rate-planning 对应"。

### TF-K5 判决:2×2 完成(2026-07-12,链路 01:04 落地)

| 臂 | 乘积梯度 | encoder | rate(2 种子) | smax8 | planning(6 评测) |
| --- | --- | --- | ---: | ---: | ---: |
| K=1 | ✗ | ✗ | 1.286/1.310 | 16.5/19.4 | 83.0 |
| **TF-K5** | ✗ | ✓(多步监督)| **1.390/1.359** | 24.0/21.4 | **69.0** |
| sg-K5 | ✓ | ✗ | 1.166/1.165 | 5.3/4.9 | 54.7 |
| K=5 开环 | ✓ | ✓ | 0.977/0.994 | 1.6/1.7 | 87.0 |

**P-tf-1 ✓✓(主判决):没有乘积梯度就没有任何驯化。**
TF-K5 的 rate(1.36-1.39)甚至略高于 K=1;drift 曲线(1.05→6.0)与
K=1(1.05→6.12)逐点重合;动作回声曲线同样重合;一步 drift 也无代价。
**多步监督(数据层)对驯化贡献为零——起作用的是自复合的梯度,
且(结合 sg 臂)必须到达 encoder 才能压到临界。**

**P-tf-2 ✗(信息量更大的方向):TF-K5 planning = 69.0(64/60/80/68/74/68),
比 K=1 低 14 分**,而非预注册的"≈83"。两个"单翼"臂都显著差于两个
"对称"臂(sg 55 / TF 69 << K=1 83 / K=5 87):把多步性只塞进
encoder-predictor 的一侧,无论塞哪侧都伤规划——共适应结论从第二个
方向被加强。

**诚实缺口(重要):证书集不完备。** TF-K5 在我们所有证书上
(rate/drift/信号/对齐/一步精度)都是 K=1 的孪生,planning 却差 14 分
——存在证书未捕获的 planning 相关性质。候选后续:probe R²/特征内容对比
(TF 的多偏移监督可能改变了 encoder 的信息选择)。在方法论文里这条
要么补上,要么明写为边界。

**Fast-LeWM 机制分界线(本对照的原始目的):**
prefix 型目标(无自复合)不修复 latent 误差动力学——若 Fast-LeWM
效果好,来源只能是**推理接口绕开复合**(测试时从不自复合),
不是治好了动力学。与我们不对称故事一致:它逃逸问题,K 步训练修复问题。
(注:TF-K5 是机制代理而非其完整架构,该结论限于机制层。)
