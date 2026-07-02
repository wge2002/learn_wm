# 理论稿:从 LeWM 的高斯预测目标到"充分动力学"损失

> 目的:把多步 unroll 低 drift 但 planning 崩的实验事实,推成一个更深的理论结论:
> **LeWM + SIGReg 学到的不是"充分状态",而是"高斯化的可预测坐标"**。
> 这个偏置在接触/混合动力学里会系统性丢掉难预测但任务关键的自由度。
>
> 与旧稿 [theory_sufficiency_loss.md](../../theory_sufficiency_loss.md) 的关系:
> 旧稿已经给出了"多步项只训 dynamics module"这个最小方案;
> 本稿尝试把它的理论地基补硬:从 Gaussian/SIGReg + MSE 的最优解写出
> **可预测性选择偏置**,再推出为什么 PushT 角度会丢、为什么 self-drift 会骗人,
> 以及为什么新 loss 必须把**表征充分性**和**动力学稳定性**拆开训练。
>
> 实验证据见 [multistep_unroll_drift.md](../../multistep_unroll_drift.md)、
> [regime_direction_results.md](../../regime_direction_results.md)、
> [diagnosis_world_model_drift.md](../../diagnosis_world_model_drift.md)。

---

## 0. 一句话

> LeWM 的原 loss 可以理解为一个 **Gaussian predictive-coordinate objective**:
> SIGReg 把 latent 边缘压成近似 $N(0,I)$,预测 MSE 在这个固定高斯体积里选择
> **条件方差最低、最可预测**的坐标。它防的是维度坍缩,不是信息坍缩。
> 接触动力学里,block 角度这类自由度恰好"难预测但控制关键";
> 多步 unroll 会进一步放大这类方向的条件方差,于是 encoder 有动力把它们换成更容易滚的替代坐标。
> 所以低 self-drift 可以和差 planning 同时出现。新 loss 的核心不是"再压 drift",
> 而是一个分离原则:**encoder 只从充分性/grounded 的短程目标学习,多步稳定性只训练 dynamics module,不再改写坐标系**。

---

## 1. LeWM 目标:高斯化的 JEPA

记

```text
O_t       = observation/history
Z_t       = φ(O_≤t) ∈ R^D
C_t       = (Z_{t-h+1:t}, A_{t-h+1:t})     # predictor 的上下文
f(C_t)    = Ẑ_{t+1}
```

代码中 LeWM 的实际训练形式是:

$$
\mathcal{L}_{\text{LeWM}}(\phi,f)
=
\mathbb{E}\lVert f(C_t)-Z_{t+1}\rVert_2^2
+ \lambda\,\mathcal{R}_{\text{SIGReg}}(Z).
$$

这里有两个结构性事实:

1. **target 也是 encoder 产生的。** 右边的 $Z_{t+1}=\phi(O_{\le t+1})$ 不是外部真状态,
   而是同一个 encoder 移动出来的目标。预测器和 encoder 可以合谋改变"要预测什么"。
2. **SIGReg 约束的是边缘分布,不是语义。** 实现上 SIGReg 随机采很多单位方向 $u$,
   让一维投影 $u^\top Z$ 像标准高斯。按 Cramer-Wold 直觉,这逼近:

$$
Z \sim N(0,I).
$$

这件事很强,但强在"形状":均值、方差、相关性、投影分布都像高斯。
它没有规定哪个坐标必须是 block angle、哪个坐标必须是 contact state,
也没有规定 latent 欧氏距离必须和控制距离同构。

因此 LeWM 的目标不是:

```text
学一个对物理状态/任务充分的 z。
```

而更接近:

```text
在一个边缘近似 N(0,I) 的 latent 里,找一组容易被 f 预测的坐标。
```

下面这条等式是整个方向的理论核心。

---

## 2. 核心命题:高斯边缘 + MSE = 选择可预测坐标

先固定 encoder $\phi$,只优化 predictor。平方损失下的最优 predictor 是条件均值:

$$
f^*(C_t)=\mathbb{E}[Z_{t+1}\mid C_t].
$$

代回得到不可约预测误差:

$$
\inf_f \mathbb{E}\lVert f(C_t)-Z_{t+1}\rVert_2^2
=
\mathbb{E}\,\mathrm{Tr}\,\mathrm{Var}(Z_{t+1}\mid C_t).
$$

若 SIGReg 已经把 $Z_{t+1}$ 的边缘标准化到均值 0、协方差 $I$,则由全方差公式:

$$
\mathbb{E}\,\mathrm{Tr}\,\mathrm{Var}(Z_{t+1}\mid C_t)
=
D
-
\mathrm{Tr}\,\mathrm{Var}\left(\mathbb{E}[Z_{t+1}\mid C_t]\right).
$$

也就是说,在高斯边缘固定总方差 $D$ 后,LeWM 的 pred_loss 等价于:

```text
最大化 latent 中可由当前上下文解释的方差。
```

这不是一个充分性原则,而是一个**可预测性选择原则**。

### 2.1 Toy theorem:精确白化下的 predictive PCA

下面的推导不是声称真实 LeWM 的非线性 encoder 有闭式解,而是给一个最小 toy theorem:

```text
Assumptions:
1. 存在已白化的充分特征 Y_{t+1}, E[Y]=0, Cov(Y)=I。
2. encoder 是线性投影 Z=U^T Y,且 U^T U=I_D。
3. predictor 容量足够,平方损失下达到条件均值。
4. conditioning context C_t 在这个 toy theorem 中视为固定/外生,
   不随 U 改变。
5. SIGReg/whitening 近似成精确的 marginal covariance constraint。
```

把下一状态的某个充分特征记为已白化变量 $Y_{t+1}\in R^m$,令线性 encoder
$Z_{t+1}=U^\top Y_{t+1}$,且 $U^\top U=I_D$。设

$$
M=\mathrm{Cov}\left(\mathbb{E}[Y_{t+1}\mid C_t]\right).
$$

则最优预测误差为:

$$
\mathcal{E}(U)=D-\mathrm{Tr}(U^\top M U).
$$

最小化它的 $U$ 是 $M$ 的 top-$D$ 特征方向。结论很直接:

> 在白化/高斯约束下,预测 MSE 会选择"未来中最可由当前上下文解释的子空间"。
> 如果某个任务关键变量条件方差高,它会被低优先级处理,即使它对控制非常重要。

这就是旧稿里"难预测信息被丢掉"的严格版本。SIGReg 保证了 latent 不塌成低维,
但它允许模型把维度用在**可预测但任务次要**的因素上。

这个 toy theorem 不证明"任意非线性 LeWM 必然丢任务变量"。它证明的是一种
**selection pressure**:当 latent 维度/容量/正则化形成竞争时,prediction loss 偏好
explained variance 高的因素,而不是 task value 高的因素。若 $D$ 足够大、任务变量本身也低条件方差,
或另有显式 control-sufficiency 约束,任务变量当然可以被保留。

还要注意,真实 LeWM 里 $C_t$ 包含 $Z$ history,所以 $M$ 一般应写成 $M(U)$。
也就是说,top-eigenvector 结论只在"conditioning context 固定"的 toy setting 中严格成立;
在完整模型里它是一个局部压力/一阶解释,不是全局闭式最优解。

### 2.2 从 Gaussian loss 看同一件事

若把 latent transition 写成高斯似然

$$
p_f(Z_{t+1}\mid C_t)=N(f(C_t),\sigma^2 I),
$$

则负对数似然就是 MSE 加常数。若允许 full covariance,结论取决于 covariance 的参数化:
固定/共享 covariance 时仍主要惩罚条件不确定性;若 covariance 可由模型自由预测,
模型可能把难预测方向解释成高方差而不是改变 representation。本文主张只针对 LeWM 当前的
isotropic-MSE 训练形态,不声称覆盖所有 Gaussian NLL 设计。

```text
让 Z 的条件不确定性尽可能小。
```

但 planning 需要的是另一件事:

```text
Z 对任务状态/goal/cost-to-go 足够充分。
```

最小化 $H(Z_{t+1}\mid C_t)$ 或 $\mathrm{Tr}\,\mathrm{Var}(Z_{t+1}\mid C_t)$
并不推出 $I(Z_t;S_t)$ 大,也不推出 $Z$ 保留所有控制关键变量。
这正是 JEPA 自指目标的漏洞:它可以通过改变 target 表示来降低条件熵。

---

## 3. 为什么接触动力学会把问题放大

PushT 不是平滑单模态系统。更合适的局部模型是 hybrid dynamics:

$$
S_{t+1}=F_{R_t}(S_t,A_t)+\xi_t,\qquad R_t\in\{\text{free},\text{contact},\text{push},\text{release}\}.
$$

其中 regime $R_t$ 由几何接触条件决定,在边界附近会出现小状态差异导致的大动力学分叉。
线性化一段开环 rollout:

$$
\delta S_{k+1}
\approx
J_{R_k}\delta S_k
+ B_{R_k}\delta A_k
+ w_k
+ \Delta F_k\,\delta R_k.
$$

对应的误差协方差近似满足:

$$
P_{k+1}
\approx
J_{R_k}P_kJ_{R_k}^\top
+ Q_k
+ p_k(1-p_k)\Delta F_k\Delta F_k^\top.
$$

最后一项是切换不确定性:同一段动作在接触边界附近可能进入不同动力学分支。
在 PushT 里,block angle 正是这种项最强的自由度之一:

```text
接触点/摩擦/推力臂的小差异 -> 角速度和角度未来大差异。
```

于是对 encoder 来说,角度方向有两个坏属性:

1. **条件方差高。** 给定当前 latent 和 action,未来角度仍更难预测。
2. **开环放大强。** 多步 rollout 下,角度误差会通过接触 Jacobian 和 regime switching 放大。

但对 planner 来说,角度又是好变量:

```text
PushT 的成功条件高度依赖 block pose,尤其是朝向。
```

这就形成了一个理论冲突:

```text
LeWM/SIGReg/MSE 喜欢低条件方差坐标;
PushT planning 需要保留高条件方差但任务关键的接触坐标。
```

实验里的线性探针结果正落在这个冲突上:
位置 R² 基本保住,角度 R² 从约 0.80 掉到 0.68。
这不是普通 collapse,而是**选择性充分性损失**。

---

## 4. 为什么多步 unroll 会"低 drift,坏 planning"

多步训练把目标从一步改成:

$$
\mathcal{L}_K(\phi,f)
=
\sum_{k=1}^K w_k\,
\mathbb{E}\lVert f^{(k)}_\phi(C_t,A_{t:t+k-1})-\phi(O_{t+k})\rVert^2.
$$

若 encoder 也吃这个梯度,上一节的选择偏置会变得更强:

```text
一步预测惩罚的是一步条件方差;
多步预测惩罚的是开环传播后的条件方差。
```

在局部线性近似下,某个 latent 方向 $u$ 的 k 步不可约误差大致含有:

$$
u^\top P_k u.
$$

接触/角度方向的 $P_k$ 增长更快,所以它在多步 loss 中变得更"昂贵"。
如果 encoder 可以重写坐标,最省 loss 的做法不是一定把 $f$ 学得更物理,
而是把这些高放大方向从 latent 中降权、缠绕、替换成更容易滚的特征。

这解释了看似矛盾的实验。注意:下面的 planning gap 仍需用真正的 3-frame planner history
重跑确认,因为当前工程审查发现 multi-step 模型可能被 1-frame cold-start 评测污染。

| 现象 | 理论解释 |
| --- | --- |
| 多步模型 mse@8 0.315 -> 0.177 | 它确实学到了更自洽、更容易开环滚的 latent dynamics |
| PushT planning 82% -> 22% | 这个 latent 的欧氏几何不再充分表达 goal-critical pose |
| action spread 0.374 > 0.307 | 不是动作不敏感,而是动作敏感地滚向一个任务不充分的坐标系 |
| 角度 R² 0.80 -> 0.68 | 高条件方差的接触角度被选择性牺牲 |

关键是 self-drift 的定义:

$$
D_k=\mathbb{E}\lVert\hat Z_{t+k}-\phi(O_{t+k})\rVert^2.
$$

它同时使用同一个模型的 predictor 和 encoder。只要二者合谋换一个更容易预测的坐标系,
$D_k$ 就能下降。它测的是**自一致性**,不是**对真实任务状态的保真性**。

---

## 5. planning 需要的充分性到底是什么

LeWM 的 MPC/CEM cost 近似是:

$$
J_z(a_{t:t+H-1})
=
\lVert\hat Z_{t+H}(a)-Z_g\rVert^2.
$$

这个 cost 有用,需要 latent 满足一个比预测更强的性质:

```text
对候选动作序列 a,b,
若 a 的真实终态更接近目标,latent terminal distance 也应给 a 更低 cost。
```

形式化地说,至少要有 ranking preservation:

$$
d_{\text{task}}(S^a_{t+H},G)
<
d_{\text{task}}(S^b_{t+H},G)
\quad\Rightarrow\quad
\lVert\phi(S^a_{t+H})-\phi(G)\rVert^2
<
\lVert\phi(S^b_{t+H})-\phi(G)\rVert^2
$$

在大多数候选对上近似成立。

SIGReg 只给出 $Z\sim N(0,I)$,即全局边缘距离被白化;
它没有给出上面的 ranking preservation。
一个变量可以在高斯 latent 中满秩、各向同性、动作敏感,
但仍然把 goal-critical angle 压到一个对欧氏 goal cost 不够敏感的位置。

所以我们需要区分三种东西:

| 名称 | 含义 | LeWM 原 loss 是否保证 |
| --- | --- | --- |
| 维度非坍缩 | latent 占满维度,边缘近似各向同性 | SIGReg 基本保证 |
| 自预测充分 | $Z_{t+1}$ 能由 $Z_t,A_t$ 预测 | pred_loss 保证 |
| 控制充分 | $Z$ 保留任务状态/goal ranking/cost-to-go 所需信息 | **不保证** |

旧稿里说的"充分性 loss"真正要补的是第三项,不是第一项或第二项。

---

## 6. 由理论推出的设计原则

上面的推导给出一个很明确的原则:

> **不能让多步 self-consistency 的梯度同时优化 encoder 和 predictor。**
> 因为这个梯度天然偏好"更可预测的坐标",会和控制充分性冲突。

更一般地,我们要把两个目标拆开:

```text
Representation / encoder:
  学一个对任务状态和 goal geometry 足够充分的坐标系。

Dynamics / predictor:
  在这个已经固定语义的坐标系里,学会长程稳定 rollout。
```

这不是工程 trick,而是由第 2 节的等式推出的分离原则。
如果坐标系可以被长程预测误差改写,loss 就有捷径;
如果坐标系被 grounded/sufficient 的目标锚住,多步误差才会被迫用于改进 $f$。

---

## 7. 主损失:Dynamics-only multi-step as anti-erosion

最小改法是保留 LeWM 原生的一步目标来塑形 encoder,但把多步项变成 dynamics-only。
这里的 dynamics module 不只包括 Transformer predictor,也包括 action encoder 和 prediction projection head;
它不包括 visual encoder 和 projector,因为后两者定义 latent 坐标系本身。

$$
\mathcal{L}_{\text{sep}}(\phi,f)
=
\underbrace{
\mathbb{E}\lVert f(C_t)-Z_{t+1}\rVert^2
+\lambda \mathcal{R}_{\text{SIGReg}}(Z)
}_{\text{LeWM one-step}}
+
\beta
\underbrace{
\sum_{k=1}^K w_k
\mathbb{E}
\left\lVert
f^{(k)}(\mathrm{sg}[C_t],A_{t:t+k-1})
-
\mathrm{sg}[Z_{t+k}]
\right\rVert^2
}_{\text{dynamics-only rollout}}
$$

其中 `sg` 是 stop-gradient。

更准确地说,这个多步项本身不是"充分性 loss";
它是一个**anti-erosion loss**:

```text
它不创造充分性,但阻止多步自一致性梯度侵蚀 encoder 已有的充分性。
```

如果单步 LeWM encoder 已经足够支持 planning(当前 baseline 82%),这个最小方案就可能够用:

```text
encoder 维持单步 LeWM 的 planning geometry;
predictor 额外学习开环稳定性;
drift 和 planning 有机会第一次同向改善。
```

### 7.1 为什么这比"普通多步 loss"更理论干净

普通多步:

```text
β L_K(φ,f)  同时更新 φ 和 f
```

会优化:

```text
坐标系本身是否容易滚。
```

stop-gradient 多步:

```text
β L_K(sg[φ], dynamics)
```

只优化:

```text
在既定坐标系中 f 是否能滚准。
```

这正好切断第 2-4 节里的捷径。

实现上还有一个关键细节:seed embeddings 和 target embeddings 要 detach,但 rollout 中预测出来的
`nxt` 不能 detach。否则多步 loss 只会训练最后一步,不能通过 open-loop BPTT 约束早期 dynamics。

### 7.2 这条方案的可证伪预测

1. 角度 probe R² 应接近单步 baseline,而不是多步共训模型。
2. 多步 drift 应低于单步 baseline,因为 $f$ 仍吃到了 open-loop 训练。
3. planning 成功率应接近或超过单步 baseline。
4. 若 drift 降了但 planning 仍崩,说明单步 encoder 的充分性并未被保住,
   或者 planner cost geometry 本身需要显式 sufficiency anchor。

---

## 8. 若主损失不够:显式控制充分性

第 7 节的方案依赖一个经验事实:当前单步 LeWM encoder 在 PushT 上 planning 好。
理论上更完整的目标应写成:

$$
\mathcal{L}_{\text{sufficient-dynamics}}
=
\mathcal{L}_{\text{LeWM-1step}}
+\beta \mathcal{L}_{\text{rollout}}^{\text{sg}}
+\gamma \mathcal{L}_{\text{suff}}.
$$

$\mathcal{L}_{\text{suff}}$ 才是真正的充分性项。它有三种强弱不同的实现:

### S1. 状态 probe sufficiency(最干净,PushT 可用)

如果环境给真状态或关键状态:

$$
Y_t=(x_{\text{agent}},x_{\text{block}},\theta_{\text{block}},\ldots),
$$

加一个小 probe $q$。如果只是训练后 probe,它是诊断;如果作为 loss,必须让梯度回到
encoder/projector:

$$
\mathcal{L}_{\text{suff-state}}
=
\min_{\phi,q}\mathbb{E}\lVert q(\phi(O_t))-Y_t\rVert^2.
$$

这在理论上最直接:防止高条件方差但任务关键的 $Y$ 被 encoder 丢掉。
缺点是引入监督,不如纯 JEPA 干净;优点是诊断和论文论证最硬。

### S2. Goal-ranking sufficiency(最贴 planning)

用同一批 candidate action,要求 latent terminal distance 的排序接近真实任务距离或环境成功距离:

$$
\mathcal{L}_{\text{rank}}
=
\mathbb{E}_{a,b}
\max\left(0,m+
J_z(a)-J_z(b)\right)
\quad
\text{when } d_{\text{task}}(a)<d_{\text{task}}(b).
$$

它直接优化 planning geometry,但更贵,也更像 planner-level loss。

### S3. Action-realizable / inverse-dynamics sufficiency(弱监督或自监督)

要求相邻 latent pair 能解释动作:

$$
\mathcal{L}_{\text{idm}}
=
\mathbb{E}\lVert g(Z_t,Z_{t+1})-A_t\rVert^2.
$$

它能防止 encoder 丢掉动作造成的可控差异。PLDM 类方法也用这一项。
但在当前 PushT 证据下要谨慎:多步模型并没有失去 action sensitivity,
真正丢的是角度这样的 goal-critical DOF,所以 IDM 可能不够。

### 推荐顺序

```text
先跑 L_sep:最小、最符合 LeWM、只验证"多步腐蚀 encoder"这个机制。
若 planning 不回来,再加 S1 state sufficiency probe:
  因为 PushT 的失败变量已经被定位到 angle,这是最小的显式补强。
```

---

## 9. 与已有实验的对账

| 已有事实 | 本稿理论解释 |
| --- | --- |
| SIGReg 下 z 无聚类,Jacobian 有 contact regime | SIGReg 高斯化边缘,不约束 transition;离散只会藏在 $f$ 的局部算子里 |
| oracle regime 路由能降 drift,可学门控不划算 | regime 确实解释一部分动力学方差,但路由错误代价高;这不解决 encoder 充分性 |
| multi-step drift 低但 planning 崩 | 多步共训选择更可预测坐标,降低 self-drift,破坏 control sufficiency |
| action spread 没降反升 | 失败不是动作无关,而是动作敏感的 latent geometry 不再对 goal ranking 充分 |
| angle probe 掉、position 保住 | 角度是高条件方差+高任务价值方向,正是理论预测会被牺牲的变量 |
| ID drift 多数是 on-manifold diffusion | 问题不是简单投影回流形;需要在坐标学习阶段保留 control-relevant uncertainty |

---

## 10. 实验计划

### A. 最小主方案

实现:

```text
cfg.wm.unroll_sg:
  K = 5
  β ∈ {0.25, 0.5, 1.0, 2.0}
  targets and seed embeddings stop-grad for multi-step term
  one-step pred_loss + SIGReg keep normal encoder gradients
```

评测:

1. `regime_lewm_iter2_eval.py`: mse@k,特别是 k=5/8。
2. `eval_wm.py`: PushT planning 50 ep。
3. 线性 probe:agent xy、block xy、block angle。
4. action-quality ranking:Spearman/top1/top5,确认 latent cost landscape 是否保住。

判据:

```text
drift < 单步 baseline
planning ≈ 单步 baseline 或更高
angle R² ≈ 单步 baseline
```

三条同时成立,主机制立。

### B. 若 A 失败

加最小显式 sufficiency:

```text
L = L_sep + γ L_suff-state
Y = agent xy + block xy + block angle
γ sweep small: {0.01, 0.03, 0.1}
```

判据不变,但额外看:

```text
angle R² 是否随 γ 单调恢复;
planning 是否随 angle R² 恢复而恢复;
drift 是否还能保持低于 baseline。
```

如果 angle R² 回来了 planning 仍不回,问题就不只是 representation sufficiency,
而是 terminal latent L2 本身不是合适的 planning geometry。

---

## 11. 最终收敛的结论

这条线现在可以收敛成三个可写进 paper 的命题:

1. **LeWM/SIGReg 的盲点不是维度坍缩,而是可预测性偏置。**
   在 $Z\sim N(0,I)$ 约束下,MSE transition loss 等价于选择条件方差低的坐标;
   它保证 latent 好预测,不保证 latent 对控制充分。

2. **接触动力学把这个盲点变成系统性 failure mode。**
   hybrid/contact switching 让角度等任务关键自由度具有高条件方差和高开环放大;
   多步共训会更强地惩罚这些方向,于是出现"低 self-drift、高 planning failure"。

3. **正确的 loss 结构是分离表征充分性和动力学稳定性。**
   多步 rollout 应训练 predictor 在固定语义坐标系里稳定滚动,
   而不应反过来改写 encoder 坐标系。
   最小实现是 $\mathcal{L}_{\text{LeWM-1step}}+\beta\mathcal{L}_{\text{rollout}}^{sg}$;
   若单步 encoder 的充分性不够,再加一个小的 control-sufficiency anchor。

一句话版本:

> **LeWM 缺的不是更强的自一致 drift loss,而是防止高斯预测目标把"难预测但可控关键"信息换掉的充分动力学原则。**
