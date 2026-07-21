# Temporal 方向备忘录：Horizon-Bundle World Model（2026-07-17）

> **2026-07-21 导航更新：**本文现作为 chronological lab notebook，记录
> Horizon-Bundle → OE-WM → BP-OE → topology/lineage 的实验演化。开头的
> Horizon-Bundle proposal 已被后续 Gate 拒绝，BL-WM 也被最新 causal refit
> 降级。当前结论、停止清单和下一 Gates 统一见
> [LeWM planning 研究现状总账](lewm_planning_status_20260721.md)。
>
> **状态：探索性假说 / 实验预注册草案，不是当前项目的既成结论。**
>
> 本文暂存一条比 CritWM 标量临界控制更高上限的方向，以及在实现新模型前必须完成的
> 两个低成本审计。若 Gate 不通过，应保留负结果，但不继续包装方法；若通过，再把结论
> 合并进[实验主账本](lewm_gaussian_dynamics_direction.md)和
> [统一文献地图](worldmodel_literature.md)。

---

## 0. 一句话决定

当前最值得赌的不是继续把一个共享 latent 的全局 `rate` 调到 1，而是检验下面这个更强的命题：

> **同一个物理状态没有一个对所有递归深度都最优的 latent 表示；world model
> 应该按“剩余想象 horizon”提供一束相互连接的状态表示。**

工作名：

```text
Horizon-Bundle World Model
World Models Need a State for Every Horizon
```

但现在不能直接开做大模型。先完成：

1. **horizon-matching audit**：固定训练 horizon 的模型是否在不同 planning horizon
   上发生稳定、可复现的交叉，而不是存在一个通吃所有 horizon 的 `K`；
2. **iso-rate audit**：`rate` 近乎相同的模型是否仍有显著不同的候选排序和 planning，
   从而确认 CritWM 当前只控制了一个不充分的标量。

这两个审计分别回答“是否真的需要 horizon-specific state”和“为什么单一临界率不够”。

---

## 1. 为什么当前方向的上限不够高

### 1.1 我们已经有的强事实

当前实验最稳的发现不是“K-step drift 更低”，而是：

1. `rate(K)` 随训练 horizon 单调下降，且跨 seed 稳定；
2. 多步收益主要住在 **encoder / latent gauge**，不是 predictor 容量；
3. teacher forcing、predictor-only self-composition、独立 gain penalty 都不能复现完整收益；
4. 有效机制需要 self-composition gradient 真正到达 encoder，并允许
   encoder–predictor co-adaptation；
5. K-step 训练对不可预测误差方向和 action signal 做非对称分配，而非统一收缩所有方向；
6. 远 goal 会放大 `K=1` 与 `K=5` 的 planning 差异，但 planning horizon 从 5
   拉到 8/10 又出现候选排序悬崖，且现有 `rate` / separation 证书没有预警。

理论最小模型进一步说明：

```text
K=1 risk 对“误差在哪里被后续动力学放大”精确盲；
K>1 risk 按累计不可预测放大量重新分配表示预算。
```

详见[理论最小模型笔记](theory_minmodel_notes.md)。

### 1.2 CritWM v1 给出的上限警告

CritWM v1 的确定性 `rate≈0.998`，但标准 CEM 下：

| 模型 | goal offset 25 | goal offset 40 |
| --- | ---: | ---: |
| fixed `K=5` | 87.0 | 56.0 |
| CritWM v1 | 82.7 | 42.0 |

因此：

```text
rate≈1 既不是充分的 planning 目标，
也不能唯一确定“哪一种临界 latent geometry”是好的。
```

CritWM v1 还有 sensor dropout 偏置、BatchNorm buffer 被探针改写、`gamma`
长期饱和在上界等训练环问题；这些问题应该修，但即使全部修掉，也只能证明
“一个 controller 能稳定追踪某个标量”，不能证明这个标量足以决定 control quality。

### 1.3 更深一层的可能问题：shared latent 本身是隐藏约束

目前所有固定 `K`、multi-`K`、CritWM 方案都默认：

```text
phi_0 = phi_1 = ... = phi_H = phi
```

也就是要求同一个 latent 同时服务于：

- 下一步局部拟合；
- 5 步 self-composition；
- 8/10 步候选排序；
- 短 goal 与远 goal；
- error suppression 与 action sensitivity。

但我们的理论恰恰说明，不同 horizon 对表示方向的风险权重不同。把所有 horizon
绑进一个 `phi`，可能不是正则化，而是在强迫一组互不兼容的 representation quotient
做折中。固定 `K` 的 planning-horizon 悬崖可能就是这个约束第一次显形。

---

## 2. 新 idea：Horizon-Bundle World Model

### 2.1 核心对象

对最大规划深度 `H_max`，不再只有一个 latent state，而是：

```text
z_t^(r) = phi_r(o_t),       r ∈ {0, 1, ..., H_max}
```

其中 `r` 表示**从当前节点到规划终点还剩多少次递归 transition**，不是物理时间尺度。
相邻 gauge 之间由跨层动力学连接：

```text
F_r : (z_t^(r), a_t) -> z_(t+1)^(r-1)
```

一个长度为 `h` 的 rollout 是：

```text
phi_h(o_t)
  --F_h,a_t--> phi_(h-1)(o_(t+1))
  --F_(h-1),a_(t+1)--> ...
  --F_1,a_(t+h-1)--> phi_0(o_(t+h))
```

它仍是逐步、同物理分辨率的 dynamics，只是每走一步，表示所需承担的
“剩余递归风险”也减一。

### 2.2 训练目标

对长度 `h` 的训练片段，定义：

```text
zhat_0^(h) = phi_h(o_t)
zhat_(j+1)^(h-j-1)
    = F_(h-j)(zhat_j^(h-j), a_(t+j))

L_h = (1/h) sum_(j=1..h)
      || zhat_j^(h-j) - phi_(h-j)(o_(t+j)) ||²
```

总目标：

```text
L_bundle
  = E_(h~p(h))[L_h]
  + lambda * sum_r SIGReg(phi_r(o))
  + beta * L_action_guard
  + eta * L_adapter_budget
```

关键约束：

- 预测点全部来自真正的递归 rollout，不退化为 teacher forcing；
- gradient 必须穿过整条 self-composition path；
- encoder 与 predictor 必须联合更新；
- 每个 `phi_r` 都单独接受防 collapse 的 marginal regularization；
- `L_action_guard` 只保护 counterfactual action distinguishability，不直接把 gain
  压小，避免 EchoReg 式“关掉动作通道”的作弊；
- 所有 horizon 使用同一套数据，不增加未来标签。

若令所有 `phi_r` 共享、所有 `F_r` 共享，就退化回普通 shared-latent multistep LeWM。
因此该模型把“一个 universal state 是否足够”变成可直接 ablate 的结构假设。

### 2.3 Planning 时怎么用

长度为 `h` 的 CEM candidate 从 `phi_h(o_t)` 出发，经 `F_h,...,F_1`
到达 `phi_0`。terminal goal cost 为：

```text
c_H = || zhat_(t+h)^0 - phi_0(o_goal) ||²
```

若使用中间 stage cost，则第 `j` 步只能在匹配的 gauge 内比较：

```text
c_j = d_(h-j)(zhat_(t+j)^(h-j), phi_(h-j)(o_goal))
```

不能把 `phi_8` 的状态直接和 `phi_0` 的 goal 做欧氏距离。

MPC 每次真实执行一步后重新规划时，重新把新 observation 编码到
`phi_h`；因此不需要让跨规划周期的 latent 永久停留在同一 gauge。

### 2.4 最小可实现版本

第一版不应复制 `H_max+1` 个完整 encoder。建议：

```text
shared visual trunk
  + small residual/FiLM horizon adapters for phi_r
shared transition trunk
  + remaining-horizon embedding / small adapters for F_r
```

这样可以做严格的参数匹配，并区分三种可能：

1. 只需 horizon-conditioned predictor；
2. 真正需要 horizon-conditioned encoder；
3. encoder 与 predictor 都需要 bundle co-adaptation。

建议先只支持稀疏 knots `r∈{0,1,3,5,8,10}`，中间 horizon 用共享或插值 adapter，
避免第一版在架构复杂度上失控。

---

## 3. 这个 idea 与现有方向的差异

| 方向 | 已经解决什么 | Horizon-Bundle 的不同问题 |
| --- | --- | --- |
| Fast-LeWorldModel | action-prefix、多 horizon terminal prediction、并行加速，绕开 autoregressive compounding | 保留真实 self-composition；研究不同剩余递归深度是否需要不同 state quotient |
| Temporal Straightening | 在单一 latent 中拉直真实轨迹 tangent，改善 planner conditioning | 处理 off-trajectory recursive error transport 及其 horizon-dependent representation allocation |
| HWM / MTS3 | 按物理时间尺度建 fast/slow dynamics 或 hierarchy | 所有层仍走一个环境 step；索引是 time-to-go，不是 temporal abstraction |
| UWM-JEPA / multimodal successor | 部分可观测下表示多种未来 belief | 当前假说在确定性 LeWM 内也成立；多样性来自 planning query，不是 hidden-future mode |
| CritWM | 在一个 shared latent 内调节全局 scalar `rate` | 放松“一个 gauge + 一个 setpoint 可服务所有 horizon”的前提 |
| Control Theory of Predictability / Sensing Clocks | 诊断 planner-facing cost error、validity horizon、何时应重感知 | 改变被诊断的 representation architecture，而不只给已有模型画有效边界 |

因此不能把贡献写成“首次发现 prediction-control gap”“首次做 multi-horizon”
或“首次认证 imagination horizon”。唯一有机会成立的新主张是：

> **prediction horizon 不只是 loss 的长度，而是 state representation 的查询变量；
> shared latent 是可被实验否证的 tying constraint。**

相关 novelty 边界以[统一文献地图](worldmodel_literature.md)最后一节为准。

---

## 4. Gate A：horizon-matching audit

### 4.1 要回答的问题

不是验证简单的“`K_train = H_plan` 一定最好”，而是验证：

```text
不同 training horizon 选择的 latent gauge，
是否拥有不同且稳定的 planning-horizon operating envelope？
```

如果一个固定 `K` 在所有 `H_plan`、goal distance、seed 上都占优，那么 bundle 没有必要；
最多做一个更好的 shared latent 即可。

### 4.2 零训练主矩阵

先只使用已有 checkpoint：

```text
K_train     ∈ {1, 2, 3, 5, 10}
H_plan      ∈ {1, 3, 5, 8, 10}
goal_offset ∈ {25, 40, 60}
seed        = 所有现有训练 seed
```

每个格子同时跑两套 compute protocol：

1. **fixed candidates**：固定 CEM candidate 数，反映实际 wall-clock / capacity；
2. **fixed model calls**：固定 `candidate × horizon × iteration`，排除长 horizon
   仅因模型调用数更多而占便宜或吃亏。

另保留一个大预算 stress test。已有 `h8` 从 300 提到 1000 samples 仍不能救
`K=5`，这是先验线索，但不能替代完整矩阵。

### 4.3 必须补的 candidate-rank oracle

只看 episode success 无法知道是 representation、CEM 搜索还是环境容错。对每个规划状态：

1. 保存同一批 candidate action sequence；
2. 记录每个 checkpoint 的 predicted terminal cost 和完整 latent path；
3. 用可克隆 simulator 执行分层 candidate 子集：
   top-ranked、near-tie、随机、预测 gain 分位；
4. 得到 true terminal cost / return；
5. 计算：

```text
Spearman / Kendall rank correlation
pairwise inversion rate
top-k precision
simple regret = true_cost(predicted_best) - true_cost(oracle_best)
```

同一 candidate bank 应被所有 `K_train` 模型复用，形成 paired comparison。

### 4.4 同时记录的机制量

- deterministic audited `rate` 及其局部分布，而不只报均值；
- teacher-forced one-step residual、`drift_k`；
- free-vs-composed `D*` gap；
- action gain / counterfactual action separation；
- state/goal observability probes；
- contact/free regime、candidate 是否离开数据 manifold；
- rollout depth 上第一次发生 rank inversion 的位置。

### 4.5 预注册判决

**Bundle Gate 通过**需同时满足：

1. `K_train × H_plan` 对 candidate regret/rank inversion 有稳定 interaction；
2. 至少两个 planning-horizon 区间由不同 `K` 显著占优，而非一个 `K`
   在误差条内通吃；
3. interaction 在 fixed-model-calls、共享 candidate bank、held-out seed 下仍在；
4. 交叉主要体现在 ranking，而不只是某个 planner 的搜索效率。

建议把“显著占优”预注册为：成功率差至少 5 个百分点，或 paired candidate-regret
bootstrap 95% CI 不跨 0；主判据以 candidate ranking 为准，success 为能力终点。

**Kill / 降级条件：**

- 一个固定 `K` 在所有 horizon 上 Pareto-dominant；
- 交叉在匹配 model calls 或共享 candidates 后消失；
- 差异完全由 CEM 维度和采样预算解释；
- representation-side 指标没有任何 horizon interaction。

发生以上情况时，停止 Horizon-Bundle，转向“找一个更好的 universal latent /
planner”而不是堆 horizon adapters。

---

## 5. Gate B：iso-rate audit

### 5.1 为什么必须做

`K=5` 与 CritWM v1 已经给出一个很强的 discovery pair：

```text
rate(K5)       ≈ 0.98–0.99
rate(CritWM)   ≈ 0.998
goal40 success = 56 vs 42
```

但这还不能直接写成结论，因为二者训练路径、loss mixture、sensor 状态和 checkpoint
选择不同。iso-rate audit 的目标是构造“同 rate、不同其他属性”的受控模型组，
确认：

```text
scalar rate 是否对 planning state 具有识别性？
```

### 5.2 先统一测量协议

所有模型用同一批 observation/action/perturbation、同一 horizon 和同一 estimator：

- `model.eval()`，关闭 dropout；
- sensor 全程无梯度、无状态；
- 禁止改写 BatchNorm running buffers；
- 报 bootstrap CI 和局部 rate 分布；
- rate-matched 定义为 `|Δrate|≤0.03` 且 CI 大量重叠；
- checkpoint 选择规则在看 planning 前锁定。

否则“iso-rate”可能只是测量器不等价。

### 5.3 受控模型组

第一组复测已有：

- fixed `K=5`；
- CritWM v1 final checkpoint；
- CritWM v1 训练轨迹中所有 rate-matched checkpoints。

第二组补最小训练：

```text
L = L_1step + gamma * L_K5_openloop + lambda * L_SIGReg
gamma ∈ {0.3, 1.0, 3.0, 5.0}
```

每个 `gamma` 至少 3 seeds，训练预算、初始化族、scheduler 和数据顺序匹配。
再加入：

- 修复 sensor 后的 CritWM v2 controller；
- 一个 controller-gamma 轨迹回放对照：把 CritWM 的 `gamma_t` 序列离线重放，
  但不读在线 sensor，用来分离 feedback 与训练 curriculum；
- fixed `K=1` / `K=5` 作为外部锚点。

### 5.4 不只测 planning success

对每一对 rate-matched 模型，测四层差异：

| 层 | 指标 | 它回答什么 |
| --- | --- | --- |
| local fidelity | one-step residual、`drift_1`、refit `D*` | 是否为追 rate 牺牲了局部模型真实性 |
| action sufficiency | action gain、inverse-action probe、counterfactual separation、local controllability | 收缩是否关掉了可控方向 |
| state/goal sufficiency | state/goal probes、goal-distance ordering | latent 是否仍保留 planner 需要的信息 |
| planner-facing | candidate rank correlation、pairwise inversion、top-k regret、success | 相同 rate 是否导向不同决策 |

另外比较 local gain 的 tail、regime 条件分布和 non-normal product，而不是只比较
几何平均 scalar。两个模型可以拥有同一个均值，却把风险放在完全不同的候选路径上。

### 5.5 固定 gamma 的判决表

| 结果 | 判决 |
| --- | --- |
| fixed `gamma=5` 也接近 rate 1 且 planning 差 | 问题在 target/objective；修 controller 不能救核心 claim |
| fixed `gamma=5` planning 好、CritWM 差 | 主要是 sensor、feedback 或训练路径问题 |
| 较低 `gamma`、`rate>1` 反而 planning 更好 | “临界 1 是最优 setpoint”被否证 |
| rate 相同但 action/goal sufficiency 不同 | rate 只控制 error channel，缺少 control-relevant axis |
| 上述 probe 也相同但 candidate ranking 不同 | 需要路径条件、分布尾部或 horizon-specific certificate |
| rate-matched 模型在完整指标上也等价 | CritWM scalar 假说获得支持，Bundle 动机明显变弱 |

### 5.6 预注册判决

**“rate 不充分”确认**要求：

1. 不只依赖 `K=5` vs CritWM v1 一个事后 pair；
2. 至少在一组预先锁定的 fixed-gamma / controller 对照中，rate-matched 模型仍有
   稳定 candidate-ranking 差异；
3. 差异跨 seed，并在共享 candidate bank 上成立；
4. 找到至少一个能解释差异的额外轴，或明确证明 scalar mean 丢掉了局部 tail / path 信息。

建议能力效应阈值为：远 goal success 差至少 8 个百分点，或 paired candidate-regret
95% CI 不跨 0。若只有 success 差而 ranking 没差，不能归因给 world-model geometry。

---

## 6. Gate C（便宜但关键）：cross-horizon gauge transport

即使 Gate A 出现交叉，也要排除“这些 encoder 只是同一个 state 的简单旋转/缩放”。
对已有 `phi_K` 冻结后，用同一批 observation 拟合：

```text
T_(K->K') : phi_K(o) -> phi_K'(o)
```

按复杂度递增比较：

1. orthogonal Procrustes；
2. affine / ridge map；
3. 小型 2-layer MLP。

held-out 上不只测 latent reconstruction，还测：

```text
state alignment
action displacement alignment
dynamics commutativity
candidate goal-distance / ranking preservation
```

其中 dynamics commutativity 的核心问题是：

```text
T(f_K(z, a))  是否约等于  f_K'(T(z), a)
```

**判决：**

- 若一个简单 affine map 就能保留 dynamics 和 candidate ranking，完整
  Horizon-Bundle 过重；优先做 shared latent + horizon conditioning；
- 若 state 可对齐但 action displacement / rollout ranking 不可对齐，说明差异确实住在
  dynamical gauge，而不是普通 feature basis；
- 若只有非线性 map 能部分对齐，且 residual 随 horizon/contact regime 系统变化，
  才支持“每个 horizon 需要不同 quotient”的强解释。

---

## 7. Gates 通过后的最小原型实验

### 7.1 Baselines

同参数量、同数据、同 optimizer、同总 transition loss evaluations 比较：

1. best fixed `K` LeWM；
2. shared encoder + multi-`K` objective；
3. shared encoder + horizon-conditioned predictor；
4. horizon-conditioned encoder + shared predictor；
5. full Horizon-Bundle；
6. `K=1 + Temporal Straightening`；
7. Fast/prefix terminal predictor（能力参照，不混同机制）；
8. CritWM v2（若训练环先通过自己的硬门）。

### 7.2 必做 ablations

- adapter 参数量匹配；
- bundle knots 数量；
- 是否对每个 `phi_r` 单独做 SIGReg；
- full self-composition gradient vs stop-gradient；
- joint co-adaptation vs frozen encoder；
- 有无 action guard；
- terminal-only loss vs intermediate matched-gauge targets；
- 相邻 gauge 独立参数 vs shared trunk + small adapters。

其中 stop-gradient、frozen encoder 和独立 gain penalty 已有强负先验，应作为机制对照，
不再当主要候选方法。

### 7.3 评价协议

开发环境用 PushT，把现有压力面完整展开：

```text
goal_offset × planning_horizon × latent_capacity × seed
```

主终点不是某一个 `h=5` 数字，而是：

- across-horizon success / regret curve；
- 最坏 horizon performance；
- candidate ranking frontier；
- fixed-compute Pareto curve；
- one-step fidelity 与 action sufficiency 是否保住；
- 是否把 `h8/h10` 的 imagination-depth cliff 推远。

在写泛化 claim 前，必须在第二环境盲测；优先使用已有 pipeline 中真正需要长程
决策的 OGBench-Cube 设置。若第二环境没有可观测的 horizon interaction，只能写
PushT mechanism paper，不能写 universal architecture。

### 7.4 方法生死线

Horizon-Bundle 只有同时满足以下条件才算 work：

1. 比 best fixed `K` 和 shared multi-`K` 更好，而不只是比 `K=1` 好；
2. 提升发生在至少两个 planning horizon，不是把最优点从 5 搬到 8；
3. fixed model calls 和参数匹配后仍成立；
4. candidate ranking 的改善先于或伴随 success 改善；
5. encoder-conditioned ablation 提供不可由 predictor conditioning 替代的增益；
6. 第二环境至少复现“shared latent tying 有代价”这一方向性结果。

以下任一成立就停止：

- shared multi-`K` 与 full bundle 等价；
- 只有增加参数量后才赢；
- gain 仅来自 goal encoder / cost scale 改变；
- training 更稳但候选 ranking 不变；
- 只在单 seed、单 horizon、单 goal distance 上有效。

---

## 8. 可能的论文形态

### 8.1 强版本

```text
Observation:
Prediction horizon changes encoder-side dynamical gauge, and no fixed gauge
dominates across recursive depths.

Theory:
Different remaining horizons induce different weights on unpredictable
amplification, so shared state tying creates an avoidable representation
compromise.

Method:
A horizon-indexed bundle of co-adapted latent states and cross-gauge
transitions.

Result:
Better candidate ranking and planning across horizons at matched compute.
```

### 8.2 中版本

若 bundle 方法收益一般，但 Gate A/B/C 很强，可以写机制论文：

```text
There Is No Horizon-Agnostic State for Recursive World Models
```

贡献是交叉矩阵、iso-rate 反例、gauge transport 与理论分配，而不是硬包装一个模型。

### 8.3 负结果版本

若一个 fixed `K` 或 shared multi-`K` 通吃，则诚实结论是：

```text
horizon-induced encoder geometry is real,
but it does not require horizon-indexed state;
the remaining failure is planner-facing ranking or optimization.
```

这时回到[imagination frontier 提案](imagination_frontier_flagship.md)中的
candidate-level ranking/certificate，而不是继续加 representation hierarchy。

---

## 9. 最短执行顺序

```text
Day 1–2  horizon-matching matrix + shared candidate oracle logs
Day 2–3  deterministic rate re-audit + existing K5/CritWM iso-rate pair
Day 3–4  cross-horizon gauge transport
Day 4    Gate review：决定是否启动 fixed-gamma confirmation
Week 2   fixed-gamma / controller replay confirmation
Week 3   只有 Gates A+B 通过才实现 Horizon-Bundle minimal prototype
```

实验记录必须按下面顺序解释：

```text
先判 ranking interaction
→ 再判 representation interaction
→ 再判 scalar rate 是否不充分
→ 最后才判新 architecture 是否值得做
```

这样即使新模型失败，也会留下一个清楚、可复现、不会被近期文献轻易覆盖的科学结论。

---

## 10. A100 首轮执行日志（2026-07-17）

> 本节记录 Week-1 kit 在新 A100 节点上的第一次真实执行。这里的 smoke 数字只用于
> 验证测量链路，不作为 Horizon-Bundle / CritWM 的科学结果。

### 10.1 机器、数据与代码状态

```text
machine        = 4 × NVIDIA A100-SXM4-80GB
remote repo    = /225010117/code/learn_wm
remote commit  = f2dbc7e
dataset        = /225010117/data/pusht_expert_train.h5
pixel sidecar  = /225010117/data/pusht_expert_train_pixels.npy
checkpoint dir = /225010117/stablewm/checkpoints
```

新节点初始只有 CritWM v1 checkpoint，没有 Gate A 所需的
`K_train∈{1,2,3,5,10}` 全套旧 checkpoint。因此本轮先启动 verification training
wave；Gate A matrix 在旧 checkpoint 被传入前不应空跑。

### 10.2 实跑暴露的问题与判决

| 问题 | 直接后果 | 修复 / 当前判决 |
| --- | --- | --- |
| `candidate_oracle.py` 调用了不存在的 `World.reset_to`、`World.step_env`、`World.get_state` 和 `WorldModelPolicy.plan_once` | 原脚本无法进入第一个 oracle state | 改为复用 dataset-eval 的 `_extract_init_goal`、history seeding 和 rendered-image refresh，再直接调用已配置的 solver |
| 原 `--bank` 只复用 start rows，不复用 action candidates | 不同模型实际比较不同 CEM candidate set，不能称 paired audit | reference 模型保存量化后的真实 candidate tensor；其他 checkpoint 直接对同一 tensor 重新打分 |
| 原合法起点用全局 `step_idx.max()` 判断 | 可能跨 episode 取 goal | 改为逐 episode 最大 step，保存 `(episode,start,row)` |
| 原 true cost 直接取 `state[:5]` 欧氏距离 | angle 未 wrap，且“block pose”命名与实际维度不一致 | 使用与 PushT success 对齐的 position L2 + wrapped/threshold-scaled angle，并同时保存 position、angle、success |
| Gate A 只覆盖 `plan_config.horizon`，没有同步短 horizon 的 `receding_horizon` | `H=1/3` 时 policy 试图执行比 plan 更长的 action buffer，reshape 会失败 | `receding_horizon=min(H,5)`；`H≥5` 保留历史五步 MPC commitment |
| verification runbook 依赖未跟踪的 `outputs/pd/make_eval_dir.py` 与若干 certificate scripts | 新 checkout 会在训练后才晚失败 | train/eval/cert 分 phase；checkpoint 用显式 `name/weights_epoch_30.pt`；cert phase 先做依赖 preflight |
| 三个训练进程同时使用 `6 workers × prefetch 2` | `/dev/shm` 出现 DataLoader worker bus error；三项任务均在 sanity check 阶段退出 | 降到每进程 `2 workers × prefetch 1`；共享内存稳定在约 1.1/4.0GB |
| 第一轮重启忘记显式传 `pixels_path` | 直接读 44GB HDF5 pixels，只有约 `0.4 it/s` | 加 `+data.dataset.pixels_path=...npy`，lossless mmap 后稳定到约 `1.3–1.4 it/s` |
| 仅凭进程 exit code 判断训练成功 | Lightning 收到 SIGTERM 后可能正常退出，使 driver 误记 `DONE` | 成功条件增加 `weights_epoch_30.pt` 产物检查 |

两次失败启动都在 epoch 0 内主动停止，没有产生可用于比较的 checkpoint，不能混入正式结果。

### 10.3 Candidate-oracle smoke 结果

使用 CritWM epoch-30 checkpoint、一个 dataset state、8 个候选、`H=3` 做 reference
生成与 bank replay：

| 审计项 | 结果 |
| --- | ---: |
| 同一 state/action 重置后 terminal state 最大绝对差 | `0` |
| normalized action → env action → normalized action 最大 round-trip error | `2.38e-7` |
| reference 与重新加载 bank 后的 predicted cost 最大差 | `0` |
| true cost 最大差 | `0` |
| Spearman/Kendall/inversion/top-k/regret 是否逐项一致 | 全部一致 |

该单 state 的具体 rank 数字（例如 Spearman `0.071`）没有统计意义；这里唯一的判决是：

```text
reset-to-state、action denormalization、true-cost execution、
candidate serialization 和 paired rescoring 链路已经闭合。
```

### 10.4 正式 verification wave 状态

正式 driver 于北京时间 2026-07-17 16:08 左右启动。当前第一波：

```text
GPU 0  mix_d192_g10_s1   seed=1
GPU 1  mix_d192_g10_s7   seed=7
GPU 2  mix_d192_g05      seed=3072
GPU 3  保留给 oracle / early evaluation
```

三项均已通过 sanity validation 和第一次 backward，显存约 `26.7GB/GPU`，
训练速率约 `1.3–1.4 it/s`，未再出现 shared-memory、CUDA OOM 或 traceback。
第一波完成后 driver 自动启动：

```text
mix_d192_g20
pd_d192_k5_s7
pd_d192_k1_s7
```

远端日志：

```text
/225010117/logs/week1_verify_driver_retry2.log
/225010117/logs/week1_verify_retry2/train_<model>.log
```

### 10.5 为什么完整结果需要约 5–6 天

这不是一次训练本身需要 5–6 天，而是**六个独立的 30-epoch、全数据训练**
在四张卡上的总 makespan：

```text
每个 epoch 的 train batches = 11,306
实测吞吐                    ≈ 1.3–1.4 batch/s
纯 train 时间               ≈ 2.3 h/epoch
+ validation/checkpoint     ≈ 2.4 h/epoch
单模型 30 epochs            ≈ 70–75 h
六模型至少两波              ≈ 140–150 h ≈ 5.8–6.3 天
```

`batch_size=128`、30 epochs 和单卡训练是为了与已有 anchor 保持优化协议一致。
直接增大 batch、减少 epoch 或改成不同 global batch 的 DDP 虽然更快，但会把
iso-rate 对照重新引入 training-budget / optimization confound。

“什么时候有结果”应拆成：

| 里程碑 | 从启动起的预计时间 | 可做什么 |
| --- | ---: | --- |
| epoch 1 checkpoint | `~2.5 h` | 只做 pipeline/loss sanity，不判方法 |
| epoch 5 | `~12 h` | 第一轮 early planning，观察方向和灾难性失败 |
| epoch 10 | `~24 h` | 初步趋势，可决定是否继续明显失败的 arm |
| epoch 20 | `~48 h` | 较可信的中期比较，但仍需对齐最终 checkpoint |
| 第一波 epoch 30 | `~72 h` | 两个 gamma=1 seeds + gamma=0.5 的正式结果 |
| 第二波 epoch 30 | `~144 h` | gamma=2、K1/K5 seed anchors 到齐，完成预注册比较 |

因此最早在当天约 2–3 小时后就会有 checkpoint，约 12–24 小时能看到初步方向；
5–6 天指的是**六个模型全部到 30 epoch 并可做最终统计**，不是在此之前没有结果。

---

## 11. 5090 Gate A 首轮结果：shared-bank 偏置与 cross-bank 判决（2026-07-17）

> **当前判决：Horizon-Bundle Gate A 尚未通过，不应据此开始实现 bundle。**
>
> 本节使用已有单 seed、epoch-30 checkpoint 做零训练筛查。它足以否定“现在就开做”
> 的决策，但不能替代 A100 上的完整 end-to-end、fixed-calls 和 held-out-seed 结果。

### 11.1 机器、数据和产物

```text
machine       = 1 × NVIDIA GeForce RTX 5090 32GB
remote repo   = /mnt/data/wge/learn_wm
commit        = 81caff0
eval dataset  = /mnt/data/wge/data/pusht_eval_state_only.h5
checkpoints   = /mnt/data/wge/stablewm/checkpoints
matrix        = outputs/week1/oracle_matrix_5090_deterministic_n40_c24
cross-bank    = outputs/week1/oracle_crossbank_5090_deterministic_n40_c24
```

完整 shared-bank matrix 含 `15 cells × (1 bank + 5 scorers) = 90 NPZ` 和
`75 logs`；cross-bank 含 `3 cells × 5 generators × (1 bank + 5 scorers)
= 90 NPZ` 和 `75 logs`。两者均完整结束，无 traceback、CUDA OOM 或失败单元。

每个 cell 使用：

```text
40 paired planning states
24 stratified final-CEM candidates / state
K_train ∈ {1,2,3,5,10}
H_plan  ∈ {1,3,5,8,10}
offset  ∈ {25,40,60}
```

### 11.2 5090 与 A100 的初始硬件一致性

六个 5090 end-to-end anchor 为：

| checkpoint | H | offset | protocol | success | time |
| --- | ---: | ---: | --- | ---: | ---: |
| K1 | 1 | 25 | fixed candidates, `s=300` | 60% | 25.01s |
| K2 | 3 | 40 | fixed calls, `s=500` | 36% | 44.83s |
| K5 | 5 | 40 | fixed candidates, `s=300` | 58% | 30.14s |
| K3 | 8 | 60 | fixed candidates, `s=300` | 12% | 76.16s |
| K5 | 8 | 60 | fixed candidates, `s=300` | 4% | 79.34s |
| K10 | 10 | 60 | fixed candidates, `s=300` | 6% | 96.39s |

已能直接跨机器核对的 `K10/H10/off60/s300` 在 A100 与 5090 上拥有**完全相同的
50-episode success vector**（3/50），A100 为 `132.58s`，5090 为约 `96.59s`，
5090 快约 `1.37×`。这支持把 5090 用作分流机器，但目前只是一格 exact anchor；
“总体跨硬件一致”仍需等 A100 matrix 到齐后再对至少数个不同 K/H 的格子复核。

严格启动器下 `K5/H8/off60` 两次 end-to-end 重跑也得到完全相同的 success vector
（2/50）；运行时间为 `82.75s` 与 `75.93s`。因此 success 终点对当前硬件波动稳定。

### 11.3 确定性审计暴露的边界

首版临时 deterministic launcher 用 `runpy` 执行目标脚本时，没有模拟 Python
直接执行脚本的 `sys.path[0]` 语义，使 `candidate_oracle.py` 无法导入同目录的
`eval_wm.py`。现已加入跟踪版
`scripts/plan/deterministic_launcher.py`，同时固定：

```text
Python hash seed
Python / NumPy / Torch / CUDA seeds
deterministic algorithms
cuBLAS workspace
cuDNN deterministic mode
TF32 off
```

但“严格 Torch 配置”还不等于整个 planner+simulator bitwise deterministic。
同一 `K5/H5/off40` reference bank 跨进程生成两次时：

```text
39 / 40 states 逐数组完全相同
唯一分叉 state = row 1055885, episode 8502, start 60
```

该 state 的 CEM candidate 分支不同，但对 40-state 汇总的影响很小：

| metric | repeat 1 | repeat 2 | absolute delta |
| --- | ---: | ---: | ---: |
| Spearman | 0.27909 | 0.28613 | 0.00704 |
| Kendall | 0.20127 | 0.20598 | 0.00471 |
| inversion | 0.39937 | 0.39701 | 0.00236 |
| top-k precision | 0.24583 | 0.24583 | 0 |
| regret | 12.12736 | 12.16240 | 0.03504 |

完整 strict-config matrix 与此前普通 matrix 的最大 mean delta 也只有：

```text
Spearman 0.0091
inversion 0.0040
normalized regret 0.0231
```

所以本节的方向性判决不依赖该分叉；但后续不能把 candidate generation 宣称为
bitwise exact。需要继续区分 Python hash、PushT `_set_state` 的 physics step、
接触态重置和 CEM elite tie-breaking。

### 11.4 单一 K5-generated shared bank 的表面结果

对每个 cell 用同一个 K5 生成的 candidate bank 给五个 checkpoint 重打分，并对
planning state 做 20,000 次 paired bootstrap。除原始五项外，还从保存的
`pred/true/success` 导出：

```text
normalized regret
selected candidate 的 true-cost percentile
有成功 candidate 时 predicted-best 是否成功（success hit）
```

15 个 cell 的描述性 winner credits 为：

| metric | K1 | K2 | K3 | K5 | K10 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Spearman | 1 | 0 | 2 | 4 | **8** |
| Kendall / inversion | 1 | 1 | 3 | 1 | **9** |
| normalized regret | 0 | 2 | 2 | 4 | **7** |
| selected true percentile | 0 | 2 | 3 | 3 | **7** |

更重要的是：

```text
在 15 cells × 8 metrics 中，
没有任何一个描述性 best checkpoint 相对 runner-up 的
paired bootstrap 95% CI 排除 0。
```

跨三个 offset 平均后，K10 的 Spearman 在每个 H 都最高：

| H | K5 | K10 |
| ---: | ---: | ---: |
| 1 | -0.053 | **0.137** |
| 3 | 0.102 | **0.181** |
| 5 | 0.195 | **0.263** |
| 8 | 0.358 | **0.365** |
| 10 | 0.442 | **0.455** |

因此这份 shared-bank 结果并没有给出“短 horizon 一个 K、长 horizon 另一个 K
显著占优”的证据；点估计反而更像 K10 是较强的 universal scorer。

同时必须注意，Spearman 随 H 上升不能解释为长规划变好了。H 越大，24 个候选的
true-cost spread 也越大，远端坏候选更容易做全局排序；实际 end-to-end success
却会下降。这里应优先看 normalized regret、elite selection 和闭环状态，而不能
把全局相关系数当能力指标。

### 11.5 Cross-bank factorial：真正的新信息

单一 K5 bank 仍可能把 generator 本身混进 scorer 比较。为此在三个关键格子跑：

```text
candidate generator K_g ∈ {1,2,3,5,10}
candidate scorer    K_s ∈ {1,2,3,5,10}
```

即每格完整 `5 generator × 5 scorer` factorial。先在每个 state 内对五种
generator bank 的指标取平均，再 bootstrap states。结果是：

| cell | robust best scorer | Spearman: best vs K5 | best-vs-runner 95% CI | norm-regret: best vs K5 | 95% CI |
| --- | --- | ---: | ---: | ---: | ---: |
| H1 / off25 | K10 | 0.304 vs 0.241 | `[-0.026, 0.155]` | 0.324 vs 0.351 | `[-0.022, 0.078]` |
| H5 / off40 | K10 | 0.318 vs 0.303 | `[-0.039, 0.070]` | 0.294 vs 0.299 | `[-0.047, 0.059]` |
| H10 / off60 | K10 | 0.349 vs 0.310 | `[-0.015, 0.093]` | 0.265 vs 0.297 | `[-0.010, 0.080]` |

表中 normalized-regret 的 advantage 已转成“正数为 best 更好”；所有 CI 仍跨 0。
所以 balanced-bank 后仍是 K10 点估计最好，但没有足够证据宣称它显著通吃。

各 generator 实际找到的 candidate coverage 却明显不同。表中每格是
`有至少一个成功 candidate 的 state 比例 / 每 state oracle-best true cost`
（前者越高越好，后者越低越好）：

| cell | G1 | G2 | G3 | G5 | G10 |
| --- | ---: | ---: | ---: | ---: | ---: |
| H1 / off25 | .150 / 62.3 | .125 / 71.8 | .225 / 57.1 | .200 / **54.3** | **.325** / 62.0 |
| H5 / off40 | .400 / 101.9 | .525 / 74.9 | .625 / **67.4** | **.675** / 73.0 | .475 / 113.8 |
| H10 / off60 | .250 / 192.8 | .250 / 153.2 | **.350** / 138.4 | .325 / 116.5 | **.350 / 104.2** |

最值得注意的是 `H5/off40`：K10 是跨 bank 最强的点估计 scorer，但 K10 自己驱动
CEM 时产生的 bank 反而拥有最差的 oracle-best cost。也就是说：

```text
固定候选上的平均排序能力
≠
这个 cost landscape 被 CEM 自适应查询后能否找到好候选。
```

selection bias 在 `H1/off25` 最清楚。同一 scorer 在自己的 final-CEM bank 上，相对
它在另外四个 generator bank 上的平均 Spearman advantage 为：

```text
K5  = -0.392, 95% CI [-0.525, -0.257]
K10 = -0.492, 95% CI [-0.631, -0.353]
```

因此单一 reference bank 会系统性改变 scorer 的难度；K5-generated bank 在短 H
上尤其惩罚 K5 自己。原 shared-bank matrix 中一部分看似 `K×H` 的交互，很可能是
`generator×scorer×H` 的自适应选择效应，不能直接归因为 horizon-specific latent。

### 11.6 当前 idea 判决与下一步

按第 4.5 节的预注册标准：

1. 不同 H 的点估计有变化，但尚无稳定的不同-K显著 winner；
2. balanced cross-bank 的三个 anchor 都是 K10 点估计最好，且均未显著胜过 K5；
3. held-out seed 与完整 fixed-model-calls 仍未到齐；
4. 当前最强交互住在 candidate generator / adaptive CEM，而不是已证明的
   representation-side horizon interaction。

所以当前判决是：

```text
Horizon-Bundle：HOLD / 不实现
“一个 universal latent 已被否证”：不成立
“单一 global rate 足以决定 planning”：仍不成立，但需等 iso-rate 受控组
当前更值得追的机制：optimizer-induced query distribution
```

下一轮最有信息量的补充不是再加一层 horizon adapter，而是：

1. 保存**每一轮** CEM population、elite boundary、mean/variance，而不只保存 final
   population；
2. 在真实 evaluation rollout 的早/中/晚 replanning states 做 snapshot replay，
   不再只审计 expert-dataset start states；
3. 分解“proposal 找不到好 candidate”和“scorer 在已有 candidate 上选错”；
4. 检验 selection-induced rank collapse 是否先于 end-to-end failure。

只有当这些控制后仍出现稳定的 representation-side `K_train×H_plan` 交叉，才恢复
Horizon-Bundle。若交叉消失，下一条更合理的 idea 是
**selection-aware / optimizer-compatible world-model cost geometry**，而不是
horizon-indexed state bundle。

---

## 12. A100 Gate A 完整闭环矩阵：无 horizon matching，转向 selection-aware（2026-07-17）

> **当前判决：本轮 Gate A 不通过。停止实现 Horizon-Bundle；保留 Gate B
> iso-rate 训练，用于判断 CritWM scalar 假说和下一条 selection-aware 方向。**
>
> 这里的“不通过”限定于当前 PushT、checkpoint family 和评估协议。它否定的是
> “现有证据足以支持立刻实现 bundle”，不是宣称所有任务都存在 universal state。

### 12.1 完整性、协议与可复现产物

A100 driver 于北京时间 `2026-07-17 16:55:58` 开始，`23:11:31` 结束：

```text
5 K_train × 5 H_plan × 3 goal offsets × 2 compute protocols
= 150 runs

每个 offset 内，所有 K/H/protocol 共享同一有序的 50 个 evaluation starts
不同 offset 使用不同 physical start rows
DONE   = 150
FAILED = 0
```

五个 K checkpoint 来自当前已有的单训练-seed family；因此这是立项 gate 和闭环
能力筛查，不是“K3 跨 seed universal”的确认实验。K1/K5 held-out-seed anchor
仍由正在运行的 verification training 提供。

两套协议为：

```text
fixed candidates:   samples = 300
fixed model calls:  samples = floor(1500 / H)
                    H1=1500, H3=500, H5=300, H8=187, H10=150
```

原始结果仍在 A100：

```text
/225010117/stablewm/checkpoints/gateA_*.txt
/225010117/logs/week1_gateA_driver.log
```

仓库保存了全部 150 个 ordered success vectors、run-level rate/time、聚合表和
20,000 次 paired bootstrap 结果：

```text
docs/knowledge/horizon_bundle_gateA_a100_20260717/
```

汇总脚本为：

```text
scripts/plan/summarize_gate_a_end_to_end.py
```

### 12.2 主结果：K3 近似 universal winner，不是 K 与 H 匹配

下表对三个 goal offset 取平均，数值为 success `%`。

**Fixed candidates**

| `H_plan` | K1 | K2 | K3 | K5 | K10 | 描述性 winner |
| ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 1 | 30.7 | 24.7 | **31.3** | 26.7 | 20.7 | K3 |
| 3 | 46.7 | 47.3 | **52.7** | 46.7 | 48.0 | K3 |
| 5 | 48.7 | 50.0 | **57.3** | 56.7 | 55.3 | K3 |
| 8 | 16.0 | 17.3 | **20.0** | 16.7 | 15.3 | K3 |
| 10 | 12.0 | 11.3 | 11.3 | **14.7** | 10.7 | K5 |

**Fixed model calls**

| `H_plan` | K1 | K2 | K3 | K5 | K10 | 描述性 winner |
| ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 1 | 28.0 | 24.0 | **32.0** | 26.7 | 20.0 | K3 |
| 3 | 48.0 | 47.3 | **56.0** | 48.0 | 47.3 | K3 |
| 5 | 48.7 | 50.0 | **57.3** | **57.3** | 55.3 | K3/K5 |
| 8 | 16.7 | 19.3 | **20.0** | 18.0 | 16.7 | K3 |
| 10 | 9.3 | 8.7 | **13.3** | 12.7 | 10.7 | K3 |

跨全部 `15 cells` 的均值同样由 K3 最高：

| protocol | K1 | K2 | K3 | K5 | K10 |
| --- | ---: | ---: | ---: | ---: | ---: |
| fixed candidates | 30.80 | 30.13 | **34.53** | 32.27 | 30.00 |
| fixed model calls | 30.13 | 29.87 | **35.73** | 32.53 | 30.00 |

这不是 `K_train=H_plan` 的 pattern。对存在 exact matching checkpoint 的
`H∈{1,3,5,10}`：

```text
fixed candidates:
  matched K 相对 best non-matched K 的平均差 = -2.0 pp
  12 cells 中 3 win / 2 tie / 7 lose

fixed model calls:
  平均差 = -1.0 pp
  12 cells 中 5 win / 2 tie / 5 lose
```

以并列 winner 平分 credit，K3 在 fixed-candidates / fixed-calls 下分别得到
`5.33 / 6.25` 个 cell credits，均为五个 checkpoint 中最高。长 horizon 也没有
由 K10 接管：K10 的全局均值在两种协议下都只有 `30.0%`。

### 12.3 不确定性：局部 winner 多数不可区分，且没有跨协议复现

在每个 `(protocol,H,offset)` 内，按 50 个 paired episodes 对描述性
best-vs-runner 做 20,000 次 bootstrap：

```text
fixed candidates:  1 / 15 cells 的 95% CI 排除 0
fixed model calls: 1 / 15 cells 的 95% CI 排除 0
```

两个 cell 还不是同一个条件：

```text
fixed candidates: H1/off60，K3 18% vs K1 10%，
                  advantage CI [2, 16] pp

fixed model calls: H3/off40，K3 60% vs K1 44%，
                   advantage CI [4, 28] pp
```

把三个 offset 聚合、在每个 offset 内保持 pairing 并分层 bootstrap 后，只有
`fixed-calls H3` 的 K3 相对 K5 为：

```text
56% vs 48%，advantage 8.0 pp，95% CI [1.3, 14.7]
```

对应的 fixed-candidates H3 best-vs-runner interval 仍跨 0。又因为 winner 是看完
同一批结果后选择、且这里没有 multiple-comparison correction，这些 interval 只能作
描述性筛查，不能包装成 confirmatory significance。

因此可靠结论不是“K3 已被证明显著 universal”，而是：

```text
没有稳定证据表明不同 H 需要不同 K；
点估计反而更接近一个 K3 在大部分 H 上共同占优。
```

### 12.4 真正稳定的交互是 planning horizon cliff

对 K 和 offset 再取平均：

| protocol | H1 | H3 | H5 | H8 | H10 |
| --- | ---: | ---: | ---: | ---: | ---: |
| fixed candidates | 26.8 | 48.3 | **53.6** | 17.1 | 12.0 |
| fixed model calls | 26.1 | 49.3 | **53.7** | 18.1 | 10.9 |

goal offset 的主效应也稳定：

| protocol | off25 | off40 | off60 |
| --- | ---: | ---: | ---: |
| fixed candidates | 54.1 | 27.0 | 13.6 |
| fixed model calls | 54.6 | 27.0 | 13.4 |

也就是说，`H8/H10` cliff 和远 goal 压力在所有 K 上都存在；改变训练 horizon
没有把 cliff 系统性推远。fixed-model-calls 与 fixed-candidates 的总 pattern
几乎不变，排除了“只是长 H 模型调用数更多”这一简单解释。

### 12.5 自然重复与跨硬件一致性

`H5` 时两套协议都恰好使用 `300 samples`，因此 15 个格子构成同配置重复：

```text
14 / 15 episode-success vectors 完全一致
```

唯一分叉为 `K5/H5/off40`：

```text
fixed candidates = 56%
fixed calls      = 58%
3 / 50 episode bits 翻转，净差 2 pp
```

这给出了当前 end-to-end evaluator 的小样本重复边界：单 cell 的 `2 pp` 差异不能
过度解释，但不影响上述大尺度 pattern。

此前在 5090 预先选定的六个硬件 anchor 与本次 A100 matrix 对照：

| anchor | 5090 | A100 |
| --- | ---: | ---: |
| K1/H1/off25/fixcand | 60 | 60 |
| K2/H3/off40/fixcalls | 36 | 36 |
| K5/H5/off40/fixcand | 58 | 56 |
| K3/H8/off60/fixcand | 12 | 12 |
| K5/H8/off60/fixcand | 4 | 4 |
| K10/H10/off60/fixcand | 6 | 6 |

即 `5/6` success rate 完全一致；唯一的 `2 pp` 差正好落在 A100 自然重复已观察到的
范围内。K10/H10/off60 的 50-bit vector 先前也已逐位确认一致。因此 A100/5090
总体方向一致，可以继续分流实验，但不应要求每个非严格 CEM run 都 bitwise exact。

### 12.6 与 candidate oracle 合并后的机制判决

5090 balanced cross-bank oracle 给出的点估计是 K10 通常为最强固定-bank scorer，
但本次 A100 闭环矩阵却更偏向 K3。尤其在 `H5/off40`：

```text
cross-bank fixed scorer: K10 点估计最好；
K10 自己生成的 bank:    oracle-best true cost 最差；
A100 end-to-end:         K3=62%，K5=56/58%，K10=52%。
```

所以“在外生固定 bank 上平均排序更好”不能推出“让同一个 cost 驱动自适应 CEM
就更好”。现在最一致的解释是：

```text
world-model geometry
  → 改变每轮 CEM 的 elite / proposal distribution
  → planner 查询进入不同区域
  → 最终 candidate coverage 与 closed-loop success 改变
```

这比“不同 planning horizon 需要不同 latent state”更贴合现有全部证据。

### 12.7 最终 Gate A 判决与后续分工

Gate A 的预注册通过条件要求：稳定 `K×H` interaction、不同 H 由不同 K 显著占优、
matched compute 后保持、且交叉主要住在 candidate ranking。本轮结果逐项不满足：

1. 没有 horizon matching；K3 点估计在多数 H 上共同更强；
2. 局部 winner 几乎都不可区分，且仅有的 CI 排除 0 条件不跨协议复现；
3. matched compute 不改变结论；
4. fixed-bank scorer 与 adaptive CEM 闭环 winner 明显不一致。

因此：

```text
Horizon-Bundle Gate A = FAIL for the current formulation
Horizon-Bundle implementation = STOP
```

仍在运行的六模型 verification training 不再承担“救 Bundle”的任务。它只服务：

1. Gate B：同 rate、不同 gamma 路径是否仍产生不同 ranking/planning；
2. K1/K5 held-out-seed anchors：确认已有 multi-step 机制是否跨 seed；
3. 决定 CritWM 的问题是 scalar target、controller path，还是二者都有。

下一条优先机制实验保持为：

1. 保存每轮 CEM population、elite、mean/variance；
2. 在真实闭环 early/mid/late replanning states 做 snapshot replay；
3. 因果分解 proposal coverage failure 与 scorer selection failure；
4. 检验 rank/elite collapse 是否先于 episode failure。

只有未来在这些 selection controls 后重新出现稳定的 representation-side
`K×H` crossing，才重新打开 Horizon-Bundle。

## 13. 5090 selection 闭环：从 verifier/portfolio 负结果到 Optimizer-Equivalent WM（2026-07-18）

> **本轮判决：Horizon-Bundle 继续关闭；“再加一个 verifier”关闭；
> multi-model portfolio 保留为机制 probe 和强 baseline，但证据不足以独立成为
> flagship。当前唯一值得进入训练 Gate 的新方向是
> Optimizer-Equivalent World Model（OE-WM）：不要求模型在外生 candidate bank
> 上全局拟合真实 cost，而要求它在自己诱导的 planner query distribution 上，
> 产生与真实动力学相同的 CEM elite-moment update。**
>
> 这里的 “GO” 是“核心假设已有因果证据、上限明确、可以花训练预算”，不是宣称
> 方法已经训练成功。

### 13.1 协议与本轮新增控制

主机为 `5090lan`，数据为
`/mnt/data/wge/data/pusht_eval_state_only.h5`，三个 checkpoint 为：

```text
K3  = pd_d192_k3_eval
K5  = iter2_multistep_eval
K10 = pd_d192_k10_eval
```

正式 snapshot audit 使用普通 CEM：

```text
N=300 candidates / round
R=30 rounds
topk=30
H5/off40 与 H8/off60
3 generators × 3 scorers
saved rounds = {0,1,2,4,9,19,29}
```

其中 `H5/off40` 另用 evaluator 完全相同的 `50` 个起点保存最终完整 population。
端到端 MPC 控制均为 `50` episodes；关键 matched-compute cell
`H5/off40` 补到 seeds `42/7/123`。所有比较保留逐 episode pairing。

本轮不再只问“哪个 scorer 在固定 bank 上相关性高”，而是依次审计：

1. proposal 中是否存在成功 candidate；
2. scorer 能否选中或 refit 它；
3. scorer 诱导的 elite mean/variance update 是否与 true-cost CEM 同向；
4. snapshot 上的正结果能否穿过完整 closed-loop MPC；
5. 增益是否在相同 world-model call budget 下仍然存在。

### 13.2 最重要的新事实：learned cost 与 true cost 给出不同的 CEM update

对同一个 population，定义 learned scorer 的 top-30 elite mean update
`Δμ_model`，以及 simulator true cost 的 top-30 elite mean update
`Δμ_oracle`。记录：

```text
elite overlap = |E_model ∩ E_oracle| / 30
update cosine = cos(Δμ_model, Δμ_oracle)
relative error = ||Δμ_model - Δμ_oracle|| / ||Δμ_oracle||
```

generator 使用自己的 scorer 时，跨 12 个状态的平均值为：

| cell/model | overlap step 0 → 29 | cosine step 0 → 29 | relative error at 29 |
| --- | ---: | ---: | ---: |
| H5/off40 K3 | .342 → **.097** | .525 → **.150** | 1.175 |
| H5/off40 K5 | .494 → **.064** | .580 → **.054** | 1.230 |
| H5/off40 K10 | .319 → **.075** | .422 → **.074** | 1.205 |
| H8/off60 K3 | .481 → **.219** | .608 → **.311** | 1.047 |
| H8/off60 K5 | .544 → **.231** | .617 → **.192** | 1.194 |
| H8/off60 K10 | .511 → **.111** | .552 → **.217** | 1.095 |

也就是说，CEM 越优化，learned elite 与 true elite 通常越分离；在
`H5/off40` 最后一轮，三个模型的 elite overlap 都低于 `10%`，update direction
几乎正交，relative error 全部大于 `1`。

这个量第一次解释了之前看似矛盾的两组结果：

```text
外生 fixed bank：K10 经常是较好的 pointwise scorer；
自适应 closed loop：K3 却是更稳定的 generator / planner。
```

K3 在自己诱导的最终 bank 上虽然 pointwise ranking 并不最好，但它的
`update cosine` 在 H5/H8 都是三个 generator 中最高，方向与端到端 winner 一致。
世界模型在 CEM 中不是被动 evaluator；它定义了下一轮会查询哪里。

### 13.3 Oracle ceiling 存在，但 post-hoc verifier 无法兑现

在与 evaluator 起点完全一致的 `H5/off40, n=50` 最终 K3 population 上：

| first-plan open-loop rule | success | 相对 K3 returned mean |
| --- | ---: | ---: |
| K3 returned top-30 mean | 58% | — |
| true-cost top-1 candidate | **70%** | 6 gains / 0 losses |
| true-cost top-5 refit mean | **68%** | 5 / 0 |
| true-cost top-10 refit mean | **68%** | 5 / 0 |
| true-cost top-30 refit mean | **68%** | 5 / 0 |
| K10-scored top-5 refit mean | 66% | 4 / 0 |

所以当前 population 里确有约 `+10–12 pp` 的第一步可兑现空间；问题不是
“根本没有好动作”。但同一个 K10 top-5 rule 放入完整 MPC 后：

```text
ordinary K3 CEM       = 62%
K10 top-5 refit MPC   = 60% (0 gains / 1 loss)
K10 top-10 refit MPC  = 58% (0 / 2)
K10 top-30 refit MPC  = 60% (0 / 1)
```

其余 single-verifier 变体也没有一次 paired gain：

| K3 proposal 后的选择 | success | gains/losses vs 62% |
| --- | ---: | ---: |
| K10 select sparse means | 58% | 0 / 2 |
| K10 select final population | 58% | 0 / 2 |
| K5 select sparse means | 60% | 0 / 1 |
| K5 select final population | 56% | 0 / 3 |
| K5 top-30 refit | 56% | 0 / 3 |
| K3 self sparse-means control | 62% | 0 / 0 |
| K3 self top-30 refit control | 62% | 0 / 0 |

因此不能再用“snapshot 上 K10 top-5 到 66%”讲方法结果。它恰好证明：

```text
expert/dataset start 上的固定-bank正结果
≠
方法自己执行后进入的新状态上的 closed-loop 正结果。
```

一个不改变后续 query distribution、也不在该分布上训练的 verifier，无法修复
adaptive optimizer 的 Goodhart / planner-overfitting。

### 13.4 Portfolio 到底有多少是真模型信号，多少只是多算

三个模型各自运行独立 CEM，保留七个 smooth means，最后由三个模型的 fractional
rank consensus 选择。seed 42 的 full-compute 结果：

| cell | K3 `1×300` | diverse portfolio `3×300` | K3 `1×900` |
| --- | ---: | ---: | ---: |
| H5/off25 | 86 | **94** | 90 |
| H5/off40 | 62 | **72** | **72** |
| H5/off60 | 24 | 34 | **36** |
| H8/off60 | 12 | **14** | **14** |

这说明 portfolio 本身能工作，但 `3×` search compute 几乎解释了全部增益，而且
没有修复 H8 cliff。

更严格的 `H5/off40` equal-call audit 中，每轮总共都使用约 `300` 次
world-model trajectory scores：

```text
ordinary K3:             1 path × 300 candidates
same-model multistart:   3 K3 paths × 100 candidates
diverse portfolio:       K3/K5/K10 paths × 100 candidates
shared rank ensemble:    1 path × 100 candidates × 3 models
```

三 seed 完整 MPC：

| method | s42 | s7 | s123 | pooled |
| --- | ---: | ---: | ---: | ---: |
| K3 `1×300` | 62 | 68 | 58 | 62.7 |
| K3 `3×100` multistart | 66 | 62 | 60 | 62.7 |
| K3 `1×900` | 72 | 60 | 66 | 66.0 |
| diverse `3×100` portfolio | 66 | 74 | 58 | **66.0** |
| shared-population rank ensemble `3×100` | 6 | 14 | 4 | **8.0** |

相对 ordinary K3，diverse portfolio 的三个 seed delta 为 `+4/+6/+0 pp`，
pooled `+3.3 pp`，episode bootstrap CI 仍跨 0；相对 same-model multistart
也是 `+3.3 pp`，CI 仍跨 0。因此存在**弱但一致的 model-specific diversity
信号**，却还远未达到 flagship 方法的证据强度。

更有信息量的是 shared-population ensemble 的崩溃。单独 K3 用相同 `N=100`
时三 seed 为 `58/68/52`，而三个 scorer 的 rank 在每轮直接平均后变成
`6/14/4`。这排除了“只是 candidate 数少”；当前最直接的解释是不同模型支持的
elite modes 被塞进同一个 Gaussian 后，其 mean 成为不可执行的折中，但这个具体
机制仍需 shared-ensemble population trace 确认。

seed 42 的三个 ordinary planner 单独成功率为 K3/K5/K10=`62/56/52%`，
但 episode-wise oracle union 为 `76%`。这 `+14 pp` 说明互补模式真实存在；
现有证据支持先保留各 search branch、再选择，而不是在搜索中提前平均；它还不能
证明某一种 mixture 实现必然有效。

### 13.5 文献边界：为什么不能把它写成普通 search、verifier 或 ensemble

截至 2026-07-18 的直接近邻：

1. [IMWM](https://arxiv.org/abs/2606.01626) 已证明即使 perfect world model，
   finite-budget search 也可能失败，并用 demonstration-trained intuition 做
   retrieval initialization、hybrid cost 和 reliability gate。因此不能 claim
   “首次发现 search bottleneck”或只加 action proposal network。
2. [EV-WM](https://arxiv.org/abs/2606.13053) 已用 predicate-grounded verifier
   引导 sampling、gate action、选择 proposal；[World Action Verifier](https://arxiv.org/abs/2604.01985)
   已用 state plausibility、action reachability 和 cycle consistency 修复
   under-explored actions。普通 verifier/可达性监督不是空位。
3. [Closing the Train-Test Gap](https://arxiv.org/abs/2512.09929) 已用 online /
   adversarial data synthesis 平滑 gradient-based planning landscape，也报告对
   CEM 的影响。因此“在 planner distribution 上 finetune”本身也不是新意。
4. [Differentiable CEM](https://arxiv.org/abs/1909.12830) 已提供把 CEM 放入
   end-to-end learning 的可微工具；不能把 soft top-k 本身当贡献。
5. [PETS](https://arxiv.org/abs/1805.12114) 是 probabilistic ensemble +
   trajectory sampling 的经典先例；普通 ensemble planning 没有 novelty。
6. [Imperfect World Models are Exploitable](https://arxiv.org/abs/2605.15960)
   已形式化 model exploitation；2026-07-15 的
   [RENEW](https://arxiv.org/abs/2607.14180) 又用 uncertainty-targeted human
   preferences 直接修 world-model dynamics。不能 claim 首次修 planner exploitation。

本轮检索没有找到直接把下面这个对象定义为 latent world-model 训练目标的工作：

> **在模型自己诱导的 CEM proposal distributions 上，匹配 learned cost 与
> true dynamics 所诱导的 elite sufficient-statistic update。**

这个 novelty 判断仍是检索后的暂定边界，不是“未搜到即证明首次”。必须在正式写作前
继续对打 decision-focused learning、learning-to-optimize、DCEM 和 planner-overfitting。

### 13.6 新 idea：Optimizer-Equivalent World Model（OE-WM）

设第 `r` 轮 CEM proposal 为

```text
q_r(a) = N(μ_r, diag(σ_r²))
```

对任意 candidate cost `c`，定义 optimizer update operator：

```text
U_c(q_r) = ( μ_c⁺(q_r), log σ_c⁺(q_r) )
```

其中 `μ_c⁺, σ_c⁺` 是由 `c` 的 soft top-k / entropic elite weights 得到的
下一轮 sufficient statistics。令：

```text
c_θ(a | o,g) = learned latent world-model goal cost
J(a | s,g)   = candidate 在真实动力学/高保真 teacher 中执行后的 goal cost
```

则在 proposal `q` 上定义 planner equivalence：

```text
c_θ ≡_q J    iff    U_cθ(q) = U_J(q)
```

训练目标为：

```text
L_OE =
  Σ_r || μ_θ⁺(q_r) - μ_*⁺(q_r) ||²_W
  + β || log σ_θ⁺(q_r) - log σ_*⁺(q_r) ||²
  + γ L_elite-boundary
```

`L_elite-boundary` 只强调 oracle elite boundary 附近的 pairwise order/margin，
不浪费容量拟合 population 尾部所有 pair。原 LeWM prediction + SIGReg loss
继续保留，`L_OE` 是 planner-facing auxiliary，不把模型退化成纯 task value head。

关键不是只在一次外生 bank 上训练，而是 planner-query data aggregation：

```text
pretrained WM
  → 用当前 c_θ 跑 CEM，收集其 q_0...q_R
  → 在少量候选上获得 true/teacher rollout cost
  → 构造每轮 oracle elite moments
  → 更新 encoder/predictor，使 U_cθ 接近 U_J
  → 用更新后的模型重新收集 query distribution
```

这使训练分布跟随模型自己的 optimizer path，直接堵住 §13.3 的
“first-plan positive、closed-loop negative”漏洞。inference 时仍是单模型普通 CEM，
不需要 verifier、三模型 ensemble 或额外 predicate。

一个自然的理论目标是：若 `U_J` 对 proposal metric 是 `L`-Lipschitz，且每轮
`||U_cθ(q_r)-U_J(q_r)||≤ε_r`，则 R 轮 proposal 偏差可由

```text
d(q_R^θ, q_R^*) ≤ Σ_{r=0}^{R-1} L^(R-1-r) ε_r
```

控制，再通过 terminal cost 的 Lipschitz / elite coverage 条件转成 optimizer regret。
这个 bound 约束的是实际搜索递推，而不是再给 latent prediction error 换一个名字。

shared-rank-ensemble 的崩溃还给出实现约束：第一版只在同一个 coherent oracle
elite set 上匹配 update；若 oracle elite 自身多模态，再升级为 mixture update heads。
不能先把不同模型的互斥 modes 平均成一个 mean。

### 13.7 为什么这个方向的上限高于当前几个备选

它同时解释并利用本轮所有结果：

1. **比 Horizon-Bundle 更贴因果对象。** 当前没有稳定 `K×H` matching，却有稳定的
   learned-vs-oracle update divergence。
2. **比 verifier 更早介入。** verifier 只改最终选择；OE-WM 改每一轮下一批 candidate
   从哪里来。
3. **比 portfolio 更可扩展。** portfolio 用 `3×` checkpoint/compute 暴露互补 proposal
   modes；OE-WM 的目标是把这种搜索质量蒸馏进一个模型、一个推理 budget。
4. **有可测上限。** first-plan final-population oracle 为 `68–70%`，ordinary planner
   union 为 `76%`；不是在没有 headroom 的 cell 上凭空造 loss。
5. **能正面回答 efficiency。** 真正成功应让 `N=300` 的 OE-WM 匹配或超过
   baseline `N=900`，而不是靠更多 samples 赢。
6. **有跨 planner 扩展。** CEM 先做干净验证；若核心成立，可把
   `U_c` 换成 MPPI weight update 或 gradient optimizer step，而不是绑定某个 latent
   horizon adapter。

它的主要代价也必须写清：oracle update 需要 simulator、高保真 teacher 或有限真实
执行；纯 fixed offline dataset 无法给任意 counterfactual candidate 提供真标签。
如果 active query efficiency 做不下来，这条线不适合宣称通用 offline world model。

### 13.8 训练前预注册 Gate

第一版只允许小规模 finetune，不直接开完整多周 training：

```text
base checkpoint: K3
cell: H5/off40
planner states: 先 200–500 个，必须来自 current-policy closed loop
candidate labels: 每轮 oracle-boundary + disagreement 的小子集
loss: original LeWM + soft elite mean/cov update + boundary rank
inference: one checkpoint, N=300, R=30
```

**OE Gate 通过**至少满足：

1. 三 seed `H5/off40` 相对 ordinary K3 平均提升 `≥8 pp`，且 paired CI 不靠单 seed；
2. `N=300` 匹配或超过 K3 `N=900`，或在 `≤2 pp` 内用约三分之一 model calls；
3. final-round update cosine 至少增加 `0.20`，relative update error 至少降低 `30%`；
4. 改善主要发生在 baseline-failed 且 population 有 oracle success 的 states；
5. 完整 MPC 保持正增益，不能只在 expert-start snapshot 上有效；
6. 至少一个第二环境和 H8 pressure cell 同向，H8/off60 不能仍停在 `12–14%`。

**立即停止**条件：

- update metric 变好但 proposal coverage / MPC 不变；
- 只拟合 PushT pose oracle，换 goal/task 就失效；
- 需要三模型或 `3×` inference compute 才有效；
- active label 数量接近重新收集完整 expert dataset；
- 普通 adversarial/online WM、IMWM initialization 或 PETS-style ensemble 在同预算下
  完全解释增益。

### 13.9 当前最终路线

```text
Horizon-Bundle                         STOP
post-hoc cross-model verifier          STOP
shared-population score ensemble       STOP (catastrophic negative control)
independent multi-model portfolio      KEEP as probe / strong baseline
Optimizer-Equivalent WM training Gate  GO
```

所以本轮找到的不是“又一个能把 PushT 加 2–4 分的小 planner trick”，而是一个更严格的
world-model sufficiency 定义：

> **一个 world model 对某个 optimizer 有用，不是因为它在固定 actions 上平均更准，
> 而是因为在它自己会访问的 proposal distributions 上，它把 optimizer 的下一步
> 送向与真实动力学相同的区域。**

## 14. OE 因果干预与第一版训练 Gate（2026-07-18）

### 14.1 先问 oracle-equivalent update 是否真的改变后续搜索

训练前先做反事实干预，避免在一个对 task outcome 无因果作用的 metric 上继续堆
loss。对 K3 自己产生的 CEM population，分别用 learned cost 和 simulator true
cost 算 top-30 elite mean/std，再按

```text
(μ_α, log σ_α) =
    (1-α) (μ_model, log σ_model)
  + α     (μ_oracle, log σ_oracle)
```

构造下一轮 proposal。所有 α 分支使用相同 Gaussian noise；simulator 每次从完全
相同的 initial/goal state 重放。

one-step exploratory 结果覆盖 12 states × source rounds `{4,9,19,29}` ×
`N=100`：

| α | next-population min true cost | Δ vs α=0 | model-refit success |
|---:|---:|---:|---:|
| 0.00 | 80.57 | — | .354 |
| 0.25 | 76.57 | -4.01 | .354 |
| 0.50 | 74.91 | -5.66 | .354 |
| 0.75 | 73.15 | -7.43 | .354 |
| 1.00 | 71.11 | **-9.47** `[-16.38,-3.83]` | .354 |

这证明单次 oracle update 会把下一批 proposal 推向更低 true cost 的区域，但一次
update 尚不足以改变 learned scorer 选出的 success。随后从 source step 4 的
updated distribution 开始递归 5 轮；α=1 的 final-mean true cost 相对 α=0
下降 `52.00`，success 从 `2/12` 到 `3/12`。这两项早期实验的 model scorer 使用
float32 candidate、simulator cache 使用 float16 candidate，因此只保留为
exploratory；正式长递归统一先量化 candidate，再同时送给 scorer 和 simulator。

严格的长递归使用：

```text
H5 / offset40
12 paired states
start after source CEM step 4
25 counterfactual rounds
N=100, topk=30
α ∈ {0, 0.5, 1}
candidate precision: scorer == simulator == float16-quantized
```

结果为：

| α | avg proposal coverage | last coverage | final mean true | Δ final true | final mean success |
|---:|---:|---:|---:|---:|---:|
| 0.00 | .303 | .417 | 105.72 | — | `3/12 = .250` |
| 0.50 | .410 | .583 | 45.29 | **-60.42** `[-115.65,-14.06]` | `7/12 = .583` |
| 1.00 | .447 | .583 | 38.22 | **-67.50** `[-123.62,-20.26]` | `7/12 = .583` |

α=0.5 与 α=1 相对普通 learned update 都是 `+4/12 = +33.3 pp` success，
paired state bootstrap CI 为 `[+8.3,+58.3] pp`；final true-cost、average
coverage 和 mean-success dose slope 的 CI 也都在有利方向。最大 candidate
float16 量化误差为 `3.906e-3`，state/goal mismatch 为 0，action scaler
roundtrip error 为 `9.537e-7`。

这个实验给出两个重要判决：

1. **核心因果对象通过。** update equivalence 不只是与 planning 相关；连续多轮地
   修正它会改变 proposal coverage，并最终改变 task success。
2. **不需要完美 oracle 才有 headroom。** α=0.5 已取得与 α=1 相同的 7/12
   success，说明可学习的近似 correction 可能足够。

但这仍是每轮执行所有 candidates 的 oracle intervention，不是可部署方法结果，
也没有回答跨状态学习是否可行。

### 14.2 12-state fixed-trace 微调：能拟合，不能泛化

第一版 feasibility trainer 冻结 image encoder / goal representation，仅更新
`action_encoder + predictor + pred_proj`，用以下 loss 拟合保存的
`{4,9,19,29}` 四轮：

```text
balanced oracle-elite boundary BCE
+ soft elite mean / log-std matching
+ base-score anchor
```

严格做 3-fold state cross-fit：每折 8 train states、4 completely held-out
states，每个状态只由一个没有见过它的模型计分。结果不是“小幅不显著”，而是方向
一致的过拟合：

| setting | epoch | cross-fit cosine | Δ cosine | relative error | Δ rel | overlap | Δ overlap |
|---|---:|---:|---:|---:|---:|---:|---:|
| all dynamics modules, lr=1e-5 | 0 | .121 | — | 1.201 | — | .147 | — |
| all dynamics modules, lr=1e-5 | 1 | .078 | **-.044** | 1.270 | **+.070** | .122 | **-.025** |
| pred-proj only, lr=2e-6, strong anchor | 0 | .121 | — | 1.201 | — | .147 | — |
| pred-proj only, lr=2e-6, strong anchor | 1 | .119 | -.002 | 1.208 | +.007 | .141 | -.006 |
| pred-proj only, lr=2e-6, strong anchor | 10 | .075 | -.046 | 1.259 | +.058 | .132 | -.015 |

全 dynamics setting 在训练 states 上的 cosine 能从 `.055` 升到 `.362`，所以
gradient 和 loss 确实能拟合 target；失败发生在 state generalization。降低容量、
学习率并加强 anchor 只能减慢退化，不能反转它。

因此当前判决必须拆开：

```text
oracle-equivalent update as causal target       GO
8-state fixed-trace dynamics fine-tuning        STOP
12-state checkpoint → full MPC                  DO NOT RUN
```

不能拿 train metric 变好就去跑完整 MPC，更不能从三个 fold 里挑一个好看的
checkpoint。下一步不是继续在同 12 states 上调参，而是用新 seed 收集更多独立
planner-query states。

### 14.3 锁定的 60-state bridge gate

在看新数据前锁定一个中等容量 setting：

```text
new source seed = 20260719
60 independent first-plan states
K3 only; source rounds = {4,9,19,29}
full N=300 candidates and simulator labels
3-fold state cross-fit: 40 train / 20 held out
modules = action_encoder + predictor + pred_proj
lr = 2e-6; anchor weight = 0.2
primary epoch = 5 (不按曲线挑 checkpoint)
```

bridge gate 通过要求 epoch 5 同时满足：

```text
cross-fit Δ update cosine       >= +0.10
cross-fit Δ relative error      <= -0.10
cross-fit Δ elite overlap       >= +0.05
state-paired bootstrap direction consistent
```

它仍低于 §13.8 预注册的 200–500 个 current-policy closed-loop states，因此即使
通过也只允许训练一个单模型 checkpoint 进入 recursive resampling 与 50-episode
MPC；不允许直接宣称 OE-WM work。

### 14.4 60-state bridge 结果：有可泛化方向信号，但 Gate 未通过

新 trace 使用 `seed=20260719`，保存 K3 在 H5/off40 的完整
`60 states × 4 rounds × 300 candidates` population。它与早先 12-state
诊断、evaluator-matched 50-state archive 的 row overlap 都为 0；原始 archive
SHA-256 为
`fd8e3623e10c51db37ae14a2f42f6da29b656552079d9f090d13541c82ac1e07`。
独立数据重现了 late-round optimizer divergence：

| source step | elite overlap | update cosine | relative update error |
|---:|---:|---:|---:|
| 4 | .253 | .295 | 1.090 |
| 9 | .218 | .217 | 1.159 |
| 19 | .135 | .132 | 1.159 |
| 29 | .079 | .043 | 1.198 |

因此 12 states 上看到的 OE mismatch 不是小样本偶然。锁定的 3-fold
`40 train / 20 held-out` setting 在 epoch 5 得到：

| metric | epoch 0 | epoch 5 | paired state Δ |
|---|---:|---:|---:|
| update cosine | .172 | .248 | **+.076** `[+.030,+.123]` |
| relative update error | 1.152 | 1.157 | +.005 `[-.034,+.044]` |
| elite overlap | .172 | .207 | +.036 |
| selected-elite true cost | 102.65 | 100.61 | -2.04 |

这不是纯过拟合：held-out state 上的 update direction 和被选 candidate 的真实
质量都同向改善。但它没有达到预锁定的 `Δ cosine ≥ +.10`、
`Δ relative error ≤ -.10`、`Δ overlap ≥ +.05`，而且 epoch 10 的 relative
error 已恶化 `+.055`，所以 bridge gate 判为 **FAIL**。

针对第一版 soft elite mass 约为 40、而 hard top-k 为 30 的实现偏差，又做了一次
development-only 修复：

```text
calibrate sigmoid threshold so total elite mass == 30
+ direct relative update-vector loss
```

epoch 5 的结果为 `Δ cosine=+.080 [+.034,+.128]`、
`Δ relative error=-.005 [-.043,+.034]`、`Δ overlap=+.034`、
`Δ selected true cost=-2.16`。relative error 不再像 v1 后期那样明显恶化，
但仍远未达到 Gate。不能继续在同一份 60-state 数据上调 loss。

当前更精确的判决是：

```text
OE update as causal target                         GO
fixed-trace OE gradient has held-out direction     WEAK POSITIVE
OE-only dynamics fine-tuning                       STOP
v1/v2 checkpoint -> recursive MPC                  DO NOT RUN
next: OE auxiliary + original dynamics replay      TEST
```

失败点不是“完全学不到”，而是只用 planner-query loss 更新大 dynamics backbone
时，模型能学到一部分方向，却不能稳定学到 oracle update 的相对大小，并逐步牺牲
原有 dynamics geometry。下一版必须恢复 §13.6 原本要求的 prediction replay，
而不是把 OE loss 当成唯一训练目标；如果 replay 后仍不能跨状态、跨 horizon
泛化，就应关闭直接微调 world-model dynamics 这条实现。

### 14.5 H8/off60 因果压力测试：第二 horizon 同向

为了判断 H5 的 causal ceiling 是否只是特定 horizon 偶然，使用独立的
H8/off60 12-state trace，保持严格 candidate precision，再从 source step 4
递归 25 轮。这里只比较普通 learned update 与完整 oracle update：

| update | avg coverage | last coverage | final mean true | Δ final true | final mean success |
|---|---:|---:|---:|---:|---:|
| learned | .443 | .500 | 97.27 | — | `3/12 = .250` |
| oracle | .487 | .583 | 47.94 | **-49.32** `[-101.39,-14.24]` | `6/12 = .500` |

mean-success across rounds 的 slope 为 `+.153 [+.023,+.330]`，final success
为 `+3/12 = +25 pp`，但 12-state final-success bootstrap 下界恰为 0。
true-cost 的 paired CI 完全有利，candidate 量化误差 `3.902e-3`，
state/goal mismatch 为 0，action roundtrip error 为 `9.537e-7`。原始 archive
SHA-256 为
`244533b036d2a8f153f485ee9731a15d86a68aae9114dd755537b7500ce7793a`。

联合 H5 与 H8，oracle-equivalent recursive update 分别得到：

| cell | learned success | oracle success | Δ final true cost |
|---|---:|---:|---:|
| H5/off40 | `3/12` | `7/12` | -67.50 |
| H8/off60 | `3/12` | `6/12` | -49.32 |

因此“每轮 optimizer update 是否值得修”已经不再是当前不确定性；两个 horizon
都给出实质 outcome headroom。真正的瓶颈收缩为：能否在保持原始 prediction
能力的前提下，把这个 update correction 学成一个跨 state、跨 horizon 的单模型。

### 14.6 正式 on-policy aggregation：只防止退化，不产生新能力

为了检验 fixed-bank 失败是否主要来自 train/test proposal shift，在 A100 上运行
四臂正式 on-policy 聚合：

```text
frozen cumulative / latest-only / cumulative / trust-region cumulative
H5 / offset40
12 paired states
3 generators × 3 scorers
7 CEM rounds × 300 complete candidates
fixed validation bank; state-paired bootstrap
```

固定验证集的共同起点为
`cos=.242, rel=1.111, overlap=.207, selected-true=98.79`。最终结果为：

| arm | cosine | relative error | overlap | selected true |
|---|---:|---:|---:|---:|
| frozen cumulative | .253 | 1.134 | .214 | 98.96 |
| latest-only | .254 | 1.153 | .212 | 98.83 |
| cumulative | .263 | 1.129 | .217 | 98.88 |
| trust cumulative | .263 | 1.116 | .217 | 98.64 |

最好的 trust arm 相对共同起点只有
`Δcos=+.0205 [+.001,+.041]`、`Δrel=+.0044 [-.0129,+.0221]`、
`Δoverlap=+.0100 [+.0007,+.0197]` 和
`Δselected-true=-.15 [-.74,+.41]`。latest-only 的 relative error 反而恶化
`+.041 [.003,.079]`；trust 相对 frozen 的 `-.0186 [-.0334,-.0037]`
只说明它能抑制 online 更新造成的退化。

H8/off60 使用严格 shared-start 重新审计后，起点为
`.1788 / 1.1757 / .1867 / 168.84`，所有 formal arm 相对它的最终增量 CI
都跨 0。因此先前 H8 的表观收益主要是 shared bootstrap，而不是训练本身。

判决：

```text
proposal distribution shift as dominant failure mode    REJECT
exact scalar LeWM + elite-moment on-policy recipe        STOP
on-policy trust aggregation                              KEEP only as safeguard
```

### 14.7 独立 operator head：幅度可修，但方向是多模态的

下一步冻结 world model，把 correction 从大 dynamics backbone 中分离。严格
nested state-held-out probe 使用 H5/off40 的
`240 states × 4 rounds × 300 candidates` query bank，并在 H8/off60 做独立
pressure cell。直接回归 CEM update 的小 head 得到：

| H5 feature | Δ cosine | paired 95% CI | Δ relative error |
|---|---:|---:|---:|
| planner | +.004 | `[-.002,+.009]` | -.142 |
| planner history | +.001 | `[-.005,+.007]` | -.145 |
| planner + frozen latent | +.026 | `[+.012,+.040]` | -.141 |
| history + frozen latent | +.021 | `[+.008,+.034]` | -.146 |
| planner + true state, privileged | +.070 | `[+.051,+.090]` | -.149 |

H8 上 deployable frozen-latent families 只有 `+.011` 与 `+.013` cosine，CI
均跨 0；privileged true-state 也只有 `+.027 [-.002,+.057]`。

但对同一批 oracle updates 做离散 correction codebook ceiling 时，K=16 达到：

| cell | Δ cosine | paired 95% CI | Δ relative error |
|---|---:|---:|---:|
| H5/off40 | +.252 | `[+.238,+.268]` | -.130 |
| H8/off60 | +.214 | `[+.183,+.245]` | -.099 |

所以 direct head 的 `relative error` 改善主要来自收缩 update magnitude；它把
彼此不兼容的方向平均掉了。真正剩余的对象不是一个连续平均向量，而是需要根据
state/candidate 识别的离散方向 mode。

### 14.8 candidate-level cost head Gate：接口有信号，当前 deployable 表示失败

为避免直接平均 update，训练独立小 MLP 给每个 frozen-WM CEM candidate 预测
residual logit，再由 CEM 对 corrected logits 取 top-30。candidate features
包含 proposal-normalized action、相对 model-elite action、frozen score/rank；
context 分别使用 planner、frozen LeWM latent、causal history latent 和
privileged true state。epoch 与 residual blend 都只在 inner held-out states
选择，每个最终预测来自 3-fold outer state-held-out model。

H5/off40 的第一轮严格结果为：

| context | Δ cosine | paired 95% CI | Δ relative error | Δ overlap |
|---|---:|---:|---:|---:|
| planner | -.004 | `[-.019,+.011]` | +.031 | +.001 |
| planner + frozen latent | +.013 | `[-.022,+.048]` | **+.049** | +.010 |
| causal history + latent | -.010 | `[-.041,+.022]` | **+.073** | -.006 |
| planner + true state, privileged | +.088 | `[+.049,+.129]` | +.015 | **+.052** |

其中正的 `Δ relative error` 是恶化。为排除一次 neural training seed 偶然，
对 frozen-latent 和 privileged-state 两臂各补 3 个 seed；与原 seed 合并后的
四-seed training variation 为：

| context | mean Δ cosine (seed range) | mean Δ rel (seed range) | mean Δ overlap (seed range) |
|---|---:|---:|---:|
| frozen latent | +.0079 `[+.0016,+.0139]` | +.0562 `[+.0485,+.0614]` | +.0041 `[-.0001,+.0103]` |
| true state, privileged | +.0605 `[+.0434,+.0882]` | +.0171 `[+.0117,+.0253]` | +.0363 `[+.0278,+.0523]` |

预锁定 Gate 要求 `Δcos≥+.10`、`Δrel≤-.10`、`Δoverlap≥+.05`。deployable
frozen-latent arm 三项均失败，而且 relative error 在四个 seed 上都恶化；因此
不扩 H8，也不进入 recursive MPC/full training。privileged state 的方向和 overlap
改善跨 seed 保持同号，但仍只满足一部分 Gate，说明 candidate-ranking interface
本身不是完全错误，关键缺口是可泛化、control-relevant 的 state/goal geometry。

本轮最终判决：

```text
continuous averaged update head                    STOP
candidate-level reranking with raw frozen latent   STOP
candidate-level reranking with explicit geometry  WEAK POSITIVE UPPER BOUND
next: learned object/state geometry bottleneck
      + candidate-level cost correction            TEST
```

对应复现实现在
`scripts/plan/oe_candidate_cost_head_probe.py`、
`run_a100_candidate_cost_head_gate_20260720.sh` 和
`run_a100_candidate_cost_head_seed_audit_20260720.sh`；A100 compact outputs
位于
`/225010117/logs/candidate_cost_head_gate_a100_20260720/` 与
`/225010117/logs/candidate_cost_head_seed_audit_a100_20260720/`。

### 14.9 correction 不是任意高维噪声，而是两条反向 spatial axes

对 H5/off40 `240 states × 4 rounds` 的 oracle-minus-model CEM mean update
做 K=4 spherical codebook。四个 prototype 的前两个奇异轴解释 `98.75%`
energy，且形成两对近似反向 modes：

```text
pair cosine ≈ -.963 / -.981
dominant action pattern ≈ long-horizon ±x / ±y correction
adjacent-round same mode = 61.25%
all-four-rounds same mode = 28.3%
```

mode counts 近似平衡。这把 §14.7 的“direct head 平均方向”从解释升级成具体机制：
同一 compressed input 下的 squared regression 会返回 conditional mean，而相反
correction 的 mean 接近零或落入不可执行方向。

但 exact one-mode router 仍失败。nested state-held-out K=4 routing 得到：

| features | Δ cosine | Δ relative error |
|---|---:|---:|
| planner | -.006 | +.014 |
| planner + frozen CLS latent | +.017 | +.003 |
| planner + true state, privileged | +.016 | +.029 |

连 true physical state 也不能决定 mode，说明 label 主要依赖当前 sampled
population 的 counterfactual outcomes，而不是静态 state quadrant。

### 14.10 Set-valued optimizer operator：branch set 成立，hard top1 不成立

据此把 operator 改成 permutation-equivariant population-to-set map。完整
300-candidate population 先由 attention 编码；M=5 输出 no-op 加四个
correction hypotheses，训练用 best-of-M physical update loss。所有 codebook、
normalization、epoch/blend selection 仍严格在 outer training states 内完成。

第一轮结果：

| context | output | Δ cosine | Δ relative error |
|---|---|---:|---:|
| frozen CLS latent | M=1 top1 | +.028 | +.017 |
| frozen CLS latent | M=5 top1 | +.007 | +.001 |
| frozen CLS latent | M=5 top2 coverage | +.051 | -.034 |
| frozen CLS latent | M=5 all coverage | +.106 | -.073 |
| true state, privileged | M=1 top1 | +.076 | -.050 |
| true state, privileged | M=5 top1 | +.058 | -.032 |
| true state, privileged | M=5 top2 coverage | +.113 | -.081 |
| true state, privileged | M=5 all coverage | **+.174** | **-.128** |

`top2/all coverage` 用 oracle 在 retained hypotheses 中选最好者，只回答 branch
set 是否覆盖正确 basin，**不是 deployable selection**。raw latent 的 top1
best-mode rate 为 `21.1%`（chance `20%`），top2 contains-best 为 `39.5%`
（chance `40%`）；true state 只提高到 `28.3% / 49.8%`。

因此当前第一个可靠正结论是：

```text
single averaged optimizer update                 falsified
set-valued optimizer-equivalent representation   supported
learned hard top1 route                           not supported
```

### 14.11 dense spatial read path 与 vector imagined outcomes

raw cache 只含 SIGReg projected CLS。为测试 spatial coordinates 是否被 CLS
压掉，从同一 frozen ViT encoder 保存三帧 history 与 goal 的完整
`16×16×192` patch grid；240 states cache 的 state/goal reconstruction mismatch
均为 0。

dense goal-relative cross-attention 的 M=5 结果（两个 seed 同向）：

| output | Δ cosine | Δ relative error |
|---|---:|---:|
| learned top1 | +.012 | -.043 |
| top2 coverage | +.058 | -.075 |
| all-mode coverage | **+.111** | **-.111** |

它把 best-mode rate 提到约 `27%`，接近 privileged state 的 `28%`，说明 frozen
patch grid 的确保留了 spatial state；但它仍不能从静态几何决定
population-specific mode。

下一步保留 LeWM 对每个 candidate 的 signed imagined terminal-minus-goal vector，
而不是只保留其 squared norm scalar cost。K3 vector outcome 得到：

```text
M=1 top1          +.037 / -.045
M=5 top1          +.025 / -.047
M=5 top2 coverage +.078 / -.083
M=5 all coverage  +.122 / -.112
```

其中成对数值依次为 `Δcos / Δrel`。vector outcome 确有额外 signal，但仍不足以
做 hard routing。把相同 vector 喂给 independent candidate reranker 的四个 seed
反而一致恶化 relative error `+.042~+.064`，所以不能退回“再训一个 cost head”。

one-step observable latent innovation、regret-aware/softmin router、以及 frozen
dense moments 上的 structured pose dynamics 也全部失败：

```text
innovation M5 top1                    +.005 / -.011
dense+innovation+outcome M5 top1      +.014 / -.033
dense structured relative dynamics   Δrel ≈ +.147
dense structured terminal dynamics   Δrel ≈ +.184
```

这排除了 raw CLS、静态 geometry、最近一步 prediction residual 或 routing loss
任何一个单独作为剩余解释。

### 14.12 cross-model vector disagreement 是当前最强 deployable signal

用户提出“不同类型/尺度的 WM 可以与 LeWM 组合，而不是一概视为冲突”后，进一步在
同一 K3 population 上缓存 K5 与 K10 的 signed candidate imagined outcomes。
这与此前 shared-rank ensemble 不同：不平均 scalar ranks，而是让 set operator
观察各模型保留方向的 counterfactual disagreement。

| inputs | top1 `Δcos/Δrel` | top2 coverage | all-mode coverage |
|---|---:|---:|---:|
| K3 + K5 vectors | `+.037 / -.051` | `+.087 / -.085` | `+.147 / -.124` |
| K3 + K10 vectors | **`+.045 / -.033`** | **`+.097 / -.079`** | **`+.164 / -.125`** |
| K3 + K5 + K10 vectors | `+.029 / -.057` | `+.064 / -.081` | `+.111 / -.110` |
| dense + all three vectors | `+.034 / -.026` | `+.091 / -.074` | `+.165 / -.125` |

K3+K10 top1 的 cosine CI 为 `[+.024,+.066]`，首次得到稳定的 deployable
cross-model signal，但仍低于 locked `+.10` Gate。只给 operator 多模型 scalar
cost/rank 时 top1 为 `-.009~+.014`；有效部分来自 signed vector geometry，
不是普通 score ensemble。

这也解释为何“模型组合”本身不与现有 WM idea 冲突，而 naive ensemble 会冲突或
失败：

```text
scalar opinion averaging                  catastrophic earlier control
independent proposal portfolio            weak positive, compute-heavy
vector disagreement → optimizer branches  new surviving mechanism
```

### 14.13 新的 surviving idea 与下一 Gate

暂定方法名：

> **Branch-Preserving Optimizer-Equivalent World Model（BP-OEWM）**

定义从 scalar cost 改为 population-conditioned set operator：

```text
Fθ(o,g,q) = { (Δμm, Δlogσm, πm) }m=1…M
```

best-of-M OE loss 保证至少一个 branch 接近 true-dynamics optimizer update；
多个 WM 的 signed imagined outcomes 提供可泛化 disagreement。关键实现约束是：
不能在每轮 hard top1，也不能把 modes 重新平均成一个 Gaussian；必须递归保留
proposal branches。

下一步锁定 disjoint 12-state recursive Gate：

```text
train/calibrate: current 240-state H5 bank
eval: week1 H5/off40 12 states, row overlap = 0
inputs: K3 + K10 vector outcomes
carry: B=2 branches, total N=300 (150 each)
controls: ordinary K3 1×300; K3 2×150 multistart
selection: model-side only; simulator/oracle only reports hidden ceiling
```

只有 learned branch reduction 在 recursive resampling 后同时超过 ordinary K3
与 matched-compute multistart，才进入 H8/full MPC。若只剩 oracle union 增益，
则 set-valued OE 仍只是 uncertainty ceiling，不能宣称方法 work。

本轮 compact reports 已下载到
`docs/knowledge/optimizer_equivalence_a100_20260720/`；大体积 caches 留在 A100。

### 14.14 recursive + sparse + iso-compute：生成成立，选择目标错位

12 个 row-disjoint states 上连续做了三轮 recursive re-sampling，并用 CRN
重跑消除方法间 population noise。matched BP 的 deployable K10 selector 相对
ordinary K3 `1×300` 为 `-2.87`、8/12 wins，但 CI
`[-10.39,4.75]` 跨零；same-model `2×150` 为 `+2.61 [.32,5.19]`，所以
branch mechanism 没有退化成普通 multistart，但仍未达到 work 标准。

随后用 K3 先看全部 candidates、K10 只查询 K3 elite 10%：

```text
fixed trace top1/top2/all Δcos = +.026 / +.075 / +.133
recursive K10 selector delta = -1.59, 6/12 wins
recursive oracle union delta = -11.04, 7/12 wins
oracle median / trimmed delta = -4.23 / -4.16
```

static primary/consensus 却为 `+3.91`。这说明 sparse cross-horizon signal
生成的 branch set 有明显高 ceiling，但当前 router 学的是局部 oracle-moment
mode，不是三轮后的 branch outcome；剩余 bottleneck 是 value-aligned pruning，
不是继续改 winner-CE 或增加 mode 数。

等算力 audit 又补了 K3 `1×600/2×300`：

```text
K3 1×600 vs 1×300              -0.01 [-1.60,1.60]
K3 2×300 vs 1×300              +1.42
BP full K10 vs K3 1×600        -3.31
BP full oracle vs K3 1×600     -7.21
sparse oracle vs K3 2×300     -12.46 [-31.18,-.04]
```

额外 K3 samples 已基本饱和，因此 branch rescue 不是多算力假象。新的 locked
实验不再复用此前 12-state test 调 selector：

1. 四折 OOF operator 在 240 training states 上生成 recursive branch pairs；
2. simulator 只标两条最终 branch mean，避免再次执行完整 populations；
3. 以 final outcome 训练 antisymmetric branch-value head；
4. 超参只由 240-state shard-held-out CV 锁定；
5. 新生成 60 个同时排除 training240 和旧 eval12 的 H5/off40 rows，一次性报告
   ordinary K3、learned selector、K10 selector 与 hidden oracle union。

同时保存每轮 proposal mean/std。若两条 branch 的前两个 action blocks 明显比
末端更接近，则下一层方法可改为 observation-conditioned deferred commitment；
若从第一步就同样分离，则关闭该路线，转向真正的 recursive value/pruning loss。

### 14.15 fresh120 falsification：不要再调 selector，转向 basin topology

完成了四折 OOF 的 240-state branch label bank，并连续开封两套互相以及与所有旧
数据都 row-disjoint 的 fresh60。锁定 action-trajectory ridge selector 在 A 上
相对 K3 `1×300` 为 `-2.83`，到 B 只剩 `-.04`；静态 K10 也在
`+.73/-1.13` 间波动。说明当前 240 labels 学不到跨 state distribution 稳定的
branch value。

显式保留 ordinary K3 no-op anchor 在 A 上给出很强 post-hoc ceiling，于是用 B
预注册确认：

```text
proposed calls = K3 anchor 300 + sparse BP 300
adopt BP iff final K10(BP) < final K10(anchor)
control = ordinary K3 600
```

confirm-B 的 adoption delta 为 `+.98 [-3.86,6.38]`，actual success
`26.7% vs 33.3%`，失败。hidden anchor+two-branch oracle 的 true-cost delta
仍为 `-6.31 [-9.47,-3.52]`，但 success 与 K3 `1×600` 同为 `33.3%`。

同时，final branches 的 action-step normalized divergence 在 train/A/B 都几乎
平坦，first-two/last-two ratio 为 `1.003/.969/.995`。branch 从第一 action
block 就分开，无法先执行共同 prefix 再靠 observation 消歧。

这轮最重要的不是又一个负数，而是把两层问题拆开：

```text
branch generation / continuous-cost coverage  reproducible
deployable selection / task success           not reproducible
```

所以 frozen LeWM 上的 planner-side BP patch 到此停止。新的高上限问题改为：
真实动力学的 low-cost set 是否本来就含多个 persistent basins，而 standard
next-state training、AWM 式 convexification 或 latent straightening 会发生
basin merge/drop？下一实验应在同一 action population 上做 lower-star/connected
component iso-rate topology audit，比较 true、K3、K10；不是继续训练 final
reranker。

### 14.16 static topology → adaptive basin lineage

完成 `240 states × 4 CEM rounds` 的 iso-rate `H0` audit 后，真实 top-10% action
set 在 correction/full geometry 中平均都有约 `5` 个 connected basins，不支持把
正确 planning landscape 预设成单一 convex basin。K3/K10 对这些 basins 的覆盖都
很低，但 K10 在三种 geometry 上都一致优于 K3：

```text
                          correction  full-50D  random-2D
K3 component coverage       .400        .432       .308
K10 component coverage      .499        .544       .435
K3 successful coverage      .454        .552       .383
K10 successful coverage     .520        .627       .483
```

需要保留一个重要的 robustness 修正：K3 在 held-out correction subspace 中会
fragment（persistent-count delta `+.50`），但在 full action graph 中反而比 K10
merge/drop 更多（top-rate count delta `-3.00 vs -1.15`）。因此可靠现象是
**basin correspondence/coverage failure**，不是简单的全局
“short horizon rough、long horizon smooth”二分。

naive horizon combination 仍失败。average ranks 的 top-rate recall `.235` 低于
K10 `.296`；min-rank union 产生更多 spurious components。更强的是，把每个 scorer
的 top-30 真正 refit 为 next CEM proposal，再在相同 10% support rate 下审计，
K10 的 one-step mean/log-std error及 successful-basin coverage 仍稍好于 K3。

这与早先“K10 fixed-bank 好、K3 recursive/MPC 更稳”形成新的严格矛盾：

```text
static candidate fidelity       K10 > K3
one-step proposal fidelity      K10 > K3
adaptive closed-loop planning   K3 > K10
```

所以 horizon-matching 的下一层对象不是某个固定 rate 或单张 landscape，而是
proposal 随 CEM iteration 改变时的 **basin lineage**。更精确的 idea 是：

> 在 planner-induced query sequence 上，匹配真实 dynamics 的 persistent basin
> birth/death/continuation；smoothness 只在同一真实 basin 内使用，不能跨 barrier
> 把多个有效 contact strategies 拉成 conditional mean。

暂名 **Basin-Lineage World Model（BL-WM）**。它与原 LeWM 不冲突：LeWM 继续负责
visual latent dynamics，新增 supervision 只约束其 action-cost pushforward 在
planner proposal sequence 上的拓扑。实现优先使用 persistent pairing 的稀疏
birth/saddle witnesses，而不是每步反传完整 persistence diagram：

```text
L = L_LeWM
  + λcover · L_each_true_basin_has_low_model_witness
  + λbarrier · L_true_saddles_remain_above_basin_minima
  + λlineage · L_component_transport_across_optimizer_rounds
```

K3/K10 signed disagreement 的新角色是低成本 query acquisition：优先向 simulator
查询可能发生 basin birth/death 的候选，不再作为 frozen inference ensemble。

当前已启动同一套 fresh60 rows 上的 K3/K10 paired on-policy full-population trace。
只有它证明 earlier basin death 能预测 final failure、且 K10 的 static advantage
确实在自己的 adaptive path 中翻转，才进入 BL-WM fine-tune。这比直接拿静态
topological loss 训练更慢一步，但能避免重复 fixed-trace OE 的同类错误。

## 15. 2026-07-21 consolidation：paired path 为正，topology causal Gate 为负

paired trace 确认了 K10 的 self-induced tail reversal：final true-top30 recall 在
K3 path 是 `K3/K10=.096/.272`，在 K10 path 变为 `.197/.079`；path interaction
为 `-.293 [-.381,-.207]`。但 counterfactual refit 表明这不是一个 K3 scorer swap
或 component split 可以修的 failure：

| K10-path final intervention | success |
|---|---:|
| stored / K10 global mean | `43.3%` |
| K3 global mean | `45.0%` |
| true global mean | `58.3%` |
| true component oracle | `60.0%` |
| candidate-support ceiling | `63.3%` |

K3 swap 只净增 `+1.7pp [-6.7,+10.0]`，component oracle 在 true global 之上只救
1/60 state。因此 §14.16 的 BL-WM promotion Gate **未通过**；true multi-basin
structure 保留为 diagnostic，generic topology training 不启动。

当前唯一开放的问题收紧为：如何在模型自己诱导的完整 proposal sequence 上维持
low-cost-tail / elite-update validity，或对不可信 update 给出风险证书。它仍需先做
新一轮 literature collision，不是已经锁定的新方法。统一停止清单、证据账本和
下一 Gates 见
[LeWM planning 研究现状总账](lewm_planning_status_20260721.md)。
