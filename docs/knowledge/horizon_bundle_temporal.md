# Temporal 方向备忘录：Horizon-Bundle World Model（2026-07-17）

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
