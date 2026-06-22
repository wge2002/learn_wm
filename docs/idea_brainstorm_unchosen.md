# Idea Brainstorm（B 线候选池，尚未 choose）

> 本 MD = **与离散承诺锚点并行的第二条 paper seed 候选池**。A 线离散实验继续推进；这里不是为了替换 A 线，也不是为了给离散找竞品，而是寻找另一个基于 world model / latent world model 的、同等有意义的独立主张。
> A 线见 [commitment_anchor_discrete.md](commitment_anchor_discrete.md);诊断依据见 [diagnosis_world_model_drift.md](diagnosis_world_model_drift.md)。

## 为什么需要重写这个池子

早期 brainstorm 的问题是把"同级"理解成了"同样宏大的名词"：吸引子、组合性、时序抽象、因果最小等都能说出一套故事，但很多落地后会变成 loss、metric、head、planner trick，或者只是给 LeWM 某个 bug 打补丁。

离散承诺锚点之所以有同级感，不是因为它能提升某个 metric，而是因为它提出了一个 world model 内部应该存在的对象：

```text
continuous latent 负责精度
discrete commitment token / plan-word 负责分支承诺、对齐、组合、复用
```

也就是说，它改变的是 world model 的内部 ontology，而不是外层 planner 的评分方式。B 线也必须达到这个级别。

## "和离散同级"的判据

一个能和离散同级的 idea，必须满足：

1. **提出 WM 缺的某种内部结构**，不是 loss、metric、head、planner trick。
2. **这个结构不是 policy/Q 可以直接替代的**，因为它要支持 counterfactual imagination、composition、reuse。
3. **它有一个外部经验地基**，像离散借语言/符号；不是只从 LeWM bug 推出来。
4. **它能改变我们理解 WM planning 的方式**，不是只修一个现象。

因此，下面这些方向先从主候选里删除：

- **Action ranking**：适合作为诊断指标，不适合作为核心 idea。否则会瞬间塌成 action selector / Q / policy。
- **纯 cost/metric patch**：例如只把 L2 换成 quasimetric、加 uncertainty penalty、加 ranking loss。如果没有新的内部结构，意义不够。
- **普通时序抽象 / options**：如果只是"把 horizon 缩短"，很容易是老问题重做；除非能提出新的 WM 内部事件结构。
- **视觉因果最小 latent**：如果只解释 visual shift，会太窄，像 robustness/bisimulation 补丁。

## 保留候选总表

| 候选 | WM 缺的内部结构 | 外部地基 | 为什么保留 | 最大风险 |
| --- | --- | --- | --- | --- |
| Event / Phase WM | 事件边界、阶段、regime transition | 人类事件认知、接触动力学、hybrid systems | 最像一个新的推理单位 | 容易退化成 options/分段预测 |
| Action-Realizable Geometry | 可行动几何、affordance/reachability graph | 生态心理学 affordance、控制理论、机器人 motion planning | 直接挑战 latent L2 planning 的前提 | 容易退化成 metric learning |
| Compositional Controllable Dynamics | 实体、关系、局部可控子系统 | 物体物理、object-centric learning、systematic generalization | 有足够大的外部地基 | 赛道拥挤，必须和 planning/control 绑定 |
| Attractor / Energy WM | 吸引子盆地、能量面、回复力 | Hopfield/联想记忆、EBM、动力系统 | 和 rollout drift 的 ontology 对立 | 容易变成 denoiser/EBM patch |
| Intervention / Controllability Factors | 可干预因子、不可控背景、动作影响边 | 因果干预、controllability、active inference | 比"去背景"更大，有 WM 意义 | 容易被 bisimulation/Q 吃掉 |

---

## 候选 1：Event / Phase World Model

### 核心主张

长程 planning 不是 dense latent state 一步步外推，而是由稀疏事件和阶段切换组织的。世界模型缺的不是更准的下一帧，而是：

```text
state -> event boundary -> new regime -> event-conditioned dynamics
```

例如 PushT 里的接触/脱离/对齐，TwoRoom/Corridor 里的过门/撞墙/进入下一段，机器人任务里的抓住/释放/插入/锁定。这些事件改变了后续 dynamics 和可行动集合；如果 WM 只滚 dense latent，它会在事件边界附近糊、漂、或者给 planner 错误的 counterfactual。

### 为什么保留

- 它提出了明确的内部结构：**event variable / phase state / regime transition**。
- 它不是 policy/Q：事件结构可以被不同 goal、reward、constraint、planner 复用。
- 外部地基很强：人类事件认知、接触丰富的机器人学、hybrid dynamical systems 都把世界理解为阶段/模式切换，而不是纯连续流。
- 它能解释很多 long-horizon failure：不是每一步都错，而是在事件边界处错一次，后面整段 counterfactual 都换了。

### 可能的第一轮信号

- 在已有 PushT/TwoRoom/Corridor 轨迹里标出或无监督发现事件边界，测 WM 的 cost-ranking / rollout error 是否在 event boundary 前后显著恶化。
- 学 event-conditioned dynamics，比单一 predictor 更能保持 long-horizon planning ranking。
- 用 event graph 做 planning：先预测事件序列，再在阶段内用 continuous latent 控制。

### 最大问题

- 很容易被 reviewer 认为只是 options / skills / temporal abstraction。
- 必须证明事件不是手工分段，也不是 planner trick，而是 WM 内部可复用的推理对象。
- 如果只在单个环境上用接触标签，会显得太工程；最好要有无监督/弱监督事件发现，或跨任务复用。

## 候选 2：Action-Realizable / Affordance Geometry

### 核心主张

当前 latent WM planning 默认：

```text
latent L2 close = planning close
```

但控制世界里，"表征近"和"可行动地近"不是一回事。两个 latent 很近，可能隔着墙、接触模式、不可逆门；两个 latent 看起来远，可能沿 action manifold 很容易到达。WM 缺的是一个由动作可达性定义的内部几何：

```text
z-space should be organized by feasible action paths, not perceptual similarity.
```

这不是简单换一个距离函数，而是要求 WM 内部显式表示 affordance / reachability / feasible geodesic。

### 为什么保留

- 它提出的内部结构是**可行动几何**：哪些方向可控、哪些状态可达、哪些路径不可行。
- 它不是 policy/Q：可达几何可以支持不同目标、不同 reward、不同 constraints 下的 planning。
- 外部地基很强：Gibson affordance、motion planning、controllability、robotics 中的 configuration-space geometry。
- 它直接挑战 LeWM 类方法的核心假设：用 encoder latent 的欧氏几何当 planning geometry。

### 可能的第一轮信号

- 复用 Phase 8c 的失败：latent interpolation waypoint 全面变差，说明几何中点不是 action-realizable 中途。
- 构造"latent 近但 action 不可达 / latent 远但 action 易达"的 pair，测 planner cost 和真实可达性的错位。
- 学一个 action-realizable chart / geodesic / reachability graph，并证明它比 terminal latent L2 更能支持 planning。

### 最大问题

- 如果最后只是学一个新的 distance head，会降级成 metric learning。
- 要避免变成 Q/value：必须强调它表示的是可复用的 state-to-state reachability geometry，而不是某个 reward 下的 action value。
- 理论上漂亮，但实验设计要很小心：需要证明"几何结构"能换 goal / 换 constraint 复用。

## 候选 3：Compositional Controllable Dynamics

### 核心主张

Dense monolithic latent 把整个世界压成一个向量，长程 rollout 的误差会全局扩散；但真实世界的动力学通常是实体、关系、局部接触和局部可控子系统组成的。WM 缺的不是更大 latent，而是：

```text
objects/entities + relations + local controllable dynamics
```

planning 不应该在一个纠缠整体向量里做，而应该能按实体和关系组合 counterfactual。

### 为什么保留

- 它提出明确内部结构：entity slots、relations、object-wise controllability。
- 它不是 policy/Q：对象关系模型天然支持 counterfactual composition，例如换目标、换障碍、换对象数。
- 外部地基极强：物理世界是对象和关系组成的；object-centric learning 和 systematic generalization 已经提供很多经验基础。
- 它可能解释 Phase 6 里 error vector 铺满全维的现象：整体 latent 一漂全维一起漂，而组合式 dynamics 应该更局部。

### 可能的第一轮信号

- 在 PushT/Scene/Cube 里测 perturbation 是否应该局部化到 object/relationship，但 LeWM latent error 是否全局扩散。
- 用 object-wise dynamics 或 relation graph 做 planning，比较 geometry shift / contact shift 下的泛化。
- 证明 factorized WM 对新对象配置或新 goal 更可复用，而不是只提高 reconstruction。

### 最大问题

- 赛道很拥挤，容易被认为是 object-centric world model 的常规延伸。
- 必须把 novelty 放在 **planning/control 需要的可控分解**，不是普通 slot reconstruction。
- 对 PushT 这类低对象数任务，效果可能不够戏剧；需要找对象组合性真正 binding 的环境。

## 候选 4：Attractor / Energy World Model

### 核心主张

自回归 latent rollout 的问题可能不是"预测器不够准"，而是它缺少动力系统里的回复力。一个支持长程推理的 WM 不应该只做：

```text
z_t, a_t -> z_{t+1}
```

而应该有能量面 / 吸引子盆地 / 稳定 manifold，让 imagined trajectory 在长期内回到可解释、可规划的结构上。

### 为什么保留

- 它提出内部结构：energy landscape、basins、fixed points / attractors。
- 它不是 policy/Q：能量面可以约束多种 planning 查询下的 imagination。
- 外部地基有一定重量：Hopfield networks、associative memory、EBM、dynamical systems。
- 它和现有 LeWM 诊断有自然张力：Phase 6/7 的 on-manifold diffusion 可以被理解为缺少 restoring force。

### 可能的第一轮信号

- 证明 LeWM rollout error 是沿 manifold 扩散，而不是离开 manifold；然后测试 learned contraction / energy projection 是否能保持 planning-relevant structure。
- 比较 autoregressive rollout vs energy refinement rollout：同样训练数据下，谁更能保持 long-horizon counterfactual consistency。
- 在多目标或长程任务中，把 planning 写成 energy minimization over trajectory，而不是逐步 rollout。

### 最大问题

- 非常容易降级成 denoiser、diffusion refinement 或 EBM patch。
- 外部地基虽大，但和 robotics WM planning 的连接需要讲清楚，否则像借名词。
- 如果只证明 latent MSE 降了，不够；必须证明它改变了 imagination 的稳定机制和 planning 复用能力。

## 候选 5：Intervention / Controllability Factor World Model

### 核心主张

Planning 关心的不是"未来会怎样"，而是"我通过动作能改变什么、不能改变什么、改变某个因素会如何影响其他因素"。普通 predictive latent 容易把可控因素、不可控背景、偶然相关特征混在一起。WM 缺的是：

```text
controllable factors + uncontrollable context + action intervention edges
```

这比"只表示最小充分状态"更强，因为它要求模型内部知道动作的干预边界。

### 为什么保留

- 它提出内部结构：intervention axes、controllability partition、action-effect graph。
- 它不是 policy/Q：可干预结构可以被多个目标和 reward 复用，支持 counterfactual queries。
- 外部地基明确：causal intervention、controllability、active inference、robotics 中的 action-effect models。
- 它能把 visual shock / geometry shift / distractor robustness 统一到一个更大的问题：WM 没分清哪些变量对动作和目标有因果作用。

### 可能的第一轮信号

- 构造同 observation similarity 但 controllability 不同的状态对，测普通 latent 是否混淆。
- 学 action-intervention factorization，验证它在换颜色、换背景、换不可控 distractor 时保持 planning。
- 测 factor 是否支持 counterfactual editing：固定不可控背景，改变可控因素，rollout 是否只改变应改变的未来。

### 最大问题

- 如果只做 visual invariance，会退回"因果最小 latent / bisimulation"老问题，太窄。
- 如果只服务一个 reward，会被 Q/value 吃掉。
- 需要设计出真正体现 intervention/composition/reuse 的任务，否则意义不够。

---

## 当前倾向（未定）

按"idea 意义"而不是"最容易出结果"排序，目前最值得认真展开的是：

1. **Event / Phase WM**：最像离散那种"WM 里缺一个推理单位"的主张，外部地基也强。
2. **Action-Realizable Geometry**：最直接挑战 latent L2 planning 的基础假设，理论味道好，但要防止降级成 metric learning。
3. **Compositional Controllable Dynamics**：意义大，但赛道拥挤，必须把切口钉在 planning/control 的可控分解上。

Attractor/Energy 和 Intervention/Controllability 暂时保留为备选：它们都有机会，但最容易被做小、做旧、做成 patch。除非能找到一个非常干净的内部结构和可复用实验，否则不优先展开。

下一步不是马上开实验，而是对前三个各写一页：

- 核心 ontology：WM 里多了什么对象？
- 外部地基：为什么这个对象不是我们硬凑的？
- 与 policy/Q 的边界：为什么不能直接用 policy/Q 替代？
- 最小证伪实验：什么结果会让这个 idea 当场死亡？
