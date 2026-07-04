# LeWM / stable-worldmodel 后续方向调研（arXiv + 顶会，2026-07-04）

> 范围：截至 2026-07-04，重点看和 LeWM / stable-worldmodel 最相关的
> reconstruction-free latent world model、JEPA world model、MPC/planning、robotics world model、
> benchmark/platform 工作。结论按“能不能成为我们后续 paper seed”来读，而不是普通读书笔记。

## 0. 执行结论

最近确实有一批新 paper 直接压到 LeWM 这条线上。最关键的是：

1. **Fast-LeWorldModel（2026-06-24）已经把“LeWM rollout 慢、长程 latent error 累积”作为主问题做了。**
   它用 action-prefix prediction 替代一步一步 autoregressive rollout，报告平均成功率从 85.8% 到
   90.5%，加 self-consistency 到 92.0%，CEM 求解时间约从 54.4s 到 28.3s。
   因此“多步监督 / 降 latent drift / 加速 rollout”本身已经不是新意。

2. **JEPA-WM 系统性 ablation 已经出来。**
   *What Drives Success in Physical Planning with Joint-Embedding Predictive World Models?*
   已被 TMLR 接收，系统扫了 encoder、predictor、multi-step rollout、context length、proprioception、
   planner 等组件。它的结论会成为后续 LeWM 论文必须对齐的 related work。

3. **新的空位不是“更低 self-drift”，而是“为什么更低 self-drift 有时反而坏 planning”。**
   你们已有实验已经看到 pure multistep 把 drift 压低但 PushT planning 82/86% 掉到 22/40%。
   这和当前 literature 主线形成缺口：多数论文仍把 latent prediction error、rollout error、speed 当正指标，
   少有人正面定义 **predictive-control gap**。

4. **stable-worldmodel 的后续价值更像 benchmark/evaluation substrate。**
   近期 benchmark 明显转向“decision-relevant fidelity”：WorldArena、WorldModelGym、ACT-Bench、
   WorldMark、iWorld-Bench 都在说视觉好看不等于可决策。stable-worldmodel 可以抓住这个趋势：
   把“latent error、action ranking、FoV/OOD、planning success、wall-clock”统一成可复现评测。

5. **下一篇更 solid 的方向应是：Control-Sufficient / SER-aware Gaussian JEPA。**
   主张不要写成 stop-gradient trick，也不要再写 regime-MoE。应写成一个 failure mode：
   Gaussian/self-targeted JEPA 会选择“可预测坐标”，但 planning 需要“控制充分坐标”。
   方法上用 ranking / control-Fisher / controllability-gated sufficiency 保护 task-critical high-uncertainty directions。

## 1. 最相关论文速览

| paper | 状态 | 核心做法 | 主要结论 | 对我们的影响 |
| --- | --- | --- | --- | --- |
| [LeWorldModel](https://arxiv.org/abs/2603.19312) | arXiv 2026-03，v3 2026-06 | end-to-end JEPA；next-embedding MSE + SIGReg；无 EMA/stop-grad/pretrained encoder | 15M 参数；loss 超参从 PLDM 类多项降到 1；planning up to 48x faster than DINO-WM；PushT 等任务竞争力强 | 我们的基线对象。它的弱点是 one-step teacher forcing + self-targeted latent |
| [stable-worldmodel](https://arxiv.org/abs/2605.21800) | arXiv 2026-05 | 统一数据层、baseline、solver、FoV 环境、评测协议 | 解决 world model 研究代码碎片化、数据加载慢、泛化基准不足 | 可以成为后续实验平台和 benchmark contribution |
| [Fast-LeWorldModel](https://arxiv.org/abs/2606.26217) | arXiv 2026-06-24 | action-prefix encoder + parallel latent predictor；不滚中间每一步；可加 self-consistency scoring | 平均成功率 LeWM 85.8 -> 90.5，+SC 到 92.0；dynamics module 31.4s -> 8.0s；CEM 54.4s -> 28.3s | 把“加速 + 降长程 latent error”占掉了。我们不能再把多步/前缀预测作为主创新 |
| [What Drives Success in Physical Planning with JEPA-WMs?](https://arxiv.org/abs/2512.24497) | TMLR accepted，v3 2026-05 | 系统 ablation：encoder、AdaLN/RoPE predictor、rollout steps、context、proprioception、planner | 推荐 recipe：CEM/L2；sim nav 较短 rollout/context，real manipulation 更深 predictor、更长 rollout/context；提出比 DINO-WM/V-JEPA-2-AC 更强组合 | 必须作为强 related work。它覆盖“JEPA-WM 工程 recipe”，但未解决 self-drift 与 planning 可反向 |
| [Causal-JEPA / C-JEPA](https://arxiv.org/abs/2602.11389) | ICML 2026 accepted | object-centric latent masking；mask object slots，让 masked object 由上下文推断 | counterfactual reasoning 约 +20%；control 中用 patch-based WM 1% latent features 达到可比 planning | 给我们一个“结构化部分可观测 / counterfactual query”的方向，但它偏 object-centric，不是 Gaussian JEPA sufficiency |
| [Learning Invariant Visual Representations for Planning with JEPA-WMs](https://arxiv.org/abs/2602.18639) | arXiv 2026-02 | 在 DINO-WM 类 objective 外加 bisimulation encoder，压 slow visual features / distractors | background/distractor robustness 改善；latent 维度可小到 DINO-WM 的 1/10 | 与我们的 FoV shift 诊断强相关：慢特征不变性和 control relevance 是同一大问题 |
| [V-JEPA 2](https://arxiv.org/abs/2506.09985) | arXiv 2025-06 | 互联网视频自监督预训练 + 少量 robot interaction alignment，得到 V-JEPA-2-AC | motion understanding、VQA、robot planning 都展示 scaling potential | foundation encoder 路线；不适合直接当 LeWM 轻量 end-to-end follow-up，但会是强 baseline |
| [V-JEPA 2.1](https://arxiv.org/abs/2603.14482) | arXiv 2026-03，v3 2026-06 | dense predictive loss、deep self-supervision、image/video tokenizer、scaling | 更强 dense features；偏 representation，不直接解决 MPC 目标 | 对 frozen encoder baseline 和 dense latent probing 有参考意义 |
| [DINO-WM](https://arxiv.org/abs/2411.04983) | arXiv 2024/2025 | 冻结 DINOv2 patch features，训练 latent dynamics，CEM/MPPI goal planning | reward-free zero-shot planning，在 mazes/PushT/particles 等任务强 | 仍是 LeWM/JEPA-WM 标准 baseline |
| [PLDM](https://arxiv.org/abs/2502.14819) | arXiv 2025 | JEPA latent dynamics + VCReg/temp/idm，reward-free offline planning | latent planning 对 suboptimal offline data 和 unseen layouts 有优势 | LeWM 的“多 loss end-to-end alternative”参照 |
| [TD-MPC2](https://arxiv.org/abs/2310.16828) | ICLR 2024 | decoder-free latent dynamics + reward/Q/actor；online RL + planning | 104 online tasks；317M multi-task agent；连续控制强 | 说明 continuous latent + value-equivalent planning 仍是强基准 |
| [DCWM / DC-MPC](https://arxiv.org/abs/2503.00653) | ICLR 2025 | discrete codebook stochastic latent + MPC | 在 continuous control 上和 TD-MPC2、DreamerV3 竞争 | 离散 latent 不是完全死，但它是 RL/value/planning 设置，不是 SIGReg JEPA 的表征离散 |
| [Closing the Train-Test Gap in World Models for Gradient-Based Planning](https://arxiv.org/abs/2512.09929) | arXiv 2025-12 | 训练时合成数据，让 WM 更适合 test-time action optimization | gradient planner 用约 10% time budget 匹配或超过 CEM | 强化“训练目标必须对齐 planner 使用方式”这一点 |
| [WorldPlanner](https://arxiv.org/abs/2511.03077) | arXiv 2025-11 | action-conditioned visual WM + diffusion action sampler + MCTS + MPC | action sampler 降低 WM hallucination，实机 3 个任务验证 | 说明 planning 不只是 WM；action proposal / sampler 也很关键 |
| [EV-WM](https://arxiv.org/abs/2606.13053) | arXiv 2026-06 | pretrained-feature WM rollout 后 decode structured events，再用 task-progress/physics/uncertainty verifier scoring | 长程 manipulation 中 predicate-level verifier 改善规划 | 和我们的“latent MSE 不够，需 task predicate/ranking”同向 |
| [Foresight](https://arxiv.org/abs/2606.23085) | arXiv 2026-06 | action-conditioned WM latents 做 long-horizon failure detection；只用 episode success/failure label | 给 VLA policy failure detection 统一 latent monitor | WM latent 可用于 monitor/verification，而不只是预测 |
| [DiWA](https://arxiv.org/abs/2508.03645) | CoRL 2025 | 用 world model offline fine-tune diffusion policy | 用一次训练好的 WM 替代大量真实交互，提升 sample efficiency | stable-wm 可支持 policy refinement 方向，但离 LeWM/JEPAs 远 |
| [World4RL](https://arxiv.org/abs/2509.19080) | arXiv 2025/2026 | diffusion WM 作为 imagined environment，直接做 RL policy refinement | 不只 planning，还能在 imagination 中优化 policy | 说明 generative WM 的主战场偏 policy refinement / simulator |

## 2. 直接命中 LeWM 的后续工作

### 2.1 Fast-LeWM：LeWM 的最直接 follow-up

**问题设定。** LeWM 在 planning 时对每个 candidate action sequence 做 autoregressive latent rollout。
这有两个问题：慢；长 horizon 时误差累积。

**方法。**

- 不再逐步预测 `z_{t+1}, z_{t+2}, ...`。
- 对 action sequence 的 prefixes 编码：`a_{t:t+k}` 是一个 token/query。
- 并行预测每个 prefix 执行后会到达的 future latent。
- planning 时直接用最后一个 prefix token 的 predicted latent 算 goal cost。
- 可选 self-consistency term：同一个 terminal latent 在不同 prefix decomposition 下应一致。

**结果。**

- 四任务平均 success：LeWM 85.8%，Fast-LeWM 90.5%，Fast-LeWM+SC 92.0%。
- Two-Room 87 -> 98，Reacher 86 -> 88/90，PushT 96 -> 96/98，OGBench-Cube 74 -> 80/82。
- dynamics module 时间约 31.4s -> 8.0s，full CEM solve 54.4s -> 28.3s。

**我们的判断。**

Fast-LeWM 是“工程上合理且结果直接”的 LeWM follow-up，但它也把最自然的一条路占了：

```text
 multi-horizon supervision
+ non-autoregressive rollout
+ lower open-loop latent error growth
+ faster planning
```

如果我们的后续只做 fixed encoder multistep、prefix loss、stop-grad rollout，很容易被认为是 Fast-LeWM
或 JEPA-WM ablation 的小变体。要绕开它，核心 claim 必须是：

```text
不是降低 latent rollout error，
而是证明/修复 latent rollout error 与 control sufficiency 的错位。
```

### 2.2 What Drives Success：JEPA-WM recipe 已经被系统扫过

这篇很重要，因为它把“哪些组件让 JEPA-WM planning 成功”做成了系统 ablation。

**它怎么做。**

- 固定视觉 encoder 路线，训练 proprio/action/predictor。
- 比较 DINOv2、DINOv3、V-JEPA、V-JEPA2。
- 比较 predictor conditioning：feature conditioning、sequence conditioning、AdaLN、RoPE 等。
- 比较 rollout steps、context length、proprioception、planner（CEM、NeverGrad、Adam、GD）。
- 在 simulated navigation、MetaWorld、DROID/Robocasa 等 manipulation 数据上测试。

**重要 recipe。**

- Planner：CEM 仍是最稳；cost 用 L2。
- Predictor：AdaLN + RoPE 是强组合，real manipulation 可以用更深 predictor。
- Multistep rollout：不是越长越好；sim navigation 最优较短，real manipulation 最优可更长。
- Proprioception：很关键，视觉 embedding 对精确 metric state 不够。

**我们的判断。**

它覆盖了“工程上怎么把 JEPA-WM 做好”。但它仍把 planning success 当最终指标，把 error propagation
当可优化对象，并没有正面处理下面这个反例：

```text
self-prediction / self-drift 更好，但 action ranking / control 更差。
```

我们的新意应该站在这篇之后：不是再扫 recipe，而是提出一个它没定义的 failure mode 和 metric。

### 2.3 C-JEPA：object-level masking 给了 counterfactual 训练信号

**方法。** 从 patch masking 改成 object latent masking。训练时遮住某个 object slot，要求 predictor
从周围 object 和动作/辅助变量推断它。这个 structured partial observability 相当于给模型制造
counterfactual-like query，逼它学 interaction-dependent dynamics。

**结论。**

- 反事实 reasoning 的 VQA 约 +20%。
- control 中只用 patch-based world model 约 1% 的 latent input features 就能做到可比 planning。

**我们的判断。**

它说明“把预测查询设计成 counterfactual / interaction-relevant”很有价值。
但它需要 object-centric representation。PushT/OGBench-Cube 可做，DMC/Atari/通用 FoV 不一定自然。
适合作为补强或 baseline，不适合直接替代 Gaussian JEPA 主线。

### 2.4 Invariant JEPA-WM：slow feature 是 visual shift 的一个已发表切口

**方法。** 在 DINO-WM 类 latent predictive objective 上加 bisimulation encoder，令 transition dynamics
相似的 state 更近，减少 background/distractor 这类 slow visual features 对 latent 的影响。

**结论。** 在背景变换和 distractor 下 robustness 更好，latent space 更小。

**我们的判断。**

它和我们 FoV diagnosis 对上：visual FoV shift 的本质不是 drift，而是 encoder shock。
不过这篇处理的是 nuisance visual invariance；我们的更宽：有些 task-critical variable 本身难预测，
不能被 predictive objective 牺牲。

## 3. 规划 / 控制类 world model 趋势

### 3.1 Train-test alignment 正在变成主线

*Closing the Train-Test Gap in World Models for Gradient-Based Planning* 的论点很直：

```text
world model 训练时做 next-state prediction，
测试时却用来反推 action sequence。
```

它通过 train-time data synthesis 改善 gradient-based planning，并在多个 object manipulation/navigation
任务上用约 10% time budget 达到或超过 CEM。

这和我们强相关：LeWM 也是 train/test mismatch，只不过 mismatch 更具体：

```text
训练目标：预测同一个 encoder 给出的 future latent
测试目标：用 terminal latent distance 排序 action candidates
```

因此我们的 PCG/ranking-sufficiency formulation 很容易和这条 literature 对齐。

### 3.2 Verification / event / predicate scoring 在起来

EV-WM 和 Foresight 都说明一件事：latent/video prediction 本身不够。

- EV-WM：WM rollout 后 decode structured event state，再按 task progress、semantic consistency、
  physical feasibility、uncertainty 打分。
- Foresight：用 action-conditioned WM latents 监控 long-horizon manipulation failure，只用 episode-level
  success/failure label，不依赖 dense failure annotation。

这两篇的共同点：

```text
把 WM latent 当作预测 substrate，
但最终信号是 task/event/failure，而不是 raw latent MSE。
```

这支持我们把 LeWM 后续方向从 “latent MSE drift” 转为 “decision-relevant latent geometry / action ranking”。

### 3.3 Discrete / codebook 方向没有死，但位置不一样

DCWM（ICLR 2025）说明 discrete codebook latent 在 continuous control 中可以有效，甚至和 TD-MPC2、
DreamerV3 竞争。这里要区分：

- DCWM 是 stochastic discrete codebook + model-based RL/MPC。
- LeWM 是 SIGReg Gaussian JEPA，边缘 latent 被正则成近 isotropic Gaussian。

所以“离散可以做”不等于“在 LeWM 的 `z` 上聚类/VQ 可以做”。对 LeWM 而言，离散更适合当 dynamics
diagnostic（regime in `f`），而不是 representation method。

## 4. Benchmark / platform 趋势：stable-worldmodel 的机会

### 4.1 新 benchmark 都在从 perceptual realism 转向 decision fidelity

| benchmark | 状态 | 测什么 | 对 stable-worldmodel 的启发 |
| --- | --- | --- | --- |
| [WorldModelBench](https://arxiv.org/abs/2502.20694) | arXiv 2025 | video generation 作为 world model 的 instruction following + physics adherence | 视觉生成好坏不等于物理/指令可用 |
| [WorldScore](https://arxiv.org/abs/2504.00983) | ICCV 2025 | controllability、quality、dynamics 的 unified world generation benchmark | 评测维度要分开，不要只给单一 FVD |
| [ACT-Bench](https://arxiv.org/abs/2412.05337) | arXiv 2024 | autonomous driving world model 的 action fidelity | action controllability 是 world model 必测项 |
| [WorldArena](https://arxiv.org/abs/2602.08971) | arXiv 2026 | perceptual + embodied functional utility；EWMScore | 功能效用和视觉质量要一起评 |
| [WorldBench](https://arxiv.org/abs/2601.21282) | arXiv 2026 | disentangled physics concepts / laws | 诊断必须把物理因素拆开 |
| [WorldMark](https://arxiv.org/abs/2604.21686) | arXiv 2026 | interactive video WM 的统一 action interface | action interface 标准化是公平比较前提 |
| [iWorld-Bench](https://arxiv.org/abs/2605.03941) | ICML 2026 | interactive WM 的 visual generation、trajectory following、memory | 交互能力需要统一 action generation framework |
| [WorldModelGym](https://www.reka.ai/news/worldmodelgym) | Reka, 2026-07-03 | decision-based fidelity；看模型预测对真实决策后果是否有用 | 这和我们的 PCG/action-rank 指标高度一致 |

### 4.2 stable-worldmodel 的独特位置

stable-worldmodel 本身已经主张：

- Lance-based data layer。
- MP4/HDF5/LeRobot conversion。
- LeWM、PLDM、DINO-WM、TD-MPC2 等 baseline。
- CEM/MPPI/gradient/PGD/Lagrangian 等 solvers。
- FoV environments：visual、geometry、physics factors of variation。

这使它非常适合做下面这种 contribution：

```text
Benchmarking world models by control-relevant latent fidelity:
  not just success rate,
  not just open-loop prediction MSE,
  but action-ranking preservation, controllability, FoV robustness, and wall-clock planning cost.
```

换句话说，stable-worldmodel 后续不要只做“更多环境/更多 baseline”。更有 paper 感的是：

```text
一个统一评测协议证明：
  self-prediction fidelity、visual fidelity、decision fidelity 可以系统性分离。
```

这正好承接你们 LeWM negative result。

## 5. 与我们当前实验结论的对账

### 5.1 已经被新论文覆盖的部分

这些不适合再作为主创新：

- 纯 multi-step rollout loss。
- stop-gradient separation 本身。
- 降低 open-loop latent MSE。
- 加速 autoregressive rollout。
- 普通 JEPA-WM architecture ablation。
- Frozen foundation encoder + predictor 的 recipe 搜索。

理由：

- Fast-LeWM 覆盖 prefix/multi-horizon/加速。
- What Drives 覆盖 JEPA-WM recipe。
- Dreamer/TD-MPC 类文献早已有 dynamics/representation stop-gradient 和 imagined rollout。

### 5.2 仍然开放、而且我们手里有证据的部分

真正有新意的现象是：

```text
lower self-referential latent drift can make planning much worse.
```

你们已有数据：

- single-step baseline：drift@8 0.315 / 0.251，planning 82% / 86%。
- pure multistep：drift@8 0.177 / 0.130，planning 22% / 40%。
- sgmulti β=1/2：planning 50/52%，仍低于 baseline，drift 还更差。

这不是普通 compounding error 问题。它说明：

```text
JEPA 的 target 由同一个 encoder 给出，
encoder + predictor 可以共同把 latent 改造成“好预测但不好控制”的坐标。
```

这个点在现有新论文里没有被充分形式化。

## 6. 建议的 LeWM 后续 paper framing

### 6.1 不建议的标题/主张

不建议：

- “Multi-step LeWorldModel”
- “Stable LeWM with Stop-Gradient Rollout”
- “Fast/Accurate Long-Horizon LeWM”
- “Regime-MoE LeWM”

这些要么被 Fast-LeWM / What Drives 覆盖，要么被我们自己的实验判死。

### 6.2 建议的主张

建议主线：

```text
Predictive-Control Gap in Gaussian JEPA World Models
```

或者：

```text
Control-Sufficient Gaussian JEPA for Latent World Model Planning
```

核心 claim：

1. Gaussian JEPA / SIGReg 防 representation collapse，但不防 **control-sufficiency erosion**。
2. Self-referential prediction error 可以被 encoder-predictor co-adaptation 刷低。
3. 低 latent drift 与好 planning 可以系统性反向。
4. 要优化的是 action-ranking / control-sufficient geometry，而不是 self-drift 本身。

### 6.3 方法对象

方法不要太像 auxiliary probe。建议分三层：

**Metric / certificate：PCG。**

定义两维 Pareto：

```text
P_k(phi): self-predictability / open-loop latent fidelity
C_H(phi): action-ranking agreement / planning sufficiency
```

PCG opens when:

```text
P_k improves, C_H degrades.
```

**Direction-level diagnosis：SER。**

对任务变量或 latent/probe direction 定义风险：

```text
SER_ctrl(v) = task sensitivity(v) * controllability(v) * predictive uncertainty(v)
```

高风险方向就是“难预测、可控、又影响目标排序”的变量，例如 PushT block angle/contact DOF。

**Training principle：control-sufficiency anchor。**

候选实现从保守到激进：

1. fixed planning-good encoder `phi0`，只训练 dynamics `f`，先做机制验证。
2. rank-preserving latent cost loss：候选 action pair 的 latent terminal ranking 对齐 true/task ranking。
3. SER-weighted predictive loss：高 SER 方向的 pred/cost 保真加权。
4. EMA/teacher anchor：把 fixed `phi0` 软化成 teacher，避免完全冻结。

### 6.4 实验清单

必须对打：

- LeWM。
- Fast-LeWM。
- DINO-WM。
- PLDM。
- JEPA-WM recipe baseline（尽量复现 What Drives 的强配置）。
- TD-MPC2 / DCWM 作为 RL/value-style 参照，至少在 state/continuous control 上给背景。

必须报告：

- Planning success。
- CEM/solver wall-clock。
- Open-loop latent MSE / self-drift。
- Action-ranking Kendall tau / pairwise AUC。
- Cost-rank top-k agreement / regret。
- State probes：position、angle、contact。
- FoV robustness：visual、geometry、physics。
- Ablation：fixed encoder、rank loss、SER weighting、EMA anchor。

关键判据：

```text
若方法只降 drift，不升 action-rank/planning：失败。
若 action-rank/planning 升，drift 不降：仍可能是成功，因为指标目标就是 decision fidelity。
若 fixed phi0 成功：说明问题主要是 moving encoder geometry。
若 fixed phi0 仍失败：说明 latent MSE 本身和 action ranking 不一致，主线转 rank/cost supervision。
```

## 7. 建议的 stable-worldmodel 后续 paper framing

如果走 stable-worldmodel 平台后续，不建议只写“我们支持更多格式/更多环境”。更好的 framing：

```text
stable-worldmodel: a decision-fidelity benchmark suite for latent world models
```

核心贡献：

1. **统一接口。** 同一 dataset / same candidate actions / same solver budget 下对比 LeWM、Fast-LeWM、
   DINO-WM、PLDM、TD-MPC2、DCWM、V-JEPA-2-AC。
2. **统一 FoV。** visual、geometry、physics、contact、object shape、camera shift。
3. **统一 decision metrics。** success、action-rank、latent drift、cost regret、wall-clock。
4. **负结果很值钱。** 证明 latent MSE / visual realism / success 在 ID 下可以互相脱钩。

这和 WorldModelGym / WorldArena / ACT-Bench 的趋势一致，但 stable-worldmodel 的优势是：

- 它不是只评 video/foundation WM。
- 它能直接跑 MPC 和 latent planning。
- 它能控制 FoV，做因果式诊断。

## 8. 最终路线建议

**LeWM 方法线：**

先跑 fixed-`phi0` 机制测试，不要急着做完整新 loss。

```text
baseline phi0 fixed
train f with one-step + multistep / rank-aware loss
evaluate drift + action-rank + planning
```

如果 fixed-`phi0` 下 drift 和 planning 同向，主方法就是 EMA/anchor + rank/SER。
如果 fixed-`phi0` 下仍反向，主方法直接转 ranking/cost sufficiency，不再追 drift。

**stable-worldmodel 平台线：**

把现有 diagnosis 融入一个公开 benchmark protocol：

```text
World model should be evaluated by decision fidelity under controlled variation,
not by self-prediction alone.
```

**论文定位：**

最强组合不是二选一，而是：

1. 用 stable-worldmodel 做复现平台和 benchmark。
2. 在 LeWM 上提出 predictive-control gap。
3. 用 SER/ranking method 修复。
4. 对打 Fast-LeWM 和 JEPA-WM recipe，证明我们不是“更低 drift”，而是“更可控制”。

## 9. 来源索引

- [LeWorldModel: Stable End-to-End Joint-Embedding Predictive Architecture from Pixels](https://arxiv.org/abs/2603.19312)
- [stable-worldmodel: A Platform for Reproducible World Modeling Research and Evaluation](https://arxiv.org/abs/2605.21800)
- [Fast LeWorldModel](https://arxiv.org/abs/2606.26217)
- [What Drives Success in Physical Planning with Joint-Embedding Predictive World Models?](https://arxiv.org/abs/2512.24497)
- [Causal-JEPA: Learning World Models through Object-Level Latent Masking](https://arxiv.org/abs/2602.11389)
- [Learning Invariant Visual Representations for Planning with Joint-Embedding Predictive World Models](https://arxiv.org/abs/2602.18639)
- [V-JEPA 2: Self-Supervised Video Models Enable Understanding, Prediction and Planning](https://arxiv.org/abs/2506.09985)
- [V-JEPA 2.1: Unlocking Dense Features in Video Self-Supervised Learning](https://arxiv.org/abs/2603.14482)
- [DINO-WM: World Models on Pre-trained Visual Features enable Zero-shot Planning](https://arxiv.org/abs/2411.04983)
- [Learning from Reward-Free Offline Data: A Case for Planning with Latent Dynamics Models](https://arxiv.org/abs/2502.14819)
- [TD-MPC2: Scalable, Robust World Models for Continuous Control](https://arxiv.org/abs/2310.16828)
- [Discrete Codebook World Models for Continuous Control](https://arxiv.org/abs/2503.00653)
- [Closing the Train-Test Gap in World Models for Gradient-Based Planning](https://arxiv.org/abs/2512.09929)
- [WorldPlanner: Monte Carlo Tree Search and MPC with Action-Conditioned Visual World Models](https://arxiv.org/abs/2511.03077)
- [EV-WM: Event-Verified World Models for Long-Horizon Robotic Manipulation](https://arxiv.org/abs/2606.13053)
- [Foresight: Failure Detection for Long-Horizon Robotic Manipulation with Action-Conditioned World Model Latents](https://arxiv.org/abs/2606.23085)
- [DiWA: Diffusion Policy Adaptation with World Models](https://arxiv.org/abs/2508.03645)
- [World4RL: Diffusion World Models for Policy Refinement with Reinforcement Learning for Robotic Manipulation](https://arxiv.org/abs/2509.19080)
- [ACT-Bench: Towards Action Controllable World Models for Autonomous Driving](https://arxiv.org/abs/2412.05337)
- [WorldModelBench: Judging Video Generation Models As World Models](https://arxiv.org/abs/2502.20694)
- [WorldScore: A Unified Evaluation Benchmark for World Generation](https://arxiv.org/abs/2504.00983)
- [WorldArena: A Unified Benchmark for Evaluating Perception and Functional Utility of Embodied World Models](https://arxiv.org/abs/2602.08971)
- [WorldBench: Disambiguating Physics for Diagnostic Evaluation of World Models](https://arxiv.org/abs/2601.21282)
- [WorldMark: A Unified Benchmark Suite for Interactive Video World Models](https://arxiv.org/abs/2604.21686)
- [iWorld-Bench: A Benchmark for Interactive World Models with a Unified Action Generation Framework](https://arxiv.org/abs/2605.03941)
- [WorldModelGym: a decision-based fidelity benchmark for world models](https://www.reka.ai/news/worldmodelgym)

---

## 10. Round-3 审稿补遗(2026-07-04 晚):本调研漏掉的直接威胁

3 个独立审稿 agent 实际检索后新发现、且必须纳入对打的工作:

| paper | 为什么危险 | 我们的差异化生死线 |
| --- | --- | --- |
| [RC-aux: Predictive but Not Plannable](https://arxiv.org/pdf/2605.07278) | **同 base model(LeWM)+ 同 headline gap**("预测准但 latent 不可规划"),用 reachability 监督修 | 无收缩理论/接触分析/证书;必须引用并正面击败 |
| [TRM: Beyond Euclidean Proximity](https://arxiv.org/html/2605.22164v1) | **同 base model**,post-hoc 轨迹可达 terminal metric,LeWM TwoRoom 7%→97% | "latent L2 不是对的决策度量"正在被挖;时间窗收紧 |
| [Invariant JEPA-WM](https://arxiv.org/abs/2602.18639) | **最危险**:JEPA WM 内联合训练 reward-free bisimulation encoder(1-step transition 相似) | 我们的 H-step 开环分歧 target 携带复合增益信息,1-step 对 G_K 梯度盲(Thm B 可证);需 head-to-head ablation |
| [NCDS](https://openreview.net/forum?id=iAYIRHOYy8) (ICLR'24) | 已在学出的 latent 空间用微分同胚不变性做收缩 | 杀死"首次 latent 收缩"措辞;我们的新意是反转:hybrid 系统分岔处不可收缩,均匀收缩是错的 |
| [MICo](https://arxiv.org/pdf/2106.08229) + [Robust Bisim](https://arxiv.org/abs/2110.14096) (NeurIPS'21) | reward=0 的 MICo ≈ 我们的度量项;**坍缩病理已被证明并修复** | 必须继承其修复,不能只用 BN |
| [Asadi et al.](https://proceedings.mlr.press/v80/asadi18a.html) (ICML'18) | Lipschitz 控制复合误差已是 established practice | 单独的收缩惩罚项无新意;新意在三难+margin |
| [When Does LeJEPA Learn a World Model?](https://arxiv.org/abs/2605.26379) | 证明 Gaussian marginal 唯一给出线性可辨识性并支持最优 latent planning | **反对丢弃 SIGReg 的理论依据**;主形态保留 SIGReg |

另:本调研 §0.3/§5.2 引用的"低 drift 反而坏 planning(82→22)"已于 2026-07-02 被
Gate 0 证伪(评测伪影);幸存的 gap 是"self-referential 指标跨模型不可识别
(ρ=0.26)vs refit-D* 证书(ρ=0.94)",见 phase_diagram_results.md。
