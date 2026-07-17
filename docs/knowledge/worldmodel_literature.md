# Latent World Model / LeWM 文献地图（持续维护）

> 本文是仓库中**唯一的文献综述与竞品地图**。初始系统检索完成于
> 2026-07-04，之后发现的论文继续按主题合并到本文，不再建立“旧 survey /
> 新补充”文档。实验、理论和方法文档只保留必要引用与本文链接。
>
> 范围：与 LeWM / stable-worldmodel 最相关的 reconstruction-free latent
> world model、JEPA world model、MPC/planning、robotics world model 和
> benchmark/platform 工作。结论按“它占了什么、留下什么、如何影响当前
> paper seed”来读，而不是普通读书笔记。
>
> **阅读约定：**早期判断会保留用于追踪思路演化，但被后续 Gate 或新文献
> 推翻的结论必须在原处或后文显式标记。当前项目的 claim 边界以本文最后的
> “当前项目对账”以及实验主账本的最新判决为准。

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

3. **“prediction error 不等于 control”已经不是空位。**
   早期记录的 multistep 22/40% planning 是 cold-start 评测伪影，不能再作为
   predictive-control gap 证据；RC-aux、TRM、Control Theory of Predictability、
   Operator-on-F 等工作也已从 reachability、cost discrepancy 和 operator fidelity
   多侧正面占据这一问题。

4. **stable-worldmodel 的后续价值更像 benchmark/evaluation substrate。**
   近期 benchmark 明显转向“decision-relevant fidelity”：WorldArena、WorldModelGym、ACT-Bench、
   WorldMark、iWorld-Bench 都在说视觉好看不等于可决策。stable-worldmodel 可以抓住这个趋势：
   把“latent error、action ranking、FoV/OOD、planning success、wall-clock”统一成可复现评测。

5. **当前项目仍有区分度的是 horizon-induced representation co-adaptation。**
   现有证据表明 prediction horizon 主要改变 encoder-side finite-horizon geometry，
   对 state error 与 action signal 做不对称分配，并且不能被 teacher forcing、
   predictor-only training 或独立 gain regularization 拆解。但这目前是一组强机制
   结果，不等于已经找到足够大的 paper framing 或成功方法。

6. **Temporal Straightening（ICML 2026）是当前在“latent geometry 改善 planning”叙事上最接近的工作。**
   它在 reconstruction-free one-step JEPA 上显式惩罚真实轨迹的 latent curvature，并把
   straightness 与线性系统的 action Hessian / controllability Gramian 联系起来。因此
   “更好的 representation geometry 让规划更易优化”已经被占据；当前项目必须把差异落在
   **真实轨迹的切向几何**与**递归想象误差的横向传播几何**之间，并正面对打
   `K1 + curvature` 与 multistep co-adaptation。

## 1. 最相关论文速览

| paper | 状态 | 核心做法 | 主要结论 | 对我们的影响 |
| --- | --- | --- | --- | --- |
| [LeWorldModel](https://arxiv.org/abs/2603.19312) | arXiv 2026-03，v3 2026-06 | end-to-end JEPA；next-embedding MSE + SIGReg；无 EMA/stop-grad/pretrained encoder | 15M 参数；loss 超参从 PLDM 类多项降到 1；planning up to 48x faster than DINO-WM；PushT 等任务竞争力强 | 我们的基线对象。它的弱点是 one-step teacher forcing + self-targeted latent |
| [stable-worldmodel](https://arxiv.org/abs/2605.21800) | arXiv 2026-05 | 统一数据层、baseline、solver、FoV 环境、评测协议 | 解决 world model 研究代码碎片化、数据加载慢、泛化基准不足 | 可以成为后续实验平台和 benchmark contribution |
| [Fast-LeWorldModel](https://arxiv.org/abs/2606.26217) | arXiv 2026-06-24 | action-prefix encoder + parallel latent predictor；不滚中间每一步；可加 self-consistency scoring | 平均成功率 LeWM 85.8 -> 90.5，+SC 到 92.0；dynamics module 31.4s -> 8.0s；CEM 54.4s -> 28.3s | 把“加速 + 降长程 latent error”占掉了。我们不能再把多步/前缀预测作为主创新 |
| [What Drives Success in Physical Planning with JEPA-WMs?](https://arxiv.org/abs/2512.24497) | TMLR accepted，v3 2026-05 | 系统 ablation：encoder、AdaLN/RoPE predictor、rollout steps、context、proprioception、planner | 推荐 recipe：CEM/L2；sim nav 较短 rollout/context，real manipulation 更深 predictor、更长 rollout/context；提出比 DINO-WM/V-JEPA-2-AC 更强组合 | 必须作为强 related work。它覆盖“JEPA-WM 工程 recipe”，但未解决 self-drift 与 planning 可反向 |
| [Causal-JEPA / C-JEPA](https://arxiv.org/abs/2602.11389) | ICML 2026 accepted | object-centric latent masking；mask object slots，让 masked object 由上下文推断 | counterfactual reasoning 约 +20%；control 中用 patch-based WM 1% latent features 达到可比 planning | 给我们一个“结构化部分可观测 / counterfactual query”的方向，但它偏 object-centric，不是 Gaussian JEPA sufficiency |
| [Learning Invariant Visual Representations for Planning with JEPA-WMs](https://arxiv.org/abs/2602.18639) | arXiv 2026-02 | 在 DINO-WM 类 objective 外加 bisimulation encoder，压 slow visual features / distractors | background/distractor robustness 改善；latent 维度可小到 DINO-WM 的 1/10 | 与我们的 FoV shift 诊断强相关：慢特征不变性和 control relevance 是同一大问题 |
| [Temporal Straightening for Latent Planning](https://arxiv.org/abs/2603.12231) | ICML 2026 poster，v2 camera-ready | one-step reconstruction-free JEPA + latent velocity cosine curvature；主配置用 128-D learnable aggregation head 计算 curvature | 多个设置下改善 GD/CEM；在线性系统中连接 action Hessian 和 controllability Gramian，但 long-horizon PushT 结果并不单调 | “geometry improves planning”已被占据；但它约束真实轨迹切向，不直接控制 recursive error/Jacobian product、non-normal gain 或 imagination frontier |
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

### 2.5 Temporal Straightening：轨迹切向几何与递归误差几何的最近邻

这篇已作为 ICML 2026 poster 接收。它的方法非常简单，但问题选得准：
即使 one-step prediction 准，弯曲的 latent trajectory 仍会让 action optimization
条件很差；因此直接把可行轨迹在 latent 中“拉直”。

**方法与实现。** 训练主体仍是 reconstruction-free joint encoder-predictor JEPA：

```text
L_pred = || z_hat[t+1] - sg(z[t+1]) ||²
v[t]   = z[t+1] - z[t]
L_curv = 1 - cos(v[t], v[t+1])
L      = L_pred + λ L_curv
```

decoder 不参与 world-model training，只在训练后冻结 encoder、单独拟合，用于可视化
latent 是否还保留可重建信息。官方实现还有几个容易读错的细节：

- prediction training 的 `num_pred=1`，即只做 one-step target；没有把 predictor
  自回归展开 K 步再反传，因此没有 self-composition gradient。
- 主 spatial 配置把 frozen DINOv2 的 `196×384` patch feature 经 trainable CNN
  projector 变为 `196×8` latent，`L_pred` 仍作用在完整 spatial latent 上。
- 主 curvature 配置 `aggcos` 另用一个 learnable MLP，把每帧 latent 聚合成
  128-D 向量 `g[t]=h_phi(z[t])`，再计算
  `g[t+1]-g[t]` 与 `g[t+2]-g[t+1]` 的 cosine。也就是说，**相似度主实验确实经过
  MLP aggregation head，但它不是把 predictor 使用的 latent 全局降维后再预测**；
  patch-direct、mean、flatten 和 learnable aggregation 都作为 ablation。正文写法近似
  `h(v[t])`，代码实际是先聚合每帧再做差，这一点复现时应以代码为准。

**第四节到底强在哪里。** 在线性 latent dynamics
`z[t+1]=Az[t]+Ba[t]` 下，论文定义 `||A-I||₂≤ε` 为近似 straight，
把 horizon action Jacobian 写成由 `A^kB` 组成的矩阵，进而得到
action loss Hessian `H=2JᵀJ` 与有限时域 controllability Gramian 的联系，并给出
`A≈I` 时 condition number 的界。这为“straight trajectory 为何更适合 GD”
补上了一条干净的 control-theoretic 桥。

但它不是一条无条件的 nonlinear guarantee：

- 从 cosine curvature 推到 `A-I` 小，需要速度模长近似稳定、动作变化平滑，
  并且训练轨迹覆盖相关方向。
- 主实验可在单独的 aggregation space 计算 curvature，而 theorem 讨论的是
  planner/predictor latent；两者之间还缺少严格等价关系。
- nonlinear state-dependent Jacobian products、高阶项和 off-trajectory perturbation
  被留作 future work。
- cosine 对尺度不敏感。例如 `A=2I`、动作增量为零时，连续 velocity 完全同向，
  curvature 可为零，但 K-step error 会按 `2^K` 放大。反过来，旋转动力学可以
  轨迹很弯，却保持误差范数稳定。

因此第四节是很好的**解释桥和审稿加分项**，但不像是单靠一个新 theorem 撑起接收。
更完整的原因是：问题与结论一句话就能讲清、regularizer 即插即用、PointMaze/PushT
上有明显提升，并同时给出多 encoder、global/spatial latent、GD/CEM、open-loop/MPC、
curvature/PCA/距离热图/loss landscape、长 horizon 和 aggregation ablation。
理论把这套实验从“一个好用的正则项”升级成完整故事；v2 camera-ready 新增的
Gramian 形式化也说明它更像关键支撑，而不是唯一贡献。

经验结论也要保留边界：long-horizon PushT 并非随 curvature 单调改善，部分设置下
显式 curvature 反而更差；相应实验还在 aggregation space 增加 goal cost。因此它证明了
straightening 是有用 planning bias，但还没有解决一般的 long-horizon drift、候选排序反转
或可信想象 frontier。

**与当前项目的真正分界。**

```text
Temporal Straightening：约束真实可行轨迹的 tangent / velocity geometry，
                        目标是让 latent goal landscape 更适合求解。

当前项目：            约束或诊断递归想象偏离真实轨迹后的 error transport，
                        研究 nonlinear Jacobian product、共适应和可信想象视距。
```

所以不能再泛称“我们首次发现 representation geometry 影响 planning”或“首次改善
long-horizon conditioning”。更稳的定位是：

> They straighten feasible trajectories; we stabilize and certify
> counterfactual recursive imagination.

**必须加入的 head-to-head baseline。** 在完全相同的 LeWM/SIGReg protocol 下至少比较：

1. `K1`；
2. `K1 + L_curv`；
3. `K5`；
4. `CritWM`；
5. 可选 `K5 + L_curv`，检验两种几何是否互补。

同时报告 trajectory cosine、`rate(K)` / Jacobian-product gain、action gain、
refit-`D*`、goal 25/40/60 的 candidate rank inversion 和 planning success。
决定性的结果不是谁的单一 loss 更低，而是验证 **curvature 与 recursive error
amplification 是否是两个独立轴**：若 curvature 变好但 error gain/frontier 不变，
正好支持我们的分界；若二者同步改善，则需进一步做 frozen-encoder、frozen-predictor
与 aggregation-head ablation 归因。

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

早期“更低 drift 反而让 planning 从 82/86% 掉到 22/40%”已被 Gate 0
判定为 cold-start 评测伪影，不再使用。当前真正站住的证据是：

- matched-history 下 K=5 planning 高于 K=1，且容量越小优势越大；
- frozen/refit 证明主要收益住在 encoder，不是 predictor capacity；
- teacher-forced one-step MSE 基本相同，open-loop composition 相差约 2 倍；
- `rate(K)` 随 K 单调且跨训练种子稳定，但不是充分的 planning predictor；
- state-error propagation 被强烈压制，action discrimination 基本保留；
- TF-K5、sg-K5 与 EchoReg 的失败共同说明 representation、transition、
  rollout accuracy 与 gain pressure 不能任意解耦。

因此仍开放的窄问题是：

```text
在同一 SIGReg marginal constraint 下，
prediction horizon 如何通过 encoder-predictor co-adaptation
选择 finite-horizon representation 与局部动力学？
```

这一问题尚未被直接覆盖，但仅靠它本身还不足以保证旗舰级意义或方法贡献。

## 6. 建议的 LeWM 后续 paper framing

> 本节保留 2026-07-04 当时的候选 framing，供追踪思路演化；它不是当前推荐。
> 当前 claim 边界以 §11.5 为准，具体实验判决以
> [lewm_gaussian_dynamics_direction.md](lewm_gaussian_dynamics_direction.md)
> 的后续章节为准。

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
- [Temporal Straightening for Latent Planning](https://arxiv.org/abs/2603.12231)
- [Temporal Straightening official implementation](https://github.com/agentic-learning-ai-lab/temporal-straightening)
- [Temporal Straightening at ICML 2026 (OpenReview)](https://openreview.net/forum?id=Ik1mKtUYlZ)
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

## 10. 直接近邻与 novelty 生死线

以下工作最初在 2026-07-04 Round-3 审稿检索中定位，现作为持续维护的
直接近邻表；后续发现同类工作直接补入，不再单列“补遗”。

| paper | 为什么危险 | 我们的差异化生死线 |
| --- | --- | --- |
| [RC-aux: Predictive but Not Plannable](https://arxiv.org/pdf/2605.07278) | **同 base model(LeWM)+ 同 headline gap**("预测准但 latent 不可规划"),用 reachability 监督修 | 无收缩理论/接触分析/证书;必须引用并正面击败 |
| [TRM: Beyond Euclidean Proximity](https://arxiv.org/html/2605.22164v1) | **同 base model**,post-hoc 轨迹可达 terminal metric,LeWM TwoRoom 7%→97% | "latent L2 不是对的决策度量"正在被挖;时间窗收紧 |
| [Invariant JEPA-WM](https://arxiv.org/abs/2602.18639) | **最危险**:JEPA WM 内联合训练 reward-free bisimulation encoder(1-step transition 相似) | 我们的 H-step 开环分歧 target 携带复合增益信息,1-step 对 G_K 梯度盲(Thm B 可证);需 head-to-head ablation |
| [Temporal Straightening](https://arxiv.org/abs/2603.12231) (ICML'26) | **叙事最近邻**：同为 reconstruction-free joint encoder-predictor，以显式 latent geometry 改善 planning，并已连接 linear action Hessian / Gramian | 它管真实轨迹 tangent curvature，我们管 off-trajectory recursive error transport；必须用 `K1+curv` 对打，不能只靠措辞区分 |
| [NCDS](https://openreview.net/forum?id=iAYIRHOYy8) (ICLR'24) | 已在学出的 latent 空间用微分同胚不变性做收缩 | 杀死"首次 latent 收缩"措辞;我们的新意是反转:hybrid 系统分岔处不可收缩,均匀收缩是错的 |
| [MICo](https://arxiv.org/pdf/2106.08229) + [Robust Bisim](https://arxiv.org/abs/2110.14096) (NeurIPS'21) | reward=0 的 MICo ≈ 我们的度量项;**坍缩病理已被证明并修复** | 必须继承其修复,不能只用 BN |
| [Asadi et al.](https://proceedings.mlr.press/v80/asadi18a.html) (ICML'18) | Lipschitz 控制复合误差已是 established practice | 单独的收缩惩罚项无新意;新意在三难+margin |
| [When Does LeJEPA Learn a World Model?](https://arxiv.org/abs/2605.26379) | 证明 Gaussian marginal 唯一给出线性可辨识性并支持最优 latent planning | **反对丢弃 SIGReg 的理论依据**;主形态保留 SIGReg |

另:早期版本 §0.3/§5.2 引用的"低 drift 反而坏 planning(82→22)"已于
2026-07-02 被 Gate 0 证伪(评测伪影)；本文现已在原处更正。幸存的 gap 是
"self-referential 指标跨模型不可识别(ρ=0.26)vs refit-D* 证书(ρ=0.94)"，
见 [contact_contraction_main.md](contact_contraction_main.md)。

---

## 11. LeWM / SIGReg 动力学表示重点对账

这一节承接前面的广域调研，专门记录和当前 LeWM Gaussian dynamics 实验
直接相关的工作。它不是另一份 survey；后续论文继续按主题补进本节或前面的
对应主题。

### 11.1 基础与直接近邻

| paper | 它已经占据的部分 | 对当前项目仍开放的部分 |
| --- | --- | --- |
| [LeWorldModel](https://arxiv.org/abs/2603.19312) | end-to-end JEPA world model；next-embedding MSE + SIGReg；latent L2 + CEM | prediction horizon 为什么主要改变 encoder，以及同一 Gaussian marginal 下的局部动力学几何 |
| [When Does LeJEPA Learn a World Model?](https://arxiv.org/abs/2605.26379) | Gaussian marginal、线性可辨识性和 latent planning 的理论地基 | action-conditioned transition、operator product、非线性局部增益与 encoder-predictor 共适应 |
| [A Generalization Theory for JEPA-Based World Models](https://arxiv.org/abs/2606.27014) | action-conditioned co-occurrence 低秩分解、pretraining error 与 planning regret、维度权衡 | 不同 horizon 在同一边缘约束下如何选择 representation |
| [Delta-JEPA](https://arxiv.org/abs/2606.31232) | 用 latent displacement 解码 action，强化一阶 action sensitivity | finite-horizon composition 和 error/action channel 的不对称变化 |
| [Fast-LeWorldModel](https://arxiv.org/abs/2606.26217) | action-prefix、多 horizon 并行预测、避免 autoregressive compounding、加速 CEM | 它主要绕开自复合；没有解释自复合目标如何重写 encoder geometry |
| [Sub-JEPA](https://arxiv.org/abs/2605.09241) | 随机低维子空间 Gaussian regularization | 给定 marginal regularizer 后，horizon 如何使用剩余自由度 |
| [Temporal Straightening](https://arxiv.org/abs/2603.12231) | reconstruction-free one-step JEPA + 显式 trajectory curvature；主实现经 128-D MLP aggregation head 算 velocity cosine；线性 action Hessian / Gramian 分析 | SIGReg Gaussian marginal、self-composition gradient、nonlinear Jacobian-product error dynamics、encoder-predictor co-adaptation 与 imagination frontier |
| [Predictive Objectives Discard Exogenous Control-Relevant Features](https://arxiv.org/abs/2606.30068) | predictive objective 可能丢失不可预测但控制相关变量 | 当前实验中的 angle 下降、agent/action 信号增强并不是简单的“控制信息侵蚀” |
| [AdaJEPA](https://arxiv.org/abs/2606.32026) | MPC 期间的 test-time self-supervised adaptation | prediction update 在部署时究竟把 latent geometry 推向哪里 |
| [ScratchWorld](https://arxiv.org/abs/2606.31689) / [WorldModelGym](https://www.reka.ai/news/worldmodelgym) | decision fidelity / executable consequence 评测趋势 | stable-worldmodel 可提供受控、可规划的机制诊断协议 |
| [World-Model Collapse as a Phase Transition](https://arxiv.org/abs/2606.31399) | horizon/state load 附近的 phase-transition 式 failure | 当前容量 × horizon 相图能否给出 representation-side 机制 |

### 11.2 当前最直接的竞争与补强

| paper | 核心说法 | 对当前 claim 的约束 |
| --- | --- | --- |
| [A Control Theory of Predictability in Latent World Models](https://arxiv.org/abs/2607.10362) | planner committed plans 上的 predicted-vs-true cost discrepancy；on-manifold residual、off-manifold divergence 与非正规放大 | 不能再声称首次提出 prediction-control gap、non-normal amplification 或 planner-facing certificate；差异必须落在 horizon 如何改变 representation 本身 |
| [Operator-on-F complements value-equivalence](https://arxiv.org/abs/2607.04464) | 用 model predictor 比较 k-step pushforward，在可观测函数集上诊断 fidelity | 占据通用 operator diagnostic；当前项目若使用 Jacobian/operator，必须解释训练选择压力而非只做 checkpoint metric |
| [Adaptive Compute in Latent World Models](https://arxiv.org/abs/2607.10203) | predictor depth 存在 help/hurt/flat 三种 regime，浅深误差比可预示 CEM depth 效果 | “更深想象不一定更好”不是独立新意；当前深度悬崖只能作为具体机制证据 |
| [Mind the Gap: Promises and Pitfalls of Hierarchical Planning in LeWorldModel](https://arxiv.org/abs/2607.12547) | high-level subgoal generation 与 low-level search distribution mismatch；data-supported macro actions 修复部分 long-horizon failure | 简单 hierarchy 已被占据；仍可追问 shared latent point subgoal 是否是错误抽象 |
| [MoP-JEPA](https://arxiv.org/abs/2607.05238) | 单 regressor 在 stochastic transition 下预测无效 conditional mean；K-head successor set 改善 graph planning | 多模态 successor / mixture predictor 已被占据；不能把“多个未来”本身当主创新 |
| [The SIGReg Objective as Variational Free Energy](https://arxiv.org/abs/2607.13612) | 在特定假设下解释 SIGReg 的 information-bottleneck 与 latent goal-cost 地位，并指出 state-epistemic 缺口 | 补强 SIGReg 的规范性，但没有解释 action-conditioned finite-horizon operator 或 horizon-dependent representation |
| [Qantara](https://arxiv.org/abs/2607.04978) | 同一 JEPA checkpoint 支持 latent planning、BC action sampling、inverse dynamics | 若做方法 paper，需要作为强工程 baseline 或至少正面讨论 |
| [Grounding Spatial Relations in a Compact World Model](https://arxiv.org/abs/2607.06925) | goal-conditioned dynamics 会产生 instruction leakage；goal 应只进入 planner/read path | 支持 goal/cost 不应污染 dynamics representation 的边界 |
| [Write-Protected Discrete Bottlenecks](https://arxiv.org/abs/2607.08312) | 端到端语言梯度会让 discrete symbols collapse；使用 detach、外部 memory 与 DP-Means | 离散符号更适合作为受保护接口，而非当前 LeWM 的端到端 waypoint |
| [Learning Task-Sufficient World Models](https://arxiv.org/abs/2607.04409) | active exploration + structured modeling 学 task-specific minimal sufficient latent | 支持 task sufficiency 视角，但离 SIGReg horizon/co-adaptation 机制较远 |

### 11.3 Decision fidelity、认证与 planner-facing metric

| paper | 已覆盖内容 | 对当前项目的含义 |
| --- | --- | --- |
| [RC-aux](https://arxiv.org/abs/2605.07278) | multi-horizon open-loop + reachability supervision + planner gate | reachability-aware training 不是空位 |
| [TRM](https://arxiv.org/abs/2605.22164) | 固定 world model 上的 terminal reachability metric | candidate ranking / latent metric repair 不是空位 |
| [Predicting Closed-Loop Performance of Latent World Models](https://arxiv.org/abs/2607.01736) | validation loss 和 multi-step RMSE 不能可靠选出 closed-loop checkpoint；ROF/CROF | open-loop error 不够已经有直接证据 |
| [Certified World Models as Sensing Clocks](https://arxiv.org/abs/2607.01537) | 把 validity horizon 用作 re-sensing deadline | 不能泛泛声称首次提出“认证想象视距” |
| [The Rank-One Corner](https://arxiv.org/abs/2607.06640) | scalar value equivalence 只保留 task closure 的低维投影 | 支持多维控制充分性问题，但不是当前项目独有发现 |
| [Imagined Rollouts are Kinematic, Not Dynamic](https://arxiv.org/abs/2607.05966) | imagined kinematic consistency 对真实 dynamics regime failure 不敏感 | 支持接触/动力学诊断，但不直接解释 LeWM representation co-adaptation |
| [Validate the Dream Before You Trust Its Verdict](https://arxiv.org/abs/2607.07196) | world model 作为 simulator/test oracle 前需要 admissibility ladder | 可作为 benchmark/assurance 背景 |
| [Reduced-Order Models: The Mother of World Models](https://arxiv.org/abs/2607.03198) | 把 world model 放回 model reduction、control、verification 与 error-bound 传统 | 为 coarse-graining、memory 和 verification 提供历史背景 |

### 11.4 时间尺度、抽象与记忆

| paper | 核心对象 | 对潜在 scale-dependent state 问题的约束 |
| --- | --- | --- |
| [Hierarchical Planning with Latent World Models](https://arxiv.org/abs/2604.03208) | 多时间尺度 dynamics，但各层共享 latent，高层预测作为低层 point subgoal | 已占据 multiscale hierarchy；其 shared-latent 假设和 level-specific abstraction limitation 是仍可追问之处 |
| [Multi Time Scale World Models](https://openreview.net/forum?id=fY7dShbtmo) | 概率 state-space model 中的 slow task state 与 fast latent state | 多时间尺度 latent 本身不是新意；必须区分 temporal decomposition 与 horizon-specific quotient |
| [Learning Markov State Abstractions for Deep RL](https://arxiv.org/abs/2106.04379) | 学习保留 Markov property 的 state abstraction | 为 lumpability/Markov abstraction 提供直接前史 |
| [Model Reduction with Memory and the Machine Learning of Dynamical Systems](https://arxiv.org/abs/1808.04258) | Mori-Zwanzig 视角下，coarse-graining 会诱导 memory 与 noise | 支持“被删除的 state information 会以 memory/stochasticity 回来”，但这一原则不是 world-model 新发现 |
| [World Models as Group Actions](https://arxiv.org/abs/2605.24578) | 用 identity、inverse、composition consistency 把 action-conditioned WM 形式化为 group action | 占据纯粹的 action algebra / composition headline |
| [Back to Parsimonious Latents / TC-WM](https://arxiv.org/abs/2605.25620) | 从 foundation embedding 学 compact task-centric latent，并给 task-sufficiency 理论 | compact/task-sufficient representation 已有直接工作；差异需要落在 temporal scale 或 coarse-to-fine semantics |

### 11.5 当前项目的 claim 边界

截至当前实验与检索，不能把下面这些作为“首次”主张：

- prediction error 与 control 脱钩；
- latent L2 不等于 reachability；
- operator/pushforward diagnostic；
- non-normal amplification 或普通 contraction；
- validity/imagination horizon；
- 多 horizon、prefix prediction、hierarchy 或 multimodal successor。

当前仍由本项目直接证据支持、但尚未自动构成旗舰贡献的是：

1. 在同一 SIGReg LeWM family 内，prediction horizon 主要通过 encoder 改变
   finite-horizon composition geometry，而非只增强 predictor。
2. `rate(K)` 随 K 单调且跨训练种子稳定，但它是机制量，不是充分的 planning
   predictor。
3. multi-step objective 强烈压制 state-error propagation，同时保留 action
   discrimination；这种不对称需要 representation 与 transition 共适应。
4. teacher-forced、predictor-only 和显式 gain regularization 分别打开不同的
   失败/作弊通道；完整 open-loop objective 是目前唯一成功的耦合实现。

因此，后续新 idea 必须解释或利用上述整组事实，而不能只把其中一个 metric、
regularizer 或 planner patch 重新包装成主问题。
