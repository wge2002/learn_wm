# Planner-equilibrium / adaptive tail validity：复核证据底稿（2026-07-22）

> 本文是 `research_tail_validity_collision_20260722.md` 的证据账本，不是 ≤2500 字的回传正文，也不作 GO/NO-GO。检索截止 2026-07-22；方法与实验判断只采用论文原文、arXiv/PMLR/OpenReview 官方页面。摘要级定向扫描不能证明“全网不存在”，所以统一写成“本次精确词检索未检出”。

## 0. 复核口径

- 五个被审要素：①把 learned-vs-true 的 low-cost-tail 误排本身作为对象；②刻画误差如何经 CEM elite/refit 的 proposal-update flow 累积；③有效性沿 planner 实际 query path 自适应；④形式化 planner-model 的均衡/固定点；⑤落在 latent WM + CEM/MPPI。
- 记号：`✓`=直接覆盖；`△`=结构相邻但对象/反馈/时序不一致；`—`=未覆盖。特别区分“执行后更新 WM”“对抗采数”“每轮 CEM 配权”“固定全序列 verifier”“按部署残差做 conformal 更新”，它们不能都简称 adaptive。
- 第一版确实漏了五个关键近邻：AWM 的显式 min-max、GeoWorld 的 planner-aware JEPA+CEM、WM-VAE 的全序列 novelty cost、Who Moved My Distribution 的闭环固定点，以及与本仓库语境最贴近的 IMWM proposal/rank 诊断；本次均已补入。

## 1. 覆盖矩阵

| 工作 | ① | ② | ③ | ④ | ⑤ | 严格边界 |
|---|---:|---:|---:|---:|---:|---|
| AdaJEPA 2606.32026 | — | — | △ | — | ✓ | 先执行、拿到真实 next state 后才更新；不看同一轮 CEM 内部候选 |
| PROWL 2605.18803 | — | — | △ | △ | — | adversarial policy 挖错例并离线式交替微调；无 CEM cost/rank |
| AWM 2607.10630 | — | — | △ | ✓* | — | `*` planner–world-model min-max 是直接博弈，但 WM 扮演交通 adversary，不是 cost-validity evaluator |
| Temporal Straightening 2603.12231v2 | — | — | — | — | ✓ | 离线 representation/cost geometry；无在线校准 |
| Navigable WM OpenReview fnLHZcXZUW | — | — | △ | — | — | 为梯度可导航性静态塑能量；arXiv 未确认 |
| PaD 2512.17846 | — | — | △ | — | — | 训练对齐梯度 descent dynamics，不是 CEM query/refit |
| GeoWorld 2602.23058 | — | — | △ | — | ✓ | predictor 经几何目标训练，推理才跑冻结 CEM；未反传 elite/refit |
| Performative prediction/RL | — | — | — | ✓ | — | 均衡在 decision/policy 与外部数据生成环境之间 |
| Who Moved My Distribution 2511.11567 | — | — | ✓* | ✓* | — | `*` 数据来自已部署 controller，固定点为 CP/controller–reactive agents，不是内部 WM cost |
| FCP-MPC 2607.00776 | — | — | ✓* | — | △ | `*` 延迟真实残差更新安全场标量；MPPI 有、latent WM 无 |
| ACID 2607.02403 | △ | — | △ | — | ✓ | 全序列、逐 CEM-round 重排最接近；自适应只等化两项 cost 的候选集 spread |
| WM-VAE 2508.06096 | △ | — | — | — | ✓ | 全序列 novelty verifier 固定；不从 planner 反馈学习 |
| IMWM 2606.01626 | △ | △ | — | — | ✓ | 诊断 rank/coverage 与 refit，改 proposal/cost；组件和 gate 在 query 前冻结 |
| Imperfect WMs 2605.15960 | △ | — | — | — | — | 形式化 model/true policy preference reversal，但无 low-tail/CEM |
| Verified WM Still Loses 2607.14169 | △ | — | — | — | — | 关键少数转移导致 play failure；对象是 code WM/classical search |

`△` 不表示论文自称覆盖该要素，只表示它提供了可碰撞的机制或诊断。

## 2. 线 1：adaptive / adversarial world models

### AdaJEPA — arXiv:2606.32026

- 原文：[arXiv abstract](https://arxiv.org/abs/2606.32026)，[full HTML](https://arxiv.org/html/2606.32026)。
- §3.1：JEPA latent rollout 的 goal cost 可由 GD 或 CEM 优化。§3.2：plan → execute first action/chunk → observe transition → self-supervised prediction-loss update → replan；buffer 可取 recent-N 或 hard-N，更新 encoder/predictor 的选定层。
- §4：PushT/PushObj 与 PointMaze；视觉、形状、动力学、layout shift；每次 MPC replan 一步更新，GD/CEM 都测。关键句（§3.2）： “AdaJEPA continuously updates itself using the transitions caused by its own actions.”
- 边界：反馈来自已经执行的环境转移；论文不检查/校准同一 CEM round 的未执行 candidates，更不学习 elite 排序在多轮 refit 中的误差传播。因此是“executed-path adaptation”，只部分贴合③。

### PROWL — arXiv:2605.18803

- 原文：[arXiv abstract](https://arxiv.org/abs/2605.18803)，[full HTML](https://arxiv.org/html/2605.18803)。
- §3：先以 BASALT FindCave 训练 action-conditioned diffusion video WM；再让受 forward-KL 约束的 VPT/PPO policy 在 MineRL 主动找高 prediction-error 轨迹。PAT buffer 以 latent error、optical-flow action fidelity、learning progress 排序，随后微调 WM 的 action-conditioning 子集； held-out MakeWaterfall / BuildVillageHouse / CreateVillageAnimalPen 验证。
- 论文把它描述为 asymmetric two-player game，但算法是“固定 WM 时找错例，再微调 WM”的交替过程，并非同时求 saddle/fixed point；全文未给 CEM cost/elite。它贴合 planner-like data acquisition，不贴合①②⑤。

### World Models as Adversaries (AWM) — arXiv:2607.10630

- 原文：[arXiv](https://arxiv.org/abs/2607.10630)。
- Abstract/方法概述：把鲁棒驾驶 planner 学习写成 constrained min-max。inner 将 planner 的 predictive WM 角色化为 adversary，学习 sparse scene-adaptive attack coalitions；outer 在冻结 AWM 上做 regret-aware robust best response，含 tail-risk weighting 与 trust region。实验为 nuPlan、InterPlan。
- 这是④的真实结构碰撞，不能只叫“类比”；但其“tail”是交通 hard-case risk，WM 输出 adversarial interactions，不是在 CEM candidate low-cost tail 上校准预测 cost，也没有 proposal-refit 流。

## 3. 线 2：Temporal Straightening camera-ready 差分

- 原文：[arXiv v2 / ICML 2026 camera-ready](https://arxiv.org/abs/2603.12231)，[v2 HTML](https://arxiv.org/html/2603.12231v2)。历史为 v1 2026-03-12、v2 2026-06-11。已分别拉取官方 v1/v2 PDF/TeX 并逐段差分，不只看当前摘要。
- 主方法未变：JEPA prediction loss 加局部 trajectory curvature regularizer，使 latent Euclidean distance 更接近 geodesic、改善 action-space objective conditioning。主实验 Wall、PointMaze、PushT；主要 planner 为 GD，§5.3/B.3 另比 CEM（200 samples、10 iterations），straightening 对两者都有增益，但 CEM 更慢。
- v2 明确新增：目标距 start 50 steps 的 long-horizon 评测；`L_plan=L_spatial+0.1L_agg`，MPC 下优于只用 spatial goal cost；§5.3 新增 symmetric Euclidean cost 对 asymmetric/irreversible dynamics 的限制，以及 prediction space 与 task/geometry-aware projected planning space 可分离的讨论。
- 关键句（§5.3 Limitations）： “the planner optimizes a task- and geometry-aware objective in a projected space.”
- 边界：新增内容是固定 cost 组合与表征几何，不估计 learned-vs-true 排序，不在 CEM candidate/query 分布上适应，也不展开/训练 CEM refit。

## 4. 线 3：navigable / planner-aware EBM

### Learning Navigable World Models via Latent Energy Shaping

- 原文：[OpenReview PDF, id=fnLHZcXZUW](https://openreview.net/pdf?id=fnLHZcXZUW)，ICLR 2026 World Models Workshop；本次未确认 arXiv 版本。
- 官方 PDF 可检索文本显示：把 composed start-goal energy 塑成 convex-like basin，使 gradient planning 收敛到 meaningful next-step latent；再接 distance-preserving encoder 与 skill-conditioned actor，在 offline GCRL suite 评测。
- OpenReview 网页触发校验页，未能稳定独立读取全部表格/附录；除上述官方 PDF 可见内容外，细节标为未确认。它是静态 optimizer-aware shaping，不是部署时 query-path calibration。

### Planning as Descent (PaD) — arXiv:2512.17846

- 原文：[arXiv](https://arxiv.org/abs/2512.17846)，[full HTML](https://arxiv.org/html/2512.17846)。
- §4：对整个 latent future trajectory 学 goal-conditioned energy；hindsight relabeling、trajectory corruption 和 denoising loss把 energy landscape 塑在同一套 gradient refinement 上。推理对多个 time-to-reach hypotheses 并行 refinement、top-K 选 plan，再以独立 inverse dynamics 解码动作；在线只做 replan，不更新 energy model。
- §5：OGBench state-based single-cube，两种数据质量；论文报告 narrow expert 95% success。关键句（§1）： “The energy landscape is shaped during training around the exact descent dynamics used at test time.”
- 边界：它确实对齐 optimizer dynamics，但 optimizer 是 gradient descent；没有 latent forward WM+CEM，也没有对 CEM candidates 的 true-cost tail ranking 或 refit accumulation。

### GeoWorld — arXiv:2602.23058，CVPR 2026

- 原文：[arXiv](https://arxiv.org/abs/2602.23058)，[full HTML](https://arxiv.org/html/2602.23058)。
- §3.2–3.4：H-JEPA 在 Poincaré ball 学 dynamics；teacher forcing + two-step rollout；GRL 直接更新 predictor，以 negative hyperbolic energy/path value 和 triangle-inequality regularization 改善多步 rollout。§3.5：训练后冻结 encoder/predictor，用 CEM 搜 action sequence。CrossTask/COIN 的 3/4-step procedural/video planning，论文报告相对 V-JEPA2 约 +3%/+2% SR。
- 关键句（§3.5）： “The optimization is performed with the Cross-Entropy Method (CEM)”
- 边界：它是当前最明确的 planner-aware JEPA+CEM 近邻，但 GRL 没有把 CEM elite/refit 展开进训练，也不以 planner 实际 candidate distribution 或真实 cost rank 为监督。

## 5. 线 4：performative prediction / RL / control

- [Performative Prediction (PMLR 2020; arXiv:2002.06673)](https://proceedings.mlr.press/v119/perdomo20a.html)：参数 `θ` 诱导部署分布 `D(θ)`，定义 performative stability；给 repeated risk minimization 收敛到 stable point 的条件。
- [Performative RL (PMLR 2023; arXiv:2207.00046)](https://proceedings.mlr.press/v202/mandal23a.html)：policy 改变 reward 与 transition；重复优化正则化目标、gradient variant、有限 trajectory variant 均给稳定解收敛，实验为 grid world。
- [Linear-MDP extension (AISTATS 2025; arXiv:2411.05234)](https://proceedings.mlr.press/v258/mandal25a.html)：从 tabular 扩到 linear MDP；bounded coverage 下反复求 empirical Lagrangian saddle point 收敛到 stable solution，复杂度依 feature dimension。
- [Gradually Shifting Environments (arXiv:2402.09838)](https://arxiv.org/abs/2402.09838)：环境还依赖 previous dynamics，MDRR 混合多次部署样本并给收敛条件。
- 覆盖判断：这些论文给④最干净的数学词汇（distribution map、stable point、contraction/regularized retraining），但 model 是外部环境响应映射，不是 planner 内部的 learned WM/cost；无①②⑤，也未把 CEM query distribution 当 performative distribution。

## 6. 线 5：decision-induced shift 下的 adaptive conformal / risk

### Who Moved My Distribution? — arXiv:2511.11567

- 原文：[arXiv](https://arxiv.org/abs/2511.11567)；已拉官方 PDF 复核 §4–5。
- 机制：部署 uncertainty-aware MPC 后，reactive non-ego agents 改变行为，产生 endogenous distribution shift。算法在每轮收集新 interaction trajectories、重校 conformal regions、平滑 threshold、重部署 controller。§4.1 定义 `T=F∘CP`；若 `L_MPC L_CP<1`，Banach contraction 给 trajectory-distribution 唯一 fixed point。
- 实验：2-agent/3-agent simulator；ICP/ISCP 通常约 4 轮到停止条件，摘要报告最高 +9.6% success。重要限定：§5 的 3-agent ICP 若越过 stopping condition 继续迭代，prediction sets 会再变化，作者称 fixed point empirically unstable；因此理论是条件性的，不能写成无条件收敛。
- 边界：这是③+④的强结构碰撞，但 feedback/calibration 对象是外部 agent trajectory 的安全 prediction set，不是 latent WM candidate cost/rank。

### Conformal Validity / FCP-MPC / CP-SLS-MPC

- [Conformal Validity Guarantees Exist for Any Data Distribution (ICML 2024; 2405.06627)](https://arxiv.org/abs/2405.06627)：允许 agent actions 诱发 sequential feedback-loop shift；一般构造理论，并在 black-box optimization/active learning 上给可计算实例。不是 MPC cost-rank 方法。
- [FCP-MPC (2607.00776)](https://arxiv.org/abs/2607.00776)：offline 用 FPCA+GMM conformalize residual distance field；online AFCP 依据延迟到达的真实 residual，仅更新每个 horizon 的低维标量；将 field envelope 作为 sampling MPC/MPPI 的 hard constraint 或 soft penalty。ETH-UCY 与最多 280 dynamic obstacles 的 PyBullet quadrotor。它是“沿执行路径的风险有效性”而非“沿 CEM 内部未执行 query 的 model-cost 有效性”。
- [CP-SLS-MPC (2602.12047)](https://arxiv.org/abs/2602.12047)：state-control-dependent weighted conformal bounds 进入 robust MPC；同样校准 safety coverage，不校准候选相对 cost rank。

## 7. 三个定向扫描中的最近邻

### ACID — arXiv:2607.02403

- 原文：[arXiv](https://arxiv.org/abs/2607.02403)，[full HTML](https://arxiv.org/html/2607.02403)。这是本轮最重要的遗漏修正。
- §3.2：对 WM 预测轨迹的每一步，用独立训练的 inverse dynamics model 回推 action，与 conditioning action 的 residual 累加成 action-consistency cost。§3.3：与 terminal goal cost 相加；每个 CEM iteration 以两项 cost 在当前 candidates 上的标准差自动定权，从而直接改变 elite ranking/refit。
- §4：三种 JEPA latent WM（DINO-WM、PLDM、Le-WM）加一个 video WM；Cube/Reacher/PushT/Rope/Granular/Nav 六任务。原文明确：“CEM ranks candidate action sequences according to their augmented costs, and selects those with the lowest costs”
- 碰撞边界：它已经占据“全序列 verifier + 每轮 CEM 配权 + latent WM+CEM”。但 adaptive weight 只是 scale/spread equalization；IDM 与 WM 都冻结，没有真实 outcome/rank feedback；不测 learned-vs-true low-tail ordering，也不分析误排经 refit 的累积。因此①是 proxy、③是弱形式、②④仍空。

### WM-VAE / novelty cost — arXiv:2508.06096

- 原文：[arXiv](https://arxiv.org/abs/2508.06096)；已拉官方 PDF 复核 Algorithm 1 与 Tables I/III。
- 在 DINO-WM+CEM 上另训 VAE novelty detector。每个 candidate rollout 的每个 predicted latent 都计算 reconstruction loss；总 cost 为 terminal goal MSE 加 `w·Σ_t reconstruction_loss_t`，因此每轮 elite/refit 都受全序列 novelty 影响。NVIDIA FleX Granular/Rope/Cloth；论文表 I 的 Chamfer distance 为 DINO-WM 0.391/1.394/9.228，WM-VAE 0.362/1.014/5.372；权重过大反而伤害收敛。
- 原文（§I）：“the reconstruction loss of each predicted latent state is treated as the per-action cost for each action in a trajectory.”
- 边界：这是固定 surrogate validity，不从 planner query 或环境反馈更新；不对 true cost/rank，也不研究 proposal-flow accumulation。

### IMWM — arXiv:2606.01626

- 原文：[arXiv](https://arxiv.org/abs/2606.01626)，[full HTML](https://arxiv.org/html/2606.01626)。与当前 LeWM+CEM 语境高度接近，必须单列。
- §2/A：即使把 learned predictor 换成环境真动力学、保留 terminal latent-MSE 与固定 CEM budget，Two-Room/OGBench-Cube 仍会因 population 中没有成功 candidate 而失败；Theorem A.1 给 finite-query proposal-volume worst-case bound。
- §4：demo-trained intuition model；Retrieval Initialization 改 CEM 初始均值，Hybrid Cost 合并 intuition/WM cost，Reliability Gate 在 query 前选固定 recipe。Appendix B.2 明写 CEM 为 top-k empirical target 上的 maximum-likelihood proposal refit；B.6 显示 5-step rollout 承载 rank signal；G.5 在 oracle-dynamics logs 上做 candidate-rank diagnostic，结论为一旦成功 candidate 存在，terminal latent objective 通常排第一；B.7 则把 closed-loop replan distribution shift 明列为未审计区。
- 边界：它研究 proposal coverage、refit mechanics 与若干 rank 诊断，但不是 learned-vs-true low-tail misranking，未刻画误排跨 refit 累积；intuition/WM/bank/gate 在评测时冻结，故无③④。

### 形式化/诊断型旁证

- [Imperfect World Models are Exploitable (2605.15960)](https://arxiv.org/abs/2605.15960)：直接把“model 偏好 policy A、true dynamics 偏好 B”定义为 exploitation，并给大 policy set 上的不可避免性与 safe horizon；是①的抽象上位概念，但无 CEM/low-tail。
- [When a Verified World Model Still Loses (2607.14169)](https://arxiv.org/abs/2607.14169)：code WM 在 planner search distribution 上 ≥98% state accuracy 仍会因 <1% pivotal dynamics 系统性输掉；强调 planning adequacy 不能由平均 prediction accuracy 代替。不是连续 latent/CEM。
- [Inference Time Policy Optimization (2603.22430)](https://arxiv.org/abs/2603.22430)：对 frozen differentiable WM 的 imagined rollout 反传，在线更新 policy 参数；D4RL MuJoCo/AntMaze。适应的是 policy，不是 WM validity，也无 CEM。

## 8. 检索式与否定性陈述边界

本次对 arXiv/官方索引做了以下标题/摘要精确词与同义词组合扫描：

- `"CEM elite misranking"`、`"elite selection error" "model-based planning"`；
- `"planner-induced distribution shift" "world model"`、`world model CEM "proposal distribution"`；
- `"test-time cost calibration" MPC`、`world model "planning cost" adaptive CEM`；
- 同义扩展：`candidate rank`、`action consistency`、`novelty cost`、`proposal volume`、`endogenous distribution shift`、`adaptive conformal MPC`。

前三个精确短语本次未检出同名工作；同义扩展命中 ACID、WM-VAE、IMWM、AdaJEPA、Who Moved My Distribution、FCP-MPC 等。这个结果只支持“未检出”，不支持“绝对不存在”。
