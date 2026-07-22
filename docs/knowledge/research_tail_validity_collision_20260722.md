# Planner-equilibrium / adaptive tail validity 碰撞调研（复核版，2026-07-22）

> 仅答需求书 §2，不作 GO/NO-GO。①tail 误排；②proposal-update 累积；③query-path 自适应；④planner-model 均衡；⑤latent WM+CEM/MPPI。

## 五条线

1. **Adaptive/adversarial WM。** (a) AdaJEPA（2606.32026）行动后以真转移更新 JEPA（CEM；PushT/PointMaze）；PROWL（2605.18803）以 KL 约束策略挖错例再训视频 WM（MineRL）；AWM（2607.10630）做驾驶 min-max（nuPlan/InterPlan）。(b) Ada 部分③⑤；AWM 结构性④；PROWL 为对抗环。(c) 均非①②，未沿 CEM query 校准 cost。引（AdaJEPA §3.2）：“transitions caused by its own actions.”

2. **Temporal Straightening。** (a) 2603.12231v2 以曲率正则训 JEPA；逐页差分确认 camera-ready 新增 50-step、`Lspatial+0.1Lagg` 及非对称/投影 cost 限制，并比较 GD/CEM。(b) 覆盖⑤与 planner-facing cost shaping。(c) 不校准①，无②③④，目标离线固定。引（§5.3）：“the planner optimizes a task- and geometry-aware objective in a projected space.”

3. **Navigable/planner-aware EBM。** (a) GeoWorld（2602.23058）以 H-JEPA+GRL 整形双曲能量/三角正则，CEM 测 CrossTask/COIN；PaD（2512.17846）令整段 latent energy 的训练/梯度规划同构；Navigable WM（OpenReview:fnLHZcXZUW）arXiv 未确认。(b) 静态③，GeoWorld⑤。(c) 无①②、在线③④，也不反传 CEM refit。引（GeoWorld §3.5）：“performed with the Cross-Entropy Method”

4. **Performative prediction/RL。** (a) 2002.06673 定义参数诱导分布与 stability；2207.00046 令策略改变 reward/transition并重复优化至稳定；2411.05234 扩至线性 MDP、有限样本鞍点。(b) 形式化④及收敛。(c) 均衡是 policy–外部环境，非 planner–内部 WM；无①②③⑤。引（PRL Abstract）：“the policy chosen by the learner affects the underlying reward and transition dynamics.”

5. **Adaptive conformal/risk。** (a) 2511.11567 交替适配 conformal/controller 应对内生 shift（2/3-agent）；2405.06627 涵盖 agent-induced shift；2607.00776 在线 AFCP 风险场接 sampling MPC。(b) ③及条件性④。(c) coverage/safety 非 cost 排序；无①②⑤。引（2511.11567 Abstract）：“distribution shift which we call endogenous distribution shift.”

## 定向标题/摘要扫描（2025–2026）

- **CEM elite misranking：**精确词组未检出。最近邻 ACID（2607.02403）把逐步 inverse-dynamics residual 以每轮 CEM 自适应权重重排 elite；WM-VAE（2508.06096）把每个预测 latent 的 VAE novelty 累加进全序列 cost。二者覆盖⑤；ACID 的③仅为候选集方差配权，二者均未用真实 cost 定义①或学习②。
- **planner-induced shift：**精确词组未检出。IMWM（2606.01626）刻画有限查询 proposal-volume、改 CEM 初始 proposal/混合 cost，并明确把 closed-loop replan shift 留作未审计区；2605.15960 与 2607.14169 分别刻画 model/true 偏好反转和“高准确但错在关键少数”。均未做自适应 planner–WM 均衡。
- **test-time cost calibration MPC：**精确词组未检出。ACID 是逐候选、逐 CEM-round 的最近邻；WM-VAE 是固定全程 novelty verifier；Temporal v2/IMWM 属固定目标或静态 gate。未发现把在线真值反馈用于 low-tail rank 校准并贯穿 proposal refit 的工作。
