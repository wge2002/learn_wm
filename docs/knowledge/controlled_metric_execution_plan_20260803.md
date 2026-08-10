# 受控度量方向：合并后的执行计划

日期：2026-08-03
状态：执行计划（取代三份文档里各自的 gate 列表作为**执行顺序**的依据；
理论内容仍以原文档为准）

> **2026-08-10 协议修正：**Phase 1/2 的训练配对、seed 数、checkpoint 时间点和
> 判决阈值已由
> [正式成对训练协议](controlled_metric_paired_protocol_20260810.md) 取代。
> 旧 K1/K5 的模型初始化、window 长度和 update budget 不匹配，只能作为探索性
> 证据；正式最小集现为 `K1/K5 × 3 seeds = 6 runs`。

前置文档：

- [Gaussian Measure, Controlled Metric](horizon_induced_gaussian_gauge_20260724.md)（G0–G4）
- [Gaussian marginal 到底约束了什么](gaussian_dynamics_identifiability_20260801.md)（`K × persistence`）
- [哪一种 Gaussian 几何适合动力学](dynamics_compatible_gaussian_geometry_20260801.md)（Gate A/B/C）

---

## 0. 为什么需要这份文档

三份文档各自写了一套 gate，合起来有三处不能直接执行：

1. **前提失效。**三套 gate 都假设 D192 的 K1/K3/K5/K10 checkpoint 存在。
   这批 checkpoint 已经没有了。gauge 文档里"先用现有 checkpoints 完成 G0–G3，
   不训练新模型"这条执行指令因此**作废**——现在任何读数之前都得先重训。
2. **Gate A 与 G2 的关系没写清。**Gate A 读 Euclidean 谱（`log singular spread`、
   product max gain），而 G2 的 kill criterion 明确说*只*有 Euclidean 变化不算
   改善。所以 Gate A 不能独立判决：**Gate A 是筛子，G2 才是判决**。
3. **`K × persistence` 网格买不起。**原文写 `K∈{1,2,3,5,10}` ×
   `persistence∈{frame,2f,5f,episode}` × `seed>=3` = 60 次训练。见 §6 的削减。

另外算力变了：原计划基于 A100 / 8 卡，现在是 **2× L20Z 作 debug + DLC 跑正式**。
L20Z 的 bf16 吞吐明显低于 A100，所以"能训多少 checkpoint"是这一版的主约束。

---

## 1. 判决依赖图

```text
[免费，现在就能做]
  P0a estimator 在 toy 上校准 ──┐
  P0b 输入侧 local ID 审计 ─────┤
                               ↓
[最小 checkpoint 集: K1,K5 × 3 seed = 6 runs]
                               ↓
  G0b latent 侧刚性审计 → 若"确实是精确满维刚性区" ⇒ premise 死
                               ↓
  Gate A (Euclidean 筛子) → 只看 shear 是否与 uniform contraction 分离
                               ↓
  G1 (双向可逆性) → 区分 gauge reshaping vs 信息丢失
                               ↓
  G2 (坐标不变 pencil) ★ 主判决 ★
                               ↓
        ┌──────────────────────┴──────────────────────┐
     G2 过                                        G2 不过
        ↓                                             ↓
  G3 (2×2 因果) / Gate B (frozen gauge)      科学 claim 降级为
  G4 (head-to-head) / K×persistence          predictor-relative 坐标整形
```

关键性质：**G2 是最便宜的决定性实验**（正式设计只要 K1 vs K5 × 3 seed），而
G3/G4/Gate B 全都贵得多。所以路径必须是「先花 6 次训练拿到 G2 判决，再决定
要不要继续投」。

---

## 2. P0a — 先在 toy 上校准证书估计器（免费，无需 GPU）

所有 gate 本质上都是**测量程序**：pullback metric 估计、generalized pencil、
shear 与 uniform contraction 的分离。这些估计器如果有 bug，后面每一个读数都是垃圾，
而且在真实 latent 上**没有 ground truth 可以发现 bug**。

已有的 exact-Gaussian toy（`scripts/plan/dynamics_compatible_gaussian_geometry_toy.py`）
恰好提供了解析已知的答案：正确 gauge 下 bilinear MSE `1.00 → 0.279`（Bayes limit）、
one-step p95 gain `25.36 → 1`、H=5 product excess `17.65 → 1`。

因此 P0a 要求：把打算用在 PushT 上的**同一份估计器代码**跑在 toy 上，必须复现上面
三个数。三者任一复现不出来，就是估计器的问题，不是模型的问题。

这一步不需要 checkpoint、不需要 GPU，可以和 §3 的训练并行做。

---

## 3. P0b — 输入侧维度审计（免费，无需 checkpoint）

G0 原文把"输入侧"和"latent 侧"混在一起，但**输入侧不需要任何 checkpoint**：

- PushT physical state 是 7 维（agent xy / block xyθ / agent vxvy），history=3；
- deterministic rendering 下，latent 的理论独立随机因素上界由此确定；
- 数据集上直接测 local intrinsic dimension。

若这个上界本身就接近 192，那"D192 过完备是自由度主要来源"这条就站不住，
整条线要退回等维近似分析（这是 gauge 文档 G0 的证伪条件）。**这是最便宜的
premise kill test，应该在训练启动前就跑掉。**

---

## 4. Phase 1 — 最小 checkpoint 集（细节由 2026-08-10 协议取代）

```text
K=1, seed × 3
K=5, seed × 3        （matched one-step 对照 vs open-loop K5）
```

6 次训练。这是 G0b/Gate A/G1/G2 的正式最小输入，不多训一个。每个 seed 的
两臂必须加载同一份初始化，并使用相同 8-frame window、split、batch order 和
update budget；具体自动验配规则见 2026-08-10 协议。

`K=3/K=10` **暂缓**——它们只在"趋势单调性"上加信息，而 G2 的判决只需要 K1 vs K5
的对比。等 G2 过了再补，用来验证 §5 的单调性预测。

2 卡 debug 阶段的任务只有一个：确认 forward/backward/checkpoint/SIGReg 全通，
以及记下单步耗时用来给 DLC 排期。**不要在 2 卡上训正式 checkpoint。**

DDP 放大时要单独看的三处（2 卡测不到）：SIGReg 的 `num_proj: 1024` 投影是否跨卡
同步、`batch_size: 128` 是 per-device 所以有效 batch 随卡数变、以及 CritWM sensor
里那个 `all_reduce`（若启用）。

---

## 5. Phase 2 — 决定性读数

### Gate A（筛子，不是判决）

对 K1/K5 报 one-step 与 H-step 的 `log singular spread`、product max gain、
action/error 谱，按 free/contact regime 分层。

**唯一需要它回答的问题**：shear 下降与 uniform contraction 是否**分离**。
即 `dev log(P_H^T P_H)` 下降是否明显快于整体尺度下降，且 action Gramian 不同步塌。
若两者同步下降，直接进入降级分支，不必再花钱做 G2。

### G1（双向可逆性）

matched physical state 上拟合 K1↔K5 latent 的双向 held-out map（orthogonal /
linear / invertible nonlinear 三档）。双向近可逆且非正交 = gauge/metric selection；
单向可预测而逆向差 = 信息丢失。**只报单向 R² 不算通过。**

### G2（主判决，坐标不变）

固定四件事再算 generalized eigenvalues / `CP_H`：action covariance 来自同一标准化
planner perturbation、residual covariance 用真实 one-step latent residual、
reference `W_0` 用同一 physical perturbation 的 pushforward、逐轨迹逐终点单独算。

**kill**：若 K 只改变 Euclidean trace/singular value 而 invariant pencil 在
matched 定义下不变 ⇒ 科学 claim 降级为 predictor-class-relative 坐标整形。

### 必须同时报的充分性证书

Obsessed Encoder 的教训是 anti-collapse 不保证 dynamics。所以 Phase 2 的每个
checkpoint 都要附 block/agent/goal pose probe、contact probe、action identifiability
（paired-action effect、`||∂f/∂a||`）。**几何变好但 probe 塌 = 拿到一个语义空掉的
latent，不算通过。**

---

## 6. Phase 3 — 条件性下游（仅在 G2 过后）

`K × persistence` 网格从 60 次削减到 **12 次**：

```text
K ∈ {1, 5} × persistence ∈ {frame, episode} × seed × 3
```

依据是 identifiability 文档 §6.2 的解析 phase slice 已经定位了效应应该出现的位置，
中间档（2f/5f）和中间 K 值只在描点上加信息，不改变判决。角点设计足以测核心预测：
K 增大时 episode-persistent shortcut 的 obsession 是增强还是减弱。

这一步的 kill criterion 是整条线最重的一条：**若 K5 在 episode-square arm 上拿到
更低的 error-product gain，却同时更强编码 key、丢状态/动作、planning 失败，则
"horizon 自动选择更合适的 dynamics manifold"这个无条件说法被证伪**，幸存版本必须
写成"horizon 在给定 sufficiency anchor 之后才选择更好的 recursive metric"。

其余下游按原文执行，不在这里重复：G3 的 2×2（K1 / TF-K5 / sg-K5 / full-K5）、
Gate B 的 frozen-encoder gauge fitting（`T_theta` + predictor，encoder 冻结）、
G4 的 method/science claim 分离与 matched-compute head-to-head。

---

## 7. CritWM 的处置

**不重跑 v1。**理由不是实现 bug，是控错了量：

`scripts/train/lewm.py:346-374` 的 sensor 每 `thermo_every` 步注入单位扰动，
之后每步把 diff 重新归一化再注入——这是 power iteration，测的是 **top singular
gain σ_max**，控制器 (`thermo_target: 1.0`) 把它推向 1。而 §5 的 Gate A 判据说
σ_max 只是必要不充分：需要的是 shear 下降**且** action gain 保住。

v1 的实测正好落在预注册的失败分支上：确定性 rate 近临界，但标准 CEM 远 goal
只有 K=1 水平（42.0，对比 K=5 的 56.0）——"只得到 uniform contraction，
action gain 同步下降"。

**处置**：thermostat 机制保留，但 setpoint 要等 Gate A 给出正确的那一对
（shear ↓ + action gain 守住）之后再接。同时修 sensor 的两个污染源：probe 时
predictor 的 `dropout: 0.1` 是激活的，BN running stats 也在更新——probe 必须
`eval()` + 存恢复 BN buffer，且同一 checkpoint/batch/delta 要 bitwise 可复现。

在 Gate A 之前重跑 CritWM，最好的结果也只是把已知负结果复现一遍。

---

## 8. 每层的 kill rule 汇总

| 层 | 通过条件 | 不过则 |
| --- | --- | --- |
| P0a | toy 上复现 0.279 / 1 / 1 三个数 | 估计器有 bug，所有后续读数无效 |
| P0b | 输入侧独立因素远少于 192 | 过完备不是自由度来源，退回等维分析 |
| G0b | 非精确满维 Gaussian 刚性区 | premise 死 |
| Gate A | shear 与 uniform contraction 分离 | 直接降级，不做 G2 |
| G1 | 双向近可逆且非正交 | 是信息选择而非 gauge selection |
| **G2** | invariant pencil 在 K1→K5 改善 | **科学 claim 降级为坐标整形** |
| 充分性 | probe/action 证书不塌 | 几何漂亮但语义空，不算通过 |
| K×persist | K 增大不放大 shortcut obsession | 无条件版本证伪，须加 sufficiency anchor |
| G4 | matched-compute 赢 curvature/bisim | 方法 claim 关闭，理论方向仍可保留 |

任一层失败按原文 kill rule 降级，**不发明新 regularizer 去救叙事**。
