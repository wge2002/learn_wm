# LeWM planning 研究现状总账（截至 2026-07-21）

> **这是当前结论的唯一入口。** 2026-07-17 至 2026-07-21 的长文档保留为
> chronological lab notebook；其中早期的 `Horizon-Bundle`、`BP-OEWM`、
> `BL-WM` 等名字记录了 idea 演化，不代表仍在执行。若本文与早期段落冲突，
> 以本文和对应的最新 locked/fresh result 为准。

## 0. 当前一句话结论

我们已经确认一个真实且有控制价值的问题：

> **learned world-model cost 会在 CEM 的 low-cost tail 上产生错误的 elite/update；
> 用真实动力学给同一批 candidates 排序并递归更新 proposal，可以显著改善后续
> 搜索和最终 outcome。**

但这几天测试的实现都没有把这个 oracle ceiling 稳定转成 deployable method：

- 没有稳定的 horizon matching；
- post-hoc verifier、scalar ensemble、continuous correction head 不 work；
- set-valued branches 能生成有价值的候选，但 learned selector 在 fresh data 上不复现；
- true landscape 确实 multi-basin，但 component-wise refit 只带来极小增益；
- K10 在自己诱导的 query path 上发生 tail-fidelity reversal，但最后一轮换成 K3
  scorer 几乎救不回来，说明损失是沿 proposal flow 累积的，不是一个 final selector
  patch 可以修复。

因此截至现在：

```text
scientific problem / oracle target                         SUPPORTED
current deployable method                                  NOT FOUND
Horizon-Bundle                                             STOP
OE-only dynamics fine-tuning                               STOP
frozen BP-OE branch patch                                  STOP
topology / BL-WM as main headline                          DOWNGRADE TO DIAGNOSTIC
next open question                                         full-sequence adaptive tail fidelity
```

现在**不应该直接开一个大规模 topology training**，也不应该继续调 frozen
selector。下一步先把“planner-equilibrium / adaptive tail validity”与 AWM、
Temporal Straightening、Navigable EBM、performative prediction、adaptive
risk/conformal calibration 做严格碰撞，再决定是否还有足够独立且可实现的方法空间。

## 1. 这几天 idea 是怎样一步步变化的

| 阶段 | 当时的问题 | 新实验给出的事实 | 当前判决 |
|---|---|---|---|
| Horizon-Bundle WM | 不同 planning horizon 是否需要不同 latent bundle | 完整 `K × H` Gate 没有 horizon matching；K3 近似 universal winner，真正稳定的是 planning-horizon cliff | `STOP` |
| verifier / portfolio | 能否在现成 K3/K5/K10 中挑对 scorer 或组合 candidates | oracle portfolio 有 ceiling，但 learned verifier、rank average、shared-population ensemble 均不稳定或崩溃 | 方法 `STOP`；保留为 probe/baseline |
| Optimizer-Equivalent WM | 是否应直接匹配 true dynamics 诱导的 CEM elite moment | fixed-bank mismatch 跨独立数据重现；true update 的递归干预在 H5/H8 都有 outcome headroom | causal target `SUPPORTED` |
| OE fine-tune / on-policy aggregation | 能否把 oracle update 直接微调进原 WM | held-out direction 有小幅改善，但 locked Gate 失败；on-policy aggregation 只防退化，不产生能力 | 当前 recipe `STOP` |
| independent operator / candidate head | 冻结 WM 后能否用小 head 修 update 或 rerank candidates | continuous head 主要缩幅并平均反向 modes；raw latent candidate head 不能稳定泛化 | 当前实现 `STOP` |
| set-valued BP-OE | correction 是否应是一组 branches 而非一个平均向量 | branch set 有真实 lower-cost ceiling，cross-model vector disagreement 有信号；但 fresh60×2 selector/adoption confirmation 不 work | frozen planner patch `STOP` |
| topology / BL-WM | 更 smooth 的模型是否 merge/drop 真实 basins | true low-cost set multi-basin；K10 静态 topology fidelity 更好；自己的 query path 上又发生 tail reversal | topology 是有效 diagnostic |
| causal refit | tail reversal 是否真由多 basin / mean averaging造成 | final K3 swap只净增 1.7pp；true component oracle只比true global mean多1.7pp | basin multiplicity 不是当前主要 causal bottleneck |

这条演化不是“所有 idea 都失败了所以又换名字”。真正逐步收紧的是因果对象：

```text
representation horizon
  -> candidate selection
  -> CEM sufficient-statistic update
  -> multimodal correction set
  -> adaptive proposal path
  -> full-sequence low-cost-tail validity
```

前五层大多已经被控制实验排除或降级；最后一层仍是开放问题，但还不是完成的 idea。

## 2. 最重要的证据账本

### 2.1 Horizon-Bundle Gate：没有预期的 K/H matching

完整 A100 matrix、5090 cross-bank 与 matched-compute controls 的共同结论是：

- 没有稳定的“训练 horizon K 与 planning horizon H 对角占优”；
- K3 在大多数设置中近似 universal winner；
- local winner 多数 paired CI 跨零，也没有跨协议稳定复现；
- 最稳定的现象是 planning horizon 增大后的 cliff，而不是 representation bundle
  的 crossing。

因此原始 `World Models Need a State for Every Horizon` headline 关闭。详细记录见
[Horizon-Bundle temporal notebook](horizon_bundle_temporal.md) §§10–12 与
[A100 Gate A report](horizon_bundle_gateA_a100_20260717/README.md)。

### 2.2 Learned cost 与 true dynamics 的 optimizer update 确实不等价

在独立 `60 states × 4 rounds × 300 candidates` 的 K3 H5/off40 trace 上，
随着 CEM 进入 late rounds：

| recorded step | elite overlap | update cosine | relative update error |
|---:|---:|---:|---:|
| 4 | `.253` | `.295` | `1.090` |
| 9 | `.218` | `.217` | `1.159` |
| 19 | `.135` | `.132` | `1.159` |
| 29 | `.079` | `.043` | `1.198` |

即使 learned objective 在优化，model top-30 与 simulator true top-30 及其 proposal
update 会持续分离。这个事实跨早期 12 states、独立 60 states 与不同硬件重现。

### 2.3 True/oracle update 有真实递归 outcome ceiling

从同一 saved proposal 出发，只替换 CEM update，保留相同 candidate budget：

| cell | learned final-mean success | oracle-update success | final true-cost delta |
|---|---:|---:|---:|
| H5 / offset40 | `3/12` | `7/12` | `-67.50` |
| H8 / offset60 | `3/12` | `6/12` | `-49.32 [-101.39,-14.24]` |

样本小，所以 success CI 不能被夸大；但两个 horizon 的 true-cost 和 proposal
coverage 同向，足以支持“optimizer update 是值得修的 causal target”。详细产物见
[5090 OE audit index](optimizer_equivalence_5090_20260718/README.md)。

### 2.4 直接学习这个 target：有弱方向信号，但没有过 Gate

锁定的 60-state、3-fold state-held-out OE fine-tune 在 epoch 5 得到：

```text
Δ update cosine       +.076 [.030,.123]
Δ relative error      +.005 [-.034,.044]
Δ elite overlap       +.036
Δ selected true cost  -2.04
```

targeted mass/relative-loss 修复后也只有 `+.080 / -.005 / +.034`，仍未达到预注册
的 `+.10 / -.10 / +.05`。继续训练还会损伤原 dynamics geometry。

正式 on-policy aggregation 的最好 trust arm 也只有：

```text
Δ cosine  +.0205
Δ rel     +.0044
Δ overlap +.0100
```

所以应精确表述为：

```text
oracle OE target is causal                         YES
fixed-trace gradient contains some signal          YES, weak
proposal shift alone explains failure              NO
current OE-only / scalar on-policy recipe works    NO
```

这里的“不由 proposal shift 单独解释”与后面的 self-induced tail reversal 不冲突：
query feedback 确实存在，但简单累积 on-policy samples 并不能解决它。

### 2.5 Correction 是多模态的，cross-model vector disagreement 有信号

H5 oracle-minus-model correction 的四个 prototypes：

- 前两个奇异轴解释 `98.75%` energy；
- 形成两对近似反向 modes，cosine `-.963 / -.981`；
- 近似长时域 `±x / ±y` action corrections；
- adjacent rounds 保持同 mode 的比例为 `61.25%`。

因此单向量 L2 regression 容易返回抵消后的 conditional mean。set-valued probe 中：

| observable | learned top1 Δcos | top2 coverage Δcos | all-mode coverage Δcos |
|---|---:|---:|---:|
| raw frozen CLS | `+.007` | `+.051` | `+.106` |
| true state（privileged） | `+.058` | `+.113` | `+.174` |
| K3+K10 vector outcomes | **`+.045`** | **`+.097`** | **`+.164`** |

只有多个 scalar ranks 不 work；有用信号来自 signed imagined-outcome vectors。
K3 elite 的 30/300 个 K10 sparse queries 已保留 full cross-model signal 的约
`58% / 77% / 81%`（top1/top2/all modes）。

### 2.6 Branch set 有 ceiling，但 frozen selection 不 work

12-state recursive CRN 与 iso-compute controls 表明：

- same-model `2×150`、`2×300` 均不优于 ordinary K3；
- K3 从 300 增到 600 calls 基本饱和；
- BP full hidden oracle 相对 K3 `1×600` 仍约 `-7.21` true cost；
- sparse BP hidden oracle 相对 same-model `2×300` 为
  `-12.46 [-31.18,-.04]`。

所以 branches 暴露了 model-diversity 引出的真实 lower-cost alternatives，不只是
多采样。但 deployable selector 没有跨 fresh sets 复现：

| selector vs K3 | fresh A | fresh B confirm |
|---|---:|---:|
| learned branch-value | `-2.83 [-7.14,1.61]` | `-.04 [-6.44,6.24]` |
| hidden two-branch oracle | `-4.13 [-8.40,.23]` | `-4.72 [-10.42,.85]` |

base-anchor adoption rule 在 confirm-B 反而是 `+.98`，actual success
`26.7% vs 33.3%`。branch prefixes 也没有足够共享，deferred commitment 关闭。

判决：branch generation mechanism 保留为 scientific evidence；BP-OE 作为现成
inference patch 关闭。详细主账见
[A100 set-valued OE audit](optimizer_equivalence_a100_20260720/README.md) §§1–10。

### 2.7 Static topology 是真实结构，但不是充分的训练目标

同一 `240 × 4 × 300` population 的 iso-rate topology audit：

| geometry | true top-10% components | K3 coverage | K10 coverage | K3/K10 successful-basin coverage |
|---|---:|---:|---:|---:|
| correction 2-D | `4.97` | `.400` | `.499` | `.454 / .520` |
| full action 50-D | `4.50` | `.432` | `.544` | `.552 / .627` |
| random 2-D control | `12.58` | `.308` | `.435` | `.383 / .483` |

可以支持：

- true PushT planning surface 在 sampled action geometry 中确实 multi-basin；
- K3/K10 都 miss 一部分 true/successful basins；
- K10 是更好的 fixed-population ranking/topology model。

不能支持：

- K3 在全局都 fragment、K10 在全局都 merge；
- component 数目接近就代表 basin correspondence；
- static PH/topology loss 会自动改善 recursive planning。

同 proposal 的 one-step transport 仍是 K10 略好，因此真正矛盾被推进到 adaptive
query path，而不是单张 landscape。

### 2.8 Paired on-policy trace：出现强 tail reversal

在同一 fresh60、common initial noise 上分别运行 K3 和 K10 generator，保存
steps `4/9/19/29` 的完整 300-candidate populations，并用两模型交叉评分。
最终 step 29：

| generated path | K3 true-top30 recall | K10 true-top30 recall | K3/K10 component coverage |
|---|---:|---:|---:|
| K3 path | `.096` | **`.272`** | `.278 / .517` |
| K10 path | **`.197`** | `.079` | `.408 / .300` |

路径 interaction：

```text
true-elite recall       -.293 [-.381,-.207]
component coverage      -.346 [-.472,-.222]
successful coverage     -.265 [-.414,-.131]
```

即 K10 在 K3 path 上明显更准，却在自己连续诱导的 proposal path 上失去 low-cost
tail advantage。普通 global Spearman 在 K10 final path 仍为 `.246 vs .247`，
几乎看不到这个 reversal。

实际 recursive success 为：

```text
K3:  .200 -> .317 -> .533 -> .517
K10: .150 -> .383 -> .417 -> .433
```

final K10−K3 为 `-.083 [-.200,+.033]`，所以 outcome winner 本身仍不是 60 states
下的显著结论。完整报告见
[paired basin-lineage audit](optimizer_equivalence_a100_20260720/basin_lineage_pair_a100_20260720/audit_v1/REPORT.md)。

### 2.9 Counterfactual refit：topology headline 被因果降级

对 paired bank 的每个 population 执行 K3/K10/true top-30 global mean，以及
每个 connected component 的 mean。final results：

| path | K3 elite mean | K10 elite mean | true elite mean | true component oracle | candidate support |
|---|---:|---:|---:|---:|---:|
| K3-generated | `51.7%` | `55.0%` | `63.3%` | `65.0%` | `66.7%` |
| K10-generated | `45.0%` | `43.3%` | `58.3%` | `60.0%` | `63.3%` |

在最关键的 K10 path 上：

- 用 K3 global refit 替换 K10，只是 `+1.7pp [-6.7,+10.0]`；4 states 被救，
  3 states 被损失；
- 用 true global refit 是 `+15pp [6.7,25.0]`，净救回 9 states；
- true component oracle 只比 true global mean 再多 1 state；
- 22/60 states 的 population 根本没有成功 candidate。

最终 K3/K10 的 5-state success 差距可以完全拆成：

```text
K3: 40 states have a successful candidate - 9 conversion failures  = 31
K10: 38 states have a successful candidate - 12 conversion failures = 26
```

这意味着：

1. self-induced tail reversal 是真实 diagnostic；
2. 但 K3 的 alternate ranking 并不是足够的 causal repair；
3. multi-basin mean averaging 不是当前主要损失；
4. 大头是沿多轮 proposal flow 累积的 support + tail/update error；
5. standard true-ranked single-Gaussian CEM 已能取得大部分 oracle ceiling，暂时不需要
   把 topology preservation 当 headline。

完整反事实报告见
[counterfactual refit audit](optimizer_equivalence_a100_20260720/counterfactual_refit_a100_20260721/audit_v1/REPORT.md)。

## 3. 当前统一解释

### 3.1 已支持的因果链

```text
learned dynamics / representation
  -> distorted low-cost candidate tail
  -> wrong CEM elite sufficient statistics
  -> proposal moves into a model-specific query distribution
  -> later tail fidelity and candidate support deteriorate
  -> final success/cost gap
```

其中每一箭头的证据强度不同：

- `wrong elite -> wrong update`：强；
- `oracle update -> better recursive outcome`：强 true-cost、small-N success evidence；
- `model-specific path -> tail reversal`：强 observational interaction；
- `tail reversal -> final failure`：存在 headroom，但 K3 swap 不是充分 repair；
- `basin multiplicity -> final failure`：当前 PushT evidence 弱。

### 3.2 为什么以前看起来互相矛盾

以下三件事可以同时为真：

1. K10 在固定 K3 bank 上比 K3 准；
2. K10 在同 proposal 的 one-step refit 上也略好；
3. K10 完整递归后反而较差。

因为 scorer 不只是评估外生数据；它通过 CEM 改变下一轮数据。fixed-bank fidelity
不是这个 coupled dynamical system 的稳定性保证。另一方面，简单 on-policy replay
没有 work，说明“多收自己路径上的数据”也不是充分解法。

### 3.3 目前真正缺的不是哪个模块

不是：

- 更长 prediction horizon；
- 再加一个 verifier；
- scalar ensemble；
- final-round scorer swap；
- ordinary multistart；
- generic PH/topology loss；
- 只拟合一个平均 correction vector。

真正缺的是一个能在**模型自己诱导的 proposal sequence**上维持 low-cost-tail
validity，同时不破坏原 dynamics prediction 的训练对象或风险控制机制。

## 4. 当前 working question，而不是又一个过早命名的方法

暂时不再给新方法取 acronym。当前值得研究的问题写成：

> 给定 `q_{t+1}^M = T(M, q_t^M)`，如何让 learned model 在整个自诱导序列上
> 保持 true low-cost tail / elite update 的有效性，或在无法保证时显式拒绝一次
> 不可信的 proposal update？

可能的两个实现家族只是候选：

1. **proposal-flow / planner-equilibrium training**：不是逐轮独立拟合 candidates，
   而是匹配多轮 proposal sufficient statistics，并用原 prediction replay 保持
   dynamics geometry；
2. **adaptive tail-risk certificate**：给 elite update 一个校准的可信度/风险上界，
   在 model-induced adaptive queries 下控制 false optimistic candidates，而不是
   再输出一个未经校准的 point scorer。

它们在 promotion 前必须回答：

- 与 AWM 的 planner-aware/adversarial data synthesis 是否只是同一 pipeline？
- 与 Temporal Straightening / Navigable EBM 的 landscape shaping 有何实质不同？
- 与 ensemble uncertainty、pessimistic MPC、adaptive conformal/risk control 是否重合？
- 相比 ordinary candidate-wise rank loss，sequence-level object 是否真有额外收益？
- 能否在第二个明确 multi-route/contact task 上复现，而不只解释 PushT？

如果严格碰撞后没有独立位置，就不应为了延续项目再造名字。

## 5. 明确停止清单

以下实验除非出现新的反例证据，不再调参重跑：

- Horizon-Bundle 的 K/H matching；
- K3/K5/K10 scalar rank average、min-rank union；
- shared-population ensemble 与 ordinary verifier；
- OE-only dynamics fine-tune；
- exact scalar latest/cumulative on-policy aggregation；
- continuous averaged update head；
- raw frozen CLS candidate cost head；
- independent top1 correction router；
- frozen BP branch selector、K10 adoption rule；
- shared-prefix deferred commitment；
- ordinary mixture/multistart 作为主 idea；
- generic persistent-homology loss / component-count matching；
- 只在 final population 上换 K3/K10 scorer。

可保留为 baseline/diagnostic 的有：

- K3 ordinary CEM；
- K10 fixed-bank scorer；
- true/oracle update 与 true refit ceiling；
- K3+K10 signed imagined outcomes；
- branch hidden oracle；
- iso-rate topology 与 paired path interaction；
- same-model multistart、AWM、straightening、uncertainty controls。

## 6. 下一轮进入 GPU 前的 Gate

### Gate 1：严格文献碰撞

先确认 planner-equilibrium / adaptive tail certificate 是否有独立 claim。若只是
AWM 或已有 robust/pessimistic planning 的改写，停止。

### Gate 2：便宜的 held-out ceiling

在现有 row-disjoint query banks 上，比较：

```text
ordinary candidate rank loss
K3/K10 disagreement
uncertainty / calibration baseline
sequence-aware tail-risk or proposal-flow objective
true refit ceiling
```

必须 state-held-out，并直接报告 recursive proposal/support/outcome，而不是只报告
fixed-bank AUC、MSE 或 Spearman。

### Gate 3：最小训练

只有 Gate 2 明显优于普通 rank/uncertainty baseline 后，才训练一个小 checkpoint：

- OE/tail loss 只能是 auxiliary；保留 original prediction replay；
- train/calibration/eval rows 完全分离；
- 预锁定 epoch、risk threshold、planner calls；
- 先 H5/off40，再 H8/off60；
- 必须过 recursive resampling，不能从 fixed trace 直接跳 full MPC。

### Gate 4：方法 promotion

最终至少需要：

- fresh state-paired success 和 true cost 同向；
- iso-WM-calls 优于 K3、multistart、AWM/straightening/uncertainty；
- 第二任务复现；
- 不依赖 simulator/oracle inference selection；
- 原 prediction/dynamics 能力没有显著退化。

## 7. 文档与产物地图

### 当前入口

- **本文件**：当前事实、停止清单与下一 Gates。
- [Horizon-Bundle temporal notebook](horizon_bundle_temporal.md)：从 7 月 17 日起的
  chronological experiment log；用于追溯，不把开头 proposal 当当前结论。
- [World-model literature map](worldmodel_literature.md)：文献与 novelty 边界。

### 主要实验账本

- [A100 Gate A](horizon_bundle_gateA_a100_20260717/README.md)：完整 horizon matrix。
- [5090 OE audit](optimizer_equivalence_5090_20260718/README.md)：selection decomposition、
  oracle refit、recursive intervention、fixed-trace training。
- [A100 set-valued OE audit](optimizer_equivalence_a100_20260720/README.md)：operator、
  branches、fresh selector、topology 与 transport。
- [Paired K3/K10 path audit](optimizer_equivalence_a100_20260720/basin_lineage_pair_a100_20260720/audit_v1/REPORT.md)：
  self-induced tail reversal。
- [Counterfactual refit audit](optimizer_equivalence_a100_20260720/counterfactual_refit_a100_20260721/audit_v1/REPORT.md)：
  final scorer/component causal decomposition。

### 关键复现脚本

- `scripts/plan/cem_round_oracle.py`：完整 CEM populations 与 simulator outcomes；
- `scripts/plan/oe_fixed_trace_train.py`：fixed-trace OE fine-tuning；
- `scripts/plan/oe_set_valued_operator_probe.py`：set-valued operator；
- `scripts/plan/oe_branch_preserving_rollout.py`：recursive branch Gate；
- `scripts/plan/oe_basin_topology_audit.py`：static iso-rate topology；
- `scripts/plan/oe_basin_transport_audit.py`：one-step transport；
- `scripts/plan/oe_paired_basin_lineage_audit.py`：paired path interaction；
- `scripts/plan/oe_counterfactual_refit_eval.py` 与
  `oe_counterfactual_refit_audit.py`：global/component counterfactual refit。

大体积 raw archives 只保存在 5090/A100；仓库中保存 compact JSON、report、必要的
paired/refit NPZ。远端路径与 SHA-256 见各实验目录 README。

## 8. 后续实验统一协议

为了不再因为 protocol drift 产生“看起来又有新结论”，后续统一遵守：

1. **state pairing**：比较必须用相同 rows；shared-bank 与 generator-specific bank
   不能混写成 paired result；
2. **row disjointness**：train/calibration/dev/fresh confirm 明确排除 overlap；
3. **common random numbers**：proposal/candidate 对照尽量共享 noise；
4. **iso-rate / iso-calls**：top-k rate、candidate count、WM calls 分开报告；
5. **oracle labeling**：hidden oracle 只能写 ceiling，不能写 deployable result；
6. **双 outcome**：连续 true cost 与 task success 同时报告；
7. **state bootstrap**：candidate/round 不能冒充独立样本；
8. **locked confirmation**：任何 post-hoc improvement 都必须在第二套 fresh rows 上
   原样确认；
9. **recursive first**：fixed-bank improvement 先过 recursive resampling，再开 full
   MPC/训练；
10. **stop discipline**：未过预注册 Gate 的 branch 不通过继续调同一 bank 来“救”。

## 9. 最终保留的科学贡献种子

即使后续方法仍不 work，这几天已经形成一条可以独立整理的 scientific result：

> **World-model planning failure is an optimizer-feedback problem, not merely a
> prediction-error problem. A model can be more faithful on a fixed candidate
> bank yet lose low-cost-tail fidelity on the proposal distribution induced by
> its own planner. Oracle-equivalent CEM updates have recursive control value,
> while static reranking, branch selection and topology matching do not reliably
> realize that value.**

要把它升成方法论文，还缺一个真正能跨 fresh states、第二任务和强 training
baselines 的 deployable mechanism；在那之前，最诚实的状态是“问题已定位，方法未找到”。
