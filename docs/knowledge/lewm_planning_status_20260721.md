# LeWM planning 研究现状总账（截至 2026-07-23）

> **这是当前结论的唯一入口。** 2026-07-17 至 2026-07-21 的长文档保留为
> chronological lab notebook；其中早期的 `Horizon-Bundle`、`BP-OEWM`、
> `BL-WM` 等名字记录了 idea 演化，不代表仍在执行。若本文与早期段落冲突，
> 以本文和对应的最新 locked/fresh result 为准。

## 0. 当前一句话结论

我们已经确认：**world-model planning 的宏观对象不是 prediction error 本身，而是
prediction、proposal support、adaptive elite/update 与最终 action conversion 组成的
闭环。** 当前最紧凑的表达是：

```text
control value = representation/dynamics fidelity
              × task-aligned candidate support
              × adaptive tail/update validity
              × elite-to-action conversion
```

PushT 上，learned world-model cost 会在 CEM 的 low-cost tail 上产生错误的
elite/update；用真实动力学排序并递归更新 proposal，有显著 outcome ceiling。新加入的
OGBench Cube transfer 又给出两个关键校准：K5 的长时 latent MSE 明显更低，但 H5/H10
closed-loop success 没有稳定提升；同时，K5 在 K1 path 上较好、在自己的 K5 path 上
反而较差的 self-induced tail reversal 跨任务复现。随后完成的 hidden-oracle support
intervention 又把层次进一步拆开：保证 task-aligned proposal support 能把 K1/K5
closed-loop success 从 `37.3%/38.0%` 提高到 `69.3%/65.3%`，但没有解锁 K5 winner；
高 support 的 CEM path 仍会随迭代侵蚀可成功 mean。

但这几天测试的实现都没有把这个 oracle ceiling 稳定转成 deployable method：

- 没有稳定的 horizon matching；
- post-hoc verifier、scalar ensemble、continuous correction head 不 work；
- set-valued branches 能生成有价值的候选，但 learned selector 在 fresh data 上不复现；
- true landscape 确实 multi-basin，但 component-wise refit 只带来极小增益；
- K10 在自己诱导的 query path 上发生 tail-fidelity reversal，但最后一轮换成 K3
  scorer 几乎救不回来，说明损失是沿 proposal flow 累积的，不是一个 final selector
  patch 可以修复；
- OGBench 的 vanilla Gaussian CEM 在 late rounds 只有约 `31–38%` states 含成功
  candidate；K5 path 虽含略好的 oracle candidate，却没有转成更好的 mean action 或
  closed-loop success，说明 proposal support 与 conversion 也是独立瓶颈。
- exact future-action prior 把 final candidate support 提到 `87.5–93.8%`，并让两模型
  closed-loop success 显著提高；但 K5−K1 仍为 `-4.0pp [-10.0,+2.0]pp`，且
  candidate mean 从 round 0 的 `93.8–100%` 成功降到 round 29 的 `37.5–56.2%`。
- 最乐观的 deployable tail-feedback Gate 也已关闭：执行一个 action block 后的真实
  latent residual 不能预测下一次误排；把它作为持久加性修正放进全部 30 轮 CEM，
  held-out recall 不升反降。逐 state 真值事后挑正 alpha 的 fixed-pop recall ceiling
  也只有 `+0.015 [0.009,+0.022]`，低于预锁定 `+.05` 否决线。

因此截至现在：

```text
scientific problem / oracle target                         SUPPORTED
current deployable method                                  NOT FOUND
Horizon-Bundle                                             STOP
OE-only dynamics fine-tuning                               STOP
frozen BP-OE branch patch                                  STOP
topology / BL-WM as main headline                          DOWNGRADE TO DIAGNOSTIC
cross-task adaptive tail reversal                          SUPPORTED AS DIAGNOSTIC
proposal support as causal control lever                   SUPPORTED
next deployable gate                                       GCBC prior + update-retention/early-stop
executed-prefix additive tail feedback                     CLOSE
tail-validity method line                                  CLOSE -> PROBLEM-DEFINITION PAPER
```

现在**不应该直接开一个大规模 topology/tail training**，也不应该继续调 frozen
selector。oracle support Gate 已说明 support 很重要，也说明它不是 K5
prediction/control gap 的唯一 missing link；随后完成的文献碰撞与 honest-feedback
否决实验又关闭了当前 tail-validity 方法线。GCBC prior + early-stop/retention 仍可作为
deployable planning baseline 补齐，但不再是自动重开新 WM objective 的跳板。主线转为
① learned-vs-true low-tail 测量、②误排沿 proposal flow 的累积规律与 oracle ceiling。

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
| OGBench Cube transfer | 长时 prediction 优势能否跨 benchmark 转成 control | K5 H10 latent MSE 低 22.5%，但 H5/H10 success 无显著优势；self-path tail reversal 复现，且 natural proposal support 低 | prediction/control gap 与 cross-task diagnostic `SUPPORTED` |
| OGBench support intervention | proposal support 是否 causal、能否解锁 K5 | hidden expert prior 使 K1/K5 success `+32.0/+27.3pp`；K5 未成为 winner；高-support mean 仍随 CEM 侵蚀 | support causal `SUPPORTED`；下一步 GCBC prior + update-retention |
| Tail-validity feedback Gate | 已执行前缀的诚实残差能否预测并环内修正下一次 low-tail 误排 | 60-state OOF prediction 失败；fixed-pop positive-alpha oracle gain 仅 `+.015`；recursive recall `-.006` | 当前 additive feedback 方法族 `CLOSE`；转问题定义线 |

这条演化不是“所有 idea 都失败了所以又换名字”。真正逐步收紧的是因果对象：

```text
representation / dynamics fidelity
  -> task-aligned candidate support
  -> CEM tail ranking and sufficient-statistic update
  -> self-induced adaptive proposal path
  -> elite-to-action conversion
  -> closed-loop task outcome
```

现在不是继续向某个更极端的内部模块钻，而是用便宜干预拆清这四层中哪一层先限制
control。adaptive tail 是跨任务复现的真实现象，但还不能被直接升级成唯一主因或完成的
方法 idea。

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

### 2.10 OGBench transfer：MSE 不转 success，但 self-path reversal 跨任务复现

在 OGBench Cube Single 上，先用 `3 eval seeds × 50 paired episodes` 比较同一训练
seed 的 K1/K5。共享 starts、goals 与 CEM noise：

| protocol | K1 success | K5 success | K5−K1 |
|---|---:|---:|---:|
| H5，全部 150 | `62.7%` | `63.3%` | `+0.7pp [-4.7,+6.0]` |
| H5，排除 initial-easy | `41.1%` | `42.1%` | `+1.1pp [-7.4,+9.5]` |
| H10，全部 150 | `43.3%` | `43.3%` | `0.0pp [-6.0,+6.0]` |
| H10，排除 initial-easy | `30.3%` | `30.3%` | `0.0pp [-7.4,+7.4]` |

与此同时，fixed expert-action endpoint latent MSE 在 H10 上是 `.00703` 对
`.00545`，K5 低 `22.5%`；H20 上约低 `40%`。因此这是明确的
prediction/control non-conversion，而不是模型指标本身没有差异。

随后对 16 个 fresh nontrivial states 保存 K1/K5 完整 CEM path，在 steps
`0/4/9/19/29` 重放每轮 300 个自然 candidates，并 post-hoc 注入 32 个有效的 expert
neighborhood anchors。自然 population 到 final round 只有 `.312/.375` 的 states 含
任一成功 candidate；K5 path 的 oracle-min distance 反而略好
`-.0064 [-.0144,-.0006]`，但 actual mean 都只有 `3/16` 成功。

在 support-controlled bank 上，top-30 真实成功质量出现与 PushT 同构的 path
interaction：

```text
(K5−K1 scorer gap on K5 path) − (gap on K1 path)

step 0     .000
step 4    -.060 [-.152,-.004]
step 9    -.110 [-.244,-.023]
step 19   -.233 [-.465,-.083]
step 29   -.246 [-.504,-.073]
```

K5 final scorer 在 K1 path 的 top-30 成功率是 `.579`，在自己的 K5 path 上却只有
`.487`；对应 K1 scorer 是 `.483/.637`。这支持 self-induced adaptive-tail reversal
跨任务存在，但也同时显示：低 natural support、tail erosion 与 mean conversion 是三层
不同问题，不能只凭这项 interaction 宣称已找到单一 causal repair。

完整协议、paired bootstrap、审计与 artifacts 见
[OGBench Cube transfer audit](ogbench_cube_transfer_20260722/README.md)。

### 2.11 OGBench support intervention：support 是 causal，但 adaptive erosion 独立存在

用 exact future expert continuation 只初始化第一次 CEM mean，在 fresh
H10/budget50 四格中保持 `300 candidates × 30 rounds`、WM calls、models 与 150
paired states 不变。这是 hidden-oracle causal diagnostic，不是 deployable benchmark
result；绝对值不与前一份 budget100 H10 baseline 混比。

| condition | K1 | K5 | K5−K1 |
|---|---:|---:|---:|
| zero | `37.3%` | `38.0%` | `+0.7pp [-4.0,+5.3]` |
| expert prior | `69.3%` | `65.3%` | `-4.0pp [-10.0,+2.0]` |
| prior−zero | `+32.0pp [24.0,40.0]` | `+27.3pp [19.3,35.3]` | DID `-4.7pp [-12.7,+3.3]` |

排除 initial-easy 后，K1/K5 gain 为 `+39.3/+34.4pp`，且连续 final distance 同向
下降。proposal support 因而是很大的 causal lever；但 K5 没有得到更大的 gain，较低
latent MSE 仍未转成 robust control winner。

同一 16-state candidate path 上，prior 将 round-0 support 提到 `100%`，但 actual
mean success 从 `93.8%/100%` 逐轮降到 final `37.5%/56.2%`，此时 support 仍有
`87.5%/93.8%`。support-controlled self-path interaction 也仍为
`-.1125 [-.2187,-.0292]`。这比 baseline 数值上缓和，却证明 support、adaptive
tail/update 与 mean conversion 是三个可独立失效的环节。

完整结果见
[OGBench proposal-support causal gate](ogbench_cube_support_intervention_20260722/README.md)。

## 3. 当前统一解释

### 3.1 已支持的因果链

```text
learned dynamics / representation
  -> task-aligned candidate support (may already be missing)
  -> distorted low-cost candidate tail
  -> wrong CEM elite sufficient statistics
  -> proposal moves into a model-specific query distribution
  -> later tail fidelity/support deteriorate
  -> elite mean may fail to convert hidden candidate quality
  -> final success/cost gap
```

其中每一箭头的证据强度不同：

- `wrong elite -> wrong update`：强；
- `oracle update -> better recursive outcome`：强 true-cost、small-N success evidence；
- `model-specific path -> tail reversal`：强 observational interaction；
- `prediction MSE -> control success`：OGBench 给出明确反例；
- `proposal support -> closed-loop outcome`：OGBench hidden-oracle intervention 给出强
  paired causal evidence；
- `high support -> adaptive erosion / conversion failure`：OGBench intermediate mean
  replay 给出直接证据，但尚无 deployable repair；
- `tail reversal -> final failure`：存在 headroom，但 K3 swap 与 support prior 都不是
  充分 repair；
- `basin multiplicity -> final failure`：当前 PushT evidence 弱。

### 3.2 为什么以前看起来互相矛盾

以下三件事可以同时为真：

1. K10 在固定 K3 bank 上比 K3 准；
2. K10 在同 proposal 的 one-step refit 上也略好；
3. K10 完整递归后反而较差。

因为 scorer 不只是评估外生数据；它通过 CEM 改变下一轮数据。fixed-bank fidelity
不是这个 coupled dynamical system 的稳定性保证。另一方面，简单 on-policy replay
没有 work，说明“多收自己路径上的数据”也不是充分解法。

### 3.3 目前不能再把缺口压成单个模块

不是：

- 更长 prediction horizon；
- 再加一个 verifier；
- scalar ensemble；
- final-round scorer swap；
- ordinary multistart；
- generic PH/topology loss；
- 只拟合一个平均 correction vector。

现在能确定的缺口是端到端 **decision fidelity**，但还不能把它等同于一种 tail
training objective。OGBench 要求先区分：

1. proposal 根本没有覆盖 task-relevant success region；
2. 有 support，但 adaptive scorer/update 把 success tail 挤出；
3. population 内有好 candidate，但 Gaussian mean/refit 没有转成有效 action。

hidden-oracle intervention 已经确认：受控增加 support 后第 2/3 层仍持续。随后的一票
否决实验又确认：最密、最低延迟的诚实执行反馈仍不足以校准下一次 tail rank。因此
sequence-level tail/update validity 是成立的科学对象，但当前没有被证据授权的方法
objective。

## 4. 当前 working question：问题定义与测量

暂时不再给新方法取 acronym。当前主问题收紧为：

> learned-vs-true low-tail 误排如何沿 CEM elite/refit proposal flow 累积；这个规律能否
> 跨 PushT 与 OGBench 复现；在 support、rank 和 mean conversion 三层分别有多大的
> oracle ceiling？

下一阶段优先把现有仪器整理成问题定义论文的证据链：

1. 用统一的 state-paired 指标报告每轮 support、true-top-k recall、oracle minimum、
   refit mean 与 closed-loop outcome，不再用 global Spearman 代替 low-tail 测量；
2. 在 PushT 与 OGBench-Cube 上复现 proposal-flow 累积曲线，并明确
   prediction fidelity、support、rank 与 conversion 的条件关系；
3. 把 true refit、hidden support prior、final scorer swap、executed-residual feedback
   放进同一因果上限表，区分“问题存在”“可部署信号存在”“方法能转成 outcome”；
4. 保留 GCBC prior + early-stop/retention 作为 deployable planning baseline；它若有效，
   记作强 baseline，而不是自动升级成新 WM 方法。

当前关闭的是预注册的 one-block persistent-additive feedback family，不是关于所有可能
反馈函数的不可行性定理。但除非出现一个事前定义、只用部署时可得信息、在 held-out
states 上同时提升 rank 与 returned action 的新信号，否则不重开
proposal-flow training / adaptive tail-risk certificate 方法线。

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

## 6. 已完成 Gate 与剩余工作

### Gate 0：hidden-oracle support intervention（已完成）

exact expert prior 在 iso-WM-calls 下让 K1/K5 closed-loop success 分别提高
`+32.0/+27.3pp`，同时 final candidate support 提高 `+56.3pp`。但 K5 未成为 winner，
且 high-support mean 随 CEM rounds 明显恶化。判决：support causal `SUPPORTED`；
support-only explanation `REJECTED`。

### Gate 1：deployable action prior 与 update-retention

训练或复用 current-observation/goal-only GCBC prior，严格禁止 test-time future action。
保持 K1/K5 checkpoints、paired rows、candidate count 与 outcome 指标不变，比较：

```text
zero vs GCBC initial proposal
1 / 5 / 10 / 30 CEM rounds
ordinary update vs prior-retention/trust-region
```

先确认 deployable support gain，再判断简单 early stopping/retention 是否已能保住
outcome。若可以，就把它作为 planning baseline，不进入新 WM training。

### Gate 2：文献碰撞（已完成）

15-work 矩阵确认 ① learned-vs-true low-tail 误排与 ② proposal-flow 累积没有被
直接覆盖；但 ACID、IMWM、performative prediction 与 adaptive conformal 已占据相邻
方法词汇。三审稿人均分 `6.3/10`，只给出“诚实执行反馈先过一票否决实验”的窄门。

### Gate 3：honest executed-prefix feedback（已完成，`CLOSE`）

60-state、5-fold held-out、20,000 次 state bootstrap 的 A100 实验未发现可预测或可
修正信号。fixed-pop per-state positive-alpha oracle gain 为
`+.015 [+.009,+.022] < +.05`；recursive recall 为
`-.006 [-.018,+.003]`，returned action 也无改善。详见
`tail_validity_feedback_gate_20260723/REPORT.md`。

### Gate 4：问题定义证据链（当前）

不训练新 checkpoint；统一 PushT/OGBench 的 round-wise 仪器与因果上限表，直接报告
support、true-top-k recall、oracle minimum、refit mean 和 closed-loop outcome。GCBC
prior + retention 可补作 deployable planning baseline，但不承担“救活方法线”的任务。

### Gate 5：方法线重开条件

仅当出现一个事前锁定、只使用部署时可得信息的新信号，并在 fresh held-out states 上
同时满足以下条件时重开：

- low-tail rank 相对普通 K3/uncertainty baseline 有非零且有量级的 paired 增益；
- recursive returned true cost 或 success 同向改善；
- 第二任务复现且不依赖 simulator/oracle inference selection；
- 原 prediction/dynamics 能力没有显著退化。

## 7. 文档与产物地图

### 当前入口

- **本文件**：当前事实、停止清单与下一 Gates。
- [Horizon-Bundle temporal notebook](horizon_bundle_temporal.md)：从 7 月 17 日起的
  chronological experiment log；用于追溯，不把开头 proposal 当当前结论。
- [World-model literature map](worldmodel_literature.md)：文献与 novelty 边界。
- [Tail-validity 碰撞判决](tail_validity_verdict_20260723.md)：文献窄门、三审稿人
  判决与 honest-feedback 否决实验回填。
- [Tail-validity feedback Gate](tail_validity_feedback_gate_20260723/REPORT.md)：A100
  60-state compact 结果；协议见
  [locked protocol](tail_validity_feedback_gate_protocol_20260723.md)。

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
- [OGBench Cube transfer audit](ogbench_cube_transfer_20260722/README.md)：closed-loop
  success、natural support、MuJoCo candidate replay 与 cross-task path interaction。
- [OGBench proposal-support causal gate](ogbench_cube_support_intervention_20260722/README.md)：
  hidden-oracle support intervention、150-state paired outcome 与 high-support path erosion。

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
- `scripts/plan/ogbench_candidate_fidelity.py`：OGBench 完整 CEM population 与 MuJoCo
  candidate replay；
- `scripts/plan/summarize_ogbench_candidate_fidelity.py`：state-paired support、fidelity、
  path interaction 与 conversion summary。
- `scripts/plan/compare_ogbench_support_intervention.py`：zero/prior candidate-path paired
  comparison；
- `scripts/plan/summarize_ogbench_support_closed_loop.py`：四格 closed-loop paired
  bootstrap、McNemar 与 difference-in-differences。
- `scripts/plan/tail_validity_feedback_gate.py`：执行前缀残差 + 下一次 recursive CEM
  四 arm 采集；
- `scripts/plan/summarize_tail_validity_feedback_gate.py`：state bootstrap、5-fold
  held-out alpha 与预锁定 OPEN/HOLD/CLOSE 判决。

大体积 raw archives 只保存在对应 5090/A100 server；仓库中保存 compact JSON、report、
必要的 paired/refit NPZ。远端路径与 SHA-256 见各实验目录 README。

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
10. **support/conversion decomposition**：分别报告 population 是否含成功 candidate、
    scorer elite 质量、refit/mean action 与 executed outcome；
11. **prediction/control separation**：MSE 只能作为 mechanism metric，方法 promotion
    必须看 paired closed-loop success；
12. **stop discipline**：未过预注册 Gate 的 branch 不通过继续调同一 bank 来“救”。

## 9. 最终保留的科学贡献种子

即使后续方法仍不 work，这几天已经形成一条可以独立整理的 scientific result：

> **World-model planning value is not determined by prediction error alone. It is
> jointly limited by task-aligned candidate support, low-cost-tail validity on
> the planner's self-induced proposal distribution, and elite-to-action
> conversion. A model can have lower long-horizon prediction error and be more
> faithful on another model's candidate bank, yet gain no closed-loop success and
> lose tail fidelity on its own adaptive path. A controlled support intervention
> materially raises success, yet does not unlock the lower-MSE model as a winner
> and still permits high-support plans to erode under repeated CEM updates. This
> interaction now appears on both PushT and OGBench Cube; oracle updates expose
> control headroom, while static reranking, branch selection and topology matching
> do not reliably realize it.**

要把它升成方法论文，还缺一个真正能跨 fresh states、两任务和强 planning/training
baselines 的 deployable mechanism；在那之前，最诚实的状态是“现象与边界已定位，
方法未找到”。
