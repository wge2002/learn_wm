# A100 set-valued optimizer-equivalence audit（2026-07-20）

> 当前跨机器总判决、停止清单与下一 Gates 见
> [`../lewm_planning_status_20260721.md`](../lewm_planning_status_20260721.md)。
> 本文保留 A100 逐 Gate 的详细实验账。

## 1. Protocol

本目录保存本轮 A100 的 compact reports / JSON；大体积 candidate、dense token
和 imagined-outcome caches 未纳入仓库。

共同主单元：

```text
PushT H5 / goal offset 40 / K3 generator
240 independent states × 4 CEM rounds × 300 complete candidates
top-k = 30
outer 3-fold state holdout
inner state holdout selects epoch and residual blend
state-paired 20,000-sample bootstrap
```

基线 frozen LeWM：

```text
update cosine          .181
relative update error  1.141
elite overlap          .190
```

`top1` 是 learned single-route output。`top2 coverage` 与
`all-mode coverage` 都用 oracle 在已保留分支中挑最好者，**只测 branch set
是否覆盖正确 basin，不是 deployable planner 结果**。

## 2. Correction geometry

H5 的四个非零 correction prototypes 不是任意 50-D 噪声：

- 前两个奇异轴解释 `98.75%` prototype energy；
- 两对 prototype 近似反向，pair cosine 为 `-.963 / -.981`；
- 作用形态近似长时域的 `±x / ±y` action correction；
- 相邻 CEM rounds 保持同一 mode 的比例为 `61.25%`；
- 四轮完全相同为 `28.3%`，显著高于平衡四类的 chance。

因此单向量平方损失产生 conditional mean、并把相反 correction 抵消，是直接
operator head 失败的机制，而不只是容量或学习率问题。

## 3. Main results

下表统一写成 `Δ cosine / Δ relative error`；cosine 越高越好，relative error
越低越好。

| observable / model | top1 | top2 coverage | all-mode coverage |
|---|---:|---:|---:|
| raw frozen CLS latent, M=5 | `+.007 / +.001` | `+.051 / -.034` | `+.106 / -.073` |
| privileged true state, M=5 | `+.058 / -.032` | `+.113 / -.081` | `+.174 / -.128` |
| frozen dense patch grid, M=5 | `+.012 / -.043` | `+.058 / -.075` | `+.111 / -.111` |
| K3 vector imagined outcome, M=5 | `+.025 / -.047` | `+.078 / -.083` | `+.122 / -.112` |
| K3+K5 vector outcomes, M=5 | `+.037 / -.051` | `+.087 / -.085` | `+.147 / -.124` |
| K3+K10 vector outcomes, M=5 | **`+.045 / -.033`** | **`+.097 / -.079`** | **`+.164 / -.125`** |
| dense + K3/K5/K10 vectors, M=5 | `+.034 / -.026` | `+.091 / -.074` | `+.165 / -.125` |

重要分解：

1. `M=1` dense readout 仅为 `+.006 / -.017`；改善不是普通更强 regressor。
2. raw latent 的 learned top1 best-mode rate 为 `21.1%`（chance `20%`）；
   dense patch 约 `27%`，privileged state 约 `28%`。dense read path 确实恢复了
   一部分 spatial state，但 mode 仍主要取决于 counterfactual population。
3. K3+K10 的 signed imagined-outcome vectors 把 deployable top1 提到
   `+.045 [.024,.066]`，是当前最强单路 signal；只保留多个模型的 scalar
   cost/rank 时 top1 为 `-.009 / -.009`。有效信息来自**向量化 counterfactual
   disagreement**，不是普通 score ensemble。
4. 把 K3/K5/K10 scalar rank 直接平均在此前完整 MPC 中曾崩到
   `6/14/4%`；这里没有平均模型意见，而是用 disagreement 形成独立 proposal
   branches。

## 4. Falsified alternatives

以下路线没有通过 locked fixed-trace Gate：

- exact one-mode router：raw latent `Δcos=+.017`，privileged state
  `+.016`；true state 也不能从 state alone 决定 population-specific mode。
- candidate cost head + vector imagined outcome：四 seeds
  `Δcos=+.007~+.031`，但 `Δrel=+.042~+.064`，方向改善来自更激进、错误的
  reranking。
- one-step observable latent innovation：M=5 top1 `+.005 / -.011`；
  加 outcome 与 dense 后仍只有 `+.014 / -.033`。
- regret-aware / softmin router：没有超过 winner-CE，部分设置退化到 chance
  以下；不是把 CE 换成另一个 routing loss 就能解决。
- frozen dense spatial moments + structured terminal/relative dynamics：
  `Δcos≈0`，`Δrel=+.147~+.184`；一个浅层 task-pose sidecar 不足以泛化
  counterfactual contact dynamics。

这些负结果共同排除：

```text
raw CLS bottleneck alone
state geometry alone
last-step model residual alone
scalar cross-model disagreement alone
another independent candidate reranker
another top1 routing loss
```

## 5. Surviving idea

暂定名称：

> **Branch-Preserving Optimizer-Equivalent World Model（BP-OEWM）**

核心对象不是一个 corrected cost，也不是一个平均 update，而是给定完整 planner
population `q` 后的一组 optimizer sufficient-statistic updates：

```text
Fθ(o, g, q) = { (Δμm, Δlogσm, πm) }m=1…M
```

训练使用 best-of-M optimizer-equivalence loss，确保至少一个分支逼近 true-dynamics
诱导的 CEM update；输入保留多个 world model 的 signed imagined outcomes，使
counterfactual disagreement 不会在 scalar rank 或 Gaussian mean 中提前消失。

它与当前证据一致：

- 单路 conditional mean 被反向 modes 抵消；
- branch set 已达到 `Δcos=+.164 / Δrel=-.125` 的 held-out coverage；
- cross-model vectors 提供可泛化 routing signal，但不足以支持每轮硬 top1；
- 因而 inference 必须递归保留 2–5 个 proposal branches，不能把 oracle coverage
  误写成完成的方法结果。

## 6. Next locked test

下一步不再调 fixed-trace top1 head，改做 disjoint 12-state recursive Gate：

```text
train/calibration: current 240-state bank
evaluation: week1 H5/off40 12 states, row overlap = 0
branch input: K3 + K10 signed imagined terminal vectors
planner: carry B=2 branches recursively
total candidates per round = 300 (150 per branch)
controls:
  ordinary K3 1×300
  same-model multistart 2×150
selection must use model-side signals only
```

必须同时报告：

- learned final selection；
- retained two-branch oracle union（只作 ceiling）；
- final true cost、success、coverage、branch collapse/diversity；
- matched K3/K10 world-model calls。

Promotion 条件：

```text
learned branch planner beats ordinary K3 and 2×150 multistart
gain survives recursive re-sampling, not only fixed trace
no simulator/oracle branch selection at inference
H5 positive before opening H8 or full MPC
```

若只有 oracle union 提升、任何 deployable branch reduction 都不提升，则
BP-OEWM 仍只是一个 uncertainty ceiling，不进入完整训练。

## 7. Row-disjoint recursive Gate（两轮随机 population）

已在与 240-state 训练 bank **零 row overlap** 的 12 个 H5/off40 状态上，从
saved step-4 proposal 递归重采样 3 轮。每格同时保存 candidate-level hidden
simulator coverage，但 simulator 不参与 update、pruning 或 deployable selector。

主等调用协议为：

```text
K3 baseline:       1 branch × 300 K3 calls
K3 multistart:     2 branches × 150 K3 calls
BP matched:        2 branches × 75 candidates × (K3+K10)
BP full ceiling:   2 branches × 150 candidates × (K3+K10)
```

下表是 final branch mean 的 paired true-cost delta，相对 K3 `1×300`；负数更好。
`v1` 的方法间 population 独立，`v2` 使用嵌套 common random numbers：

| method / selector | v1 delta (wins/12) | v2-CRN delta (wins/12) |
|---|---:|---:|
| K3 `2×150` primary | `+4.26` (4) | `+2.61 [.32,5.19]` (3) |
| BP matched primary | `+2.07` (5) | `-2.03 [-9.89,5.79]` (8) |
| BP matched K10 selector | `-.41` (8) | `-2.87 [-10.39,4.75]` (8) |
| BP matched oracle union | `-.87` (8) | `-3.37 [-10.87,4.12]` (8) |
| BP full K10 selector | `-13.81` (9) | `-3.32` (7) |
| BP full oracle union | `-16.92` (9) | `-7.23` (8) |

判读：

1. matched BP 的第二轮 point estimate、median 和 wins 都为正向，且 K10 selector
   已取回大部分 oracle-union ceiling；因此 recursive resampling 没有直接证伪
   branch mechanism。
2. matched primary 在两轮分别为 `+2.07/-2.03`，CI 均跨零，**尚未达到 work
   标准**；不能用第二轮单独宣称胜过 ordinary K3。
3. BP full 在两轮都保留更强 oracle ceiling，但 v1 的均值被一个困难状态
   `-116.5` rescue 明显放大；在补 K3 `1×600/2×300` 前，它只是一条
   compute-unmatched mechanism ceiling。
4. same-model `2×150` 两轮都差于 ordinary `1×300`；当前信号不能简化成普通
   multistart。

Compact shards 与 pooled paired reports 位于
`bp_oe_recursive_a100_20260720/`。

## 8. Sparse cross-horizon query：10% K10 已保留大部分 branch signal

当前 matched BP 为支付 K10，把每轮独立 physical candidate 数从 300 降到
150。为区分“branch idea 不行”与“预算分配太差”，新增严格 deployable
query Gate：

```text
K3 evaluates all 300 candidates
K10 evaluates only K3-selected 30 or 60 candidates
K10 cost normalization/rank uses only the queried subset
unqueried K10 vector features are masked explicitly
outer state-cross-fit protocol unchanged
```

最强且最省的 `K3 elite 30 / 300` 得到：

| output | Δ update cosine | Δ relative error |
|---|---:|---:|
| learned top1 | `+.026 [.010,.043]` | `-.053 [-.063,-.042]` |
| top2 coverage | `+.075 [.059,.091]` | `-.086 [-.096,-.077]` |
| all-mode coverage | `+.133 [.119,.148]` | `-.123 [-.132,-.114]` |

相对 full K3+K10 vector 的 cosine gain，30 个 K10 queries 保留约
`58% / 77% / 81%` 的 top1 / top2 / all-mode signal，却只增加 `10%`
world-model calls。elite-30 也优于本轮的 rank-stratified-30，说明有用的
cross-horizon vector disagreement 主要集中在 K3 即将用于 update 的候选上，
不是必须重算整个 population 的普通 ensemble。

这把下一条高优先级方案从 `150 K3 + 150 K10` 改成：

```text
first recursive round:  272 K3 + 27 K10 = 299 calls
after B=2 split:        2 × (136 K3 + 14 K10) = 300 calls
```

对应 12-state / 3-round sparse recursive Gate 的结果为：

| selector | paired true-cost delta vs K3 `1×300` | wins/12 |
|---|---:|---:|
| learned cumulative branch score | `+3.91 [-4.21,11.58]` | 4 |
| final K3 score | `+4.11` | 3 |
| final K10 score | `-1.59 [-8.34,4.97]` | 6 |
| hidden oracle union | **`-11.04 [-29.66,.87]`** | 7 |

oracle-union 的 median/trimmed delta 为 `-4.23/-4.16`，相对 same-model
`2×150` 为 `-13.65 [-31.79,-1.71]`、9/12 wins。也就是说，稀疏 K10 query
确实生成了有意义且不是普通 random restart 的 branch set；失败点已收窄为
**recursive branch selection**。当前 operator log-probability 只学习“哪条局部
CEM update 更像 oracle moment”，与三轮递归后的真实 branch value 不对齐，
所以 learned primary 反而恶化。

## 9. 600-call 等算力对照：branch ceiling 不是多采样造成的

为关闭 `bp_full` 每轮使用 600 次 WM calls 的混淆，在同一 12 states、相同
proposal noise 上补了 K3 `1×600` 和 `2×300`：

| method / selector | delta vs K3 `1×300` | delta vs K3 `1×600` |
|---|---:|---:|
| K3 `1×600` | `-.01 [-1.60,1.60]` | — |
| K3 `2×300` | `+1.42` | `+1.43` |
| BP full K10 selector | `-3.32` | `-3.31` |
| BP full oracle union | `-7.23` | `-7.21` |

K3 从 300 加到 600 calls 几乎完全饱和，same-model `2×300` 还略差；BP full
的 K10 point estimate 与 oracle ceiling 在等 calls 后仍保留。更强的是，
只有约 300 calls 的 sparse BP oracle 相对 K3 `2×300` 仍为
`-12.46 [-31.18,-.04]`。因此困难状态 rescue 来自 cross-horizon
disagreement 诱导的不同 basin，而不是“多抽一倍 candidates”。

但 deployable K10 selector 的区间仍跨零，这仍然不是 work。下一步直接用
OOF 生成的 240-state recursive branch pairs 与其最终 simulator outcome
训练 outcome-aligned value selector，并锁定后在全新、与训练及此前 12 states
都 row-disjoint 的 60-state H5 source 上只开一次测试。同时保存 proposal
轨迹，审计 branches 是否有足够共享 action prefix 支持 observation-conditioned
deferred commitment。

## 10. 两套 fresh 60-state selector Gate：局部 ceiling 真实，静态选择不 work

用 model-only collector 在不到一分钟内各生成两套新 H5/off40 source；第二套同时
排除 training240、旧 eval12 和第一套 fresh60。两个 source 都有 60 个 unique
rows、全部 overlap 为 0。240-state selector bank 则由四个 OOF operators 生成，
每个 operator 排除自己负责标注的 60 rows。

selector 的 feature family / target / ridge strength 只由四折 shard-held-out CV
锁定：

```text
features = recursive action/proposal trajectory
target   = clipped final true-cost difference
alpha    = .1
OOF accuracy = 60.8%
OOF delta vs operator primary = -1.38
```

锁定后的两套 fresh 结果却不一致：

| result vs K3 `1×300` | fresh A | fresh B confirm |
|---|---:|---:|
| learned branch-value selector | `-2.83 [-7.14,1.61]` | `-.04 [-6.44,6.24]` |
| static final K10 selector | `+.73` | `-1.13` |
| hidden two-branch oracle | `-4.13 [-8.40,.23]` | `-4.72 [-10.42,.85]` |

第一套上 learned selector 取回约 `69%` oracle mean ceiling，甚至显著优于
operator primary；但第二套完全没有复现。不能把 fresh A 单独写成方法结果。

第一套 post-hoc 暴露了一个正确的结构修正：**不要用 correction branches 替换
ordinary K3，要显式保留 no-op/base anchor**。`base + two BP branches` 的 hidden
oracle 在 fresh A 为 `-8.46 [-11.32,-5.84]`。据此预锁定无阈值规则：

```text
run K3 1×300 anchor + sparse BP 300 calls
BP 内部仍用 locked learned selector
仅当 BP branch 的 final K10 cost < anchor 的 final K10 cost 时采用 correction
iso-compute control = K3 1×600
```

第二套确认结果为：

| confirm-B result vs K3 `1×600` | true-cost delta | actual success |
|---|---:|---:|
| locked anchor adoption gate | **`+.98 [-3.86,6.38]`** | `26.7% vs 33.3%` |
| anchor + two BP hidden oracle | `-6.31 [-9.47,-3.52]` | `33.3% vs 33.3%` |

因此 adoption gate 明确失败；即使 hidden oracle 对连续 true cost 有稳定 headroom，
也没有提升确认集 success。两条 branch 在 action horizon 上的 normalized RMS
divergence 从第一 block 到最后都约 `.5`，first-two/last-two ratio 在
train/A/B 为 `1.003/.969/.995`，所以 observation-conditioned deferred
commitment 也关闭。

当前最终判决：

```text
set-valued discrepancy branches expose real lower-cost basins   SUPPORTED
static recursive selector generalizes                          REJECTED
base-anchor K10 adoption rule                                  REJECTED
shared-prefix deferred commitment                              REJECTED
BP-OE as a frozen-planner patch                                STOP
```

剩下值得继续的不是再调一个 selector，而是把证据提升为更高层训练问题：
**真实 planning landscape 的 basin multiplicity 是否被更 smooth 的 WM
merge/drop？** 下一步先做 iso-rate basin-topology audit，并直接对账
Adversarial World Modeling / temporal straightening；只有确认 topology mismatch
与 success failure 相连，才训练 topology-preserving WM。

## 11. Basin-topology audit：静态拓扑有信号，但不能停在静态 landscape

在同一 `240 states × 4 rounds × 300 candidates` 上，用相同 action graph 分别对
true、K3、K10 cost 做 lower-star `H0` persistence 和 top-rate connected
components。cost 全部先转 rank，因此比较不受 latent-distance scale 影响；每个
state 的 correction basis 只由其它三折训练，并补了 full 50-D action geometry
与 matched random-2D control。

主结果（`kNN=12`、top/persistence rate `.1`）：

| geometry | true top-10% components | K3 component coverage | K10 component coverage | K3 successful-basin coverage | K10 successful-basin coverage |
|---|---:|---:|---:|---:|---:|
| held-out correction 2-D | `4.97` | `.400 [.375,.424]` | `.499 [.474,.525]` | `.454 [.426,.483]` | `.520 [.490,.551]` |
| full whitened action 50-D | `4.50` | `.432 [.400,.464]` | `.544 [.516,.572]` | `.552 [.513,.590]` | `.627 [.594,.661]` |
| random 2-D control | `12.58` | `.308 [.285,.331]` | `.435 [.411,.460]` | `.383 [.357,.409]` | `.483 [.452,.515]` |

static elite recall 同样是 K10 `.296 [.276,.316]` 高于 K3
`.190 [.174,.206]`；successful-candidate recall 为 `.305 vs .223`。
这确认两个模型都只覆盖真实 basin set 的一部分，且 K10 在固定 K3 population
上的 ranking/topology fidelity 更好。

但最初“`K3 fragments / K10 merges` 是全局偏差”的说法需要收回一半：

- correction 2-D 中 K3 persistent count 比 true 多 `+.50 [.15,.85]`，top-10%
  components 多 `+1.34 [.93,1.74]`；这是 optimizer-discrepancy 子空间里的
  structured fragmentation。
- full 50-D 中两者都明显 merge/drop；K3 top-10% count delta
  `-3.00 [-3.25,-2.74]`，K10 为 `-1.15 [-1.42,-.88]`。
- random-2D 虽然 component count 接近，persistence fidelity 对 shuffle 只有
  `.024/.052`；“数目碰巧相同”不等于 basin correspondence。

因此可以支持的是：

```text
true planning surface is multi-basin                         SUPPORTED
K3/K10 miss many true and successful basins                 SUPPORTED
K10 is the better fixed-population topology/ranking model   SUPPORTED
K3 globally fragments while K10 globally merges             REJECTED
```

同时测试了两个无需训练的 cross-horizon scalar combination。average rank 的
persistent count 更接近 true，但 top-10% recall 只有 `.235`，低于 K10 `.296`；
`min(K3 rank,K10 rank)` 把 persistent spurious count 放大到
`+1.15 [.77,1.54]`，top-10% recall `.263`。也就是说，horizon errors 有结构，
但 union/average 仍不能直接变成方法。

## 12. Iso-rate update transport：一轮也更偏向 K10，缺口在 on-policy lineage

又把“static scorer 更好”推进到 CEM sufficient statistic：分别用 K3、K10 与
rank-consensus 的 top-30 refit 下一轮 diagonal Gaussian，再在旧 population 上按
新 proposal likelihood 取等量 30 个 support witnesses。

在 correction/full/random 三种 geometry 下，K10 的 true-component coverage
都略高于 K3：

```text
correction: .427 vs .410
full:       .419 vs .413
random:     .279 vs .235
```

successful-component coverage 也同向；K10 refit 相对 true refit 的 normalized
mean/log-std error为 `.270/.125`，优于 K3 的 `.283/.138`。但两者都随 K3
on-policy rounds 恶化，例如 full successful-basin coverage 从 round 0 的
`.663/.670` 降到 round 3 的 `.368/.382`。

这个结果很关键：K10 不只 fixed-bank rank 更好，连**同一个当前 proposal 上的
one-step update**也稍好；然而此前完整 recursive/MPC 仍是 K3 更稳。于是剩余因果
对象不再是单张 landscape 或单步 `T(M,q)`，而是：

> **basin lineage under adaptive query transport**：一个 scorer 连续改变 proposal
> 后，哪些真实/成功 basins 被保留、合并、丢失，以及这些 birth/death 是否与真实
> dynamics 的 optimizer path 对齐。

新的 working idea 因而从 static topology loss 收紧为：

> **Basin-Lineage World Model（BL-WM）**：在 planner 自己访问的 proposal
> sequence 上，匹配 persistent basin witnesses、barrier witnesses 与跨轮
> continuation；不把真实 multi-basin surface 强行 convexify，也不只在冻结
> population 上匹配 rank。

最小训练形态应使用 stop-gradient 的 true birth/saddle indices，将稀疏
`basin-cover + barrier + continuation` inequalities 加到原 LeWM prediction
loss；K3/K10 disagreement 只用于主动发现可能发生 branch birth/death 的 query，
而不是 inference 时平均两套 costs。

为关闭当前唯一缺的 observational link，A100 已在同一套 fresh60 rows 上并行采集
K3 与 K10 各自的完整 on-policy `4 rounds × 300 candidates` trace。它会直接检验：

1. K10 的 fixed/one-step 优势在进入自己的 query distribution 后是否反转；
2. final success failure 是否由 earlier successful-basin death 预测；
3. ordinary mixture/multistart 与真正 basin-lineage supervision 是否可分。

在这套 paired lineage bank 完成前，不启动 topology fine-tune；否则仍可能把一个
静态 diagnostic 误写成训练目标。

## 13. Paired on-policy path：K10 的 static 优势在自己的 query path 上反转

上述 paired bank 已完成：同一 fresh60 rows、common initial noise，K3/K10 各自
递归生成 populations；steps `4/9/19/29` 均保存 `300` candidates，并由两 scorer
交叉评分。final step：

| generated path | K3 true-top30 recall | K10 true-top30 recall | K3/K10 component coverage |
|---|---:|---:|---:|
| K3 path | `.096` | **`.272`** | `.278 / .517` |
| K10 path | **`.197`** | `.079` | `.408 / .300` |

`(K10−K3 scorer on K10 path) − (K10−K3 scorer on K3 path)` 的 paired
interaction 为：

```text
true-elite recall       -.293 [-.381,-.207]
component coverage      -.346 [-.472,-.222]
successful coverage     -.265 [-.414,-.131]
```

K10 final path 的 global Spearman 仍是 `.246 vs .247`，说明 ordinary global
correlation 看不到 low-cost-tail reversal。实际 recursive success 为：

```text
K3:  .200 -> .317 -> .533 -> .517
K10: .150 -> .383 -> .417 -> .433
```

final K10−K3 success 为 `-.083 [-.200,+.033]`；outcome winner 本身仍不显著。
因此 paired trace 支持 adaptive path interaction，但它还没有证明 topology
birth/death 是 final failure 的主因。完整报告见
`basin_lineage_pair_a100_20260720/audit_v1/REPORT.md`。

## 14. Counterfactual refit：component averaging 不是主要瓶颈

为把 path interaction 从相关性推进到因果，对每个 saved population 执行：

- K3/K10/true top-30 global means；
- K3/K10/true elite 的 connected-component means；
- archived generator mean 完整复执行校验。

四个 shards 的 stored-mean true cost/success 与 source 完全一致。final results：

| path | K3 elite mean | K10 elite mean | true elite mean | true component oracle | candidate support |
|---|---:|---:|---:|---:|---:|
| K3-generated | `51.7%` | `55.0%` | `63.3%` | `65.0%` | `66.7%` |
| K10-generated | `45.0%` | `43.3%` | `58.3%` | `60.0%` | `63.3%` |

K10 path 上，K3 global refit 只比 K10 多 `+1.7pp [-6.7,+10.0]`：救 4 个、
损失 3 个。true global refit 则稳定净救 9 个，`+15pp [6.7,25.0]`；再拆成
true components 只额外救 1 个。22/60 states 的 population 根本没有成功
candidate。

最终 success 可以精确分解为：

```text
K3: 40 support - 9 conversion failures  = 31 successes
K10: 38 support - 12 conversion failures = 26 successes
```

当前判决更新为：

```text
self-induced low-cost-tail reversal                SUPPORTED
K3 final scorer swap as causal repair              REJECTED
multi-basin/global-mean collapse as main failure   REJECTED on current PushT cell
generic topology/PH training as next method        STOP
full-sequence proposal support + tail validity     OPEN
```

这并不否认 true landscape multi-basin，而是说明其 causal effect 在当前 cell 太小，
不足以支撑 `BL-WM` headline。当前总判决与停止清单统一记录在
[`../lewm_planning_status_20260721.md`](../lewm_planning_status_20260721.md)；
完整反事实报告位于
`counterfactual_refit_a100_20260721/audit_v1/REPORT.md`。
