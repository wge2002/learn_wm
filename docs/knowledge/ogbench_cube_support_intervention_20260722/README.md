# OGBench Cube proposal-support causal gate（2026-07-22）

## 判决

这次 Gate 把“proposal support 不足”和“support 之后的 adaptive update / mean
conversion”真正拆开了：

```text
proposal support 是 closed-loop causal bottleneck                 SUPPORTED
oracle support 会解锁 K5 相对 K1 的 control 优势                  NOT SUPPORTED
有 support 后，30-round CEM 仍会侵蚀可成功 plan                   SUPPORTED
当前 deployable method                                             NOT FOUND
```

最关键的结果是：用一次 hidden-oracle expert continuation 初始化 CEM 后，在
`150` 个 paired states 上，K1 success 从 `37.3%` 升到 `69.3%`，K5 从
`38.0%` 升到 `65.3%`。所以 support 不是相关性解释，而是一个很大的 causal
lever。

但它没有把 K5 较低的 long-horizon latent MSE 解锁成 winner：prior 条件下 K5−K1
为 `-4.0pp [-10.0,+2.0]pp`，support gain 的 difference-in-differences 为
`-4.7pp [-12.7,+3.3]pp`。三个 eval seeds 上 K1/K5 的排序也不稳定。

更重要的是，16-state candidate replay 中，expert prior 把 round-0 natural support
提高到 `100%`，但 actual CEM mean success 从 K1/K5 的 `93.8%/100%` 一路降到
round 29 的 `37.5%/56.2%`；此时 population support 仍有 `87.5%/93.8%`。
因此 support 缺失与 adaptive update / Gaussian-mean conversion 是两个独立瓶颈。

## 1. 干预是什么，以及它不是什么

干预只把 offline dataset 中从当前 state 开始的 exact future action continuation
设为第一次 CEM solve 的初始 mean：

```text
zero-mean Gaussian CEM
vs
expert-next initialized Gaussian CEM
```

- exact plan 含 `H10 × action_block5 = 50` 个 low-level actions；
- 只用于 episode 的第一次 replan；之后恢复普通 receding-horizon warm start；
- candidate count `300`、CEM rounds `30`、top-k `30`、random noise 与 WM calls 不变；
- CEM 本来就把 current mean 作为 population 的第一个 candidate，因此没有额外
  candidate 或额外模型调用；
- 两个模型共享 dataset rows、start/goal、round-0 random noise 与评估协议。

这会泄漏测试 state 后面的 future actions，**不是 deployable policy，也不能作为
OGBench benchmark score**。它是一个刻意强的 upper-bound intervention，只回答：
如果 task-aligned support 被保证，control 是否会上升，以及 K5 是否会因此成为
winner。

## 2. Closed-loop：support 很重要，但没有解锁 K5 winner

协议为 `3 seeds × 50 unique rows = 150` paired states，H10、goal offset 50、
eval budget 50、receding horizon 5、BF16。每格使用同一训练 seed `3072` 的 K1/K5
checkpoint。这里的四格都按 budget 50 fresh rerun；前一份 transfer audit 的 H10
baseline 使用 budget 100，绝对成功率不能跨两份协议直接比较，本节只作四格内的 paired
comparison。

| condition | K1 success | K5 success | K5−K1 | mean final distance K1/K5 |
|---|---:|---:|---:|---:|
| zero | `56/150 = 37.3%` | `57/150 = 38.0%` | `+0.7pp [-4.0,+5.3]` | `.1190 / .1156` |
| expert prior | `104/150 = 69.3%` | `98/150 = 65.3%` | `-4.0pp [-10.0,+2.0]` | `.0759 / .0752` |
| prior−zero | `+32.0pp [24.0,40.0]` | `+27.3pp [19.3,35.3]` | DID `-4.7pp [-12.7,+3.3]` | `-.0431 / -.0404` |

prior 的 paired gain 很明确：K1 `50` 个 prior-only success、`2` 个 zero-only，
McNemar `p=6.1e-13`；K5 为 `45` 对 `4`，`p=8.2e-10`。连续 task distance 与
binary success 同向下降。

排除 `28` 个 initial-easy states 后，结论更强而不是更弱：

| condition | K1 | K5 |
|---|---:|---:|
| zero，N=122 | `28/122 = 23.0%` | `29/122 = 23.8%` |
| expert prior，N=122 | `76/122 = 62.3%` | `71/122 = 58.2%` |
| prior−zero | `+39.3pp [30.3,48.4]` | `+34.4pp [25.4,43.4]` |

各 seed 的 K1/K5 prior success 分别为 `72/62%`、`84/76%`、`52/58%`。因此
“support 帮助两个模型”跨 seed 稳定；“哪个 K 更好”并不稳定。

## 3. Candidate path：support 保住了，但 mean 被逐轮推坏

在与 baseline 完全相同的 `16` 个 fresh nontrivial states 上，保存完整 CEM path，
并在 MuJoCo 中重放所有 `300` 个 natural candidates。下表的 support 表示该 state
的 natural population 是否含至少一个成功 candidate；actual 表示该轮更新后的
CEM mean 是否成功。

| step | K1 support zero→prior | K5 support zero→prior | K1 actual zero→prior | K5 actual zero→prior |
|---:|---:|---:|---:|---:|
| 0 | `.250→1.000` | `.250→1.000` | `.062→.938` | `.062→1.000` |
| 4 | `.375→1.000` | `.375→1.000` | `.125→.750` | `.125→.750` |
| 9 | `.375→1.000` | `.312→.938` | `.125→.438` | `.188→.625` |
| 19 | `.312→.938` | `.312→1.000` | `.125→.438` | `.250→.625` |
| 29 | `.312→.875` | `.375→.938` | `.188→.375` | `.188→.562` |

到 final round，prior 对两条 path 的 support 都提高 `+56.3pp
[31.3,81.3]pp`。K5 actual mean 的 paired improvement 为 `+37.5pp
[12.5,62.5]pp`；K1 为 `+18.8pp [-6.3,43.8]pp`。但 K5−K1 actual mean 的
N=16 差值 `+18.8pp [-6.3,43.8]pp` 没有在 N=150 closed-loop 中复现，不能升级为
模型 winner。

真正有约束力的是同一路径上的 scorer interaction。final support-controlled top-30
真实成功率为：

| generated path | K1 scorer | K5 scorer | K5−K1 |
|---|---:|---:|---:|
| K1 path | `.444` | `.456` | `+.013` |
| K5 path | `.583` | `.483` | `-.100` |

所以 prior 下的 path interaction 仍为：

```text
(K5−K1 gap on K5 path) − (gap on K1 path)
= -.1125 [-.2187,-.0292]
```

它比 zero-prior baseline 的 `-.2458` 数值上小，但 interaction change
`+.1333 [-.0292,+.3271]` 仍跨 0。expert support 缓和了 query shift，却没有消除
K5 在自己诱导 path 上的 tail reversal。

## 4. 宏观结论与下一 Gate

现在不需要再找一个更极端的 representation idea。因果账本已经更清楚：

```text
representation/dynamics fidelity
  × proposal support                  <- hidden-oracle intervention: large causal gain
  × adaptive tail/update validity     <- high-support path still erodes
  × elite-to-action conversion        <- successful candidates often do not survive as mean
```

下一步应把这次 oracle 诊断降成 deployable baseline，而不是直接训练新的 WM：

1. 训练或复用一个只看当前 observation/goal 的 goal-conditioned BC/GCBC action
   prior；test time 不读取 future action；
2. 在相同 150 paired states 上锁定 zero/GCBC × K1/K5，并同时报告 support、mean
   success 与 continuous distance；
3. 用现有 intermediate replay 预锁定一个很小的 `1/5/10/30` CEM-round 或
   prior-trust-region ablation，判断 deployable prior 是否也被过度优化侵蚀；
4. 只有 deployable support 已提升、且 K5 self-path reversal 仍显著时，才进入
   sequence-level adaptive-tail training 或 calibrated update rejection。

这次 Gate 因而关闭了“support 是否真的影响 control”这个问题，但没有给出可部署
方法，也没有支持“K5 的 prediction 优势只差一个 proposal prior 就能转成 control”。

## 5. 产物与审计

- [closedloop/formal_report.json](closedloop/formal_report.json)：150-state paired
  bootstrap、McNemar、continuous distance、per-seed 与 source SHA-256；
- [closedloop/formal_summary.txt](closedloop/formal_summary.txt)：closed-loop 人读摘要；
- [candidate/candidate_comparison_report.json](candidate/candidate_comparison_report.json)：
  zero/prior 完整 paired path comparison；
- [candidate/candidate_comparison_summary.txt](candidate/candidate_comparison_summary.txt)：
  intervention 人读表格；
- [candidate/formal_prior_report.json](candidate/formal_prior_report.json)：prior 条件的
  scorer/support/refit 审计；
- [candidate/formal_prior_summary.txt](candidate/formal_prior_summary.txt)：prior path
  人读摘要。

candidate audit 的 round-0 K1/K5 population、true replay 与 prediction 均 bitwise
paired；最大 reset error `<2e-15`，action scaler roundtrip error `<9.6e-7`；16 rows
无重复。server raw shards 位于
`/225010117/logs/ogbench_cube_support_prior_20260722/`，大体积 NPZ 不进入仓库。
