# OGBench Cube transfer audit（2026-07-22）

## 判决

这次外部 benchmark 没有支持“更低的长时 latent MSE 会自动变成更高的
control success”，但复现了一个更有约束力的 optimizer-feedback 现象：

> **K5 的 long-horizon prediction 确实更好，但 vanilla Gaussian CEM 经常没有把
> 成功动作放进自然 population；即使显式注入可成功的 expert neighborhood，随着
> CEM 迭代，成功 tail mass 仍会被模型自己诱导的 false-low-cost candidates 挤出。
> K5 在 K1 path 上较好、在自己的 K5 path 上反而较差。**

所以当前宏观结论应从“继续修 world-model prediction 或单点 scorer”改成：

```text
control value = representation/dynamics fidelity
              × task-aligned candidate support
              × adaptive tail/update validity
              × elite-to-action conversion
```

PushT 里定位到的 adaptive tail problem 不是纯 benchmark artifact；但它也不是唯一
瓶颈。OGBench Cube 明确暴露了 proposal support 与 conversion 这两层。

## 1. Closed-loop success：MSE 优势没有转成 winner

单个训练 seed `3072`，每格 `3 eval seeds × 50 paired episodes = 150`。K1/K5
共享 starts、goal、matched history 与 CEM random noise。H5 使用 offset 25 / budget
50；H10 使用 offset 50 / budget 100。

| protocol | K1 | K5 | K5−K1 | paired bootstrap 95% CI |
|---|---:|---:|---:|---:|
| H5，全部 150 | `94/150 = 62.7%` | `95/150 = 63.3%` | `+0.7pp` | `[-4.7,+6.0]pp` |
| H5，排除 initial-easy | `39/95 = 41.1%` | `40/95 = 42.1%` | `+1.1pp` | `[-7.4,+9.5]pp` |
| H10，全部 150 | `65/150 = 43.3%` | `65/150 = 43.3%` | `0.0pp` | `[-6.0,+6.0]pp` |
| H10，排除 initial-easy | `37/122 = 30.3%` | `37/122 = 30.3%` | `0.0pp` | `[-7.4,+7.4]pp` |

H10 的 fixed expert-action endpoint latent MSE 为：

```text
K1  .00703
K5  .00545   (-22.5%)
```

H20 上 K5 约低 `40%`，而 H5 基本持平。因此这里不是“模型指标也没有差异”，而是
**明确的 prediction/control non-conversion**。

## 2. Decision-fidelity audit

### 2.1 协议

- `16` 个 fresh、initial cube-to-goal distance `>4cm` 的 dataset states；
- K1/K5 各自产生完整 CEM path，steps `0/4/9/19/29`；
- 每轮 `300` 个自然 candidates，两个模型交叉评分；
- round 0 使用 common random numbers，两个 generator population bitwise 相同；
- 每个 candidate 从完全相同的 qpos/qvel 在 MuJoCo 中重放 50 actions；
- 另加 `2` 个 exact expert continuations 与 `30` 个 normalized-noise continuations，
  只作 post-hoc support control，不改变原 CEM path；
- state 是统计独立单位；所有 CI 都在 16 states 上 paired bootstrap；
- candidate audit 使用 FP32 inference；其绝对值不与 closed-loop dtype 混作同一
  protocol，K1/K5 与 generator/scorer interaction 在 audit 内严格配对。

审计检查全部通过：round-0 candidate/true/pred 最大差异均为 `0`，reset error 为
`0`，action scaler roundtrip error `<9.6e-7`，16 rows 无重叠。

### 2.2 自然 CEM population：support 小，K5 的更好 oracle candidate 没有被转化

| CEM step | K1 path support | K5 path support | K1 actual mean success | K5 actual mean success |
|---:|---:|---:|---:|---:|
| 0 | `.250` | `.250` | `.062` | `.062` |
| 4 | `.375` | `.375` | `.125` | `.125` |
| 9 | `.375` | `.312` | `.125` | `.188` |
| 19 | `.312` | `.312` | `.125` | `.250` |
| 29 | `.312` | `.375` | `.188` | `.188` |

到 step 29：

- K5 path 的 natural oracle-min distance 比 K1 好 `-.0064
  [-.0144,-.0006]`；
- candidate support 是 `.375 vs .312`，差 `+6.25pp [0,+18.75]pp`；
- 但 actual CEM mean 都只有 `3/16` 成功，K5-only/K1-only 都是 1 个；
- actual mean distance 差 `+.0001 [-.0075,+.0074]`，完全没有 conversion；
- 只有 `56% / 50%` 的 K1/K5 late populations 有非退化 cube-distance
  variation，其余 population 对 task outcome 几乎是常数。

因此这批数据不能被概括成“K5 proposal 更差”。K5 path 甚至含有略好的 hidden
candidate ceiling；真正没有发生的是从 support 到 elite mean/action 的稳定转化。

### 2.3 Support-controlled bank：成功 neighborhood 被 CEM late tail 挤出

expert controls 是有效的：

```text
exact same-row continuation success       100.0%
exact next-row continuation success        93.8%
noise scale .02 / .05 / .10                93.8% / 93.1% / 95.0%
```

也就是说，bank 中确实存在一个有相当局部厚度的 success neighborhood。两个模型
通常都把 exact expert 放在最前面；问题不在 top-1 是否完全失明，而在整个 elite
tail 的质量和质量随自诱导 path 的变化。

模型 top-30 中的真实成功比例：

| step | K1 scorer on K1 path | K5 scorer on K1 path | K1 scorer on K5 path | K5 scorer on K5 path |
|---:|---:|---:|---:|---:|
| 0 | `.887` | `.902` | `.887` | `.902` |
| 4 | `.777` | `.835` | `.812` | `.810` |
| 9 | `.735` | `.792` | `.785` | `.731` |
| 19 | `.556` | `.667` | `.688` | `.565` |
| 29 | `.483` | **`.579`** | **`.637`** | `.487` |

关键 interaction：

```text
(K5−K1 scorer gap on K5 path) − (gap on K1 path)

step 0     .000
step 4    -.060 [-.152,-.004]
step 9    -.110 [-.244,-.023]
step 19   -.233 [-.465,-.083]
step 29   -.246 [-.504,-.073]
```

这和 PushT paired K3/K10 trace 的结构一致：K5 在外生的 K1 path 上较好，但优势
随着自己改变 proposal distribution 而反转。普通 balanced/global 指标会把它
平均掉：final balanced K5−K1 top-30 success 只有 `-.027
[-.115,+.062]`。

support-controlled top-30 mean 的 success 从 round 0 的 `.938/.938` 降到 final
`.406/.531`。K5−K1 为 `+12.5pp [-3.1,+28.1]pp`，仍不足以成为 winner；自然
CEM mean 则是严格 `3/16 vs 3/16`。

## 3. 宏观解释

### 3.1 这次结果支持什么

1. **稳定 prediction 不是假的。** K5 的长时 latent MSE 优势清楚存在。
2. **但 prediction metric 不是 control conclusion。** H10 success 在 150 paired
   episodes 上精确打平。
3. **第二任务复现了 adaptive tail reversal。** interaction 从 round 0 的 0
   单调扩大到 final `-.246`，且 state-bootstrap CI 不跨 0。
4. **top-1 正确也不够。** exact expert 常被排第一，但 elite set 中仍逐轮混入
   false-low-cost candidates，随后 global mean/refit 失败。
5. **proposal support 是独立 bottleneck。** 自然 population 只有约
   `31–38%` states 含任一成功 candidate；hidden candidate ceiling 与实际 mean
   之间仍有明显 gap。

### 3.2 这次结果不支持什么

- 不支持“K5 在 OGBench 上已经更会控制”；
- 不支持继续只优化 fixed expert-action MSE；
- 不支持把一个 final reranker 当修复；
- 不支持从 expert-injected bank 的结果宣称 deployable improvement；这些 future
  actions 是 hidden oracle diagnostic；
- 不支持把所有失败都归因于 scorer：自然 proposal support 本身经常不存在。

## 4. Follow-up Gate 已完成

后续用 exact future expert continuation 做了一次 hidden-oracle、iso-WM-calls 的
support intervention。它不是 deployable benchmark method，而是先问“如果保证
support，会发生什么”。结果是：

- 150-state closed-loop success 从 K1/K5 的 `37.3%/38.0%` 提高到
  `69.3%/65.3%`，support 因而是明确的 causal bottleneck；
- prior 条件下 K5−K1 为 `-4.0pp [-10.0,+2.0]pp`，没有解锁 K5 winner；
- 16-state candidate path 的 round-0 mean success 为 `93.8%/100%`，到 round 29
  降为 `37.5%/56.2%`，尽管 natural support 仍有 `87.5%/93.8%`；
- support-controlled self-path interaction 仍为 `-.1125 [-.2187,-.0292]`。

因此下一步不是直接开新的 representation objective，而是把 oracle intervention
替换成只看 current observation/goal 的 GCBC prior，再做 locked early-stop / prior
retention ablation。完整结果见
[OGBench proposal-support causal gate](../ogbench_cube_support_intervention_20260722/README.md)。

## 5. 产物

- [formal_report.json](formal_report.json)：compact metrics、paired bootstrap、rows、
  source shard SHA-256；
- [formal_summary.txt](formal_summary.txt)：人读表格；
- [proposal-support follow-up](../ogbench_cube_support_intervention_20260722/README.md)：
  hidden-oracle causal intervention、closed-loop paired success 与 candidate-path erosion；
- `scripts/plan/ogbench_candidate_fidelity.py`：完整 population / MuJoCo replay；
- `scripts/plan/summarize_ogbench_candidate_fidelity.py`：state-level paired summary；
- server raw shards：
  `/225010117/logs/ogbench_cube_candidate_fidelity_20260722/formal_s{420,421,422,423}.npz`。

closed-loop raw logs 位于
`/225010117/logs/ogbench_cube_eval_20260722/`。大体积 candidate archives 不进入
仓库，校验值已写入 compact JSON。
