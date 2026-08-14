# K1/K5 受控度量：v3 正式成对训练协议

日期：2026-08-13（2026-08-15 修订）
状态：**first-Inf 根因诊断已结案；正式启动改由两 seed 稳定性证据门控**

## 1. 为什么 v2 不能直接补跑

v2 的两个 K1 任务并不是模型参数已经 NaN 后才退出，而是在一次 forward loss
仍有限、backward 首次产生 non-finite gradient 时，被
`nonfinite_grad_policy=error` 主动判失败。原始 LeWM 并不使用 exact skip：它是
`bf16 + gradient_clip_val=1.0`，且没有 non-finite guard。exact skip 是 2026-08-09
才加入本分支的运行保护。因此“v2 K1 失败”不能解释为“标准 LeWM 会训练崩溃”，
也不能把有机 Inf 降格成正常噪声；first Inf 的生成算子必须另行查清。

v2 还有一个更重要的 control 缺口：两臂都编码 8 帧并对 8 帧施加 SIGReg，但 K1
prediction loss 只覆盖 clip 的前 4 帧；后 4 帧只有 SIGReg 梯度。它既不是原始
4-frame LeWM，也没有与 K5 匹配 predictor calls 和 target positions。K1 seed 13/42
在退出前已出现显著的 validation SIGReg 尾部波动，所以不能把这两个失败 seed 当成
普通 LeWM 的稳定性结论。

## 2. v3 唯一科学干预

两臂使用同一 8-frame clip、同一初始化、split、batch order、五个 target frames、
五次 predictor call、encoder/SIGReg work、optimizer、30 epochs 和 dropout=0。

| arm | 五个 transition 的 context |
| --- | --- |
| K1-TF | 每一步都使用真实的 3-frame context；每个 loss 都是标准一步 LeWM transition |
| K5 | 第一步相同；之后递归使用模型预测的 latent context |

因此 estimand 从 v2 的“一个 prefix K1 call vs 五次 K5 call”收紧为：

> **相同的一步 targets、calls 和数据下，是否让预测进入后续 context。**

K1-TF 是标准一步 LeWM objective 在公共 clip 五个位置上的 matched-compute 应用；
不能把它误写成原始论文完全相同的 batch construction。原始 4-frame K1 checkpoint
继续作为外部 LeWM baseline 报告，但不承担 v3 的 paired causal estimand。

## 3. bf16 数值协议

原始配置确实使用 `precision=bf16`，但“bf16 没有 GradScaler”只解释 first Inf 如何穿过
clipping 污染 AdamW，并不解释 first Inf 为什么产生。正式健康门因此修订为：

- `nonfinite_grad_policy=error`；
- 每 epoch 与全程允许的 organic non-finite event 都是 **0**；
- guard 现在通过 `RawGradientModule.after_manual_backward` 的 `on_raw_gradients`
  实施，**真正在 `clip_gradients` 之前**看到原始梯度，因此保存的证据是未经 clip
  的量，且失败发生在 clip/AdamW 污染参数与 optimizer state 之前；
- 证据包括 epoch/step、loss components、non-pixel batch hash、offending parameter、
  NaN/Inf 数量、model/optimizer state；根因复现实验额外保存 pre-forward RNG 和
  BatchNorm buffers，以便 exact replay；
- `skip` 只允许用于明确标注的诊断/运行保障，不能通过 formal pairing verifier；
- 正式六模型启动前，必须在历史失败构造上复现 first Inf、定位生成算子、实施最小修复，
  并让 seed 13/42 严格跨过原失败点而无 event。

### 3.1 共同的 encoder-FP32 数值岛（两臂对称）

根因诊断已结案：first Inf 由 bf16 下的 ViT encoder 产生，最小修复是把 encoder
放在 `torch.autocast(enabled=False)` 内（`encoder_fp32`，projector/predictor/action
encoder 仍是 bf16）。因此 `lewm_paired_k1.yaml` 与 `lewm_paired_k5.yaml` **都**显式
写 `encoder_fp32: true`。

这是**共同数值协议，不是 K1-TF/K5 科学干预**。只在一臂开启会把 numerics 差异混进
因果对比；两臂之外的唯一差别仍然只有 `wm.unroll_tf: 5` 对 `wm.unroll: 5`。
`tests/wm/test_controlled_metric_formal_gate.py` 静态断言这个对称性。

### 3.2 两 seed 稳定性证据门（无布尔旁路）

旧的 `LEWM_FIRST_INF_ROOTCAUSE_RESOLVED=1` 手工旁路已删除——一个可以随手 export 的
布尔值不是证据。任何包含 `train` 的 `PHASES` 现在必须通过环境变量
`LEWM_STABILITY_GATE` 指向 `MODE=stability` 真实产出的 `STABILITY_GATE_PASS.txt`。
`run_controlled_metric_paired.sh` 在**任何 init/train 工作之前**校验该文件，逐项
必须精确匹配，否则大声失败退出 2：

| 字段 | 要求 |
| --- | --- |
| `result` | `PASS` |
| `gate` | `encoder_fp32_two_seed_stability` |
| `seeds` | `13 42`（launcher 的 `SPECS` 顺序，不是排序后的集合） |
| `encoder_fp32` | `true` |
| `max_epochs` | `30`（epoch-based cosine，改了就不是同一 recipe） |
| `nonfinite_grad_policy` | `error` |
| `stop_horizon` | 数值且 `> 137496`（跨过较晚的历史失败点） |
| `commit` | 等于当前 `HEAD` |

每个键必须恰好出现一行，重复或缺失都不放行。commit 相等这一条是关键：稳定性证据
只对产出它的那个 commit 有效，改了代码就必须重跑 `MODE=stability`。DLC wrapper
`run_controlled_metric_paired_dlc.sh` 要求并转发同一路径，自身不再有布尔旁路。

`audit` / `summarize` 单独运行**不需要**这个门，它们只读已存在的 checkpoint。

## 4. 正式设计和判读

- seeds：`7, 13, 42`，共 `3 × 2 = 6` 次单卡训练；
- DLC 申请**恰好 6 GPU**：6 个互相独立的单卡训练，每张卡一个进程
  （`NGPU=6`、`GPU_IDS=0,1,2,3,4,5`、`trainer.devices=1`），
  不申请闲置的第 7/8 张，也不把它当作整节点请求；
- 唯一正式判决点 epoch 30；epoch 5/10/20 只用于轨迹解释；
- 三个 seed pair 均须有 epoch-30 checkpoint、pairing proof 和健康门通过；
- 仍使用 v2 预注册的 G2 pencil、H=5 shear、G1 和 sufficiency gates；
- 不把 v2 的 epoch-10 结果或唯一完整 seed7 pair 混入 v3；
- 任一 arm 失败时整对无效，不与旧任务拼接。

## 5. 六模型之后的强制 CEM 分析

CEM 不是可选附录。三个 seed pair、六个 epoch-30 checkpoint 全部健康完成后，必须用
同一 state/start-goal bank 与 common random numbers 做 paired CEM conversion audit：

- 标准 CEM closed-loop success，K1-TF 与 K5 使用完全相同的 starts、采样噪声和预算；
- 固定记录 round `0/4/9/19/29` 的 population、elite 与 returned mean；
- learned/true elite overlap、CEM update cosine、relative update error；
- population support（至少一个 true-success candidate）、elite true-success、returned-mean
  true-success，区分“搜到”“选对”“均值动作可执行”；
- simulator oracle reranking 与 oracle-update ceiling，只作诊断，不调方法超参；
- 以 training seed pair 为一级、state 为二级做 hierarchical paired bootstrap；
- 表征/动力学指标改善但 paired closed-loop CI 不支持成功率提升时，结论必须写成
  conversion bottleneck，不能把 CEM 结果略过。

正式报告完成条件是：training pairing proof、representation audit 和 CEM audit 三者齐全。

## 6. 入口

训练入口必须带证据门路径（见 3.2）：

```bash
LEWM_STABILITY_GATE=<.../STABILITY_GATE_PASS.txt> \
  PHASES=init,train NGPU=6 \
  bash scripts/plan/run_controlled_metric_paired.sh
```

分析入口不需要门：

```bash
PHASES=audit,summarize NGPU=2 \
  bash scripts/plan/run_controlled_metric_paired.sh
```

默认 `RUN_TAG=controlled_metric_paired_v3_20260813`。CEM conversion audit 仍是
强制项（见第 5 节）：只有 training pairing proof、representation audit 和 CEM audit
三者齐全，正式报告才算完成。
