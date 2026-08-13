# K1/K5 受控度量：v3 正式成对训练协议

日期：2026-08-13
状态：**已修订；正式训练暂停，先完成 first-Inf 根因门**

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
- guard 在 clipping 前保存证据后立即失败，避免污染参数和 optimizer state；
- 证据包括 epoch/step、loss components、non-pixel batch hash、offending parameter、
  NaN/Inf 数量、model/optimizer state；根因复现实验额外保存 pre-forward RNG 和
  BatchNorm buffers，以便 exact replay；
- `skip` 只允许用于明确标注的诊断/运行保障，不能通过 formal pairing verifier；
- 正式六模型启动前，必须在历史失败构造上复现 first Inf、定位生成算子、实施最小修复，
  并让 seed 13/42 严格跨过原失败点而无 event。

## 4. 正式设计和判读

- seeds：`7, 13, 42`，共 `3 × 2 = 6` 次单卡训练；
- DLC 申请 6 GPU，不申请闲置的第 7/8 张；
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

## 6. 入口（根因门通过后启用）

默认入口：

```bash
PHASES=init,train NGPU=6 \
  bash scripts/plan/run_controlled_metric_paired.sh

PHASES=audit,summarize NGPU=2 \
  bash scripts/plan/run_controlled_metric_paired.sh
```

默认 `RUN_TAG=controlled_metric_paired_v3_20260813`。
