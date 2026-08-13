# K1/K5 受控度量：v3 正式成对训练协议

日期：2026-08-13
状态：**预注册并实现；替代 2026-08-10 v2**

## 1. 为什么 v2 不能直接补跑

v2 的两个 K1 任务并不是模型参数已经 NaN 后才退出，而是在一次 forward loss
仍有限、backward 首次产生 non-finite gradient 时，被
`nonfinite_grad_policy=error` 主动判失败。普通 LeWM 配置使用 exact skip；因此
“v2 K1 失败”不能解释为“标准 LeWM 会训练崩溃”。

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

两臂都使用 `precision=bf16`。一次 non-finite backward 在 gradient clipping 前被截获，
所有参数的 `grad` 设为 `None`，使 AdamW 对参数、momentum、weight decay 均执行 exact
skip。这等价于 mixed-precision scaler 拒绝一次 optimizer update，不再把单次数值事件
定义成科学失败。

健康门预先固定为：

- 每 epoch skip fraction `<= 1e-4`，即当前 11,306 steps/epoch 最多 1 次；
- 全程最多 3 次 skip；
- 超过任一门立即失败；
- 每次事件记录 epoch/step、loss components、non-pixel batch hash、offending parameter、
  NaN/Inf 数量和完整 model/optimizer evidence bundle；
- skip count 是正式报告字段，不能隐藏或事后删除 seed。

## 4. 正式设计和判读

- seeds：`7, 13, 42`，共 `3 × 2 = 6` 次单卡训练；
- DLC 申请 6 GPU，不申请闲置的第 7/8 张；
- 唯一正式判决点 epoch 30；epoch 5/10/20 只用于轨迹解释；
- 三个 seed pair 均须有 epoch-30 checkpoint、pairing proof 和健康门通过；
- 仍使用 v2 预注册的 G2 pencil、H=5 shear、G1 和 sufficiency gates；
- 不把 v2 的 epoch-10 结果或唯一完整 seed7 pair 混入 v3；
- 任一 arm 失败时整对无效，不与旧任务拼接。

默认入口：

```bash
PHASES=init,train NGPU=6 \
  bash scripts/plan/run_controlled_metric_paired.sh

PHASES=audit,summarize NGPU=2 \
  bash scripts/plan/run_controlled_metric_paired.sh
```

默认 `RUN_TAG=controlled_metric_paired_v3_20260813`。
