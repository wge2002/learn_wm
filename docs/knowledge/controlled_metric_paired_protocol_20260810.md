# K1/K5 受控度量：正式成对训练协议

日期：2026-08-10
状态：**已被 2026-08-13 v3 协议取代；保留为失败波次审计记录**
取代者：[v3 正式成对协议](controlled_metric_paired_protocol_v3_20260813.md)。
取代：
[旧执行计划](controlled_metric_execution_plan_20260803.md) 中 Phase 1/2 的训练比较细节。
旧文的理论依赖和下游 kill rule 仍然有效。

## 1. 这次修掉了什么

旧的 `pd_d192_k1_s7` / `pd_d192_k5_s7` 不能承担 K1→K5 的因果比较：

1. `seed` 只传给了 split/DataLoader generator，模型构造前没有设置全局 RNG，
   所以同名 seed 不代表相同初始化；
2. K1 用 4-frame window，K5 用 8-frame window，数据集有效起点数、每 epoch batch
   数和总更新数都不同；
3. K1、K5 没有加载同一份 bitwise initialization；
4. bf16 非有限梯度原先会被清零。对 AdamW 而言，零梯度仍会推进 momentum 并做
   decoupled weight decay，并不是真正的 skip。

因此旧 checkpoint 只保留为**探索性现象**：它提示 K5 可能降低 horizon shear、提高
action/residual generalized pencil，但不能据此声称训练 horizon 是原因，也不能用
seed=1 继续做普通复现来补这个洞。

当前实现逐项修正：

- 在 dataset 和 model 构造前执行 `seed_everything(seed, workers=True)`；
- 每个训练 seed 先原子导出一份 initialization state dict，K1/K5 都 strict-load
  同一 tensor-content SHA-256 和 file SHA-256；已有同名 artifact 若不 bitwise
  一致则立即失败（不能用 `torch.save` container hash 代替 tensor hash）；
- 两臂都读取 8-frame clip，共用 split、shuffle seed、batch size、epoch 数和 update
  budget；
- K1 只执行 matched one-step predictor objective，K5 只执行 5-step open-loop
  recursive objective；encoder work 与 SIGReg 的 8-frame 输入保持一致；
- 普通训练遇到坏梯度时把所有 grad 设成 `None`，避免 AdamW 改权重/状态；正式成对
  config 使用更严格的 `nonfinite_grad_policy=error`，任何坏 update 都使该 run 无效；
- 两臂日志记录 init、split、DataLoader state 和 epoch-0 前三批非像素字段 hash；
  `verify_controlled_metric_pairing.py` 对日志、resolved config 和最终 artifact
  自动验配，不能靠人工目测。

## 2. 唯一干预与 estimand

固定项：

- model architecture、optimizer、scheduler、precision、batch size；
- predictor dropout 固定为 0；否则 K5 每 batch 调五次 predictor、K1 只调一次，
  不同数量的 dropout draws 会成为额外干预；
- 8-frame physical clip、train/validation split、batch order；
- initialization tensor；
- 30 epochs 和每 epoch update 数；
- encoder/SIGReg 看到的全部 8 frames。

唯一干预：

| arm | predictor loss |
| --- | --- |
| K1 | 3-frame true context 上的 matched one-step prediction |
| K5 | 同一 clip 上从 3-frame context 开始的 5-step open-loop recursion |

估计量是同一训练 seed 内 `K5 - K1`，训练 seed pair 是独立统计单位，physical
trajectory 只是 pair 内的配对测量单位。不能把 64 条 Jacobian trajectory 当成 64 个
独立训练复现。

## 3. 正式最小设计

- training seeds：`7, 13, 42`；
- arms：`K1, K5`，共 6 次训练；
- checkpoint trajectory：epoch `5, 10, 20, 30`；
- **唯一正式判决点：epoch 30**；其余 epoch 只解释形成过程，不能挑最好 epoch；
- 每个 audit 共用 physical-bank seed `20260810`；
- 默认 `1024` trajectories 做 representation/probe，固定其中 `64` 条做 Jacobian；
- hierarchical bootstrap `20,000` 次：先重采样 training-seed pair，再在 pair 内
  重采样 matched physical trajectory；
- 三 seed 必须全同方向，不能由 pooled trajectory 数量掩盖一个反向 seed。

两 seed 只能用于资源不足时的 debug，不允许输出正式通过。任何 arm 中断、出现
non-finite update、init/config/hash 不配，整对作废；不能只补跑其中一臂后与旧臂拼接。

## 4. 预注册读数与判决

### 主读数：G2 generalized pencil

当前可执行主读数为逐 trajectory 的
`log det(I + W_r^{-1} W_u)`。正方向是 K5 更大，即动作可控能量相对真实 residual
噪声提高。

通过条件同时满足：

1. hierarchical 95% CI 的下界 `> 0`；
2. 每个 training seed 的 pair mean 都 `> 0`。

### Gate A：不是 uniform contraction

`H=5 log_shear_rms` 的 `K5-K1` hierarchical 95% CI 上界 `< 0`，且三个 seed
全小于 0。`log_scale_mean`、max gain、action/residual energy 同时报告，但不允许拿
单独的 scale contraction 代替 shear 结论。

### G1：双向、非正交、近可逆

每个 seed 必须同时满足：

- K1→K5 与 K5→K1 held-out linear `R² >= 0.75`；
- 两个 linear cycle `R² >= 0.90`；
- 双向 `linear R² - orthogonal R² >= 0.10`。

这只构成 linear gauge-like reshaping 的证据。最终正式 G1 仍需加入 held-out
invertible nonlinear map；单向 R² 不算通过。

### 充分性 non-inferiority

- agent xy、block xy、block angle sin/cos、agent velocity probe：K5-K1 的 seed
  bootstrap CI 下界和每 seed delta 都不得低于 `-0.05 R²`；
- block-motion proxy ROC-AUC：对应阈值 `-0.03`。

几何指标改善而 probe 越界，结论为 latent 丢失任务信息，不算 controlled metric
改善。

### 状态解释

- `KILL_OR_DOWNGRADE`：任一科学 gate 失败；按原计划把 claim 降级，不训练新
  regularizer 救叙事；
- `INCOMPLETE`：少于 3 pairs 或 pairing proof 不完整；
- `PROVISIONAL_PASS`：以上可执行 gates 全过，但**仍不是最终 intrinsic CP_H
  证书**。

之所以只能 provisional：当前 HDF5 没有 exact contact bit；`W_u` 暂用标准化
five-action-block 的 identity covariance；还缺同一 physical perturbation 经 encoder
pushforward 得到的 reference `W_0`。这三个量补齐后才允许把结论写成坐标不变的
intrinsic controlled metric 改善。

## 5. 执行

仓库根目录运行：

```bash
PHASES=init,train NGPU=2 \
  bash scripts/plan/run_controlled_metric_paired.sh

PHASES=audit,summarize NGPU=2 \
  bash scripts/plan/run_controlled_metric_paired.sh
```

可用 `DS`、`PIXELS`、`STABLEWM_HOME`、`GPU_IDS`、`SEEDS`、`AUDIT_SAMPLES`、
`JACOBIAN_SAMPLES` 覆盖环境。正式结果不能改默认 seeds、epoch 30 判决点或阈值；
如果因资源缩小 sample 数，必须在启动前另起版本并记录，不能看完结果再改。

输出目录 `outputs/controlled_metric_paired_20260810/` 应包含：

- 每个 init/train/audit 的独立日志；
- 12 个 `audit_s*_e*.json`；
- `pairing_proof.json`；
- `paired_summary.json` 与 `paired_summary.md`。

## 6. 结果后的唯一分支

若 epoch-30 得到 `PROVISIONAL_PASS`，下一步不是再扫 K 或 seed，而是补 exact
contact、planner covariance 和 physical-perturbation `W_0`，把 G2 升级为真正的
`CP_H`，随后才做 paired closed-loop planning。

若得到 `KILL_OR_DOWNGRADE`，停止“horizon 选择 intrinsic controlled metric”的强
claim；保留较弱的 predictor-relative geometry / optimization effect，并回到任务
performance 评价。若是 `INCOMPLETE`，只修缺失 pair 或协议证据，不改变指标和门槛。
