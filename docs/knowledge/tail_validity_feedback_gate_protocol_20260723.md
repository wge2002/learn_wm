# Tail-validity feedback-channel 否决实验（locked protocol，2026-07-23）

## 目的

只回答一个问题：在时刻 `t` 执行一个真实 action block 后得到的模型残差，是否能
预测并在 `t+1` 的下一次 CEM replan **环内**修正 low-tail 误排。该实验不把离线
oracle、未来 expert action 或未执行候选的真值暴露给校准器。

## 为什么必须补采集

现有 A100 `dynamics_replay_k3_n2048` 与 60-state CEM oracle trace 在生成时做了
`exclusion_radius=60` 的隔离；精确 row、`row±5` 以及同 episode 的
`start±5` 配对数均为 `0`。OGBench closed-loop 文件只保存 episode 结果，没有逐次
replan 的模型预测和真实后继。因此旧文件能提供起点、动作和 oracle 仪器，但不能
直接组成所需的 `(executed residual_t, low-tail error_{t+1})` 样本。

## Locked 数据与干预

- source：PushT H5/off40 paired K3/K10 60-state bank；固定取 K3 path 的 final mean；
- 前缀：只执行第一个 abstract action block，即 `5` 个环境动作。这是反馈最密、
  延迟最小的乐观设置；若它失败，不再用更稀疏反馈救方法线；
- 反馈：`r_t = encode(o_{t+1}) - predict(o_{t+1} | o_t, a_t)`，192 维；
- next replan：相同真实后继、3-frame matched history、H5、`30×300` CEM；
- 递归 arms：`alpha ∈ {-1, 0, .5, 1}`，所有 round 使用
  `||z_hat_terminal + alpha*r_t - z_goal||²`；`alpha=0` 是 baseline，`-1` 是反号
  control；每 state/arm 使用 common random numbers；
- oracle：只在 CEM 完成后重放各 arm 的 final 300 candidates 与 returned mean；
- 统计单位：source state。前缀已终止的 state 没有 next replan，预先排除并报告数量；
- 大 archive 留在 A100，仓库只收脚本、hash、compact JSON/Markdown。

这不是 final selector：反馈 cost 从 CEM 第 0 轮开始决定 elite 和后续 proposal。

## Primary metrics 与决定规则

1. **可预测性**：baseline final `1 - true-top30 recall` 为目标；报告 residual norm 的
   Spearman/state bootstrap CI，以及固定 5-fold OOF ridge（无反馈常数预测 vs
   residual 几何特征）的 `R²/MAE`。
2. **固定 population 校准**：在 baseline population 上报告 `.5/1/-1` 的 top30
   recall 变化，分离“rank 信号”与 proposal path 变化。
3. **递归修正**：外层 5-fold 只在 train states 的 `{0,.5,1}` 中选 alpha，再在
   held-out states 读取对应 recursive arm；primary 是 true-top30 recall，必须同时
   报 candidate support、oracle minimum、returned-mean true cost/success。
4. 所有差值使用 20,000 次 state bootstrap 95% CI，不按 18,000 candidates 当作
   独立样本。

判决预先锁定为：

- **OPEN**：cross-fit recursive top30 recall 至少 `+0.05` 且 CI 下界 `>0`，同时
  returned-mean true cost 的 CI 上界 `<0` 或 success 差的 CI 下界 `>0`；可预测性
  方向一致，反号 control 不复制增益；
- **CLOSE**：即使每 state 用真值事后挑 `{0,.5,1}`，fixed-pop recall 增益仍小于
  `.05`，或 cross-fit recursive recall 的 CI 上界 `≤0` 且 outcome 无改善；
- 其余为 **HOLD / signal but insufficient**，不得写成方法已成立。

脚本：`scripts/plan/tail_validity_feedback_gate.py` 与
`scripts/plan/summarize_tail_validity_feedback_gate.py`。
