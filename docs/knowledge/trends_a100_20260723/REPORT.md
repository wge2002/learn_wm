# 2026 趋势判决：A100 严格 gate 与主线执行记录

日期：2026-07-23
结论：**方向 B（Generative LeWM）在 PushT 上正式 CLOSE；方向 A 的
head-to-head 已启动，verify wave 已排队。**

## 1. 本次回答的问题

方向 B 只有在真实转移满足下面两个条件时才值得训练：

1. 给定足够的物理状态与动作后，next latent 仍有不可忽略的条件方差；
2. 该方差在模型的高误差放大子空间中有结构性富集。

预注册的一票否决规则是：

- `median conditional std / latent scale < 0.02`：视为近确定，KILL；
- 条件协方差 `top eigenvalue / mean eigenvalue < 2`：近各向同性，KILL；
- top-3 amplification span 中的方差占比低于随机子空间基线的 `2x`：
  缺少方向结构，KILL。

任一条件触发就关闭方向 B。

## 2. 严格协议

- policy：`iter2_multistep_eval`
- probe：20,000 个长度 11 的窗口；回放和编码耗时 1,015 秒
- 去重：删除 181 个完全相同的采样窗口，实际使用 19,819 个转移
- 邻域：全局逐维标准化条件空间中的**绝对半径**，不使用分位数阈值
- 邻居：`k=1`，每组是 anchor 与最近邻组成的 pair
- 防伪重复：按半径由小到大贪心选择互不重叠的 pair
- 最小证据量：至少 30 个互不重叠的 pair 才出 PASS/KILL
- 条件键：
  - `current`：当前物理状态与动作，17 维
  - `history`：三帧物理状态与动作历史，51 维；这是控制隐速度后的主判据
- 放大方向：64 个转移 Jacobian 的 top-3 左奇异子空间
- 对齐量：完整条件协方差落入 top-3 amplification span 的方差比例；
  192 维中的随机基线为 `3/192 = 0.015625`

脚本会同时报告 leading-eigenvector overlap，但正式 gate 使用完整协方差的
方差占比，因为它不会把其余条件方差方向丢掉。

20k 主扫描的锁定参数为：

```bash
python outputs/gauge/conditional_variance.py \
  --policy iter2_multistep_eval --data probe_n20000.npz --num 20000 \
  --knn 1 --radius 0.010 0.0125 0.015 0.0175 0.020 0.025 0.030 0.040 \
  --key-mode current --disjoint-groups --amp-rank 3 --amp-num 64 \
  --min-groups 30 --out current_pair_disjoint_n20000.json

python outputs/gauge/conditional_variance.py \
  --policy iter2_multistep_eval --data probe_n20000.npz --num 20000 \
  --knn 1 --radius 0.020 0.025 0.030 0.035 0.040 0.045 0.050 0.060 \
  --key-mode history --disjoint-groups --amp-rank 3 --amp-num 64 \
  --min-groups 30 --out history_pair_disjoint_n20000.json
```

## 3. 主结果

主判据取“达到 30 组要求的最小绝对半径”：

| 条件键 | 半径 | 独立组数 | cond std / scale | 各向异性 | amp span 方差占比 | 相对随机富集 | 判决 |
|---|---:|---:|---:|---:|---:|---:|---|
| current | 0.0125 | 32 | 0.01286 | 46.29 | 0.03961 | 2.54x | **KILL：近确定** |
| history | 0.0250 | 32 | 0.01372 | 41.89 | 0.04116 | 2.63x | **KILL：近确定** |

更重要的是半径扫描呈现清楚的局部漂移模式：

- `current`：半径 0.0125–0.0250 均 KILL；到 0.0300 才跨过
  0.02 阈值（0.02362）；
- `history`：半径 0.0250–0.0400 均 KILL；到 0.0450 才跨过
  0.02 阈值（0.02062）。

随着允许的条件失配增大，所谓“条件方差”单调增大并最终 PASS。这说明宽邻域
主要测到了确定性动力学随条件变化的局部漂移，而不是同条件下的不可约随机性。

### 3.1 8,000 样本复核

8,000 样本去重后为 7,977：

- `history` 最小可判半径 0.045、23 组，比例 0.01877，已触发 KILL；
- `current` 最小可判半径 0.025、26 组，比例 0.02481，曾是边界 PASS。

20,000 样本提供了更密的真实近邻，使两个条件键都能在更小半径、至少 30 个
独立 pair 上判定；`current` 的边界疑问因此消失。三帧 history 同时给出更符合
动力学条件化语义的否决。

### 3.2 对齐结果的修正解释

严格复核没有支持早期 smoke 中“完全不对齐”的说法。完整条件协方差在 top-3
放大子空间中的占比约 4.0%–4.3%，是随机基线的 2.4–2.8 倍，确有富集。

但这不挽救方向 B：其不可约方差量级在严格邻域内只有全局 latent scale 的
1.3%–1.7%，低于预注册的 2% 下限。换言之，方向结构存在于一个太小、会随
邻域收紧继续缩小的残差中；为它训练 stochastic covariance head 没有足够的
建模信号。

## 4. 正式判决

**方向 B：CLOSE。**

准确表述是：在当前 PushT、当前观测与三帧历史条件下，没有证据表明存在足够大
的不可约转移随机性来支撑 Generative LeWM。该结论不外推到部分可观测或本身
随机的环境，也不声称数学意义上的方差严格为零。

因此不再训练 stochastic-LeWM，不把微小 innovation 残差包装成新方法。主线只走
方向 A（Coupling as Geometry）。

## 5. 方向 A：A100 执行状态

### 5.1 正则权重校准

原始 head-to-head 配置的 `aux_beta=1` 不是公平基线：

- curvature：20-batch smoke 后训练、验证和全部学习参数均出现 NaN；
- bisim：raw auxiliary 已与预测项同量级或更大，明显改变了基础 K1 优化尺度。

只做尺度匹配后锁定：

| 基线 | beta | 100-batch raw aux | 加权 aux | pred loss | checkpoint |
|---|---:|---:|---:|---:|---|
| K1 + curvature | 0.05 | 1.3522 | 0.0676 | 0.4744 | 全部 tensor finite |
| K1 + bisim | 0.01 | 44.4047 | 0.4440 | 0.7208 | 全部 tensor finite |

两次 smoke 均约 2.3 iter/s。这个校准只避免“损坏的对手”，不是按最终规划结果
调参。

### 5.2 已启动和已排队任务

总 driver 于 `2026-07-23 11:09:34 UTC` 启动，远端 PID 为 `118149`。

第一阶段已在 4 张 A100 上并行运行：

- `curv_d192`，seed 3072
- `curv_d192_s1`，seed 1
- `bisim_d192`，seed 3072
- `bisim_d192_s1`，seed 1

全部为 30 epochs；运行至 step 1,200 时四卡利用率 96%–100%，单任务约
2.5 iter/s，每 epoch 11,306 steps，未见 NaN/OOM/traceback。训练完成后自动
执行 offset 25/40、seed 42/123/7 的规划评测。

第二阶段由同一 driver 自动排队：

- anchor+dose 的补种子与 dose wave，共 6 个 30-epoch 训练；
- 新任务统一添加 `_v2_20260723` 后缀，避免覆盖机器上已有的 epoch 13–15
  不完整产物；
- 随后执行相同 near/far-goal 评测。

远端状态与日志：

```text
/225010117/logs/trends_a100_20260723/driver.pid
/225010117/logs/trends_a100_20260723/driver.log
/225010117/logs/trends_a100_20260723/head2head/
/225010117/logs/trends_a100_20260723/verify/
```

数据从 `/dev/shm/pusht_expert_train.h5` 读取，关闭 GPFS pixel sidecar，并启用
GPU 图像预处理。总 driver 采用 `nohup`，SSH 断开不会终止任务。

## 6. 审计文件

- `current_pair_disjoint_n8000.json/.log`
- `history_pair_disjoint_n8000.json/.log`
- `current_pair_disjoint_n20000.json/.log`
- `history_pair_disjoint_n20000.json/.log`
- `probe_n20000.log`
- `curv_gpupp_smoke.log`
- `curv_b005_shm_smoke100.log`
- `bisim_b1_scale_smoke.log`
- `bisim_b001_shm_smoke100.log`

NPZ probe 约 160 MB，留在 A100：

```text
/225010117/logs/conditional_variance_strict_20260723/probe_n20000.npz
```
