# 离散算子新开始：把离散放进转移，而不是状态

> 本 MD = 离散新分支。它接在 [diagnosis_world_model_drift.md](diagnosis_world_model_drift.md) 和 [commitment_anchor_discrete.md](commitment_anchor_discrete.md) 后面，但不重复 Phase 8 的路线。Phase 8 已经说明：量化全 latent、事后 snap-back、离散 commitment proposer 在 PushT LeWM latent 上都不占优。这里重新开始的点是：**状态保持连续，离散只负责选择动力学算子**。

记录日期：2026-06-23。

## 一句话

旧路线把离散当成 latent 表示或 waypoint 表示，容易撞上精度地板；新路线把离散当成 **transition operator / regime selector**，对应物理中的接触、非接触、推动、重定位等分段动力学。

连续状态负责精度，离散选择负责不连续切换。要验证的不是单步是否更准，而是长程 open-loop rollout 的误差是否更不复利。

## 为什么这条还没有被前面证伪

已证伪或明显不优的路线：

- 全 latent 量化或替换：Phase 8 显示量化地板和 drift 同量级，PushT 精细控制需要连续精度。
- 事后投影 / snap-back：Phase 7 显示 ID drift 是 on-manifold 扩散，不是简单 off-manifold 偏离。
- 离散 commitment proposer：Phase 8g/8h 中连续 proposer 明显强于离散 proposer。

这条新路线不同：

- 不量化状态 `z`。
- 不把离散码当 waypoint。
- 不修改 planner cost 作为第一步。
- 只在 latent dynamics 里让 `selector(z_history, action)` 选择一个局部算子 `A_c`。

候选机制：

```text
cont   : z_{t+1} = z_t + MLP([history, action])
disc   : z_{t+1} = z_t + sum_c p_c * A_c(z_t) + W_a(action)
disc_c : 同 disc，但 A_c 加 spectral norm 约束，试图让局部误差不复利
```

核心判据：

```text
如果 disc_c 在 k 较大时 rollout MSE 低于 cont，或曲线明显更平，
则“离散算子抗扩散”有实证支撑。

如果 cont 在所有 horizon 上都更好，
则这条离散动力学路线也应该被 kill 或收缩为负结果。
```

## 当前代码

服务器上的 AI 新增并已同步到本地的提交：

```text
b6f6d6a Add discrete operator scripts
```

文件：

```text
scripts/plan/disc_operator_gendata.py
scripts/plan/disc_operator_train.py
scripts/plan/run_disc_operator_sweep.sh
```

作用：

- `disc_operator_gendata.py`：从 PushT HDF5 抽专家轨迹窗口，使用冻结 LeWM encoder 编码 `z_0..z_K`，并保存模型动作块。
- `disc_operator_train.py`：训练 `cont / disc / disc_c` 三种 latent dynamics predictor，输出 open-loop rollout MSE 曲线和诊断统计。
- `run_disc_operator_sweep.sh`：把组合实验分发到 GPU 1-7，每张卡串行跑自己的队列。

静态检查状态：

```text
python -m py_compile scripts/plan/disc_operator_gendata.py scripts/plan/disc_operator_train.py
bash -n scripts/plan/run_disc_operator_sweep.sh
```

两项均通过。

## 结果存放约定

### 原始运行输出

默认不进 Git，因为 `outputs/` 被 `.gitignore` 忽略。

生成 latent 数据：

```text
outputs/disc_operator/latent_seq.npz
```

这个文件由 `disc_operator_gendata.py` 生成，包含：

```text
z      : train latent sequence, shape (N, K+1, D)
a      : train action blocks, shape (N, K, action_block * action_dim)
z_val  : validation latent sequence
a_val  : validation action blocks
```

sweep 输出根目录由脚本第二个参数决定，例如：

```text
outputs/disc_operator/sweep_YYYYMMDD_HHMM/
```

该目录下：

```text
orchestrator.log
cont_K0_U1_s0/run.log
cont_K0_U1_s0/result.json
disc_K8_U5_s1/run.log
disc_K8_U5_s1/result.json
disc_c_K16_U5_s2/run.log
disc_c_K16_U5_s2/result.json
...
```

每个 `result.json` 主要字段：

```text
rollout_mse_vs_k
rollout_mse_over_spread
natural_spread
selector_usage
operator_spectral_radii
elapsed_sec
```

### Curated 结果

如果实验有保留价值，把最小摘要复制进 Git：

```text
docs/knowledge/disc_operator_figures/
```

建议只放：

```text
summary.csv
summary.json
rollout_mse_vs_k.png 或 .svg
selector_usage.csv
operator_spectral_radii.csv
```

不要把完整 `latent_seq.npz`、训练中间权重、大日志、完整 `outputs/` 放进 Git。

## 推荐运行命令

在服务器仓库：

```bash
cd ~/code/wge/learn_wm
```

先确认 GPU 和 Git key：

```bash
nvidia-smi
echo "$GIT_SSH_COMMAND"
git status -sb
```

生成 latent 序列：

```bash
CUDA_VISIBLE_DEVICES=1 SWM_TORCH_THREADS=2 OMP_NUM_THREADS=2 \
  .venv/bin/python scripts/plan/disc_operator_gendata.py \
  --h5 /home/jovyan/.stable_worldmodel/pusht_expert_train.h5 \
  --output outputs/disc_operator/latent_seq.npz \
  --num-samples 40000 \
  --val-samples 4000 \
  --max-k 10 \
  --action-block 5 \
  --device cuda
```

跑 sweep：

```bash
bash scripts/plan/run_disc_operator_sweep.sh \
  outputs/disc_operator/latent_seq.npz \
  outputs/disc_operator/sweep_$(date +%Y%m%d_%H%M) \
  60
```

快速 smoke 可以把样本数和 epochs 降低：

```bash
CUDA_VISIBLE_DEVICES=1 SWM_TORCH_THREADS=2 OMP_NUM_THREADS=2 \
  .venv/bin/python scripts/plan/disc_operator_gendata.py \
  --output outputs/disc_operator/smoke_latent_seq.npz \
  --num-samples 512 \
  --val-samples 128 \
  --max-k 10 \
  --device cuda

CUDA_VISIBLE_DEVICES=1 SWM_TORCH_THREADS=2 OMP_NUM_THREADS=2 \
  .venv/bin/python scripts/plan/disc_operator_train.py \
  --data outputs/disc_operator/smoke_latent_seq.npz \
  --output-dir outputs/disc_operator/smoke_cont \
  --arm cont \
  --epochs 2 \
  --device cuda
```

## 当前注意点

1. `run_disc_operator_sweep.sh` 写死使用 GPU 1-7。真跑前必须看 `nvidia-smi`，避免占用别人正在用的卡。
2. `run_disc_operator_sweep.sh` 默认使用 `.venv/bin/python`。如果服务器环境名变了，需要改 `PY=` 或显式用目标 Python 调用。
3. `disc_operator_gendata.py` 默认 HDF5 路径是 `/home/jovyan/.stable_worldmodel/pusht_expert_train.h5`。如果数据软链或路径变了，需要用 `--h5` 覆盖。
4. `disc_c` 当前只对 `A_c` 做 spectral norm。完整更新是 `z + A_c(z) + W_a(a)`，所以“整体非扩张”还不是严格数学保证，只能作为实验性 inductive bias。
5. 当前结果目标是 latent rollout MSE。即使 `disc_c` 赢了，还需要后续接回 planner 或 behavior eval，不能直接等同于控制成功率提升。

## 服务器 Git/SSH 状态记录

服务器路径：

```text
~/code/wge/learn_wm
```

目标行为：

```text
只在 ~/code/wge/ 路径下使用 ~/.ssh/id_ed25519_wge
```

已验证：

```text
ssh -i ~/.ssh/id_ed25519_wge -o IdentitiesOnly=yes -T git@github.com
```

返回 GitHub 认证成功。

注意：

- 裸 `ssh -T git@github.com` 不一定成功，因为它不读取 Git 的 `core.sshCommand`。
- Coder 环境可能注入 `coder gitssh`，所以在 `~/code/wge/` 下需要让 shell 设置 `GIT_SSH_COMMAND` 或清掉 Coder 的 `GIT_SSH`。
- 这个限制是服务器本地路径限制；如果 key 加在 GitHub 账号级 SSH key，GitHub 权限仍是账号级。若要 GitHub 权限也限制在单仓库，使用 repo deploy key 并勾选 write access。

## 当前 Markdown 地图

研究笔记：

```text
docs/knowledge/diagnosis_world_model_drift.md       诊断线：LeWM PushT latent drift，Phase 1-7
docs/knowledge/commitment_anchor_discrete.md        解决线：commitment anchor，Phase 8
docs/knowledge/discrete_operator_new_start.md       新离散线：discrete transition operator
docs/idea_brainstorm_unchosen.md                    B 线候选池，尚未 choose 的 idea
```

项目文档：

```text
README.md                                           项目总览、安装、quick start、环境和 baseline
docs/index.md                                       文档首页
docs/quick_start.md                                 World/Dataset/Evaluation/Planning 快速上手
docs/cli.md                                         swm CLI
docs/baselines.md                                   DINO WM、PLDM、LeWM、TD-MPC2、GCBC、IQL
```

API 文档：

```text
docs/api/dataset.md
docs/api/policy.md
docs/api/solver.md
docs/api/spaces.md
docs/api/world.md
docs/api/wrapper.md
```

环境文档：

```text
docs/envs/ale.md
docs/envs/craftax.md
docs/envs/dmc.md
docs/envs/gymnasium_control.md
docs/envs/gymnasium_robotics.md
docs/envs/ogb.md
docs/envs/piecewise.md
docs/envs/pusht.md
docs/envs/tworoom.md
```

教程和指南：

```text
docs/guides/checkpoints.md
docs/guides/online_learning.md
docs/tutorial/collect_data.md
docs/tutorial/new_env.md
docs/tutorial/training_wm.md
```

