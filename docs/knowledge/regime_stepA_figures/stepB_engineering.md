# Step B 工程深挖：到底训了什么、怎么训的

> 这是 Step B 的**工程级**说明，配合结论文档 [stepB_README.md](stepB_README.md) 与总报告
> [regime_direction_results.md](../regime_direction_results.md) §4 看。目标：让人能照着把整个训练复刻出来。
> 代码：`scripts/plan/regime_moe_stepB.py`（训练器）、`regime_stepB_eval_data.py`（造数据）、
> `run_regime_stepB.sh` / `run_regime_stepB_gatesup.sh`（编排）、`regime_stepB_aggregate.py` /
> `regime_stepB_roundA_figure.py`（聚合出图）。

---

## 0. 一句话：训的是什么

- **冻结**的是 LeWM 的 ViT encoder（一帧 → latent `z∈R^192`），**全程不训**。
- **训**的是一个**轻量 latent 动力学预测器** `f`：给过去 3 个 latent + 当前动作，预测**下一个 latent 的残差** `Δz`，即 `z_{t+1} = z_t + f(history, action)`。
- 全部在 **latent 空间**里训，**不碰像素**、**不反传进 encoder**。一个 run 在单卡 ~11 秒（8000 样本、60 epoch）。
- 三个 arm：`mono`（单体 MLP，基线）、`moe`（regime 门控的混合专家）、`oracle`（门控换成真接触标签的上界）。

> 核心问题：把动力学拆成"每个 regime 一个专家"，用**原 LeWM 损失（多步 unroll latent-MSE，不加任何聚类 loss）**训，能不能压平 drift、且门控自然对上接触。

---

## 1. 训练数据：每条样本长什么样、怎么来的

由 `regime_stepB_eval_data.py` 生成两个 npz（**走的就是总报告 §2 的公共管线 a→b→c**）：

| 文件 | 样本数 N | 用途 |
| --- | --- | --- |
| `train_contact.npz` | 8000 窗口 | 训练（内部再切 90/10 train/val） |
| `eval_contact.npz` | 1500 窗口 | **独立**测 gate→contact 对齐 |

每条样本三个数组（PushT，`action_block=5`，`max_k=10`）：

```
z            (11, 192)   真 latent 序列：专家轨迹的 11 个时间点（k=0..10）各编码一帧
a            (10, 10)    model_actions：每个模型步对应 5 个环境步×2 维动作，标准化后拼成 10 维
contact_frac (11,)       每个时间点所在 block 的接触比例 → 二值 regime 标签 = contact_frac>0
```

生成步骤（复用现成函数，没有重写物理）：
1. `build_window_batch`：从 PushT 专家集随机切 N 条窗口，拿到 `init_states / raw_actions / model_actions`。
2. `replay_windows`（来自 Step A）：用 `init_state` 重置环境、按 `raw_actions` 逐步重放，每个 block 边界渲一帧，并读 `n_contact_points` 聚合成 `contact_frac`。
3. `encode_frames`：冻结 LeWM encoder 把帧编码成 `z`。

> 注意：`z` 是**真值序列**（专家轨迹编码出来的"地面真相 latent"）。训练时模型要学的是"从真起点开环滚出去，能不能贴住这条真值序列"。

复现：
```bash
$PY scripts/plan/regime_stepB_eval_data.py --num-samples 8000 \
    --output outputs/regime_stepB/train_contact.npz
$PY scripts/plan/regime_stepB_eval_data.py --num-samples 1500 \
    --output outputs/regime_stepB/eval_contact.npz
```

---

## 2. 模型结构（精确层尺寸 + 参数量）

公共维度：`D=192`（latent 维）、`hs=3`（历史帧数）、`adim=10`（动作维）。
两个派生量：

```
in_dim    = hs*D + adim = 3*192 + 10 = 586     # 历史 latent 拼 + 当前动作（喂给专家）
state_dim = hs*D        = 3*192      = 576      # 只拼历史 latent（喂给门控）
```

### mono（单体基线）
一个 3 层 MLP，输入 `[历史3帧, 动作]`(586)，输出残差 `Δz`(192)：
```
Linear(586→h) → GELU → Linear(h→h) → GELU → Linear(h→192)
z_{t+1} = z_t + 该 MLP 输出
```
- `h=512`：参数 **0.66M**（= 586·512+512 + 512·512+512 + 512·192+192 = 661,696）
- `h=1024`：参数 **1.85M**（= 1,847,488）——这是 Step B 的**主对照**（和 MoE 参数对齐）。

### moe（regime 门控混合专家，K=2）
```
gate:    Linear(state_dim→h) → GELU → Linear(h→K)        # 只看历史 latent（state）
experts: K 个 和 mono 一样的 3 层 MLP，每个看 [历史, 动作]
路由:    p = softmax/gumbel/argmax(gate logits)          # (B,K)
Δz   =   Σ_k p_k · f_k([历史, 动作])                     # 专家加权和
z_{t+1} = z_t + Δz
```
- `h=512, K=2, gate_input=state`：参数 **1.62M**
  （gate 576·512+512 + 512·2+2 = 296,450；专家 2×661,696 = 1,323,392；合计 1,619,842）。
- `gate_input=both`：门控改看 586 维（含动作），参数 ≈ 1.62M（仅多 ~5k）。

> 设计依据（Step A）：regime 是**状态/算子**属性、残差被动作污染 ⇒ **门控默认只看 state**，专家才看动作。

### oracle（上界）
复用 MoE 的专家，但**把门控的输出直接替换成真接触标签的 one-hot**（K=2），训练和评测都用真路由。参数和 moe 一样，差别只在"路由从哪来"。

---

## 3. 怎么训：损失与训练循环（关键）

### 损失 = 多步开环 unroll 的 latent-MSE
不是单步 teacher-forcing，而是**开环滚 U=5 步、预测喂回预测**，再和真值比：

```python
# 每个 batch：
t0 = 随机起点 in [0, (K1-1)-U]          # 随机选窗口内的起滚位置，覆盖整条轨迹
hist = z[t0-2 .. t0]  (hs=3 帧真值 seed)  # 见 seed_history
for s in 0..U-1:                         # unroll U=5 步
    z_next = model.step(hist[-3:], a[t0+s])   # 用最近3帧预测下一帧残差
    hist.append(z_next)                       # 关键：把预测喂回去（开环）
loss = MSE( preds[1..U],  z_true[t0+1 .. t0+U] )
```
- **为什么开环而不是单步**：要测的是 **drift（误差累积）**。单步 teacher-forcing 每步都从真值起，看不到复合误差；开环才会暴露"regime 切换处一旦预测偏了，后面越滚越歪"。这正是 LeWM 的部署场景。
- **为什么随机 `t0`**：让模型在轨迹各处都被训到，而不是只学开头。

### 软路由 + 退火（只 MoE）
训练时门控用 **gumbel-softmax**（可微的软 one-hot），温度 `tau` 线性退火：
```python
tau = max(0.5, 1.0 * (1 - ep/epochs))   # 1.0 → 0.5
p   = gumbel_softmax(logits, tau, hard=False)   # 训练
# 评测时换成硬 argmax（或 --eval-soft 时 softmax(logits/tau)）
```
退火让早期探索（软、各专家都吃到梯度）、后期逼近硬路由（接近评测时的 argmax）。

### 优化器与超参（全部默认值，几乎不调）
```
optimizer = Adam(lr=1e-3)
batch_size = 512
epochs = 60
unroll U = 5,  hist hs = 3,  hidden = 512,  experts K = 2
train/val split = 90/10（按 seed 随机 perm）
seeds = 0,1,2（每个配置 3 个种子）
```

---

## 4. 四个探针的工程实现（Step B 的精髓）

为了把"regime 有没有用"和"门控学不学得到"**彻底拆开**，加了四个开关。每个都是几行代码：

| 探针 | CLI | 干了什么（代码层面） | 回答的问题 |
| --- | --- | --- | --- |
| **oracle 路由** | `--arm oracle` | `route_seq` = 真接触 one-hot，在 `model.step` 里 `if route is not None: p = route` 直接覆盖门控；训练+评测都用 | regime 若**完美已知**，到底有没有价值？（上界） |
| **弱监督门控** | `--gate-sup λ` | `unroll(..., want_logits=True)` 取出每步门控 logits；加 `λ·CE(logits, 真接触标签)` 到总损失 | 给门控一个接触先验，**可学门控**能否逼近 oracle？ |
| **干净专家** | `--train-route-gt` | 训练时 `route_tr`=真路由（专家像 oracle 一样干净特化），但 `route_va=None` ⇒ **评测用学出来的门控** | 差距是"软路由把专家混脏了"还是"测试时路由不准"？ |
| **软评测** | `--eval-soft` | 评测时 `p = softmax(logits/tau)` 而非 argmax | 差距是不是"硬 argmax 太脆"造成的？ |

关键代码片段（训练循环里加监督）：
```python
preds, logits = unroll(model, zb, ab, t0, U, hs, tau=tau,
                       route_seq=rb, want_logits=args.gate_sup>0)
loss = F.mse_loss(preds, zb[:, t0+1:t0+1+U])
if args.gate_sup > 0:
    tgt = lab_tr[bidx][:, t0+1:t0+1+U]                   # 每步的真接触标签
    loss += args.gate_sup * F.cross_entropy(
                logits.reshape(-1, K), tgt.reshape(-1))   # 弱监督门控
```

> 这套"oracle 先行 → 再逐个排除可实现路径"是 Step B 能给出**有机理结论**（而不是"没调好"）的关键。

---

## 5. 怎么评：两个指标

### (a) rollout mse@k（抗漂，核心）
在 **val 集**上，从 `k=0` 起**全程开环滚 10 步**（评测用硬路由）：
```python
preds = unroll(model, z_va, a_va, t0=0, steps=10, hs, hard=True, route_seq=route_va)
per_k = ((preds - z_va[:,1:11])**2).mean(over batch & dim)   # 每个 k 的 MSE
slope = 对 per_k 线性拟合的斜率                                # drift 增速
```
- `mse@10` 越低 = 滚 10 步后越贴真值 = 抗漂越好。`slope` 越小 = drift 增速越慢。
- 还记录 `natural_spread = z 的方差`（≈0.90）作参照尺度：mse@10 0.32 ≈ 自然尺度的 1/3。

### (b) gate→contact 对齐（门控有没有"发现"接触）
在**独立的 `eval_contact.npz`** 上，对每个内部转移取**硬门控编码**，和真接触标签算 NMI / purity：
```python
for k in range(hs-1, 10):
    code = model.step(z_ev[:, k-2:k+1], a_ev[:, k], hard=True).argmax   # 门控选的专家
    label = (contact_frac[:, k+1] > 0)                                  # 真 regime
NMI  = normalized_mutual_info_score(label, code)
purity = Σ 每个 code 内多数类占比 / 总数
```
- **重要细节**：这里门控吃的是**真 latent 窗口**（teacher-forced，非漂移态）。所以测的是"门控在干净输入上能不能预测接触"——这是门控能力的**上界**。即便如此，blind MoE 的 NMI≈0.007（盲），弱监督后能到 0.55（purity 0.91）但 drift 反而更差（见结论文档的机理）。
- purity 的**平凡地板** = max(接触率, 1−接触率) ≈ 0.60。

---

## 6. 怎么跑：编排与复现

**决定性轮**（mono 阶梯 + moe，3 seed，8 卡分发）：
```bash
bash scripts/plan/run_regime_stepB.sh \
    outputs/regime_stepB/train_contact.npz outputs/regime_stepB/eval_contact.npz \
    outputs/regime_stepB/decisive 60
# 单独补 oracle（同脚本，--arm oracle，需 contact_frac 在 --data 里）
$PY scripts/plan/regime_stepB_aggregate.py        # 出表 + stepB_decisive.png
```

**Round A**（弱监督门控扫描 gate-sup∈{0.1,0.3,1.0,3.0}×{state,both}，3 seed）：
```bash
bash scripts/plan/run_regime_stepB_gatesup.sh \
    outputs/regime_stepB/train_contact.npz outputs/regime_stepB/eval_contact.npz \
    outputs/regime_stepB/gatesup 60
# train-route-gt 与 eval-soft 探针用单跑命令组合（见 stepB_README 表）：
$PY scripts/plan/regime_moe_stepB.py --arm moe --experts 2 --gate-input state \
    --gate-sup 1.0 --train-route-gt --eval-soft \
    --data .../train_contact.npz --eval-contact .../eval_contact.npz \
    --epochs 60 --seed 0 --output-dir .../probe_run
$PY scripts/plan/regime_stepB_roundA_figure.py    # 出 stepB_roundA.png
```

单卡环境前缀（PushT 渲染需无头）：
```bash
SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy CUDA_VISIBLE_DEVICES=0 \
SWM_TORCH_THREADS=4 OMP_NUM_THREADS=4  $PY ...
```

---

## 7. 一图流总结（数据→训练→评测）

```
专家轨迹 ──build_window_batch──▶ init_state + actions
   │                                   │
   └─replay_windows─▶ 帧 + contact_frac │
        │                              │
   encode_frames(冻结ViT)              │
        ▼                              ▼
  z 真值序列(11,192) ───────────  model_actions(10,10) ──┐
        │                                                │
        ▼  90/10 split                                   │
   ┌─────────────── 训练循环（60 ep, Adam 1e-3, bs 512）──────────────┐
   │ 随机 t0 → seed 3 真帧 → 开环 unroll U=5（预测喂回）            │
   │ loss = MSE(pred, z真)  [+ λ·CE(gate, contact)  若 gate-sup]    │
   │ MoE 门控 gumbel-softmax，tau 1.0→0.5 退火                      │
   └────────────────────────────────────────────────────────────────┘
        ▼
   评测：val 上开环滚10步 → mse@k / slope；eval_contact 上 → gate→contact NMI/purity
```

结论见 [stepB_README.md](stepB_README.md)：oracle 有用（p=0.030）但任何可实现门控都比单体连续差 ⇒ MoE 形式判死。
