# Step B 工程：推理 pipeline 与训练

> 配合结论 [stepB_README.md](stepB_README.md) 与总报告 [regime_direction_results.md](../regime_direction_results.md) §4 看。
> 本文回答两件事：**(1) 我们的推理 pipeline 和 LeWM 原版差在哪**（用了哪些 module、替换了哪一格、训了个什么、它在 pipeline 哪个位置）；**(2) 这个东西用什么数据、怎么训的**。
> 代码：`scripts/plan/regime_moe_stepB.py`（训练器）、`regime_stepB_eval_data.py`（造数据）、`run_regime_stepB*.sh`（编排）。

---

# Part 1 — 推理 pipeline：LeWM 原版 vs 我们的

## 1.1 LeWM 原版有哪些 module（`stable_worldmodel/wm/lewm/`）

LeWM 是一个**像素世界模型**，推理时五个 module 串起来（`lewm.py`）：

| module | 类型 | 作用 | 训练状态（预训练 checkpoint） |
| --- | --- | --- | --- |
| `encoder` | ViT-tiny（D=192，12 层） | 一帧像素 → patch 特征 → 取 cls token | 预训练好 |
| `projector` | 线性/Identity | cls token → latent `z∈R^192` | 预训练好 |
| `action_encoder` | MLP | 原始动作 → 动作 embedding `c`（条件） | 预训练好 |
| **`predictor`** | **Transformer**（`module.py:Predictor`，ConditionalBlock 受 `c` 条件，pos-emb 覆盖 `num_frames=3`） | 最近 3 个 latent + 动作条件 → 预测下一个 latent | 预训练好 |
| `pred_proj` | 线性/Identity | predictor 输出 → latent 空间 | 预训练好 |

**原版推理数据流**（`encode` → `rollout`）：
```
像素帧 ─encoder→ cls ─projector→ z
动作   ─action_encoder→ c
                         │
   (z 的最近3帧, c) ─predictor(Transformer)→ pred_proj → 下一个 z'
                         │  自回归：把 z' 喂回当历史，继续滚
   规划(planning)：rollout 出一串 z'，criterion = MSE(末步 z', goal_emb)，CEM 选动作
```

**重点：LeWM 自己就有一个 `predictor`（Transformer），它就是"在 latent 空间往前滚"的那个动力学模型。**

## 1.2 我们 Step B 的推理：复用什么、替换什么、训了什么

Step B 的问题是"**把 latent 动力学换成 regime 分段的形式，会不会更抗漂**"。所以我们：

- **复用（冻结）`encoder + projector`**：只拿它们把帧变成真值 latent `z`（即公共管线的 (c) 步）。**不微调、不反传**。
- **替换掉 `predictor` 这一格**：不使用 LeWM 自带的 Transformer predictor，而是在**同一个 latent 空间**里训一个**我们自己的、刻意做小的预测器**，放在 predictor 的位置上往前滚。
- **不使用 `action_encoder` / `pred_proj`**：我们的预测器直接吃**标准化后的原始动作向量**（10 维 = 5 环境步×2），不经过 LeWM 的动作 embedding；输出直接就是 latent 残差。

换句话说，**我们训的"额外的东西"= 一个替身 predictor**——它占据 pipeline 里 `predictor` 那一格，但形式由我们控制：

```
mono : z' = z + MLP([最近3帧 z, 动作])                       ← 单体（和 LeWM predictor 同角色，但是小 MLP）
moe  : z' = z + Σ_k g_k(最近3帧 z) · f_k([最近3帧 z, 动作])  ← regime 分段（门控 + K 个专家）
oracle: 同 moe，但门控换成真接触标签（上界）
```

## 1.3 module 对照表（一眼看清用了/没用/训了）

| LeWM module | 我们用了吗 | 训练状态 |
| --- | --- | --- |
| encoder | ✅ 用（把帧→z） | ❄️ 冻结 |
| projector | ✅ 用 | ❄️ 冻结 |
| action_encoder | ❌ 不用（直接喂原始标准化动作） | — |
| **predictor（Transformer）** | ❌ **不用，被我们的替身顶替** | — |
| pred_proj | ❌ 不用 | — |
| **我们的 mono / moe 预测器** | ✅ 这就是被测对象 | 🔥 **从零训练** |

## 1.4 为什么是"重训一个小预测器"而不是"改 LeWM 自带的 predictor"

- **受控对比**：要回答的是"动力学的**形式**（单体 vs 分段）哪个好"。用一对**参数对齐的极简模型**（mono vs moe）能把 regime 这个变量单独拎出来，不被 LeWM 那个具体 Transformer 架构 + 预训练权重的混杂因素污染。
- **便宜**：小 MLP 一个 run 单卡 ~11 秒；微调一个预训练 Transformer 又慢又难对齐参数预算。
- **诚实的 caveat**：因此我们的 `mono` **不是** LeWM 自带的 predictor，而是"一个通用的单体 latent 预测器"。结论是关于 **"在 LeWM 的 `z` 空间里、给 latent 动力学加 regime 分段这个原理"**有没有用，**不是**关于 LeWM 那个 Transformer predictor 本身。但因为 mono/moe/oracle 三者**完全同构、参数对齐**，三者之间的对比是干净的。

---

# Part 2 — 训练数据：每条样本长什么样、怎么造的

由 `regime_stepB_eval_data.py` 生成两个 npz（走的就是总报告 §2 的公共管线 a→b→c）：

| 文件 | 样本数 N | 用途 |
| --- | --- | --- |
| `train_contact.npz` | 8000 窗口 | 训练（内部再切 90/10 train/val） |
| `eval_contact.npz` | 1500 窗口 | **独立**测 gate→contact 对齐 |

每条样本三个数组（PushT，`action_block=5`，`max_k=10`）：
```
z            (11, 192)   真 latent 序列：专家轨迹 11 个时间点(k=0..10)各编码一帧（encoder 冻结产出）
a            (10, 10)    model_actions：每个模型步=5 环境步×2 维动作，标准化后拼成 10 维
contact_frac (11,)       每个时间点所在 block 的接触比例 → 二值 regime 标签 = contact_frac>0
```

怎么造（复用现成函数，没重写物理）：
1. `build_window_batch`：从专家集随机切 N 条窗口 → `init_states / raw_actions / model_actions`。
2. `replay_windows`（Step A 的）：用 `init_state` 重置环境、按 `raw_actions` 逐步重放，每个 block 边界渲一帧、读 `n_contact_points` 聚合成 `contact_frac`。
3. `encode_frames`：冻结 encoder 把帧编码成 `z`。

> `z` 是**真值序列**（专家轨迹编码出的"地面真相 latent"）。训练就是要模型"从真起点开环往前滚，能不能贴住这条真值序列"。

```bash
$PY scripts/plan/regime_stepB_eval_data.py --num-samples 8000 --output outputs/regime_stepB/train_contact.npz
$PY scripts/plan/regime_stepB_eval_data.py --num-samples 1500 --output outputs/regime_stepB/eval_contact.npz
```

---

# Part 3 — 我们训练的预测器：结构 + 参数量

公共维度：`D=192`、`hs=3`（历史帧数）、`adim=10`（动作维）。派生：
```
in_dim    = hs*D + adim = 586     # 历史3帧 + 动作 → 喂给专家
state_dim = hs*D        = 576     # 只历史3帧    → 喂给门控（Step A：regime 是状态属性）
```

**mono（单体基线）**：3 层 MLP，`[历史,动作](586) → Δz(192)`，`z' = z + Δz`
- `h=512` → **0.66M**；`h=1024` → **1.85M**（= Step B 主对照，和 MoE 参数对齐）。

**moe（K=2 门控混合专家）**：
```
gate    : Linear(576→h)→GELU→Linear(h→2)          # 只看 state
experts : 2 个和 mono 一样的 3 层 MLP（看 [历史,动作]）
Δz = Σ_k softmax/gumbel/argmax(gate)_k · f_k(...)
```
- `h=512` → **1.62M**（gate 296k + 2×expert 661k = 1,619,842）。

**oracle**：结构同 moe，仅把门控输出替换成真接触 one-hot（训练+评测都用真路由）。

---

# Part 4 — 怎么训：损失与训练循环

### 损失 = 多步开环 unroll 的 latent-MSE（LeWM 的目标，不加聚类 loss）
```python
t0 = 随机起点 in [0, (K1-1)-U]            # 覆盖轨迹各处
hist = z[t0-2..t0]  (3 帧真值 seed)
for s in 0..U-1:                          # U=5 步
    z_next = model.step(hist[-3:], a[t0+s])
    hist.append(z_next)                   # 关键：预测喂回预测（开环）
loss = MSE(preds[1..U], z_true[t0+1..t0+U])
```
- **为什么开环不是单步**：要测 **drift（累积误差）**。单步 teacher-forcing 每步从真值起，看不到复合误差；开环才暴露"regime 切换处一偏，后面越滚越歪"——这正是部署场景。

### 软路由 + 退火（只 MoE）
```python
tau = max(0.5, 1.0*(1 - ep/epochs))        # 1.0 → 0.5
p   = gumbel_softmax(logits, tau, hard=False)   # 训练用软（可微）
# 评测换硬 argmax（或 --eval-soft 时 softmax(logits/tau)）
```

### 超参（几乎全默认，不调参）
```
Adam lr=1e-3 | batch=512 | epochs=60 | U=5 | hs=3 | hidden=512 | K=2
train/val=90/10（按 seed 随机 perm） | seeds=0,1,2
```

---

# Part 5 — 四个探针（把"regime 有没有用"和"门控学不学得到"拆开）

| 探针 | CLI | 代码层面干了什么 | 回答 |
| --- | --- | --- | --- |
| **oracle 路由** | `--arm oracle` | `model.step` 里 `if route is not None: p = route`，用真接触 one-hot 覆盖门控（训练+评测） | regime 若完美已知有没有价值？（上界） |
| **弱监督门控** | `--gate-sup λ` | `unroll(want_logits=True)` 取每步 logits，损失加 `λ·CE(logits, 真接触)` | 给接触先验，可学门控能否逼近 oracle？ |
| **干净专家** | `--train-route-gt` | 训练用真路由喂专家（干净特化），`route_va=None` ⇒ 评测用学出的门控 | 差距是"软路由混脏专家"还是"测试时路由不准"？ |
| **软评测** | `--eval-soft` | 评测 `p=softmax(logits/tau)` 而非 argmax | 差距是不是"硬 argmax 太脆"？ |

```python
# gate-sup 的核心两行（训练循环里）：
preds, logits = unroll(model, zb, ab, t0, U, hs, tau=tau, route_seq=rb, want_logits=args.gate_sup>0)
loss = F.mse_loss(preds, zb[:, t0+1:t0+1+U])
if args.gate_sup > 0:
    tgt = lab_tr[bidx][:, t0+1:t0+1+U]
    loss += args.gate_sup * F.cross_entropy(logits.reshape(-1, K), tgt.reshape(-1))
```

---

# Part 6 — 怎么评

**(a) rollout mse@k（抗漂，核心）**：val 上从 k=0 全程开环滚 10 步（评测硬路由），算每个 k 的 MSE 与线性斜率 slope。还记录 `natural_spread`=z 方差(≈0.90) 作尺度。

**(b) gate→contact 对齐**：在**独立 `eval_contact.npz`** 上，对每个内部转移取硬门控编码，与真接触标签算 NMI/purity。注意门控吃的是**真 latent 窗口**（teacher-forced，非漂移态）——这是门控能力的**上界**；即便如此 blind MoE NMI≈0.007，弱监督到 0.55 但 drift 反而更差（机理见结论文档）。purity 平凡地板 = max(接触率,1−接触率)≈0.60。

---

# Part 7 — 复现 + 一图流

```bash
# 单卡前缀（PushT 渲染需无头）：
ENV="SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy CUDA_VISIBLE_DEVICES=0 SWM_TORCH_THREADS=4 OMP_NUM_THREADS=4"
# 决定性轮（mono 阶梯 + moe + oracle，3 seed，8 卡）
bash scripts/plan/run_regime_stepB.sh outputs/regime_stepB/train_contact.npz outputs/regime_stepB/eval_contact.npz outputs/regime_stepB/decisive 60
$PY scripts/plan/regime_stepB_aggregate.py            # 表 + stepB_decisive.png
# Round A：弱监督扫描 + 探针组合
bash scripts/plan/run_regime_stepB_gatesup.sh outputs/regime_stepB/train_contact.npz outputs/regime_stepB/eval_contact.npz outputs/regime_stepB/gatesup 60
$PY scripts/plan/regime_stepB_roundA_figure.py        # stepB_roundA.png
```

```
                LeWM 原版                          我们的 Step B
  像素 ─encoder→z ─action_encoder→c          像素 ─encoder(冻结)→ z 真值序列
        │                                            │（仅借 encoder 造数据）
   [z×3, c] ─predictor(Transformer)→ z'        [z×3, 原始动作] ─我们的 mono/moe(从零训)→ z'
        │ 自回归 rollout / CEM 规划                  │ 开环 unroll U=5，MSE 贴真值序列
        ▼                                            ▼
   原版世界模型推理                            测：mse@k / slope（抗漂）、gate→contact NMI
```

结论见 [stepB_README.md](stepB_README.md)：oracle 有用（p=0.030）但任何可实现门控都比参数对齐的单体连续差 ⇒ regime-as-MoE 形式判死。
