# 离散 regime 方向：完整结果与实验方法（A 绿 / B 红 / C 红）

> 本文 = 这条方向的**总报告**：把三步实验的**逻辑（为什么这么测）**和**工程（具体怎么实现、每个指标是什么）**讲清楚，让没读过代码的人也能完全看懂。
> 方向定案与理论地基见 [direction_discrete_regime_from_lewm.md](direction_discrete_regime_from_lewm.md)；各步原始 run 笔记见 [regime_stepA_figures/](regime_stepA_figures/README.md)（A）、[stepB_README.md](regime_stepA_figures/stepB_README.md)、[stepC_README.md](regime_stepA_figures/stepC_README.md)。

---

## 0. 一句话结论

> 离散 regime **真实存在**于训好的 LeWM 动力学 `f` 里（Step A，绿灯，paper 级描述性发现），但它**不转化为可用的控制杠杆**：当预测器开关会因路由错误而脆弱（Step B，红），当 re-ground 触发器会输给"无脑均匀"（Step C，红）。两个失败同源——**知道"动力学哪里特殊"帮不了"该怎么动"**。

| 步骤 | 问题 | 结论 | 关键数字 |
| --- | --- | --- | --- |
| **A 存在性** | regime 在不在训好的 `f` 里？ | **绿** | Jacobian↔接触 NMI **0.30** / silhouette **0.40**；`z` 对照 ≈0 |
| **B 可用性** | 把 regime 当预测器开关，能抗漂吗？ | **红** | oracle 路由有用（mse@10 0.32 vs 0.37，p=0.030）；但**任何可实现门控都比单体连续差** |
| **C 控制层** | 把 regime 当 re-ground 触发器，比均匀好吗？ | **红** | 连 oracle 边界也输给等预算均匀（0.095 vs **0.065**，p<0.001） |

---

## 1. 背景：为什么这条方向值得测（30 秒理论）

LeWM 的训练目标 = **预测损失 + SIGReg**。SIGReg 把 latent 的**边缘分布** `p(z)` 正则成各向同性高斯（无簇、无模、无偏好方向）。

**推论**：任何"在表征 `z` 上找离散结构"（VQ / 聚类 / 码本）注定失败——在一个被显式正则成"无结构"的分布里找簇，找不到是必然。这一条解释了之前所有表征级离散负结果。

**关键转折**：SIGReg 只约束**边缘** `p(z)`，**不约束转移** `p(z'|z,a)`。而接触物理是**分段的**（自由移动 / 接触 / 推 / 脱离 = hybrid dynamical system）⇒ 离散 regime 应该天然藏在**动力学 `f`** 里，而不是状态 `z` 里。

这给出三个可证伪的台阶：**A** 它在不在 `f` 里？→ **B** 在的话，把它当预测器开关能不能让动力学更好？→ **C** 再不行，把它当"何时重置"的监控信号能不能帮 rollout？

> **总方法论（贯穿三步）**：*分析在前、训练在后、便宜在前、贵在后；每步一个 go/kill 判据；先用 oracle（上界）做决定性测试，oracle 都不行就不必做可实现版*。Step B/C 都用了"oracle 先行"——这是省算力又不自欺的关键。

---

## 2. 公共工程地基（三步共用的数据管线）

所有实验都建立在同一条"专家轨迹 → 帧 → 接触标签 → latent `z`"的管线上。理解这一节，后面三步就只是"在 `z`/`f` 上做不同的事"。

**(a) 切窗口** `build_window_batch`（`latent_drift_phase3.py`）
- 从 PushT 专家数据集随机采 N 条窗口。每条窗口 = 一段专家轨迹。
- `action_block=5`：把 **5 个环境步打包成 1 个"模型步"**（LeWM 在抽稀的时间轴上预测）。`max_k=10` ⇒ 每条窗口 10 个模型步、11 个时间点 `k=0..10`。
- 产出：`init_states`（起始真实状态，7 维：agent xy / block xy / angle / …）、`raw_actions`（原始逐步动作）、`model_actions`（标准化+按 block 拼接的动作，形状 `(N,10,5*adim)`，喂给模型）。

**(b) 回放取接触 + 帧** `replay_windows`（`regime_existence_stepA.py`）
- 用 `init_state` 重置 PushT 环境，按 `raw_actions` **逐步重放**，在每个 block 边界渲染一帧。
- 每一步读 `env.unwrapped.n_contact_points`（pymunk 物理引擎报告的接触点数），按 block 聚合成：
  - `contact_frac[k]` = 该 block 内有接触的环境步比例；
  - `contact_max[k]` = 该 block 内接触点数最大值。
- **二值 regime 标签** = `contact_frac > 0`（接触 / 非接触）。PushT 上接触率 ≈ 0.60。
- 产出 `frames (N,11,224,224,3)` 和 `contact_frac (N,11)`。

**(c) 编码成 latent** `encode_frames`
- 冻结的 `quentinll/lewm-pusht`（ViT-tiny，`hidden_size=192`，12 层）把每帧编码成 `z`（cls token → projector），`D=192`。
- 产出真实 latent 序列 `z_true (N,11,192)`。这就是动力学 `f` 作用的空间。

> 一句话：**(a) 切轨迹 → (b) 重放拿到"每一步有没有接触" → (c) 编码拿到"每一步的 latent `z`"**。接触标签是我们的 ground-truth regime；`z` 序列是动力学的舞台。

---

## 3. Step A — 存在性：regime 在 `f` 里，不在 `z` 里（绿灯）

### 逻辑
理论说：离散**不可能**活在 `z`（被 SIGReg 正则成无结构），但**可能**活在转移 `f`。所以直接对比三种特征的可聚类性：
1. `z` 本身（**对照组**，理论预测无结构）；
2. 残差方向 `f(z,a)−z` 的方向；
3. **Jacobian** `df/dz`（`f` 的局部线性算子 = "此刻动力学长什么样"）。

如果只有 Jacobian 成簇、且簇对得上接触，而 `z` 不成簇 → 存在性成立。

### 工程
- 对每条转移，算三种特征：
  - `z`：直接拿 latent。
  - 残差方向：`(f(z,a)−z)` 归一化。
  - **Jacobian**：用 `torch.func.jacrev + vmap` 对 `f` 关于 `z` 求雅可比（192×192），再取其 **top-32 奇异值谱**作为该局部算子的指纹（谱 = 旋转/拉伸的强度分布，与坐标无关）。
- 聚类对齐：每种特征 `StandardScaler → PCA → KMeans(k=2..8)`，取 silhouette 最优的 k；再用接触标签算对齐度。
- 规模：800 窗口 → 6400 条转移，Jacobian 在其中 4000 条上算，单卡 271 秒。

### 怎么读指标（重要）
- **silhouette**（簇内紧 / 簇间分）：0≈没有簇结构，越高越成簇。
- **NMI / ARI**（簇 vs 接触标签的互信息 / 调整兰德）：0=独立，1=完全一致。
- **purity**：每个簇里多数类占比。**注意平凡地板** = `max(接触率, 1−接触率) ≈ 0.60`——光靠"全猜接触"就能拿 0.60，所以 purity 要和 0.60 比才有意义。

### 结果（绿灯）

| 特征 | 最佳 k | silhouette | 接触 NMI | ARI | purity |
| --- | --- | --- | --- | --- | --- |
| `z`（对照） | 8 | 0.068 | 0.043 | 0.043 | 0.68 |
| 残差方向 | 8 | 0.037 | 0.097 | 0.068 | 0.69 |
| **Jacobian 谱** | **2** | **0.396** | **0.298** | **0.371** | **0.81** |

读法：
- **`z` 无结构、对接触盲**——正是 SIGReg 各向同性的预测。
- **`f` 的局部算子强成簇、强对齐接触**——silhouette ≈ 6×、NMI ≈ 7× 于 `z`。最佳 k=2 = **接触/非接触二元开关**。
- **残差方向信号弱**（被动作驱动污染）⇒ 后续门控应条件在**状态/算子**而非残差。

> Step A 本身就是一条干净的描述性结论：**SIGReg 把离散从 `z` 里赶走了，但它躲进了 `f`。** 图 `regime_stepA_figures/stepA_silhouette_nmi.png`。

---

## 4. Step B — 可用性：把 regime 当预测器开关（红灯）

### 逻辑
存在 ≠ 有用。Step B 问：把动力学拆成"每个 regime 一个专家"的 piecewise 预测器，用**原 LeWM 损失（多步 unroll，不加任何聚类 loss）**训，能不能压平 drift？判据（文档）：**压平 mse@k 斜率 + 门控无监督对上接触 → 落地；压不平 → 死。**

### 工程：模型与三个关键探针
**MoE 预测器**：
```
z_{t+1} = z_t + Σ_k  g_k(state_hist) · f_k([hist, action])
```
- 门控 `g`：只看**状态历史**（Step A：regime 是状态/算子属性，残差被动作污染）。训练用 gumbel-softmax（软），评测用 argmax（硬）。
- 专家 `f_k`：各自是完整 MLP，看动作。
- 对照：**单体 mono**（一个 MLP，没有开关）。

**训练损失** = 多步开环 unroll 的 latent-MSE：从真 `z` 起步，预测 U=5 步（预测喂回预测），和真 `z` 比。这就是 LeWM 的目标，**没有加聚类 loss**——regime 必须从动力学拟合里自然涌现。

为了把"regime 有没有用"和"门控学不学得到"**拆开**，加了三个探针（这是本步的工程核心）：
1. **oracle 路由**：把门控直接换成**真接触标签的 one-hot**。回答"如果路由完美，regime 到底有没有用？"——这是**上界**。
2. **gate-sup（弱监督门控）**：训练时加一项 `CE(门控 logits, 真接触标签)`，权重可调。门控仍可学（不像 oracle 硬接死），看"给个接触先验，可学门控能不能逼近 oracle"。
3. **train-route-gt（干净专家）**：训练时用真标签路由专家（让专家像 oracle 一样干净特化），但**评测时用学出来的门控**。隔离"专家混合不干净"这个可能原因。
4. **eval-soft**：评测时用 softmax 混合专家（而非硬 argmax），看"硬路由太脆"是不是原因。

### 怎么读指标
- **mse@10**：开环滚 10 步后的 latent-MSE（越低 = 抗漂越好）。这是核心。
- **slope**：mse@k 对 k 线性拟合的斜率（drift 增速）。
- **gate→contact NMI**：硬门控编码 vs 真接触的互信息（门控有没有"发现"接触）。

### 结果 1（决定性轮，3 seeds）

| 配置 | 参数 | mse@10 | slope | gate-NMI |
| --- | --- | --- | --- | --- |
| mono-h512 | 0.66M | 0.430 | 0.0413 | — |
| **mono-h1024（连续基线）** | 1.85M | **0.368** | 0.0365 | — |
| moe-state | 1.62M | 0.403 | 0.0392 | **0.008** |
| moe-both（门控也看动作） | 1.62M | 0.392 | 0.0379 | **0.006** |
| **oracle（真标签路由）** | 1.62M | **0.320** | 0.0304 | （给定） |

- **oracle 显著压平 drift**：mse@10 0.320 vs 参数对齐的 mono-wide 0.368，Δ=−0.048，t=3.29，**p=0.030**。⇒ **regime 真有用（假设"regime 无用"被证伪）**。
- 但**学出来的门控瞎了**：NMI ≈ 0.006–0.008（即使把动作喂进门控），等参数下还输给 mono-wide。⇒ 纯 rollout loss 无法让门控发现 Step A 已证存在的 regime。

### 结果 2（Round A：可学门控能逼近 oracle 吗？三种干预全失败）

| 变体 | 需要测试时真标签? | mse@10 | gate-NMI / purity |
| --- | --- | --- | --- |
| **oracle** | **是** | **0.320** | 1.0 |
| mono-wide（连续基线） | 否 | 0.368 | — |
| blind MoE | 否 | 0.403 | 0.007 |
| 干净专家 + 可学门控（软评测） | 否 | 0.475 | 0.56 |
| 干净专家 + 可学门控（硬评测） | 否 | 0.486 | 0.56 |
| 监督门控 gs=1.0（软评测） | 否 | 0.510 | 0.55 |
| 监督门控 gs=1.0（硬评测） | 否 | 0.521 | 0.55 |

并且监督越强 drift 越差：gs0.1→0.405、gs1.0→0.521、gs3.0→0.682（门控 NMI 都到了 0.55，但 drift 单调恶化）。

> **每个可实现变体都比朴素连续单体（0.368）差**，只有需要"测试时真标签"的 oracle 能赢。图 `regime_stepA_figures/stepB_roundA.png`。

### 机理（这才是有价值的部分）
把专家**特化**到 regime，会让模型对**路由错误脆弱**：错路一个特化专家，比单体泛化器的平均误差**更糟**。状态门控在真状态上最多 ~9% 路由错（purity 0.91），而开环 rollout 里门控看到的是**漂移过的状态**，路由更差、还逐步累积。软路由不脆弱，但它把专家又**混成了泛化器**，把 regime 收益归零。**内在张力：硬路由有收益但脆弱；软路由稳健但≈单体。** oracle 赢只因它**从不错路**——而它需要的接触标签恰恰是门控无法从漂移 latent 可靠预测的量（接触@t+1 由动作和精细几何决定）。

> 结论：Step B 的 latent-MoE 形式按文档判据"压不平→方法死"判死。

---

## 5. Step C — 控制层：把 regime 当 re-ground 触发器（红灯）

### 逻辑
Step B 死了"当预测器开关"。regime 还剩一个更弱的用法：当**监控信号**——不路由脆弱的预测器，只决定**何时 re-ground**（把漂移的预测 latent 换成重新编码的真观测）。假设：接触是分段动力学，预测误差应该在 regime 边界处爆发，所以"在边界处 re-ground"应该比"固定间隔均匀 re-ground"更省、更准。判据：**等预算下，regime-timed re-ground 胜均匀 → 收益兑现。**

### 工程：什么是 re-ground，怎么做到"公平对比"
- **re-ground = 段重置**：开环 rollout 时，每隔一段就用**真帧重新编码**当锚点，从锚点继续滚，直到下一个锚点。误差在锚点归零、之后随步数增长 → 整条曲线是**锯齿状**。已有 `phase3.regrounded_rollout` 实现了固定间隔版。
- 我把它推广成 `scheduled_rollout`：接受**任意的逐轨迹 reseed 掩码**（哪些 `k` 用真帧重置），按"段长"分组批处理。**关键：所有 schedule 走完全相同的 `model.rollout`，唯一区别是 reseed 点的位置**——这样对比才是 apples-to-apples。
- 四种 schedule：
  - **open-loop**：只在 k=0 重置（0 次 re-ground）。
  - **regime@边界**：在接触标签翻转处（onset/release）重置。
  - **regime-边界前**：在翻转的前一步重置（让模型在难转移前有个新锚点）。
  - **均匀（等预算）**：每条轨迹用**和 regime 一样多**的 re-ground 次数，但**均匀铺开**。
- **等预算**是公平性核心：regime 在某条轨迹用了 3 次 re-ground，均匀也给这条轨迹 3 次，只是位置均匀。这样比的是"**把同样多的重置放哪里更好**"。

### 怎么读指标
- **area-MSE** = k=1..10 的平均 latent-MSE（锯齿曲线下的面积）。越低 = drift 控制越好。
- 配对 t 检验：只在 regime 和均匀**真正不同**的轨迹上比（n=1412）。
- 用的是**真接触边界（oracle）**——这是监控用法的**上界**，oracle 都赢不了就不必做可实现触发器。

### 结果（红灯，连 oracle 都输）

| schedule | area-MSE（越低越好） |
| --- | --- |
| open-loop | 0.2166 |
| **均匀（等预算 ~3.5）** | **0.0652** |
| regime@边界（oracle） | 0.0951 |
| regime-边界前（oracle） | 0.0994 |
| 固定每 1/2/3/5 步 | 0.026 / 0.043 / 0.056 / 0.104 |

配对（n=1412）：regime@边界 **比均匀差** Δ=+0.032，**p<0.001**；边界前略好但仍输（Δ=+0.023，p<0.001）。

### 机理（根本性的）
**re-ground 控制的是误差的累积（drift），不是单步转移的难度。** 段内误差随"距上次重置的步数"单调增长 ⇒ 最小化总 drift = 最小化重置间隔 = **均匀近最优**。而接触边界会**聚集**（onset 和 release 隔几步就来），把预算砸那里，反而在别处留下长缺口让 drift 爆掉。而且 re-ground **根本无法降低难转移本身的误差**（模型仍得预测穿过接触切换），它只重置之后累积的部分——那恰恰是均匀已经处理得更好的。**知道"动力学哪里特殊"恰恰不是该放重置预算的地方。** 图 `regime_stepA_figures/stepC_reground.png`。

---

## 6. 统一洞见

三步合起来是一个自洽的故事：

- **regime 真实存在**（A）——它是关于训好的 `f` 的一个**分析事实**，有信息量、可度量、和接触物理对得上。
- 但它**不转化为可用的控制杠杆**：
  - 当预测器开关（B）：特化让模型对路由错误**脆弱**，而路由本身不可靠 → 比不特化更差。
  - 当 re-ground 触发器（C）：重置控制的是**累积**而非**事件难度**，均匀铺开就近最优 → regime 局部化反而更差。
- **两个失败同源**：知道"动力学哪里特殊"帮不了"该怎么动"，因为成本结构（路由脆弱性 / drift 累积）**都不奖励 regime-局部化的动作**。

这是一组**有机理的负结果**，不是"没调好"——每一步都说清了**为什么失败**以及**需要什么才能成立**（B 需要测试时真标签；C 需要误差在边界处真的局部化而非累积，但它是累积）。Step D/E 不再单独跑（它们建立在 B/C 的收益上，前提已被否）。

---

## 7. 复现命令（工程：照着就能跑）

```bash
# 环境：项目内 venv；PushT 渲染需无头 SDL
PY=.venv/bin/python
ENV="SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy SWM_TORCH_THREADS=4 OMP_NUM_THREADS=4"

# Step A：存在性分析（纯分析，无训练）
$ENV CUDA_VISIBLE_DEVICES=0 $PY scripts/plan/regime_existence_stepA.py \
    --num-samples 800 --output-dir outputs/regime_stepA/run

# Step B：先生成接触标注的 latent 训练集（8000 窗口）+ 评测集（1500）
$ENV CUDA_VISIBLE_DEVICES=0 $PY scripts/plan/regime_stepB_eval_data.py \
    --num-samples 8000 --output outputs/regime_stepB/train_contact.npz
$ENV CUDA_VISIBLE_DEVICES=0 $PY scripts/plan/regime_stepB_eval_data.py \
    --num-samples 1500 --output outputs/regime_stepB/eval_contact.npz
# 决定性轮（mono 阶梯 / moe / oracle，多卡多 seed）
bash scripts/plan/run_regime_stepB.sh \
    outputs/regime_stepB/train_contact.npz outputs/regime_stepB/eval_contact.npz \
    outputs/regime_stepB/decisive 60
$PY scripts/plan/regime_stepB_aggregate.py   # 出表 + 图
# Round A：弱监督门控扫描（gate-sup / train-route-gt / eval-soft 见脚本参数）
bash scripts/plan/run_regime_stepB_gatesup.sh \
    outputs/regime_stepB/train_contact.npz outputs/regime_stepB/eval_contact.npz \
    outputs/regime_stepB/gatesup 60
$PY scripts/plan/regime_stepB_roundA_figure.py

# Step C：regime-timed vs 等预算均匀 re-ground
$ENV CUDA_VISIBLE_DEVICES=0 $PY scripts/plan/regime_reground_stepC.py \
    --num-samples 1500 --output-dir outputs/regime_stepC/run
```

关键脚本一览：

| 脚本 | 作用 |
| --- | --- |
| `regime_existence_stepA.py` | Step A：三特征聚类 + 接触对齐 |
| `regime_stepB_eval_data.py` | 生成接触标注的 `(z, a, contact_frac)` 数据 |
| `regime_moe_stepB.py` | Step B 训练器（`mono/moe/oracle` + `--gate-sup/--train-route-gt/--eval-soft`） |
| `regime_stepB_aggregate.py` / `regime_stepB_roundA_figure.py` | Step B 聚合 + 出图 |
| `regime_reground_stepC.py` | Step C：`scheduled_rollout` + 各 schedule + 对比图 |

---

## 8. Caveats（诚实边界）

- **PushT-only、latent 级、ID（同分布）条件**。跨环境 / FoV shift（原计划 Step E）未测——regime 抓的是动力学不是外观，理论上应更 shift-robust，但未验证。
- **二值接触 regime**（Step A 最佳 k=2）。Step A k=3–4 仍高（更细 regime），Step B/C 未在更细 regime 上测；但二值已是 Step A 最强信号，更细不太可能反转 B/C 的脆弱性/累积机理。
- **样本量**：B 用 3 seeds（t 检验，n 小）；C 是 1500 轨迹的配对检验（n 大，稳）。
- **C 用 latent-MSE area 作指标**；planner cost-rank（Step C i）未单独跑，但 drift 是其前提且为负。
- 原始 run 数据（`outputs/regime_stepA|B|C/...`）不在 Git；图和摘要在 `docs/knowledge/regime_stepA_figures/`。

---

## 9. 文件地图

```text
docs/knowledge/
  regime_direction_results.md            ← 本文：总报告（结果 + 方法）
  direction_discrete_regime_from_lewm.md    方向定案（理论 → 方法 → 脉络 + 各步状态）
  regime_stepA_figures/
    README.md            Step A 原始笔记 + stepA_silhouette_nmi.png
    stepB_README.md      Step B 决定性轮 + Round A + stepB_decisive.png / stepB_roundA.png
    stepC_README.md      Step C + stepC_reground.png
  diagnosis_world_model_drift.md            诊断线（病：on-manifold 各向同性扩散）
  discrete_attempts_falsified_archive.md    已结案的失败离散尝试
```
