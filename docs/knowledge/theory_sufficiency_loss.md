# 理论推导:从 LeWM loss 的退化性，到一个"充分性/动力学"结合损失

> 目的:把我们一路实验里撞出来的硬事实,推成一个**有理论支持、与 LeWM 原 loss 自然结合、少超参**的新损失。
> 状态:**推导稿(v1)**,带可证伪预测 + 小规模验证方案。配套实验证据见
> [multistep_unroll_drift.md](multistep_unroll_drift.md)、[regime_direction_results.md](regime_direction_results.md)。
>
> ⏰ **24h 后看这里(多 seed 确认 82% vs 22% 的 planning gap)**:seed-1 训练于 2026-06-29 ~21:08 起跑
> (`iter2_multistep_s1` 4卡 / `iter2_baseline_s1` 2卡,约 **2026-06-30 22:00–07-01 02:00** 跑完)。
> 进度日志 `outputs/regime_stepB2/planner_20260629_2108/train_{multistep,baseline}_s1.log`;
> 跑完后:建 `*_s1_eval` 目录(config.json=model 子树 + 软链 weights)→ `eval_wm.py policy=..._s1_eval`(planning 50ep)
> + `regime_lewm_iter2_eval.py`(drift)。**找我"查 seed 结果"即可。**

---

## 0. 一句话

> LeWM 的 `pred_loss + SIGReg` **欠定**了 encoder:它的最优解里包含"**为了好预测而丢掉难预测信息**"的退化 encoder;预测 horizon 越长,这种信息丢弃越严重。我们实测到了它(多步训练 drift↓44% 但 planning 82%→22%,且丢的正是接触驱动的 block 角度)。**缺的是一项"充分性"约束**。推导给出一个最小、可证伪的结合损失:**让 encoder 只被单步+SIGReg 塑形(实测对 planning 好),多步抗漂项只训 predictor(encoder 停梯度)**,从而 drift 和 planning 同向改善。

---

## 1. 对象:LeWM 的损失(形式化)

记 encoder `φ`(观测序列 → latent `z=φ(o_{≤t})`),predictor `f`(`z, a → ẑ'`)。LeWM(`scripts/train/lewm.py`,`num_preds=1`):

```
L_LeWM(φ, f) = E_t ‖ f(φ(o_{≤t}), a_t) − φ(o_{t+1}) ‖²   +   λ · SIGReg(φ(o))
                └────────── pred_loss(单步,teacher-forced) ──────────┘     └ 边缘各向同性 ┘
```

两个关键结构性事实:
1. **target 也是 `φ` 产生的**:`φ(o_{t+1})` 在等式右边,encoder 同时决定"输入"和"被预测的目标"。这是 JEPA,天然有"把目标变好预测"的捷径。
2. **SIGReg 只约束边缘的二阶结构**(协方差 ≈ I,各向同性),即只防**维度坍缩**;它**不防信息坍缩**——latent 可以满秩各向同性,却编码了"错误"的信息。

---

## 2. 退化性论证(理论核心)

**命题(非正式).** 在固定 `f` 容量下,`pred_loss + SIGReg` 关于 `φ` 的极小元**不唯一**,且其中包含"信息丢弃"解:对任意观测分量,若它在给定 `(o_{≤t}, a_t)` 下**条件熵高**(难预测),encoder 可以**不编码它**(在该方向编码一个可预测的替代量),从而降低 `pred_loss` 的不可约部分;只要把腾出的维度用于某个可预测量以维持各向同性,SIGReg 不增。

**直觉推导.** 把 `pred_loss` 按方向分解,某方向的不可约误差 ≈ 该方向上 `o_{t+1}` 关于 `(o_{≤t},a_t)` 的**条件方差**。encoder 有两条降它的路:
(i) 学更好的 `f`(真降);(ii) **让该方向不进入 latent**(假降,丢信息)。目标里没有任何项惩罚 (ii)。

**horizon 放大.** 多步开环 unroll 把目标换成 `φ(o_{t+k})`,而难预测分量的条件熵随 `k` **单调增**(接触类事件尤甚)。于是 (ii) 的收益随 horizon 增大 ⇒ **horizon 越长,encoder 越倾向丢弃难预测(高条件熵)分量**。

**推论.** `pred_loss(+SIGReg)` 与 **表征充分性**之间存在 trade-off,而损失里**没有充分性项**。最优会牺牲"难预测但任务关键"的分量——在接触物理里,这恰是**接触驱动的自由度**(Step A 证明离散 regime 活在接触处的 `f` 里)。

---

## 3. 实验锚点(我们已测到的硬证据)

| 现象 | 数据 | 支持推导的哪一步 |
| --- | --- | --- |
| 多步训练 drift↓ 但 planning 崩 | mse@8 0.315→0.177,PushT 成功率 82%→22% | §2 退化性真实发生 |
| 不是动作不敏感 | 反事实 z'-spread 多步 0.374 > 单步 0.307 | 排除"坍缩";是**选择性信息丢弃** |
| 位置信息没丢、**角度丢了** | 线性探针 R²:位置≈0.97 两者一致;角度 0.80→**0.68** | 丢的正是**难预测+接触驱动+任务关键**的 DOF |
| regime 活在接触处的 `f` | Step A:Jacobian↔接触 NMI 0.30 | 接触 = 难预测分量的来源 = 任务关键(推 T 转角) |
| 单步(=LeWM 原生)planning 好 | baseline 82% | 单步塑形的 encoder 是"充分"的参照 |

> 注意:`drift` 是**自指**指标(预测 vs 同模型自己的 encoder),所以它能被 encoder+predictor 协同"刷低"而不反映任务保真。**这是"低 drift ≠ 好世界模型"的根因。**

---

## 4. 缺失项:充分性约束

我们要的不是"再加一个聚类/离散 loss"(那条死了),而是**一个防止 encoder 丢弃任务关键信息的约束**,且与 LeWM 原 loss 结合、少超参。形式上:

```
要求:z = φ(o) 对(任务相关的)真状态/观测是充分的,尤其在难预测(接触/regime)方向上。
```

直接上互信息/重建违背 JEPA 无解码器的精神,且贵。我们用一个**关键观察**把它变便宜:

> **单步塑形的 encoder 实测就是"充分"的(planning 82%)。问题只在 horizon 放大了信息丢弃。**
> 所以不必新造充分性度量——**只要阻止多步项去腐蚀 encoder 即可。**

---

## 5. 推导出的损失(主方案,最小且可证伪)

把 encoder 的塑形权"只交给单步+SIGReg"(已知对 planning 好),多步抗漂项**只训 predictor**(对 encoder 停梯度):

```
L_new(φ, f) = L_LeWM(φ, f)                                  # 单步 pred + SIGReg：塑形 φ（保持充分/可规划）
            + β · E_t ‖ f^(K)( sg[φ(o_{≤t})], a_{t:t+K} ) − sg[φ(o_{t+1:t+K+1})] ‖²
              └────────── 多步开环 unroll，predictor-only，φ 全程 stop-grad ──────────┘
```

- `sg[·]` = stop-gradient。**encoder 不再从多步项收到任何梯度** ⇒ 不会为压低多步 drift 而丢角度。
- `f^(K)` = 把预测喂回滚 K 步(开环)。只有 `f` 学"长程滚得准"。
- 与 LeWM 结合干净:`β=0` 退回原 LeWM;`β>0` 加一个**predictor-only 的抗漂头**。**一个新超参 β**(+ horizon K)。

**机理对应**:§2 说"多步腐蚀 encoder";本方案精确切断那条腐蚀路径,同时保留多步对 `f` 的好处(CEM 规划本身就是多步开环滚 `f`,所以 `f` 长程准是直接有用的)。

**可证伪预测(关键):**
1. `L_new` 的 **planning 成功率 ≈ 单步 baseline(~82%)或更高**(因为 φ 由单步塑形,角度 R² 应回到 ~0.80)。
2. 同时 **多步 drift 低于单步 baseline**(因为 `f` 拿到了多步训练)。
3. 即 **drift 与 planning 同向改善** —— 纯多步(§3)做不到这一点。
若 (1) 不成立(planning 仍崩)→ 说明腐蚀不在 encoder 而在别处,主方案证伪,转 §6 备选。

---

## 6. 备选 / 加强项(若主方案不够)

- **B1 充分性正则(显式)**:加 `−γ · I(z ; o)` 的便宜下界(如 latent→关键状态的轻量探针损失,用真状态或自监督代理),直接惩罚信息丢弃。更贵、引入监督,作为主方案不足时的补强。
- **B2 regime-加权保真(接 Step A)**:对接触驱动方向(由 Jacobian 谱/接触标签标出)**加权** pred_loss,强制 encoder 保留这些难预测但任务关键的方向。把 Step A 的存在性结论用作"哪里必须保真"的先验,而非门控。
- **B3 EMA/stop-grad target**:更彻底地切断"encoder 移动自己的 target"捷径(BYOL 式),与 SIGReg 叠加。改动 LeWM 的 anti-collapse 机制,较激进。

主方案(§5)= B3 的"只对多步项施加"的最小特例,优先验。

---

## 7. 验证方案(小规模,2 张卡)

1. 实现:`lejepa_forward` 加 `cfg.wm.unroll_sg`(对 φ 停梯度的多步项)+ `β`。改动 ~10 行。
2. 训:从零 LeWM + `L_new`(β 扫 {0.5,1,2},K=5),30 epoch,2 卡;对照已有 `iter2_baseline`(单步)与 `iter2_multistep`(纯多步)。
3. 评测(已有脚本):
   - 多步 drift(`regime_lewm_iter2_eval.py`,3 帧 seed)。
   - **PushT planning 成功率**(`eval_wm.py`,50 ep)——这是判据。
   - 角度线性探针 R²(应回升到 ~0.80)。
4. 判据:§5 的三条同时成立 → 方法立;planning 不回来 → 主方案证伪,走 §6。

---

## 8. 与硬约束的对账

- **有理论支持**:退化性论证(§2)+ JEPA 信息坍缩文献(SIGReg 防维度坍缩、不防信息坍缩)。
- **结论简单少超参**:一个 β(+K),`β=0` 退回 LeWM。
- **离散自然涌现、不强行**:不加聚类/门控;Step A 的离散 regime 只作为"哪里必须保真"的先验(B2),不路由。
- **从 LeWM 自身 loss 长出来**:`L_new` = LeWM + 一个 predictor-only 抗漂项,不是另起炉灶。

---

## 9. 一句话留给下一步

主方案是个**干净的、可证伪的赌注**:"多步抗漂的好处归 predictor,encoder 的充分性归单步+SIGReg"。要么 drift 和 planning 第一次同向变好(立),要么 planning 还是崩(证伪→§6 显式充分性)。两边都推进认知。
