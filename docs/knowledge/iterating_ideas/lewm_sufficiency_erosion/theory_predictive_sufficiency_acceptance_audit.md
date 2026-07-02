# 充分动力学方向:8 分审稿门槛与漏洞审查

> 目的:把 [theory_predictive_sufficiency_dynamics.md](theory_predictive_sufficiency_dynamics.md)
> 从"有直觉的理论草稿"推进到可被严格 NeurIPS/ICLR 审稿人给到 8/10 的状态。
> 本文不是最终 paper 文案,而是**审稿门槛、漏洞清单、必须补的理论/实验 gate**。
>
> 当前结论先说在前面:这个方向有潜力,但**当前还不能诚实地说是 8 分**。
> 经过新一轮独立审查后,更诚实的内部评分是 **4/10 weak reject seed**。
> 主要原因不是想法没价值,而是:
> 当前对象仍容易被理解成 stop-gradient / auxiliary loss,action-ranking control sufficiency
> 还没被 theorem、certificate 和方法闭环真正锁住。

---

## 0. 当前 paper claim 的最小可接受版本

不要把 claim 写成:

```text
LeWM 必然丢掉所有难预测任务信息。
```

这太强,审稿人会抓住反例:如果 latent 维度足够、任务变量也可预测、或 predictor 容量足够,
MSE+SIGReg 未必丢它。

更稳的 claim 应该是:

```text
在 self-supervised joint-embedding world model 中,当 encoder target 也由同一 φ 产生,
且 marginal Gaussian/whitening regularizer 只约束 latent 分布形状时,
prediction loss 对 encoder 施加的是 predictive-coordinate selection pressure:
它优先保留 conditional variance 低、rollout amplification 小的因素。

在 contact-rich hybrid dynamics 中,任务关键自由度可能同时具有高 conditional variance
和高 open-loop amplification。此时 multi-step self-consistency training can improve
self-referential latent drift while damaging planning-relevant sufficiency.
```

这句话能被线性高斯推导和 PushT 证据支撑,也留出了合理边界。

---

## 1. 理论 gate

### T1. 定义必须分层

当前理论稿混用了三层对象:

| 层 | 对象 | 当前问题 | 需要修正 |
| --- | --- | --- | --- |
| 表示分布 | $Z=\phi(O)$, SIGReg 约束 $p(Z)$ | 写得像精确 $N(0,I)$ | 改成"finite-sample sketch regularizer encourages projected Gaussianity" |
| 可预测性 | $Z_{t+1}$ 可由 $C_t$ 预测 | 有定理支撑,但不是真正 sufficiency | 明确只在 fixed $\phi$ 时最优 predictor 是 conditional mean |
| 控制充分 | $Z$ 保留 goal/cost ranking | 当前是定义/需求,不是由 LeWM 推出 | 需要单独定义 sufficiency metric |

建议在主稿中加入一个 definition block:

```text
Predictability:
  low E Tr Var(Z_{t+1}|C_t)

Ranking control sufficiency:
  high agreement between latent terminal cost ranking and true task ranking,
  with state decodability used only as a diagnostic proxy.

Self-drift:
  predictor-vs-own-encoder consistency, not external physical accuracy.
```

### T2. "Predictive PCA" 只能作为 toy theorem

当前第 2.1 节线性高斯推导是有价值的,但不能暗示一般非线性 LeWM 完全等价 predictive PCA。
严格表述应为:

```text
Toy theorem under linear encoder, whitened sufficient feature Y, squared loss,
unlimited linear predictor, fixed/exogenous conditioning context C_t,
and exact covariance constraint.
```

它证明的是 selection pressure 的方向,不是 full model 的闭式解。

需要在理论稿里补:

1. 假设列表。
2. theorem/proof sketch。
3. corollary:若任务变量方向 $v$ 的 explained variance 低于替代变量方向 $u$,
   且 latent dimension/capacity 有竞争,则 MSE objective 优先选择 $u$。
4. limitation:若 $D$ 足够大或 $\gamma L_{\text{suff}}$ 约束存在,任务变量可同时保留。

独立审稿人指出的关键理论漏洞:

```text
真实 LeWM 中 C_t=(Z history, A history),而 Z=U^T Y。
因此 M=Cov(E[Y_{t+1}|C_t]) 通常是 M(U),不是固定矩阵。
top-eigenvector 结论只有在 C_t 固定/外生的 toy setting 下严格成立。
```

所以主稿必须把第 2.1 节写成"局部 selection pressure / toy theorem",
不能写成 full nonlinear LeWM theorem。

### T3. Multi-step 放大需要写成稳定性/雅可比命题

当前第 4 节说 $u^\top P_k u$ 增长更快,但没有给出足够严格的递推假设。
建议写成:

```text
Assume local rollout error covariance:
P_{k+1}=A_k P_k A_k^T + Q_k.
Then contribution along direction u is u^T P_k u.
If a task-critical factor aligns with high-gain directions of A_k or regime-switch uncertainty Q_k,
multi-step loss gives it larger penalty than one-step loss.
```

这里不需要声称 PushT angle 一定数学上如此,只说 contact-rich mechanics makes this plausible,
然后用 probe 结果作为 empirical support。

### T4. "SIGReg 不防信息坍缩"需要更准确

更严格说法:

```text
SIGReg prevents simple dimensional/marginal collapse, but is invariant to which semantic factors
occupy the Gaussian coordinates. Therefore it cannot by itself enforce task sufficiency.
```

不要说 SIGReg "只约束二阶结构",因为实现里的 Epps-Pulley/random projection statistic
目标比二阶更强,是 projected normality。旧稿里有这句,新稿已经稍好,但还可以更精确。

---

## 2. 方法 gate

### M1. 当前方法还只是 proposal,不是 implemented method

代码现状:

- [scripts/train/lewm.py](/Users/wge/git/stable-worldmodel/scripts/train/lewm.py:73) 只有 `cfg.wm.unroll`。
- 该分支注释明确说 multi-step open-loop unroll 会让 encoder 通过 seed/target embeddings 共同训练。
- 还没有 `cfg.wm.unroll_sg` 或 dynamics-only rollout loss。

因此任何文案都必须写成:

```text
proposed loss / next experiment
```

不能写成已验证方法。

### M2. Predictor-only 的 stop-grad 边界要定义清楚

LeWM 的参数不止 `predictor`:

```text
encoder -> projector -> emb
action_encoder -> act_emb
predictor -> pred_proj -> predicted emb
```

多步项若说"只训 predictor",必须决定:

| 模块 | 多步项是否更新 | 建议 |
| --- | --- | --- |
| encoder | no | 必须 stop-grad |
| projector | no | 必须 stop-grad,否则仍会改 latent target |
| action_encoder | yes? | 建议 yes,它属于 dynamics input side |
| predictor | yes | 必须 yes |
| pred_proj | yes | 建议 yes,它属于 transition output head |

更准确的名字可能不是 predictor-only,而是:

```text
dynamics-only rollout loss
```

其中 dynamics module = action_encoder + predictor + pred_proj。

### M3. BPTT through predicted latents 不能 detach

实现 stop-grad 时容易犯错:

```text
hist.append(nxt.detach())  # 错,这会切断 multi-step 对早期 predictor 的梯度
```

正确逻辑:

```text
seed = emb[:, :hs].detach()
target = emb[:, hs:hs+K].detach()
hist = seed frames
for step:
    nxt = model.predict(ctx, actw)[:, -1]   # keep grad through f
    hist.append(nxt)                        # do not detach predicted rollout
loss = mse(preds, target)
```

这样 encoder/projector 不吃多步梯度,但 predictor 仍通过 open-loop BPTT 学稳定滚动。

### M4. Baseline 必须拆成四个对照

要让审稿人信服,不能只比 single-step 和 normal multi-step。至少需要:

| 对照 | 作用 |
| --- | --- |
| one-step LeWM | planning-good baseline |
| co-trained multi-step | 已知 low drift / bad planning failure |
| stop-grad multi-step | 主方法 |
| frozen-encoder multi-step | 区分"完全固定 encoder"与"one-step encoder still learns" |

如果 stop-grad multi-step 赢,还要确认不是训练预算或 data window 长度导致。

### M5. P0:planner history-length confound

工程审查发现一个会直接动摇当前 82% vs 22% 解释的 confound:

- LeWM 训练默认 `history_size: 3`。
- `lewm_multistep.yaml` 也是用 3 帧 seed 训练 open-loop unroll。
- 但 planner 的 `PlanConfig.history_len` 默认是 1,PushT plan config 没显式覆盖。
- `LeWM.rollout` 又从 `info['pixels'].size(2)` 推断实际 history 长度。

因此 multi-step 模型的 planning collapse 可能部分来自:

```text
训练时 3-frame seed,评测时 1-frame cold-start。
```

这不一定推翻现象,但在重跑前不能把 planning 22% 直接归因到 encoder erosion。

必须补:

1. 实现/确认 3-frame planner history,包括过去 actions 对齐。
2. 用同样 history 设置重跑 one-step、co-trained multi-step、sg-unroll。
3. 在文档中把旧 82% vs 22% 标成 "possibly history-confounded until rerun"。

### M6. P0:unroll window shape assert

当前 `wm.unroll` 与 dataset `num_steps` 是分开配置的。`lewm_multistep.yaml` 手动设成
`history_size + unroll`,但如果用户只 override `wm.unroll=5`,窗口可能太短。

实现前必须加:

```text
assert emb.size(1) >= history_size + unroll
```

并在 config 里让 `data.dataset.num_steps` 从 `unroll_sg` / `unroll` 派生,避免 silent mismatch。

---

## 3. 实验证据 gate

### E1. 当前证据强但仍是 single-seed core

已知强证据,但注意第一条当前有 history confound:

- [multistep_unroll_drift.md](../../multistep_unroll_drift.md): drift mse@8 0.315 -> 0.177,
  planning 82% -> 22%。
- action spread 0.374 > 0.307,排除简单 action-insensitivity。
- 角度 probe 从约 0.80 -> 0.68,支持选择性充分性损失。

薄弱点:

```text
planning 82% vs 22% 目前核心表述仍主要是 1 seed;
planning 82% vs 22% 可能被 1-frame planner cold-start 污染;
angle probe 的脚本/表格证据没有像 drift/planning 一样在文档中完整落盘;
stop-grad method 尚无结果。
```

8 分需要:

1. 多 seed 确认 co-trained multi-step 的 planning collapse。
2. 多 seed 确认 stop-grad multi-step 的三联指标:
   - drift 低于 one-step;
   - planning 接近/高于 one-step;
   - angle probe 接近 one-step。
3. action-quality ranking 证明 latent cost landscape 保住,不能只看 behavior success。

### E2. 关键因果链要补成 ablation

当前故事是:

```text
multi-step co-training -> angle sufficiency loss -> planning collapse
```

但 reviewer 会问:planning collapse 是否来自别的因素?

需要 ablation:

1. **Matched one-step loss**: stop-grad multi-step 仍有正常 one-step pred_loss,防止 one-step 退化。
2. **State probe correlation**:各 checkpoint 的 angle R² 与 planning success 是否相关。
3. **Latent goal ranking**:angle R² 恢复是否伴随 CEM candidate ranking 恢复。
4. **Decoder-free check**:用 ground-truth state distance ranking vs latent terminal distance ranking,避免只依赖 learned probes。

### E3. PushT-only 风险

如果只在 PushT 上成立,可作为 contact-rich case study,但 8 分难。
更稳的 scope:

```text
Primary: PushT as contact-rich manipulation.
Secondary sanity: one additional hybrid/contact or piecewise environment.
```

候选:

- `piecewise` env:便宜,能直接控制 regime/conditional variance。
- `gymnasium_robotics` push/slide:更真实但贵。

最省力的 8 分路线:

```text
先做 synthetic piecewise Gaussian/hybrid toy,验证 theorem 的 selection bias;
再做 PushT,验证 real-world failure and fix。
```

### E4. Data split / leakage 风险

工程审查还指出几个会被 reviewer 追问的复现风险:

```text
训练 split 是 random clips,不一定 episode-level;
normalizer 可能在 split 前用全数据 fit;
planner eval 当前使用 pusht_expert_train.h5;
训练 seed 主要用于 split/DataLoader generator,未完整 pl.seed_everything。
```

如果要冲 8 分,至少要给出:

1. episode-level train/val/test split 或解释当前 setting 为什么不泄漏任务。
2. train-only normalizer 与 checkpoint 一起保存。
3. fixed held-out eval episodes。
4. run metadata:seed、git SHA、checkpoint path、CEM config、history_len。

---

## 4. 当前严格评分

### Reviewer score:4/10 paper seed

独立理论/方法审稿共同给 4/10 的理由:

- stop-gradient dynamics separation 和 multi-horizon prediction 都不是新对象。
- planning collapse 的核心证据仍像 single-seed,且可能有 history confound。
- predictive-PCA theorem 之前忽略了 $C_t$ 依赖 encoder。
- 当前 theorem 更像 passive variable selection,还不是 action-conditioned ranking theorem。
- scalar PCG normalization 任意,应改为 ranking sufficiency + Pareto-PCG。
- state-SER 容易退化为 supervised state auxiliary loss。
- SER 缺少 controllability/action-relevance,可能保护不可控噪声。
- 还没有证明 SER/PCG 比 self-drift、probe R²、inverse dynamics、uniform uncertainty 更能预测 failure。

不给 8 的理由:

- 主方法对象还没有从 state-SER 升级到 Ranking-SER / Control-Fisher-SER。
- 线性高斯推导还没有升级为 action-conditioned ranking counterexample。
- "angle lost -> planning collapse" 的因果链仍是相关性,不是预注册 certificate + causal repair。
- 实验目前 PushT-heavy,且 seed 证据不足。

### 8/10 accept checklist

必须同时满足:

```text
[ ] Theory:写出 toy theorem + assumptions + limitation,不夸大到 full nonlinear theorem。
[ ] Theory:升级为 action-conditioned theorem,证明 predictive optimum 和 ranking-sufficient representation 可冲突。
[ ] Certificate:定义 ranking sufficiency、Pareto-PCG、controllability-gated SER,并预先固定 estimator/normalization。
[ ] Method:主方法是 Ranking-SER / Control-Fisher-SER; dynamics-only rollout 只作为 baseline/supporting component。
[ ] Evidence:SER/PCG 比 self-drift、probe R²、inverse dynamics、uniform uncertainty 更早预测 failure。
[ ] Replication:至少 3 seeds for one-step / co-trained multi-step / dynamics-only / uniform state / placebo / Ranking-SER。
[ ] Ablation:wrong-variable 或 low-SER placebo anchor 不能同等修复 planning。
[ ] Scope:至少一个 cheap controlled toy/piecewise env 验证 theorem pressure。
[ ] Writing:把 self-drift, predictability, ranking control sufficiency 三者严格区分。
[ ] Planner:确认 3-frame history eval 后,planning collapse/fix 仍成立。
[ ] Probe:落盘 angle/state probe script + wrapped angle/sin-cos protocol + CI。
[ ] Repro:episode split/train-only normalizer/fixed eval episodes/run metadata。
[ ] Related work:定位 Dreamer-style stop-grad dynamics、LeWM/Fast-LeWM、PLDM/DINO-WM、inverse dynamics/control representation、decision-focused/value-equivalent models。
```

---

## 5. 下一步最有价值的工作

### Step 1:把理论稿改成 theorem + limitation 结构

优先改 [theory_predictive_sufficiency_dynamics.md](theory_predictive_sufficiency_dynamics.md):

1. 第 2.1 节改名为 "Toy theorem: predictive PCA under exact whitening"。
2. 加 assumptions。
3. 加 proof sketch。
4. 加 "what this does not prove"。

### Step 2:实现 dynamics-only rollout

配置:

```yaml
wm:
  unroll_sg: 5
  unroll_sg_weight: 1.0
```

损失:

```text
loss = one_step_pred_loss + λ sigreg + β rollout_sg_loss
```

注意不要用 `wm.unroll` 替代 one-step;两者要同时存在。

### Step 3:最小实验矩阵

```text
seed ∈ {0,1,2}
model ∈ {one-step, co-trained-unroll, sg-unroll}
metrics ∈ {drift@k, planning success, angle probe, action-quality ranking}
```

### Step 4:若 sg-unroll 失败,不要硬写方法

失败也有价值,但结论要改:

```text
stop-gradient alone is insufficient; explicit control-sufficiency anchor is required.
```

这时 paper 方向转为:

```text
Gaussian predictive JEPA needs a control-sufficiency anchor in contact-rich planning.
```
