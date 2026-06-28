# Step C 工程：re-grounding 调度实验（推理 pipeline + 评测）

> 配合结论 [stepC_README.md](stepC_README.md) 与总报告 [regime_direction_results.md](../regime_direction_results.md) §5 看。
> 和 [stepB_engineering.md](stepB_engineering.md) 同构:先讲**这测的是什么、和 LeWM 推理什么关系**,再讲**数据/调度/指标/结果**,最后**诚实列软肋**(这部分尤其要 review)。
> 代码:`scripts/plan/regime_reground_stepC.py`(全部逻辑都在这一个文件)。

---

# Part 0 — 一句话:Step C 测什么、和 Step B 什么关系

- **Step B** = regime 去**条件化预测器**(MoE 开关)。**Step C 完全不碰预测器**。
- **Step C** = 用一个**固定的、普通的** latent 预测器开环往前滚,只问一件事:**在哪些步把漂移的预测 latent 换成"真观测重新编码"(=re-ground),误差控制得最好?** 具体对比"在 regime 边界处 re-ground" vs "等预算均匀 re-ground"。
- 所以 C 在逻辑上**不依赖 B 成功**——它是 regime 的另一种(更弱的)用法:当**监控信号**,而不是预测器开关。

---

# Part 1 — 推理 pipeline:什么是 re-ground,和 LeWM 什么关系

## 1.1 什么是 re-ground(段重置)
开环 rollout 时,预测会逐步漂移(误差累积)。**re-ground = 每隔一段,用一帧真观测重新编码当锚点**,从锚点继续滚,直到下一个锚点。所以整条误差曲线是**锯齿状**:每次 re-ground 归零、之后随步数增长。

> 注意:re-ground 用的是 LeWM 自己的 `encoder`(冻结)把真帧编码成 `z`,以及 LeWM 自己的 `predictor`(`model.rollout`)往前滚。**Step C 没有训练任何东西**——它纯粹是"在一个已训好的世界模型上,改变 re-ground 的时机"。这点和 Step B 不同(B 训了个替身预测器)。

## 1.2 `scheduled_rollout`:怎么保证两种调度公平对比
现成的 `phase3.regrounded_rollout` 只支持**固定间隔**。我们推广成 `scheduled_rollout`,接受**任意逐轨迹的 reseed 掩码**:
```python
scheduled_rollout(model, frames, true_emb, model_actions, reseed_mask, ...)
#  reseed_mask[i, k] = True  → 第 i 条轨迹在第 k 步用真帧重置
```
实现要点(`regime_reground_stepC.py:48`):
- 把每条轨迹的 reseed 点切成若干"段"(segment),每段 = [锚点, 下一个锚点)。
- 按**段长(horizon)分组批处理**:同段长的轨迹一起喂 `model.rollout`,scatter 回预测数组。
- **关键:所有调度走的是完全相同的 `model.rollout` 调用,唯一区别是 reseed 点的位置。** 这样 regime vs 均匀的对比就是 apples-to-apples,差异只来自"重置放哪里"。

## 1.3 一个必须正视的框架问题(见 Part 7 软肋①)
re-ground 需要**真观测帧**。你只有在**真闭环执行、真观测到了那一步**才有真帧。**规划(planning)时你在想象未来,根本没有真帧可 re-ground。** 所以 Step C 隐含的场景是**闭环执行监控**,不是规划。这点 docs 里之前没讲清,Part 7 展开。

---

# Part 2 — 训练数据(其实是"评测数据",C 不训练)

复用总报告 §2 的公共管线 a→b→c,和 Step A/B 同源:
```
build_window_batch → replay_windows(取帧 + contact_frac) → encode_frames(冻结 encoder → z)
```
- 规模:`--num-samples 1500` 条窗口,`max_k=10`,`action_block=5`。
- 每条产出:`frames(11,224,224,3)`(re-ground 用的真帧)、`true_emb=z(11,192)`(比对基准 + 锚点)、`model_actions(10,10)`、`contact_frac(11)`(二值 regime 标签 = contact_frac>0)。
- 没有 train/val split——C 不训练,1500 条全部用来评测调度。

---

# Part 3 — 四类 re-ground 调度(对比对象)

| 调度 | 怎么定 reseed 点 | 角色 |
| --- | --- | --- |
| **open-loop** | 只在 k=0 | 下界(0 次 re-ground,纯漂移) |
| **regime @ 边界** | 接触标签翻转处(onset/release):`contact_bin[k]≠contact_bin[k-1]` | 待验证主张(oracle) |
| **regime 边界前** | 翻转的前一步 k-1(难转移前给个新锚点) | 变体 |
| **均匀(等预算)** | 每条轨迹**和 regime 一样多**的 reseed 次数,但 `np.linspace` 均匀铺开 | **公平基线** |
| 固定每 1/2/3/5 步 | 周期性 | 参照梯子 |

**等预算是公平性核心**:regime 在某条轨迹用了 3 次 re-ground,均匀也给这条 3 次,只是位置均匀。比的是"**同样多的重置放哪里更好**"。代码:`regime_schedule` / `regime_pre_schedule` / `budget_matched_uniform`(`regime_reground_stepC.py:118+`)。

---

# Part 4 — 指标

- **area-MSE** = 每条轨迹在 k=1..10 的 latent-MSE 平均(锯齿曲线下面积)。越低 = 整体漂移控制越好。`def area(m): return m[:,1:].mean(axis=1)`。
- **配对 t 检验**:只在 regime 和均匀**真正不同**的轨迹上配对比较(budget≥2 且掩码不同,n≈1412)。配对是因为同一条轨迹两种调度都跑,直接比差值,统计力强。
- 用**真接触边界(oracle)**——监控用法的**上界**;oracle 都赢不了就不必做可实现触发器(和 Step B 的 oracle 逻辑一致)。

---

# Part 5 — 结果与机理

| 调度 | area-MSE |
| --- | --- |
| open-loop | 0.2166 |
| **均匀(等预算 ~3.5)** | **0.0652** |
| regime @ 边界(oracle) | 0.0951 |
| regime 边界前(oracle) | 0.0994 |
| 固定每 1/2/3/5 | 0.026 / 0.043 / 0.056 / 0.104 |

配对(n=1412):regime@边界 比均匀**差** Δ=+0.032,**p<0.001**。

**机理(根本性)**:re-ground 控制的是误差的**累积**,不是单步转移的**难度**。段内误差随"距上次重置步数"单调增长 ⇒ 最小化总 drift = 最小化重置间隔 = **均匀近最优**。接触边界会**聚集**(onset+release 相邻几步),砸预算在那里反而别处留长缺口让 drift 爆掉。而且 re-ground **降不了难转移本身的误差**(模型仍得预测穿过接触切换),只重置之后累积的——那正是均匀已处理得更好的。

---

# Part 7 — 软肋(这部分请重点 review)

和 Step B 一样,C 也有没被仔细 review 的假设。诚实列出:

1. **"re-ground 何时可用 / 为何限预算"这个框架没论证清(最大软肋)。**
   - re-ground 要真观测帧 ⇒ 只在**闭环执行**场景成立,**规划时不适用**。
   - 而闭环里你**每步本来就有观测**,那为什么不每步 re-ground、要"限预算"?"等预算"这个前提隐含假设"re-ground 有成本"(比如 encoder 贵、或想少观测多预测),但这个场景没明确论证。如果 re-ground 免费,问题本身就不成立。

2. **代理指标:用 area-MSE(latent drift),不是方向文档真正要的 planner cost-rank。** 和 Step B 用多步 drift 代替 LeWM 原生目标是同一类问题。"latent 漂移小"不一定等于"规划/控制更好"。

3. **用的是预训练 LeWM + 它自己的 predictor**,这点其实比 B 更忠实(B 训了替身)。但也意味着结论绑定在这个特定 checkpoint 的漂移特性上。

4. **二值接触 regime**(Step A 最佳 k=2);更细 regime(k=3-4)没测。不过机理(累积 vs 难度)和粒度无关,更细不太可能反转。

> 结论强度:Part 5 的"均匀打赢 regime"在**它自己的设定里**(闭环、latent-MSE、等预算)是干净的 oracle 上界否定。但软肋 1/2 意味着它**不能**直接推广成"regime 监控在任何控制场景都没用"。

---

# Part 8 — 复现

```bash
SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy CUDA_VISIBLE_DEVICES=0 \
SWM_TORCH_THREADS=4 OMP_NUM_THREADS=4 \
  .venv/bin/python scripts/plan/regime_reground_stepC.py \
  --num-samples 1500 --output-dir outputs/regime_stepC/run
# 产出 result.json + stepC_reground.png（曲线 + 等预算对比柱）
```

一图流:
```
专家轨迹 → replay(帧+contact) → encode(冻结)→ z 真值序列 + 真帧
                                                 │
   对每种 reseed 调度： scheduled_rollout(同一个 model.rollout，只换 reseed 点)
                                                 │
   area-MSE(k=1..10 平均) → 配对 t 检验 regime vs 等预算均匀
```
