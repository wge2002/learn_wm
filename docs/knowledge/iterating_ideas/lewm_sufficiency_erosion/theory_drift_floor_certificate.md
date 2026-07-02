# v4 主线定稿:Drift Floor + 侵蚀证书 + 校准动力学(Certify, don't chase)

> ⚠️ **2026-07-02 Gate 0 判决(当日晚):锚定现象被证伪,本 spine 按 §8 预注册规则挂起。**
> matched 3-frame history 重跑后(3 eval seeds × 50ep),pure multistep planning = **88%/86%**
> (两个训练 seed),≥ baseline 83%/83%;82-vs-22 反转是 1-frame 冷启动评测伪影。
> "低 drift 但 planning 崩"不存在 → phenomenon 轴掉到 ~2,min-axis 下整个方向 ≤3。
> **幸存的事实**:`sgmulti`(predictor-only 多步)在 matched history 下仍然有害
> (53-56% vs 83-88%),即 Signal 2(f 内部目标冲突)是真实的、且是唯一活下来的现象;
> Signal 1 的"侵蚀"解释失去了它要解释的损伤。fixed-φ0 实验对原目的已无意义。
> 数据:`outputs/gate0/`,详表见 [multistep_unroll_drift.md](../../multistep_unroll_drift.md)。
> 本文档保留:三路分解 / 度量不变性 / impropriety 定理仍是正确的方法论;
> "Certify, don't chase"的教训以最讽刺的方式成立——第一个被证书杀死的 claim 是我们自己的。
>
> 状态:~~方向定稿(v4)~~ → **挂起,等待方向级决策**。2026-07-02 由 3 个独立 NeurIPS
> 审稿视角(theory / method / evidence)round-2 评审收敛产生。取代 v3 各稿中的 spine 候选讨论。
>
> 一句话:**不要追逐更低的 self-drift;先证明 drift 有一个几何决定的下界,
> 把任何 drift 改善分解为"几何重写"与"动力学改进",低于下界的 drift 本身是侵蚀警报;
> 方法上接受下界——锚定几何,用校准的分布式 predictor 建模不可约不确定性,planner 用风险感知 cost。**
>
> 主线组合 = Idea 1(侵蚀记账/漂移分解证书)+ Idea 4(校准化 rollout 目标);
> Idea 2(Ranking-SER)降级为 Tier-0 oracle 诊断,不做主方法。
> 相关背景:[theory_predictive_sufficiency_dynamics.md](theory_predictive_sufficiency_dynamics.md),
> [theory_control_sufficient_gjepa.md](theory_control_sufficient_gjepa.md),
> [theory_predictive_sufficiency_acceptance_audit.md](theory_predictive_sufficiency_acceptance_audit.md)。

---

## 0. Round-2 评审结论(2026-07-02)

3 个独立审稿视角对 5 个候选方向的共识评分(0-10,min-axis 规则):

| Idea | now | potential | 共识判决 |
| --- | ---: | ---: | --- |
| 1 侵蚀记账/漂移分解证书 | ~4.5 | **~8** | 全票第一;唯一"理论对象新 + 理论推出方法"的方向 |
| 4 校准化 rollout(接受下界) | ~4 | ~7.5 | 理论强制的方法伴生项;standalone 不成立,与 1 合并 |
| 2 Ranking-SER | ~3.5 | ~7 | 幸存新意只剩 σ²_pred 门控;乘积形式被判量纲拼凑;降级诊断 |
| 3 PSD latent 拆分 | ~3 | ~6 | RSSM/Denoised-MDP 撞车;仅作升级备选 |
| 5 反事实排序锚 | ~2.5 | ~5 | HER/GCSL/VIP/QRL 红海;至多做 Idea 2 的一个 ablation 行 |

所有 now 分数被 P0 history confound 封顶:matched-history 重跑之前,任何 idea 都到不了 6。

**Spine 定稿**:*Certify, don't chase — drift floors and sufficiency erosion in
reconstruction-free Gaussian JEPA world models.*
它是唯一 phenomenon / theory / method / evidence 四轴都能同时到 7+ 的形态,
且优雅降级:若 fixed-φ0 落在 O2(见 §4),同一套基础设施直接变成一篇干净的方法论文。

---

## 1. 对既有推理的两处纠正(round-2 审稿判定)

### 1.1 Signal 1 / Signal 2 目前是同一个 confounded run 的循环使用

sgmulti(b1/b2)里 encoder 仍被单步项训练(φ 在动),predictor 同时背单步+多步两个目标
(f 被双任务共享)。因此:

- `sgmulti drift@8 = 0.358` **不是** `D*(φ_baseline) = inf_f drift(φ_baseline, f)` 的合法估计;
- "pure multistep 的 drift 改善 ≈100% 来自几何重写"(Signal 1)和
  "单步/多步目标在 f 内部冲突"(Signal 2)由**同一个 run** 支撑,不可分离,互为循环论证;
- k=1 drift 翻倍(0.011→0.022)在 pure multistep 里同样出现(0.011→0.021),
  它本身不能区分"目标冲突"与一般多目标优化干扰。

两个 Signal 降级为**假设**,由 §4 的预注册实验判定。

### 1.2 "低 drift ⇒ 侵蚀" 不可定理化;可定理化的是弱一句

- **可证**:`drift(φ_B, f_B) < D*(φ_A)` ⇒ φ_B ≠ φ_A(至多等距)——低于下界的 drift
  只能靠改几何买到。
- **不可证**:几何改变 = 侵蚀。反例:encoder 丢弃难预测但**任务无关**的噪声
  (传感器噪声、背景闪烁)同样降低 D*,对 planning 无害甚至有益。
- 正确表述:**sub-floor drift 是强制审计的警报(alarm),不是侵蚀的证明(proof)。**
  警报触发后由 Tier-0 诊断(§6)判定丢的是 nuisance 还是 control-critical 方向。

---

## 2. 理论核心(paper 的定理层)

### T1. Drift floor(一行定理,如实标注平凡)

固定 φ,对任意可测 g:

$$
\inf_g \mathbb{E}\lVert g(C_t)-Z_{t+k}\rVert^2
= \mathbb{E}\,\mathrm{Tr}\,\mathrm{Var}(Z_{t+k}\mid C_t)
\;=\; D^*_{\mathrm{free}}(\phi, k).
$$

L2 投影一行可得。定理级内容不在这里,在 T2/T3。

### T2. 三路分解(本稿主定理对象)

对开环复合 rollout 的 drift,任何两个模型 A、B 的差:

$$
\Delta\mathrm{drift}
= \underbrace{\big[D^*_{\mathrm{free}}(\phi_B)-D^*_{\mathrm{free}}(\phi_A)\big]}_{\text{信息下界项(几何)}}
+ \underbrace{\Delta\big[D^*_{\mathrm{comp}}-D^*_{\mathrm{free}}\big]}_{\text{Markov/复合缺口}}
+ \underbrace{\text{residual}}_{\text{优化残差(动力学侧)}}.
$$

- `D*_comp`:限定"一步 predictor 自复合 k 次"的下界;`D*_free`:不限形式的直接 k 步回归下界。
  `D*_comp ≥ D*_free`,当且仅当 latent 过程对 C Markov 时取等。
- **复合缺口本身是一个可发表的诊断:latent Markov 性证书**——encoder 丢状态 ⇒ Z 非 Markov ⇒ 缺口打开。
- 使其成为定理(而非记账恒等式)需要:f-类可达性、cross-fit 估计一致性、显式 Markov 条件。

### T3. 度量不变性(没有它几何项是单位伪影)

L2 drift 不是 latent 重参数化的不变量;两个信息等价、embedding 不同的 encoder 有不同 D*。
必须:

- 在**逐 encoder whitened 坐标**下定义 drift(SIGReg 使边缘 ≈ N(0,I),whitening 有原则性),
  或在固定 probe/state 空间定义;
- 显式声明不变类(至多等距 isometry);
- 跨 encoder 比较一律用不变化后的量。

### T4. Impropriety 定理(堵死 "换 NLL 不就行了")

若 target 分布 p(Z_{t+k}|C) 由 φ 自产,且 φ 吃 proper scoring rule 的梯度,则最优处
期望 log score = H(Z_{t+k}|C),即联合最小化 NLL(φ,f) **就是**对 φ 的条件熵最小化——
与 MSE 完全相同的可预测坐标选择压力,分布式损失逃不掉;还新增退化:
φ 可把 Z 变成条件独立噪声,宽高斯对它完美校准且零信息。

**推论(方法的数学地基)**:proper scoring 在自指 latent 预测里 well-posed
**当且仅当几何被锚定**(φ 只由单步+SIGReg 塑形,或 frozen/EMA)。
Idea 4 的方法由这个定理强制推出,不是工程选择。

### T5. 估计器纪律(cross-fit 的真实含义)

任何 refit 估计满足 `D̂* ≥ D*`(有限容量/有限优化),两个上偏估计的差符号不可识别,除非:

- refit 的架构/步数/LR/early-stopping 跨几何严格 matched;
- ≥3 refit seeds,报告 CI;
- fitting/eval 样本切分(真正的 cross-fitting;不做切分就不要用这个词);
- claim (b)"几何项预测 planning 损伤"需要**模型群体**(N≈10-12 个 variants)上的
  预注册相关性,不是两个点。

---

## 3. 方法层(由 T4 推出)

$$
\mathcal{L}
= \underbrace{\mathcal{L}_{\text{LeWM-1step}}(\phi,f) + \lambda\mathcal{R}_{\text{SIGReg}}}_{\text{锚定几何(唯一塑形 }\phi\text{ 的项)}}
+ \beta\,\underbrace{\mathcal{L}_{\text{calib}}\big(f_{\text{dist}};\,\mathrm{sg}[\phi]\big)}_{\text{校准分布式 }f\text{(NLL/CRPS),分布输出}}
$$

- f_dist 输出分布(高斯头或 ensemble/particle),在锚定坐标系内用 proper scoring rule 训练;
- **planner**:CEM cost 从点估计 L2 改为风险感知(mean + risk 项),接触步的不可约不确定性
  进入决策而不是被抹掉;
- **模型选择规则(证书的方法化)**:只接受"cross-fit 后仍幸存"的 drift 改善;
  确定性 drift 低于 D̂*_free ⇒ 触发侵蚀审计;
- dynamics-only rollout(原 sgmulti)保留为 baseline/ablation 行,不是方法。

预注册预测:
1. 校准 f 的 drift 落在 floor 附近(不低于);co-trained multistep 的 drift **低于** floor(警报触发);
2. 校准 f + 风险感知 CEM 的 planning ≥ baseline(matched history);
3. 警报在 held-out seeds 上对 pure multistep 触发、对 baseline 不触发。

---

## 4. 预注册:fixed-φ0 实验的结局表(跑之前锁定)

冻结 planning-good baseline encoder φ0,只训 f(K 步目标单独训练,收敛为止,≥3 seeds):

| 结局 | drift@8 | planning | 判定 |
| --- | --- | --- | --- |
| O1 | ≈0.3+(不降) | ≈82% 保住 | floor 假设(Signal 1)成立;**Signal 2 被削弱**(好几何+受限 f 没杀 planning) |
| O2 | 向 0.177 下降 | 保住 | **Signal 1 死**;动力学侧改进在 baseline 几何内可得,"≈100% 几何侧"是 moving-φ 伪影;基础设施转向方法论文 |
| O3 | 下降 | 下降 | 固定几何下 latent-MSE 抗漂与 CEM ranking 也不对齐;"几何=损伤、动力学=安全"二分法破 |
| O4 | 不降 | 下降 | Signal 2 成立、floor 可信,但"动力学侧安全"反向证伪(off-distribution 候选覆盖问题) |

当前押注(写下以便被打脸):O1 或 O4。若 drift 在 φ0 冻结下降到 0.315 以下,两个 Signal 都要重写。

**Signal 2 的对照组电池**(判定"目标冲突"是否结构性,任选其三):
- β sweep {0.1,0.5,1,2} 看 drift@1 退化是饱和(冲突)还是单调(LR 伪影);
- predictor LR 按 1/(1+βK) 重标定;
- teacher-forced K 步稠密监督对照(同梯度质量、无自复合)——若 drift@1 同样翻倍,罪魁是"更多梯度"不是"自复合冲突";
- 2× predictor 容量;horizon-conditioned 双头;
- baseline 用相同 num_steps 窗口重训(β=0)——窗口构成对照,必做。

---

## 5. 实验矩阵与顺序(evidence 视角裁定,不可协商)

**Gate 0(先于一切,天级)**:matched 3-frame planner history 重跑
{baseline, multistep, sgmulti_b1, sgmulti_b2} × 2-3 seeds × 50ep。
所有现有 planning 数字在此之前只能标注 "possibly history-confounded"。

**Gate 1**:fixed-φ0 refit 协议(§4),3 refit seeds,matched 容量/步数,CI。
同时对 φ_multistep 做同样 refit → 得到两个几何的 D̂*,三路分解第一次真正落地。

**Gate 2(泄漏修复,在任何"预注册诊断"语言写下之前)**:episode-level split、
train-only normalizer 随 checkpoint 保存、held-out eval goals(不用 pusht_expert_train.h5 的 goal)、
run metadata(seed/SHA/CEM config/history_len)。

**Gate 3(证书的预测力)**:N≈10-12 模型群体(β×K×seed 网格,多数已有),
分解不需要 planner ⇒ **先算几何份额、锁预测、再跑 planning eval**——这是真预注册。

**Gate 4(理论试验台 + 外部咬合)**:piecewise/hybrid toy env(条件熵解析已知,
floor 唯一可对真值检验的地方);外部咬合:复现一个 Fast-LeWM 式 multi-horizon 增益,
用证书判定其中多少是几何侧——这一步把分析笔记变成"必须引用的论文"。

预算量级:~8-10 次训练当量 + ~20 次 planning eval + toy,几张卡 2-3 周。

---

## 6. Tier-0 诊断(Idea 2 的降级归宿)

- **oracle-SER**,形式改为 regret 推导出的 `threshold(ρ_v 掉出 top-D) × ω(v)κ(v)`,
  σ² 是侵蚀概率代理、ωκ 是损伤,不再用三因子光滑乘积;
- κ 的候选分布 Q **外生化**(random/prior/expert-perturbation),不用被训模型的 CEM(循环性);
- ranking sufficiency(候选动作排序一致率)作为 planning 的近端代理指标,
  在警报触发后判定"丢的是 nuisance 还是 control-critical";
- 永不门控 loss;只诊断。学习式 gate 违反本项目已确立的 oracle-over-learned-gate 立场。

---

## 7. Related work 增补(round-2 指出的引用墙)

| 工作 | 覆盖 | 我们的差异句 |
| --- | --- | --- |
| Tang et al. 2023, Understanding Self-Predictive Learning in RL | 耦合 (φ,f) 自预测 + stop-grad 的形式分析 | 他们研究 collapse;我们研究 **SIGReg 下无 collapse 的度量重写** |
| Ni et al. 2024, Bridging State and History Representations | 自预测目标选择什么 | 同上,且我们给出可测证书 |
| Lambert et al. 2020, Objective Mismatch in MBRL | "model loss 降 ≠ control 升" 总伞 | 我们给出**归因分解**:哪部分改善是假的、为什么 |
| Xu et al. 2020, V-information | D*(φ) = 受限类下可用信息 | 承认同构,贡献在方向性分解与 floor 警报 |
| VaGraM / VAML / Grimm value equivalence | 决策感知模型学习(ω 加权) | 我们不加权训练,先证书后锚定;SER 只诊断 |
| Malik et al. 2019 / PETS | 校准/分布式 dynamics | 自产 target 的 impropriety 定理:校准**必须**配几何锚定 |
| DBC / bisimulation | latent 距离对齐任务距离 | reconstruction-free JEPA + 开环 CEM 设定;答"为什么不用 bisim" |
| DreamerV3 / Fast-LeWM / DINO-WM / PLDM | stop-grad 分离、multi-horizon、frozen-φ 两段式 | frozen-φ 两段式**正是**我们证书的 refit 臂;证书反过来审计他们的 error 主张 |

---

## 8. 8/10 gates(round-2 修订版)

```text
[ ] Gate 0 通过且 82-vs-22 反转幸存(否则整个 board 重排,phenomenon 轴掉到 4)
[ ] fixed-φ0 落地,O1-O4 判定写入文档(无论哪个结局,分解基础设施都保留)
[ ] T2 三路分解 + T3 不变性以定理形式写出(assumptions + limitation)
[ ] T4 impropriety 定理写出(半页),方法 = 定理推论
[ ] Gate 3 预注册预测力:几何份额在 N≈10+ 模型群体上预测 planning 损伤,
    优于 self-drift / probe R² / inverse dynamics
[ ] toy piecewise env 上 floor 对解析真值检验通过
[ ] 校准 f + 风险感知 CEM ≥ baseline planning(matched history,3 seeds)
[ ] 外部咬合:证书审计一个 Fast-LeWM 式增益
[ ] 引用墙(§7)全部定位
```

只过前四个 ⇒ 6/10(诊断论文);全过 ⇒ 可诚实主张 8/10。

---

## 9. 一句话留给下一步

> 下一步不是发明新 loss,而是先跑 Gate 0(matched-history 重跑,天级)——
> 它决定这个方向的 phenomenon 轴是 7 还是 4;然后 fixed-φ0 按 §4 的结局表落地。
> 理论侧唯一优先事项:把 T2/T3/T4 写成带假设的定理。方法自己会从 T4 里长出来。
