# 主文档:接触动力学的收缩表示(方向代号 contact-contraction)

> **本文档是当前方向的唯一权威文本**,吸收并取代 2026-06-28 至 07-04 期间
> `iterating_ideas/lewm_sufficiency_erosion/` 全部理论稿与
> `multistep_unroll_drift.md` / `dstar_decomposition.md` / `theory_sufficiency_loss.md`
> (原文见 git 历史,原始数据见 `outputs/`)。
> 配套:竞品地图见 [arxiv_topconf_worldmodel_survey_2026-07-04.md](arxiv_topconf_worldmodel_survey_2026-07-04.md)(含 §10 审稿补遗)。
> 状态:Round-3 审稿(3 独立视角,novelty 最高权重)共识 **4/10 now → 7.5-8 potential**;
> 修复路径已收敛(§6)。"LeWM++"只是占位符,不预设增量式定位。

---

## 1. 论题(Round-3 重排后的头条)

> PushT 类接触系统是 hybrid/分岔的。学 latent world model 时:
> **(i)** 边界锐度 d 不是物理给定的,而是 **encoder 的决策变量** —— 由此单步/多步
> 训练目标在"边界跟踪 / 盆对齐 / 边界放弃"三种解之间做出不同选择(三难不等式);
> **(ii)** 单步 MSE 对复合增益**梯度盲**,K 步训练在数学上是**隐式收缩度量学习**,
> 且增益整形 ~100% 发生在 encoder(实测);
> **(iii)** 真实系统在分岔处**不可收缩**,因此均匀收缩正则是错误的 —— 这是对收缩
> 文献的反命题;margin 必须与收缩共存;
> **(iv)** self-drift 在模型种群上不预测 planning(ρ=0.26),**冻结-refit D\* 证书**
> 预测(ρ=0.94)—— 不跑 planner 即可认证可规划性。

三个检索验证过的新颖点:证书(iv)、d-as-encoder-decision(i+iii)、梯度盲/隐式收缩
度量学习(ii)。方法(§4)是它们的推论,不是新颖性本体。

## 2. 已验证事实(全部多测量/多种子/预注册)

**E0 · Gate 0(评测协议)**:planner 默认 1 帧冷启动是伪影来源。matched 3-frame
history 下(实现:`policy.py` 历史缓冲 + `LeWM.rollout` past_action,
`+plan_config.history_len=3`):

| model(D=192) | h=1 旧数 | h=3(3 评测种子) |
| --- | ---: | ---: |
| K=1 baseline | 82% | 83.3% |
| K=5 multistep | 22% | **88.0%** |
| sgmulti b1/b2 | 50/52% | 56.0/53.3% |

**E1 · 扩张定位在接触**(扰动增长 ‖δ‖/ε,ε=0.01,512 窗×4 向):所有模型的
growth@8 随接触强度上升(ρ 0.21-0.34,p<1e-20;效应量 4-11% 秩方差——注意:
**四模型共有**,非 K=1 特异,S1 简单假设"接触本来更难"未被排除,见 §6 Gate)。
K=1 free 0.61→heavy 1.61;K=5 free 0.27→heavy **0.47(重接触仍收缩)**。

**E2 · 收益住在 encoder(D\* 分解)**:冻结几何、从零 refit predictor
(11.7M,60ep,2-3 seeds,`regime_lewm_predictor_stepB2.py`):

| 冻结几何 | refit 单步 | refit K=5 | 单步 teacher-forced MSE |
| --- | ---: | ---: | ---: |
| φ_K1 | 0.381 | **0.335(下界,打不过自家 0.315)** | 0.0186 |
| φ_K5 | **0.194** | 0.158 | 0.0192 |

单步可预测性两几何**相同**;开环复合性差 2 倍 → 多步教的是 encoder 不是 predictor。
Markov gap ≈ 0(1 帧 vs 3 帧单步同精度)→ 差的不是信息是误差几何。

**E3 · planning 跟随复合稳定性**:冻结几何+refit-f 拼装上 CEM:多步 f 74% vs
单步 f 50/58%,尽管后者单步 MSE 好一倍。CEM 消费开环 rollout。

**E4 · 容量相图(6 格 × 5 量,预注册 P1-P4)**:

| cell | planning | growth@8 | D\*_multi | angle R² |
| --- | ---: | ---: | ---: | ---: |
| D=192 K=1 / K=5 | 83.3 / 88.0 | 1.46 / 0.44 | 0.335 / 0.158 | 0.934 / 0.813 |
| D=32 K=1 / K=5 | 66.7 / 80.0 | 2.19 / 0.69 | 0.397 / 0.219 | 0.571 / 0.648 |
| D=8 K=1 / K=5 | 25.3 / **52.0** | 3.11 / 1.27 | 0.716 / 0.519 | 0.396 / **0.107** |

判读:无侵蚀相边界,K=5 优势随压缩**单调放大**(+4.7→+26.7);
**D=8 K=5 丢角度线性可解码性(0.107)却翻倍 planning** —— 表征侵蚀与控制损伤解耦;
跨种群 Spearman(planning):**D\*_multi +0.94(p=0.005)** > angle R² +0.89 >
growth@8 +0.71 > **self-drift +0.26(失效)**。

**E5 · 物理前提**:PushT piecewise/接触分岔;自由块平移不变(那些方向半收缩 λ=1)。

## 3. 理论核心(Round-3 修订版)

记 piecewise 动力学 F,分岔隙 Δ,encoder φ,predictor f,局部增益 L_f,
一步一致性误差 ε;开环误差递推 δ_K ≤ Σ (Π L_f) ε(标准界,非贡献,cf. Asadi'18)。

**Thm A(三难,贡献=d 是 encoder 决策变量)**:对跨界 pair
(latent 距 d,margin 需求 m):`m ≤ 2ε + L_f·d`(+φ∘F 连续模的 slack 项;L_f 取
线段上的 Lipschitz 常数)。不等式本身是经典 Lipschitz-vs-跳变论证(接触学习文献
已有);**新意在 d 由表征选择**,三种解对应三种训练结局:边界跟踪(单步:精确、
复合爆炸)/盆对齐(多步:可复合、卖盆内细节)/边界放弃(纯收缩正则的退化,
margin 丢失)。
A(a) 修订:盆内收缩度量的存在性只在**接触啮合子空间 + 增量稳定假设**下成立,
且黎曼度量→欧式 pullback 只有局部/近似版本(平坦性缺口);自由块方向 λ=1。

**Thm B(梯度盲,严格表述)**:在等 ε 极小集内,L_1 对增益配置无选择压力
(insensitivity of the minimizer set,插值域内成立);L_K 的被积函数显式含 G²
→ 降增益在下降方向且梯度达 φ。**identification(检索无先例):多步 JEPA =
隐式收缩度量学习,且增益整形在 encoder(E2)**。
原"许可证推论"降级为**假设 P-a**:各向同性一阶增益惩罚 ≠ 增益加权方向性梯度,
是否够用只能实验判。

**Conjecture C(容量分配,由定理降级)**:容量受限下按"盆分离 > 盆内低方差 >
盆内高方差"分配。表征侧预测(角度先崩)预注册命中;控制侧原预测(planning 受损)
被相图证伪后修补,故不称定理。planning 推论 `|J_z(a)−J_z(b)| > 2 G_K ε̄` 平凡但有用。

## 4. 方法方向(占位名不定;不强求 loss 融合)

主形态 **3 项,每项有定理级存在理由**:

```
L = 单步 pred MSE            (LeWM 原生;Thm B 说它对增益盲 → 需补)
  + λ·SIGReg                 (保留:LeJEPA 可辨识性理论 2605.26379 的护盾;反坍缩)
  + μ·度量项                  (Thm A 的 margin + 收缩;定义必须钉死:
                               action-matched pairs / 有限 H / ground term,
                               继承 Kemertas'21 的反坍缩修复)
```

两-loss 统一形已被 Round-3 数学击毙(等距不动点 + telescoping 坍缩 = reward-free
bisim 已知病理),只作消融。一阶收缩惩罚(原 M2)作为 P-a 的检验臂,不是方法本体。
Money figure(Reviewer B 设计):D=8 跨盆测试,1-step bisim(Invariant-JEPA-WM 式)
与 Fast-LeWM+SC 都失败、本方法通过。

可证伪预测:P-a(一阶惩罚达成 K=5 级 growth/planning,K=1 成本)、
P-b(纯收缩在 D=8 出跨盆错,margin 修复——**承重墙,现有证据偏向反面**)、
P-d(接触稀疏任务上三项增益都小 → 机制化解释 What-Drives 的 rollout-length 规律)。

## 5. Novelty 地图(生死线)

| 对手 | 关系 | 生死线 |
| --- | --- | --- |
| Invariant-JEPA-WM 2602.18639 | **最危险**:JEPA 内 reward-free 1-step bisim | H-step 开环分歧 vs 1-step transition 相似;Thm B 证后者对 G_K 梯度盲;head-to-head 消融必做 |
| RC-aux 2605.07278 / TRM 2605.22164 | **同 base model**,同 gap 正在被挖 | 引用+正面击败;时间窗收紧 |
| NCDS ICLR'24 等收缩文献 | latent 收缩已有 | 反命题:分岔处不可收缩,均匀收缩是错的,margin 必须共存 |
| MICo / Kemertas'21 | 度量项的病理与药方已有 | 继承修复,differentiation 在 H-step 分歧 + 双向 + 收缩理由 |
| Fast-LeWM / What-Drives | 多步监督/recipe 全占 | 不做"更低 error";做归因(D\*)+机制解释(P-d) |
| Asadi'18 / Lipschitz-MBRL | 复合误差界+Lipschitz 控制已有 | 界非贡献;贡献在目标选择定理与三难 |

## 6. 下一步 Gates(顺序锁定)

1. **三难另两轴测量**(零训练,现有 checkpoint):ε|接触、margin|接触、
   **方向分辨**扰动增长(沿/垂直边界方向)—— 判 S1 vs 边界几何,E1 的效应量问题在此解决;
2. **盆-穿越实验**(理论的 Gate 0,Reviewer C 全套设计):模拟器侧盆分类(接触与否
   × 转动方向,先验容差排除带)、冻结解码规则、**误差幅度配对比较**、
   O1(K=1 跨盆多→三难活)/O2(等率→退回 S1,弱论文)/O3(K=5 跨盆多→边界放弃
   模式,叙事重写)预注册,D=192 与 D=8,≥2 训练种子;
3. **piecewise-linear 玩具系统**:Thm A/B 对解析真值检验;
4. **实现 + 对打**:3 项方法(度量项定义钉死)+ 1-step bisim 臂 + Fast-LeWM+SC 臂,
   PushT test1;Fast-LeWM 也做 D\* 分解(守对账表);
5. **OGBench-Cube**(Fast-LeWM 74→82,未饱和)主战场 + navigation 对照(P-d);
6. 统计:每格 ≥3 训练种子、加密 D=16/64 与 K=10、≥150 评测集、n≥12 证书种群。

## 7. 死路线墓志铭(几段话,详情在 git 历史)

**侵蚀叙事(sufficiency erosion,06-28~07-02)**:"低 drift 但 planning 崩
(82→22)"是 planner 1 帧冷启动伪影,Gate 0 杀死。教训:先排评测,再谈机制。
衍生的 5 篇理论稿(predictive_sufficiency_dynamics / control_sufficient_gjepa /
novelty_loop / acceptance_audit / innovation_iteration_protocol)全部废止;
幸存物:选择压力 toy theorem 的表征侧预测(D=8 角度崩塌逐字应验)与
"证书先于方法"的工作方式。

**sgmulti(stop-grad 多步)**:planning 53-56%、连 k=1 drift 都翻倍。教训:
几何不变则 f 无油水(D\* 下界);"保护 encoder"方向反了——encoder 参与共训是
收缩性的来源。

**v4 drift-floor spine("Certify, don't chase")**:方向作为论文挂起,但其
证书机器(refit-D\*、三路分解、度量不变性要求、impropriety 论证)全部并入本文档
§1(iv)/§2/§6,是现存最强资产。

**M1/M2/M3 便宜收缩冲刺(作为 headline)**:被调研判死(Fast-LeWM/What-Drives
占地)+Round-3 确认"单独的 Lipschitz 惩罚是 established practice"。M2 降级为
P-a 检验臂。

**两-loss 统一形(§5.5,07-04)**:数学错误——不动点是等距非收缩;telescoping
坍缩(Kemertas'21 已证的 reward-free bisim 病理);φ→0 精确联合极小;BN 反坍缩
脆弱且与 LeJEPA 可辨识性冲突。教训:极简 loss 是卖点不是公理,每项 loss 的存在
理由必须是定理。
