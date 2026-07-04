# 主文档:接触动力学的收缩表示(方向代号 contact-contraction)

> **本文档是当前方向的唯一权威文本**,吸收并取代 2026-06-28 至 07-04 期间
> `iterating_ideas/lewm_sufficiency_erosion/` 全部理论稿与
> `multistep_unroll_drift.md` / `dstar_decomposition.md` / `theory_sufficiency_loss.md`
> (原文见 git 历史,原始数据见 `outputs/`)。
> 配套:竞品地图见 [arxiv_topconf_worldmodel_survey_2026-07-04.md](arxiv_topconf_worldmodel_survey_2026-07-04.md)(含 §10 审稿补遗)。
> 状态:Round-3 审稿(3 独立视角,novelty 最高权重)共识 **4/10 now → 7.5-8 potential**;
> 修复路径已收敛(§6)。"LeWM++"只是占位符,不预设增量式定位。

---

## 1. 论题(2026-07-04 Gate 之后的诚实版本)

> **仍然站着的(三轮证伪未伤)**:
> **(i)** 单步 MSE 对复合增益**梯度盲**,K 步共训做的是**均匀的隐式收缩**
> (Gate 1b 直接验证:全方向同比例阻尼 ~4×,无边界差异化),且增益整形 ~100%
> 发生在 encoder;
> **(ii)** 收缩性在容量压缩下是**保护性的且单调放大**(+4.7→+26.7 @ D=8);
> **(iii)** self-drift 跨模型不预测 planning(ρ=0.26),**冻结-refit D\* 证书**
> 预测(ρ=0.94)—— 不跑 planner 即可认证可规划性;
> **(iv)** 新观察:边界方向误差放大(~3×)是所有模型共有的 latent 结构量,
> 随容量压缩消失;决策相关信息比盆结构更窄(D=8 K=5 解不出盆类别仍 52%)。
>
> **被 Gate 杀掉的**:三难/盆对齐机制(训练目标选择边界几何)——见 §6.5/§7。
>
> **方向状态(2026-07-04 晚,用户决策)**:幸存的自然方法路径 P-a
> (一阶增益惩罚以 K=1 成本复刻 K=5)**创新性上限不高,不做**
> (Lipschitz 正则是 established practice,审稿人已预警 cap≈5)。
> 方向**暂停,等待更高上限的问题重构**;资产清单见 §6。

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

## 6. 资产清单与重启点(方向暂停中;原 Gate 序列 1-2 已执行,见 §6.5)

**可复用资产(全部就绪,不随方向死亡)**:
- 证书机器:refit-D\*(`regime_lewm_predictor_stepB2.py`)、扰动增长
  (`outputs/mech/perturb_growth.py`)、方向分辨/接触条件版(`outputs/gates/`)、
  matched-history planner 评测(`+plan_config.history_len=3`,Gate 0 修复);
- 模型种群:6 格相图 checkpoint + 全部五量测量;
- 定理素材:梯度盲(Thm B,受支持)、容量分配 conjecture(表征侧命中)、
  经典复合界与三难框架(机制被证伪,框架可引);
- 竞品地图:调研 + §10 审稿补遗(7 个近邻的生死线)。

**候选重启点(品味用,未承诺)**:
1. 证书/评测线:decision-fidelity benchmark(证书 + Gate 0 案例研究 +
   种群验证),乘 WorldModelGym 风口——稳但偏 analysis;
2. 边界方向放大量(§1.iv)作为新现象深挖:它是 hybrid 结构在 latent 的痕迹,
   训练目标不利用它——"应该利用吗?怎么利用?"尚无人问;
3. 换问题域:OGBench/长 horizon 上先找一个 baseline 真死、且现有资产能解释的
   失败现象,现象先行(本 repo 全部有效进展都始于一个硬现象)。

(原 Gate 序列:1-2 已执行并判决于 §6.5;3-6(玩具系统/方法对打/OGBench/统计
加固)随方向暂停一并搁置,方法对象已不存在,重启时按新问题重排。)

## 6.5 Gate 1+2 结果(2026-07-04 晚,`outputs/gates/`)——三难机制未获支持,S1 胜出

预注册预测几乎全部落空(诚实记录):

- **1a ε|接触**:预测"K=5 边界相对精度更差"——**反了**:K=5 的 heavy/free ε 比
  在所有容量下更低(1.60/1.82/2.17 vs K=1 的 2.43/2.86/2.81);K=5 付的精度代价
  是全局的(自由空间 ε 翻倍),非边界特异。
- **1b 方向分辨增益(关键判决)**:边界方向扰动确实放大 ~3×(方向结构真实存在,
  且随容量压缩消失:2.9→1.9→1.1);但 **K=1 与 K=5 的边界/随机比几乎相同**
  (2.88/2.71,1.94/1.86,1.14/1.09)——**K=5 是均匀阻尼(全方向同比例 ~4×),
  不是盆几何重构**。S1(均匀增益抑制)胜出。
- **1c margin**:所有模型 margin 比值 2.3-2.8、ρ≈0.56 无 K 差异;无边界放弃,
  但 margin 项的经验动机(P-b)不复存在。
- **2 盆-穿越**:v1/v2 解码协议无效(oracle<0.9;根因是我把历史依赖的类别用单帧
  终态解码——协议设计错误);v3(端点类别+种子/终点对)D=192 oracle 0.85,
  幅度配对后 K1=0.187 vs K5=0.164——**近似等率(O2 倾向)**,仅最低误差档 K=5
  略优(0.171 vs 0.293)。**D=8 K=5 oracle 仅 0.59:它的 latent 线性解不出盆类别,
  planning 却 52%**——连"盆级信息"都不是 CEM 的必需品(强化 E4 的解耦)。

**理论更新**:Thm A 的"训练目标选择边界几何"机制**不成立**——三难降级为经典界+
表述框架;**Thm B(梯度盲→均匀隐式收缩)成为受支持的机制**,其预测(均匀性)
被 1b 直接命中。margin 项从方法中移除,方法收敛为"一阶增益惩罚能否以 K=1 成本
复刻 K=5"(P-a,现在是最可能成立的假设)。新增独立观察:边界方向误差放大是
所有模型共有的结构量、被压缩抹除;它是 hybrid 结构在 latent 里的痕迹,但训练
目标不差异化利用它。

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

**三难/盆对齐机制 + margin 项(07-04 晚,Gate 1+2)**:预注册预测大面积落空——
K=5 的精度代价是全局的非边界特异(1a 反向);边界/随机扩张比在 K 间无差异,
K=5 是均匀阻尼非盆几何重构(1b);margin 结构对训练目标不变(1c);幅度配对后
盆-穿越近似等率,且 D=8 K=5 解不出盆类别仍能规划(2)。机制层面 S1(均匀隐式
收缩=Thm B)胜出。教训:两小时的 Gate 杀掉一个性感理论,好过两周建在它上面。

**P-a 便宜收缩方法(07-04 晚,用户决策)**:机制上最可能成立,但创新性上限不高
(Lipschitz 正则 established practice,审稿预警 cap≈5)——**主动不做**。
教训:可行 ≠ 值得做;上限判断先于实验投入。
