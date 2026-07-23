# 从 2026 趋势反推方向:判决书(Claude,2026-07-23)

> 输入:`wm_trends_research_20260723.md`。按 §4 承诺:用成功配方反推方向 →
> 三筛子 + 碰撞 → 单轮三审稿人 → 押一个或说换赛道。

## 0. 调研给的战略信号(比任何单个 idea 都重要)

三条成功配方里,**只有配方 #2 是我们的量级能进的赛道**:
- 配方 #1(prediction+action 进同一生成图)= diffusion/AR,1B CDiT、驾驶/视频,
  我们 15M PushT 进不去,且要换整个 infra;
- 配方 #3(新闭环 + 可扩展资产)= WMPO/Newt/ScaleZero,要 policy 训练环或
  200-task 规模,也不是我们;
- **配方 #2(极简 task-relevant latent 结构 + 机制解释 + 可见能力增益)=
  Temporal Straightening / GRWM / GeoWorld,全部 ICML'26/CVPR'26 接收,
  全部小结构 + 几何选择 + 长程能力增益。这就是我们的家。**

**关键对照:GRWM(CVPR'26)的模板几乎就是我们的故事**——"给表示加一个
几何选择(时间邻近对比)→ plug-and-play → **把长程失败定位到表示而非更大
dynamics** → 长程 fidelity 提升"。我们的横向发现是:horizon 诱导的几何把长程
规划定位到 encoder,远 goal 效应 4→16pt。**同构。GRWM 中了,证明这条能中。**

**由此修正我之前"问题定义论文很弱"的判断**:错的不是机制,是我把它写成了
**否定式**(测量 gap + 三重否定)。GRWM/TS 证明**建设式**(一个结构选择 +
机制 + 长程能力)能中。我们一直缺的不是"再一个 idea",是**把已有机制配一个
在其效应最大处展示的能力增益 + 一个 plug-and-play 配方**。

## 1. 反推的三个方向

### 方向 A(推荐):Coupling as Geometry —— GRWM 模板 + 我们的机制

- **形态**:一个 plug-and-play 的训练选择(anchor+dose 配方:全权重单步 +
  剂量化耦合自复合),配一条 solid 定理(一步盲性 + 不对称分配),能力在
  **远 goal / 低容量 / 长 horizon** 处展示(4→16pt,效应最大区);
- **建立在**:耦合论题 + rate 律 + 远 goal 相图(已全部实测);
- **为什么是"模型/目标"不是补丁**:它是一个可被任何多步 JEPA WM 直接换上的
  训练目标(drop-in),GRWM/TS 同型;不是 planner 补丁、不是诊断;
- **三筛子**:(1)一行改动,更简单 ✓;(2)是训练目标/建模选择,可 drop-in ✓;
  (3)定理已有 ✓;
- **kill-test(已大半在手)**:anchor+dose 补种子(verify wave 已排程)+
  head-to-head 打 `K1+curvature`(TS)、`K1+bisim`(Invariant);预注册:
  一步几何正则改善切向几何但不产生远 goal 增益,anchor+dose 两头拿;
- **最大风险**:与 TS/GRWM 的差异是"离轨误差传播几何 vs 真轨迹切向几何",
  微妙——必须靠 head-to-head 实测赢,不能只靠措辞;
- **最近邻**:GRWM(2510.26782)、TS(2603.12231);诚实差异化 = 我们控制
  **递归想象误差的横向传播**(off-trajectory),它们控制真轨迹几何(on-trajectory),
  且我们有 rate 律把"控制多强"量化成一个数。

### 方向 B(有 hook 但风险高):Generative LeWM —— 把 JEPA 从确定性变可采样

- **形态**:LeWM 是确定性的(这是它的身份也是它的天花板——进不了配方 #1 的
  生成范式)。给它一个**由分配律塑形的 innovation**:随机转移
  `z' = f(z,a) + Σ(z)^½ ε`,Σ 在高放大方向上**低**(放大方向不该再注噪声);
- **建立在**:不对称分配 + "控制/放大共享方向"新发现(所以 Σ 不能按方向切,
  要按 source 切——innovation 是误差通道,action 保留);
- **为什么不是 BA-GWM 重演**:BA-GWM 死于"predictor 线性层吸收对角方差";
  这里 Σ 是**采样分布的形状**,predictor 吸收不了一个 sampling covariance
  (确定性缩放能吸收,随机性不能)——正好补 BA-GWM 死因;
- **三筛子**:(1)一个 covariance head,简单 ✓;(2)**从确定性变生成式,
  正对配方 #1 的主流信号** ✓✓;(3)分配律给 Σ 的闭式目标 ✓;
- **kill-test**:训 stochastic-LeWM D32 K5;查 (i) innovation 方差与放大反相关
  (分配律),(ii) 采样 rollout 给出校准的多未来,(iii) 采样 cost 规划 ≥ 确定性;
- **最大风险(致命候选)**:PushT 近确定性,innovation 可能塌到零(没有随机性
  可建模)——第一个 kill-test 就判;且"stochastic latent WM"极度拥挤
  (Dreamer/PlaNet/所有 SSM-WM),唯一新意是"Σ 由放大律导出而非 ELBO 自由学",
  审稿人可能归档为"innovation 上加了个 prior";
- **最近邻**:MoP-JEPA(多模态 successor,但离散 K-head vs 我们连续 Σ)、
  Dreamer 系(随机 latent,但非 JEPA/SIGReg、Σ 非从律导出)。

### 方向 C(降级):纯几何选择(off-trajectory 曲率)—— 并入方向 A

单独做就是"再一个 geometry regularizer",直接被 TS/GRWM 覆盖。作为方向 A
的一个 ablation 臂存在,不单列。

## 2. 单轮三审稿人(NeurIPS/ICLR 怀疑者,不迭代)

- **R1(WM 方法审稿人)方向A 7/方向B 5.5**:A 的 GRWM/TS 模板成熟、能力增益
  在远 goal 真实存在,若 head-to-head 赢就是干净的接收;B 的生成 hook 诱人
  但 PushT 确定性风险 + stochastic-WM 拥挤,像"给 innovation 加 prior"。
- **R2(理论审稿人)方向A 7.5/方向B 6**:A 的一步盲性定理 + rate 律是它比
  TS/GRWM 多出来的理论厚度(它们只有几何直觉),这是加分项;B 的"Σ 由放大律
  导出"若能证成一条 propositon 会很漂亮,但需要 PushT 有随机性可建模。
- **R3(恶意怀疑者)方向A 6/方向B 5**:A 仍要正面赢 TS,差异微妙,若 head-to-head
  平手就是增量;B 大概率第一个 gate 就死于确定性。

均分:**方向 A ≈ 6.8,方向 B ≈ 5.5**。A 过线且有成熟模板背书;B 是高方差赌注。

## 3. 判决:押方向 A,方向 B 作为"新颖性 hook"备选(先做零成本 gate)

**主线 = 方向 A(Coupling as Geometry)**,按 GRWM/TS 模板重写我们的机制资产:
不是否定式问题论文,是**建设式**——一个 drop-in 训练目标(anchor+dose)+
一步盲性/不对称定理 + 远 goal/低容量的能力增益 + 打 TS/Invariant 的 head-to-head。
这把一个月的所有资产(rate 律、不对称、耦合、远 goal 相图、失败对照)全部
变成一篇 GRWM 型论文的零件,且**能力增益在效应最大处展示**(不是 PushT 近 goal
的 1.7 分,是远 goal 的 16 分)。

**方向 B 的一票否决(零训练,先跑)**:PushT 到底有没有随机性可建模?
用现有 K1/K5 rollout 测:同一 (z,a) 下真实 next-state 的条件方差,以及它是否
沿放大方向分布——若条件方差 ≈ 0(纯确定性),方向 B 死,只走 A;若非零且
结构化,方向 B 作为 A 的生成式扩展(把"能力增益"从"远 goal 规划"升级到
"可采样的校准多未来"),新颖性更强。

## 3.5 方向 B 严格 gate(A100,2026-07-23,正式 CLOSE)

smoke 的分位近邻确实把跨条件漂移误当成了条件方差。正式协议改为:

- 20,000 probe 窗口,去掉 181 个完全重复样本,实际 N=19,819;
- 标准化条件空间中的**绝对半径**,拒绝"最紧 x%"分位规则;
- `k=1` 且贪心选互不重叠的 pair,避免一个样本被多次计数;
- 至少 30 个独立 pair 才判;同时查 current 17D 与三帧 history 51D 条件键;
- kill 阈值锁定为 `median(cond_std)/latent_scale < 0.02`。

达到 30 组要求的最小半径结果:

| 条件键 | 绝对半径 | 独立组 | cond_std/scale | anisotropy | amp 方差占比 | 随机基线富集 | 判决 |
|---|---:|---:|---:|---:|---:|---:|---|
| current | 0.0125 | 32 | 0.01286 | 46.29 | 0.03961 | 2.54x | **KILL** |
| history | 0.0250 | 32 | 0.01372 | 41.89 | 0.04116 | 2.63x | **KILL** |

半径增大后才出现 PASS:current 到 0.030 时为 0.02362,history 到 0.045
时为 0.02062。这个随半径扩张而越过阈值的形状正是**确定性局部漂移污染**,
不是固定条件下的不可约随机性。

严格结果也修正了 smoke 的另一个过度判断:完整条件协方差在 top-3 放大子空间
中约占 4.0%–4.3%,是随机子空间基线的 2.4–2.8 倍,并非完全不对齐。但方差
量级只有全局 latent scale 的 1.3%–1.7%,先触发"近确定"一票否决;一个有方向
但随邻域收紧趋小的残差,不足以支撑 stochastic covariance head。

**正式判决:方向 B 在 PushT 上 CLOSE,不再训练 stochastic-LeWM。**
完整协议、半径扫描、8k/20k JSON 和 smoke 校准日志见
`docs/knowledge/trends_a100_20260723/REPORT.md`。

## 4. 立即动作

1. **方向 B gate:已完成并 CLOSE**;不再消耗训练预算;
2. **主线(方向 A):A100 已启动**。4 个 K1+curv/bisim 任务正在四卡并行,
   完成后自动评测;verify wave 的 6 个新名字任务由同一 driver 接续执行;
3. 等 head-to-head/verify 结果落盘后,按 GRWM 模板完成论文骨架
   (现象→结构选择→定理→远 goal 能力→head-to-head)。

**不再做**:CI-GWM/BA-GWM(已死)、CritWM 闭环、tail-validity 方法线(已 CLOSE)、
任何否定式问题论文形态(被 GRWM 模板取代)。
