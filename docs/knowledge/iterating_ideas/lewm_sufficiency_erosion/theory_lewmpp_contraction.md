# LeWM++ 理论推导稿 v1:接触动力学的收缩度量表示论(2026-07-04)

> ⚠️ **Round-3 审稿判决(2026-07-04,3 独立视角,novelty 最高权重):4/10 now → 7.5-8 potential。**
> 修复指令(共识,详见文末 §8):
> **§5.5 两-loss 形作为定理已死**——不动点是等距非收缩;telescoping 坍缩 = reward-free
> bisim 已知病理(Kemertas & Aumentado-Armstrong NeurIPS'21 已证明并修复);φ→0 是精确
> 联合极小;BN 反坍缩是已知脆弱技巧且与 LeJEPA 可辨识性理论(2605.26379)冲突。
> **主形态改回 3 项(保留 SIGReg),两-loss 版降级为消融**;度量项必须限定
> action-matched pairs + 有限 H + ground term。
> **真正幸存的新颖性(三方一致)**:① refit-D* 证书 + 种群验证;② "边界锐度 d 是
> encoder 的决策变量"(三难不等式本身是经典 Lipschitz-vs-跳变,接触学习文献已有;
> 反转成"真实 hybrid 系统在分岔处不可收缩,故均匀收缩是错的、必须带 margin"才是新的);
> ③ "单步 MSE 对复合增益梯度盲 / 多步 JEPA=隐式收缩度量学习"(检索未见先例)。
> **文档目前把最弱主张放头条、把最强的埋了——重排。**
> 数学修正:Thm A(a) 黎曼收缩度量 ≠ 欧式 pullback(平坦性缺口,改局部/近似版);
> PushT 自由块平移不变 ⇒ 半收缩 λ=1,定理限定"接触啮合子空间";Thm B 推论降级为
> 假设 P-a(各向同性惩罚 ≠ 增益加权梯度);Thm C 降级为预注册 conjecture。
> **新增 novelty 威胁(调研未收录,必须对打)**:RC-aux (2605.07278, 同 base model!)、
> TRM (2605.22164, 同 base model!)、Invariant-JEPA-WM (2602.18639, 最危险:JEPA 内
> reward-free 1-step bisim)、NCDS (ICLR'24, 杀死"首次 latent 收缩"措辞)、MICo/Kemertas、
> Asadi'18、LeJEPA 可辨识性 (2605.26379)。
> 差异化生死线:H-step 开环分歧 vs 1-step transition 相似度(可用 Thm B 证明后者对
> G_K 梯度盲)+ money figure:1-step bisim 和 Fast-LeWM+SC 都过不了 D=8 跨盆测试。
>
> 目的:把已验证的五个经验事实,从 hybrid 动力学第一性推导成定理体系,
> 并由定理**推出**(而非拼凑)LeWM++ 的训练目标。
> 状态:~~推导稿 v1~~ → **v1 已审,按上述判决修订中**;盆-穿越实验(误差幅度配对版,
> O1/O2/O3 预注册)与三难另外两轴的测量(ε|接触、margin|接触、方向分辨增益)先行。
>
> 经验地基(全部多测量、预注册背书):
> E1 单步训练 → latent 误差动力学扩张,且扩张定位在接触(growth 自由 0.61 → 重接触 1.61;ρ≈0.3, p<1e-20)
> E2 K 步共训 → 收缩(0.44),重接触处仍收缩(0.47);单步可预测性不变;收益 100% 在 encoder(D* 分解)
> E3 planning 跟随复合稳定性,不跟单步精度(拼装:74% vs 50%)
> E4 容量压缩下收缩性价值单调增(+4.7→+26.7);难预测接触 DOF(角度)的线性可解码性最先崩(0.107)且不伤 planning
> E5 物理系统是 piecewise/接触分岔的;分岔处真实动力学不可收缩

---

## 1. 设定与记号

真实动力学 piecewise-smooth:

$$
s_{t+1} = F(s_t, a_t), \qquad F = F_i \ \text{on regime} \ R_i, \quad
\mathcal{B} = \bigcup_{i \ne j} \partial R_i \cap \partial R_j \ \text{(接触/分岔边界)}.
$$

**分岔隙(bifurcation gap)**:对 $s$ 靠近 $\mathcal{B}$ 的两侧 $i,j$:

$$
\Delta(s,a) = \lVert F_i(s,a) - F_j(s,a) \rVert > 0
\quad \text{(推到/没推到,后续命运分开)}.
$$

encoder $\phi: o \mapsto z \in \mathbb{R}^D$,predictor $f$,局部 Lipschitz 增益
$L_f(z) = \lVert \partial f / \partial z \rVert$。一步一致性误差(teacher-forced loss 的被积项):

$$
\varepsilon(s,a) = \lVert f(\phi(s), a) - \phi(F(s,a)) \rVert .
$$

**开环误差递推**(标准,非贡献;cf. MBRL compounding-error bounds):

$$
\lVert e_{k+1} \rVert \le L_f(z_k)\,\lVert e_k \rVert + \varepsilon_k
\quad \Rightarrow \quad
\delta_K \le \sum_{k<K} \Big( \prod_{j=k+1}^{K-1} L_f(z_j) \Big)\, \varepsilon_k .
$$

定义**复合增益** $G_K = \prod_j L_f(z_j)$。我们的扰动增长测量 `growth@K` 就是
$G_K$ 沿数据轨迹的经验估计——理论量与已有测量一一对应。

---

## 2. Thm A:增益是坐标属性;盆内可收缩、跨盆必扩张

**(a) 盆内(regime 序列固定)。** 若动作条件流在盆内是增量指数稳定的
(耗散系统 + 摩擦,PushT 盆内成立;假设需实验核对),则由收缩分析的存在性定理
(Lohmiller–Slotine:系统在某度量下收缩 ⟺ 增量指数稳定),存在 encoder 度量
(等价:存在再参数化 $\psi \circ \phi$)使共轭 latent 动力学
$f^* = \phi \circ F \circ \phi^{+}$ 满足 $L_{f^*} \le \lambda < 1$。
要点:**Jacobian 的特征值在共轭下不变,但奇异值(=误差放大率)不是**——
encoder 改的不是物理,是"误差在哪个方向被度量"。这解释 E2:多步训练做的事
在数学上就是隐式的收缩度量学习。

**(b) 跨盆(边界带)。** 设 planning 需要区分分岔后的两支
(cost margin 要求 $\lVert \phi(F_i(s,a)) - \phi(F_j(s,a)) \rVert \ge m$)。
则对边界带内 $\lVert \phi(s) - \phi(s') \rVert = d$ 的跨界对:

$$
L_f \ \ge\ \frac{m}{d}
\quad \text{或者一步误差} \ \varepsilon \ge \frac{m - L_f d}{2}.
$$

**三难困境(本推导的核心不等式)**:边界锐度 $d$、predictor 增益 $L_f$、
边界处精度 $\varepsilon$ 不可兼得。三种解:

| 解 | $d$ | $L_f$ | 后果 |
| --- | --- | --- | --- |
| 边界跟踪型 | 小(两侧贴近) | 大 | 精确但复合爆炸(E1:单步模型) |
| 盆对齐/粗粒化型 | 大(两侧拉开,盆内压缩) | 小 | 可复合,牺牲盆内细节(E2/E4:多步模型) |
| 边界放弃型 | — | 小 | 边界处 $\varepsilon$ 大,margin 丢失,planning 跨盆错(纯收缩正则的风险,LeWM++ 必须防) |

**(c) 推论(floor)**:固定 $\phi$ 后,$\inf_f \delta_K$ 由该坐标系的
$d$/边界带测度决定——这就是 D* 下界为什么是几何属性(E2 的 refit 结果)。

## 3. Thm B:单步目标对复合增益"梯度盲",K 步目标是增益加权

对比两个损失对 encoder 的梯度信号。线性化误差递推后:

$$
\mathcal{L}_1 = \mathbb{E}\,\varepsilon^2
\qquad \text{vs} \qquad
\mathcal{L}_K \approx \sum_{k} \mathbb{E}\Big[ \Big( \sum_{j\le k} G_{j\to k}\,\varepsilon_j \Big)^2 \Big].
$$

- $\mathcal{L}_1$ 中**不含 $G$**:在 $\varepsilon$ 等精度类内,任何增益配置等价,
  encoder 没有降增益的梯度(只有经过边界带数据的间接、测度稀薄的信号)。
  ⇒ 单步最优解泛型地落在"边界跟踪型"(精度优先)——预测 E1。
- $\mathcal{L}_K$ 中 $G^2$ 显式出现:**降增益在下降方向上**,且梯度流向所有
  影响 $L_f$ 的参数——包括 $\phi$(坐标系决定共轭增益)。
  ⇒ 预测 E2(收缩来自 K 步)且解释 sgmulti/refit:$\phi$ 冻结时可达最小增益
  被 Thm A(c) 的 floor 卡死,f 单独训无用。

**推论(LeWM++ 的许可证)**:$\mathcal{L}_K$ 对 $\mathcal{L}_1$ 的全部增量信息
是"增益加权"。因此一个直接惩罚 $L_f$ 的一阶项可以在单步成本下补足这个梯度——
不需要 K 步 BPTT。这把此前 ad-hoc 的 M2 变成定理推论。

## 4. Thm C:容量受限下,盆内细节最先被卖

在 SIGReg 白化 + 维度 $D$ 约束下,最小化 $\mathcal{L}_K$ 且保持跨盆 margin $m$
的表征,其维度分配序(rate-distortion 论证,sketch):

1. 盆分离方向(同时携带 cost margin 和增益惩罚,双重收益)——最先保;
2. 盆内低条件方差方向(位置类)——次保;
3. 盆内高条件方差方向(接触角度类)——最先压缩。

⇒ 预测 E4 全部内容:相图单调性(容量越小,增益惩罚越占优,收缩型解的优势越大)
+ 角度 R² 在 D=8 K=5 崩塌 + planning 不受伤(margin 保住了)。

**Planning 推论**:CEM 候选 $a,b$ 排序正确的充分条件

$$
|J_z(a) - J_z(b)| \ >\ 2\, G_K \cdot \bar\varepsilon
$$

——收缩(降 $G_K$)直接扩大正确排序的候选对集合,这就是 E3 的公式化。

---

## 5. LeWM++:由定理推出的目标

$$
\mathcal{L}_{\text{LeWM++}}
= \underbrace{\mathbb{E}\lVert f(z_t,a_t) - z_{t+1} \rVert^2 + \lambda\,\mathcal{R}_{\text{SIGReg}}}_{\text{LeWM 原生}}
+ \beta\, \underbrace{\mathbb{E}_{u}\Big[ \big( \tfrac{\lVert f(z_t+\epsilon u, a_t) - f(z_t,a_t)\rVert}{\epsilon} - \gamma \big)_+^2 \Big]}_{\text{收缩项(Thm B 推论):补上 } \mathcal{L}_1 \text{ 缺的增益梯度}}
+ \mu\, \underbrace{\mathbb{E}_{(i,j)}\Big[ \big( m \cdot \lVert z^{+H}_i - z^{+H}_j \rVert_{\text{sg}} - \lVert z_i - z_j \rVert \big)_+ \Big]}_{\text{margin 项(Thm A(b)):未来分岔的 pair 不许在 latent 里被合并}}
$$

- **收缩项**:一次额外前向,把 $L_f$ 的梯度还给单步训练($\phi$ 与 $f$ 同吃)。
- **margin 项**:采样时间上邻近/latent 上邻近的窗口对,若它们 $H$ 步后的真实
  future(encoder 编码,stop-grad)相距远(= 数据告诉我们这里有分岔),
  则当前 latent 距离不得小于比例 margin。**reward-free、label-free**,
  防御 Thm A 三难中的"边界放弃型"退化——这是纯收缩正则(以及隐式的 K 步训练
  在极端容量下)都没有的保护。
- 与 bisimulation 的关系(必须写清):DBC/Invariant-JEPA-WM 用 bisim **合并**
  无关变量(不变性,上界方向);margin 项是**下界方向**(禁止合并分岔相关变量)。
  两侧合起来 = "latent 度量 ≍ 未来分歧",我们贡献下界侧 + 收缩侧 + hybrid 理论。

### 5.5 两-loss 统一形(v2,主推形态)

收缩项与 margin 项分别是 pair 距离比的上界与下界约束;合成双边对齐即一项:

$$
\mathcal{L}_{\text{metric}}
= \mathbb{E}_{(i,j)}\Big( \lVert z_i - z_j \rVert - \lVert z^{+H}_i - z^{+H}_j \rVert_{\text{sg}} \Big)^2
\qquad \text{(latent 度量 ≍ 未来分歧)}
$$

$$
\boxed{\;\mathcal{L}_{\text{LeWM++}}
= \mathbb{E}\lVert f(z_t,a_t) - z_{t+1} \rVert^2 \; + \; \mu\,\mathcal{L}_{\text{metric}}\;}
$$

**两项、一个超参,与 LeWM 同构**:LeWM 正则化 latent 的分布*形状*(SIGReg,任务盲),
LeWM++ 正则化 latent 的*度量*(动力学感知)。三个性质成为定理而非 loss:

1. **盆内收缩**:盆内增量稳定 ⇒ 未来分歧随时间收敛 ⇒ 对齐后的度量下
   $\lVert z^+_i - z^+_j \rVert \le \lVert z_i - z_j \rVert$,共轭动力学自动收缩(Thm A(a) 的构造化);
2. **跨盆 margin**:分岔 ⇒ 未来发散 ⇒ 距离自动拉开(Thm A(b) 的构造化);
3. **反坍缩(部分)**:非平凡未来分歧强制非平凡距离;残余的全局坍缩风险
   (φ 坍缩 ⇒ target 同步坍缩)由**架构**解决——projector 的 BatchNorm 锚定 scale,
   不新增 loss;保守变体保留 SIGReg(3 项)作消融对照。

新颖性风险自查:此形态最近邻是 bisimulation/MICo/behavioral-metric 线
(reward 驱动、合并方向为主)与 temporal-distance/quasimetric 表征线(goal 距离,
非动作条件开环分歧)。差异主张:reward-free 动作条件未来分歧、双边(合并与分离
同时约束)、且由收缩理论给出"为什么恰好是这个度量"(它是使耗散 hybrid 系统的
共轭动力学收缩、且保住分岔 margin 的度量)。此点为审稿主攻位。

**可证伪预测(实现前锁定)**:
P-a 收缩项使 K=1 成本的训练达到 K=5 级 growth@8(<0.7)与 planning;
P-b 纯收缩(μ=0)在 D=8 会显现跨盆错误(margin 项修复它);
P-c LeWM++ 在 D=8 的角度 R² 高于 K=5(margin 保护边界相关方向)且 planning ≥ K=5;
P-d 在接触稀疏任务(navigation 类)三项目标增益都小——解释 What-Drives 的
    rollout-length 经验规律(manipulation 需要长 rollout ⟺ 接触分岔需要收缩)。

## 6. 不重复性对账(对 2026-07-04 调研)

| 已有工作 | 覆盖 | 我们的差异 |
| --- | --- | --- |
| Fast-LeWM | prefix 监督降 rollout error + 加速 | 我们不降 error 本身,改变 error 的**几何**;理论解释它为何 work |
| What Drives (TMLR) | recipe 扫描,含 rollout 长度 | 我们给出其 rollout-length 规律的机制与公式(P-d) |
| Invariant JEPA-WM (bisim) | 合并 nuisance(上界) | 我们是 margin 下界 + 收缩,防的是相反的失败 |
| DBC / bisimulation | reward 驱动合并 | reward-free 未来分歧 margin |
| 收缩分析 / neural contraction metrics | 真实状态动力学的稳定性认证 | 首次用于**学出的 latent 共轭动力学**,并证明多步 JEPA 隐式在做这件事 |
| MBRL 误差界 (Janner 等) | $\delta_K$ 上界本身 | 上界非贡献;贡献是"哪个训练目标对上界中的 $G$ 有梯度"与容量分配定理 |
| DreamerV3 / TD-MPC2 | stop-grad / value 锚 | 不同问题;引用定位 |

## 7. 验证路线(PushT = test1,OGBench = 主战场)

1. **盆-穿越判别实验(新叙事的 Gate 0,先跑)**:baseline rollout 误差是否
   系统性跨盆(预测终态落错结果盆)而 multistep 误差沿盆内——Thm A 三难的
   直接证据;若否,理论回炉。
2. piecewise-linear 数值系统:Thm A/B/C 的可控验证(解析盆已知,floor 可对真值)。
3. PushT test1:{LeWM, K=5, 仅收缩, 仅 margin, LeWM++} × D∈{8,32,192},
   五量 + 证书全程;检验 P-a..P-c。
4. OGBench-Cube(Fast-LeWM 在此 74→82,未饱和):LeWM++ 主效果战场;
   navigation 类任务作 P-d 对照。
