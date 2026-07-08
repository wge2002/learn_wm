# 理论最小模型笔记:horizon 如何在边缘约束的 slack 里选择局部增益场(2026-07-08)

> 目标:为 Part IV/V 的实验事实(rate(K) 单调、误差/信号不对称、
> 一步代价、各向同性 slack)提供最小的可证明模型。
> 状态:Prop 1 已证;标量模型给出结构性恒等式;非线性 toy 复现签名。

## 1. 设定与记号

线性高斯受控世界:

```text
x_{t+1} = A x_t + B a_t + eps_t,   eps ~ N(0, sig^2 I),  a_t ~ N(0, I) iid
Sigma = A Sigma A^T + Q,           Q = B B^T + sig^2 I  (平稳协方差,Q > 0)
```

encoder `z = phi(x)`,predictor `f`,K 步开环 loss(lewm_multistep 结构):

```text
L_K = (1/K) sum_{k=1..K} E || zhat^(k) - phi(x_{t+k}) ||^2,
zhat^(0) = phi(x_t),  zhat^(k) = f(zhat^(k-1), a_{t+k-1})
```

SIGReg 建模为边缘约束:硬(cov(z)=I)或软(+ lam*||cov(z)-I||^2)。

## 2. Prop 1(负结果,线性情形):精确白化 ⇒ 共轭算子非扩张,K 无事可做

**命题.** 线性 encoder `z = Wx`,W 精确白化(`W Sigma W^T = I`,W 可逆)。
则共轭单步算子 `F = W A W^{-1}` 满足 `sigma_1(F) <= 1`,与 K 无关。

**证明.** W 白化 ⇔ W = R Sigma^{-1/2},R 正交。
`F F^T = W A Sigma A^T W^T = W (Sigma - Q) W^T = I - W Q W^T ⪯ I`(Q≻0)。∎

**推论 1a(gauge 刚性).** 精确白化族内剩余自由度仅为正交 R,而
`sigma_1(R M R^T) = sigma_1(M)`:瞬态范数在族内不变。
线性可实现情形下,(i) 病理不存在,(ii) 也没有可供 K 选择的自由度。

**推论 1b(现象定位).** PushT 实测(SIGReg 下单步 sigma_1 = 1.3-3.3,
乘积增长 16x,K 步驯化)不可能是线性/全局现象。
**边缘白化约束的是全局二阶矩;局部 Jacobian 场 J(z,a) 不受它约束。
LeWM 目标中唯一对局部 Jacobian 乘积有梯度的项是 K 步开环 loss。**
理论对象:局部增益场与全局边缘约束之间的 slack,由 horizon 分配。

数值验证:`outputs/theory/minmodel_v1.py` —— 所有 K 给出 sigma_1(F)≈1.0,
K 无效应,与命题一致。

## 3. 标量两 regime 模型:telescoping 恒等式

最小非线性设定(PushT contact/free 卡通,1-D 可解析):

```text
x_{t+1} = a(x_t) x_t + eps,   a(x) = a_con > 1 (|x| < theta, "contact")
                               a(x) = a_free < 1 (|x| >= theta, "free")
```

encoder 为递增可微 `h: R -> R`,`z = h(x)`。predictor 取共轭最优。
沿真轨迹 x_0 -> x_1 -> ... -> x_K 的共轭局部增益 `g_t = h'(x_{t+1}) a(x_t) / h'(x_t)`
(一阶),其 K 步乘积**telescope**:

```text
prod_{t<K} g_t = [ h'(x_K) / h'(x_0) ] * prod_{t<K} a(x_t)        (*)
```

**读法:**
1. 共轭乘积增益 = 真实乘积增益 × 边界项 `h'(end)/h'(start)`。
   encoder 驯化 K 步放大的唯一一阶手段:让 h' 沿膨胀路径下降
   (压缩坐标,h 在扩张方向上凹)——正是 III.4/V.2 实测的
   "非正交整形/低 effective rank"的 1-D 影子。
2. **一步 loss 只看相邻比值** `h'(x_1)/h'(x_0)`;K 步 loss 看跨整段膨胀的
   累计比值,并按 (*) 中真实增益乘积加权——同一梯度方向,
   压力随 K 单调增强 ⇒ rate(K) 单调性的结构来源。
3. 硬边缘约束在 1-D 恰好唯一确定 Gaussianizing 递增映射(无自由度,
   与推论 1a 同型);**软约束(lam 有限)打开一族 h,K 与 lam 的比值
   决定花多少边缘 slack 买多少增益驯化** ⇒ 各向同性 slack 随 K 上升(V.2 实测)。
4. 误差/信号不对称(V.1)的候选解释:动作在第 j 步注入,其到 horizon 的
   增益带边界项 `h'(x_K)/h'(x_j)`;误差从 x_0 起步,带全程 `h'(x_K)/h'(x_0)`。
   压缩坐标对"晚注入"的信号衰减少、对"早起步"的误差衰减多;
   且拟合数据要求区分不同动作的未来(动作通道有监督信号),
   误差方向无监督约束、是纯 slack。待写成定量命题。

## 4. 非线性 toy 复现(`outputs/theory/minmodel_v2.py`)

两 regime 世界(A_con: sigma_1≈3 剪切,A_free: 0.9 收缩,半空间切换)
+ MLP encoder/predictor + 软 SIGReg + K 步开环 loss。
3 世界种子均值(minmodel_v2.json,2026-07-08):

| K | 每步局部增益 | 乘积 sigma_1(P_8) | 即时动作增益 | 回声 ag_8 | 一步 MSE |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 3.18 | 2.76 | 1.51 | 0.60 | 0.110 |
| 2 | 2.69 | 2.28 | 1.46 | 0.50 | 0.111 |
| 3 | 2.45 | 2.17 | 1.45 | 0.47 | 0.111 |
| 5 | 2.08 | 1.42 | 1.57 | 0.42 | 0.136 |
| 10 | 1.84 | 0.93 | 1.57 | 0.31 | 0.158 |

复现 PushT 四签名,且稳健:
1. **每步局部增益驯化在全部 3 个世界逐 K 严格单调**(最强签名);
2. 一步 MSE 与 PushT drift_1 **同形**:K∈{1,2,3} 平坦,K>=5 上升
   (PushT: 1.05/0.95/1.05 → 1.60/1.65);
3. 即时动作增益保持甚至微升;回声驯化(1.9x)弱于误差乘积驯化(3.0x)
   ——不对称方向一致,幅度弱于 PushT(37x vs 1.7x;toy 更温和);
4. 未复现的部分(诚实):toy 的 K=1 rate≈0.98 不超临界(PushT 1.29)
   ——toy 的 MLP 在稳定世界拟合良好;PushT 超临界来自像素 encoder 的
   拟合难度,签名可复现、绝对刻度不可比。effrank 在 d=8 下无信号
   (slack 证据仍以 PushT V.2 实测为准)。

## 5. 待办(理论侧)

- [ ] 标量模型:软约束下显式解 h*(lam, K),证 rate(K) 单调 + slack 单调;
- [ ] 不对称命题:监督通道(动作可辨别性)vs 无监督 slack(误差方向)
      的梯度结构分离;
- [ ] 甜点下降沿:drift_H(K) ≈ eps_1(K) * sum_j rate(K)^j,
      eps_1 随 K 升、rate 随 K 降 ⇒ 固定评测 horizon 有内部最优 K
      (等补种子波判决 K=10 掉分真伪后决定是否作为主命题);
- [ ] v2 toy 3 种子结果并入;扫 lam(预测:lam→∞ 时 K 效应塌缩,
      对应 Prop 1 的硬约束极限)——这是 toy 里最干净的可证伪预测。
