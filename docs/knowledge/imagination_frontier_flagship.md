# 旗舰提案:The Imagination Frontier of Deterministic World Models(2026-07-14)

> 来源:Claude(Fable)× Codex(gpt-5.6-sol, effort max)两轮对抗性头脑风暴,
> thread `019f5f8c-e6d5-7832-8b47-e64de4b9a768`。
> 背景:PI 判定 γ-剂量方向太局部,要求"用已有定律做出特别有意思的新工作"。

## 0. 一句话主张(HYPOTHESIS,预注册前置)

确定性 latent 世界模型存在一条**可测量的想象边界(imagination frontier)**
——被短 goal 评测掩盖、被 horizon/容量压力测试暴露;一个
**共适应感知的、逐候选的增益证书**可以利用这条边界,在不用 ensemble、
不重训的前提下改进长程规划。

## 1. 讨论过程摘要(两轮)

**Round 1(Codex 提案 + 互评):**
- Claude 的 B(rate 跨架构序参量)被杀作旗舰:度量跨架构不可比,
  且"即使相关性成立也没有新能力"。保留为后续验证轴。
- Claude 的 A(视距)被合并升级:强的一半不是 H* 测量,而是
  **"因为知道自己视距而改变行为的模型"**;H* 降为校准层。
- Codex 新提:频谱稳定调节器(测试时裁剪误差模态)、放大寻求数据采集、
  表征防火墙(吸收 Claude 的 C)、失败边界评测协议。

**Round 2(Claude 三个 pushback → Codex 回应):**
- **Pushback-1 成立**:调节器撞 sg-K5 证据(rate 1.17 优于 K=1 的 1.29,
  planning 却 83→55)——任何对冻结 gauge 的传播编辑都进共适应陷阱。
  Codex 让步并给出关键修正:**永不修改 latent 传播;证书只进 CEM 的
  信任通道**(超出认证前缀的 cost 截断/降权、缩短执行前缀、重规划、弃权),
  外加**模型级资格门**(一步保真 + 动作对齐达标才允许使用增益证书)
  ——sg-K5 会被资格门正确拒绝。我们的负结果变成了设计特性。
- **Pushback-2(MBPO/ensemble 新颖性)**:差异化钉在三点——
  (a) 单个确定性 JEPA 上的路径条件放大量,无 ensemble/似然/辅助模型;
  (b) 同一增益有文档化的、种子稳定的训练期控制律(K/γ),
  训练方式与测试期可靠规划边界被同一定律连接;
  (c) 生死线 = **冻结阈值的跨 seed/K/容量/goal 距离迁移**,
  不是"trust your model"这句话本身。
- **Pushback-3(旗舰组装)**:frontier 评测协议(暴露问题)+
  认证 horizon CEM(利用定律)= 单一旗舰;数据采集与防火墙降为后续论文。

## 2. 收敛计划(Codex 原文,verbatim)

- Flagship: **The Imagination Frontier of Deterministic World Models**.
- Problem reframe: scalar short-goal success hides horizon-dependent candidate misranking.
- Capability: non-invasive, candidate-specific certified-horizon CEM.
- Certificate: one-step fidelity/action-alignment eligibility gate × accumulated local gain.
- Never clip or rewrite propagated latent states.
- Week 1 uses existing checkpoints only; gamma arms are held-out additions.
- Primary scientific endpoint: locked-threshold prediction of candidate rank inversions.
- Primary capability endpoint: far-goal success at matched planning compute.
- Governor becomes a killed negative-control idea, not a paper.
- Data acquisition and representation firewall remain follow-up papers only if the flagship survives.

## 3. Week-1 实验清单(零训练,全部用现有 checkpoint)

1. **候选日志 + oracle 标签**:给 CEM 加逐候选日志(latent 路径、每步局部
   增益、累计 log-gain、预测 cost/排名、资格门统计);从克隆的 PushT 状态
   (评测协议已有 `_set_state`)执行分层候选子集(top/近平局/增益分位),
   测真实 return 和成对排名反转。
2. **暴露边界**:扫 `goal_offset_steps`×horizon ≈ {0,10,25,50}×{5,10,25,50},
   先 K∈{1,3,10}×D∈{8,192},在边界信息量大处补 K∈{2,5}、D=32、种子。
   报告 success、候选误排、drift、证书存活 vs 所需 horizon。
3. **锁定校准测试**:在部分 seed/model 的近 goal 数据上拟合证书
   (资格门 → 累计增益阈值),**冻结后**在held-out seeds、K、D、远 goal、
   四个解耦对照、γ-剂量臂上盲测。
4. **能力对比**(匹配 CEM 采样数与环境步数):固定 horizon vs 全局
   rate 截断 vs 候选局部增益门控 vs 完整资格门证书 vs 随机匹配截断对照。

**Week-1 kill 判据**:锁定证书对 held-out 排名反转的预测若不优于
horizon/全局 rate/一步 drift 三个基线,或自适应规划在匹配算力下不改进
远 goal 成功率——旗舰机制死。

## 4. 审稿人攻击与预答(Codex 原文要点)

- **"这就是 uncertainty-aware MPC/MBPO 换了个分数"** → 不是认知不确定性
  主张:对象是确定性的候选路径放大量,绑着可控训练律,用前瞻性排名反转
  校准评估;必须赢冻结阈值迁移 + 打过全局截断/简单 drift 基线,否则新颖性
  确实不成立(诚实承认)。
- **"sg-K5 自己反驳了 rate;frontier 是 PushT 后验工程"** → sg-K5 是
  预注册对抗对照:它正是资格门的动机,并测试证书能否拒绝假收缩;
  阈值在近 goal 锁定、在未见变体上盲测,迁移失败即 kill;
  普适性主张在第二环境前明确出界。

## 5. Claude 的独立评估(非 Codex 观点)

- 这个旗舰把我们**全部资产无一浪费**地变成零件:rate 律 → 证书与训练旋钮;
  不对称 → 资格门里的动作对齐项;四个失败对照 → 评测协议里的对抗模型群;
  γ-剂量臂 → 天然 held-out 测试集;"planning 稳健之谜"(50× 放大差只映射
  4 分)→ 论文的开场钩子(短 goal 评测掩盖边界的直接证据)。
- 最大风险不在机制而在**效应量**:如果 PushT 100 步 episode 内推不出足够远
  的 goal,边界暴露不充分。Week-1 第 2 项最先回答这个;若不够,
  需要 PushT 变体(更长 episode/更远 goal)或第二环境,提前想好。
- 与 MBPO 线的生死差 = 冻结阈值迁移,这是全文最硬的赌注,值得赌:
  因为 rate 律的种子稳定性(Δ≈0.02)是我们已经测过的最稳的东西。

## 6. 状态

- mix v2(γ 剂量)训练照跑,其结果无论如何都进旗舰的 held-out 集;
- 等 PI 审阅本提案后启动 Week-1(全部零训练)。
