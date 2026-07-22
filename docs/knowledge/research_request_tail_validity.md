# 调研需求书:planner-equilibrium / adaptive tail validity 碰撞审计(2026-07-22)

> 用法:把本文交给任意深度调研工具(或人工检索)。产出交回给 Claude 分析,
> 由 Claude 出"占了什么/没占什么"判决 + 一轮三审稿人评分 + GO/NO-GO。
> 判决标准和我们的主张见 §0,调研只需回答 §2 的问题,不需要下结论。

## 0. 我们的候选主张(被审对象,不要外传原文)

Learned world-model cost 在 CEM 候选群的 **low-cost tail** 上系统性误排 elite;
误差沿 CEM 的 proposal-update flow 累积(终端 selector/verifier patch 修不了,
我们已实验证伪);剩余方法空间 = **全序列自适应 tail-fidelity**:让 cost 的
有效性沿 planner 自己的查询路径自适应,把 planner 与 model 当作一个均衡系统。
背景:JEPA 型 latent WM(LeWM)+ CEM,已有 rate(K) 律、误差/动作不对称分配、
oracle-reranking ceiling(真动力学重排同批候选 + 递归更新 proposal 大幅改善)。

## 1. 需要拉原文的五条线(2024–2026 优先)

1. **AWM / adversarial or adaptive world models for planning**(任何"对抗
   planner 查询分布训练 WM"或"随 planner 部署自适应"的工作);
2. **Temporal Straightening for latent planning**(arXiv 2603.12231)——重点看
   它 camera-ready 是否新增了 planner-facing cost 校准内容;
3. **Navigable / planner-aware EBM**(能量模型按"可被优化器导航"训练的线);
4. **Performative prediction**(Perdomo et al. 起,含 performative RL/control
   的延伸)——决策改变数据分布、均衡存在性/收敛;
5. **Adaptive conformal / risk calibration under decision-induced shift**
   (adaptive conformal inference、决策引起的分布移位下的校准)。

另加三个定向扫描(标题/摘要级即可,2025–2026):
- "CEM elite misranking" / "elite selection error model-based planning"
- "planner-induced distribution shift world model"
- "test-time cost calibration MPC" / "cost model calibration during planning"

## 2. 每条线要回答的三个问题(照抄给调研工具)

(a) 它**具体做了什么**(对象、方法、实验设定,不要只抄摘要);
(b) 它是否覆盖以下任一要素:①low-cost tail 的误排作为对象;②误差沿
    proposal-update flow 累积的刻画;③沿 planner 查询路径的自适应有效性;
    ④planner-model 均衡形式化;⑤在 latent WM + 采样式 planner(CEM/MPPI)
    上的实现;
(c) 它明确**没做**什么(留给我们的可实现空间,如有)。

## 3. 产出格式

每条线 ≤300 字 + 关键论文的 arXiv 号;直接引用原文关键句(标 §);
不确定的写"未确认",不要脑补。总篇幅 ≤2500 字。

## 4. 回传后 Claude 的分析承诺

逐条"占了什么/没占什么"一句话判决 → 单轮三审稿人(NeurIPS 怀疑者人设)
0–10 评分 → GO(给出方法空间的精确形状)或 NO-GO(转问题定义论文,
以 oracle ceiling 为 headline)。判决对齐 lewm_planning_status_20260721.md §0。
