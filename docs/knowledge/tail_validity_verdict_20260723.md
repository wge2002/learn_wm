# Tail-validity 碰撞判决书(Claude 分析,2026-07-23)

> 输入:`research_tail_validity_collision_20260722.md` + evidence 底稿。
> 按需求书 §4:逐条判决 → 单轮三审稿人(不循环)→ GO/NO-GO。

## 1. 逐条判决(占了什么/没占什么)

| 线 | 一句话判决 |
|---|---|
| AdaJEPA/PROWL/AWM | 占"执行后适应/对抗采数/外部 min-max";没占同一 CEM 轮内未执行候选的 cost 校准(①②全空) |
| Temporal Straightening v2 | 占"离线 cost 几何 + planner 对比";没占任何在线校准,①②③④全空 |
| Navigable EBM/PaD/GeoWorld | 占"静态 optimizer-aware 塑形 + JEPA+CEM(⑤)";没占在线 query-path 适应,不反传 elite/refit |
| Performative 线 | **占④的全部数学词汇**(distribution map/stable point/contraction);没占 planner↔内部 WM cost 这个回路,①②③⑤全空 |
| Adaptive conformal 线(WMMD/FCP-MPC) | 占"内生 shift 下的安全集校准 + 条件性固定点";没占 cost **排序**有效性,对象是外部 agent/安全场非候选 rank |
| 定向扫描(ACID/WM-VAE/IMWM) | ACID 占"逐轮配权 + 全序列 verifier + LeWM/PushT(最近单点)";IMWM 占 proposal-volume 理论 + rank 诊断,**且 B.7 明写 closed-loop replan shift 未审计**;两者都没占真值反馈、tail 误排测量、跨 refit 累积 |

**矩阵级事实(本判决的地基):15 个工作里,② 一列全空,① 无一个 ✓。**
"误排沿 proposal flow 累积"这个我们已实验证实的对象,文献里没有任何刻画;
"learned-vs-true low-tail 排序"没有任何直接测量。我们的 oracle bank 恰好是
世界上现成的测量仪器。

## 2. 单轮三审稿人(NeurIPS 怀疑者,不迭代)

- **R1(MBRL 方法审稿人)6.5/10**:ACID 和 IMWM 离得很近,组合感风险真实
  存在;但"真值/执行反馈进入 CEM 环内校准 tail 排序"确实无人做,若给出与
  ACID 的 head-to-head 且赢在其冻结性上,方法成立。
- **R2(理论审稿人)7/10**:②的空白 + performative 词汇可整体移植到
  planner↔内部 cost 回路(把 CEM refit 写成 distribution map、tail-validity
  写成固定点性质)是干净的形式化贡献,与你们 rate 律风格连续;担心收敛分析
  在非线性 CEM 上只能做到条件性(WMMD 同款限定)。
- **R3(恶意怀疑者)5.5/10**:你们自己证明了终端 selector 失败——那么方法
  必须靠在线反馈,而部署时唯一诚实的反馈是**已执行前缀的真实残差**(延迟、
  稀疏,FCP-MPC 同款约束);若单集内该信号不足以修正 tail 排序,方法死;
  且目前证据全在 PushT。

均分 **6.3** → 过"值得做"线,未过"稳"线;生死在 R3 指出的反馈通道可行性。

## 3. 判决:GO(窄门),形状如下

**名字候选**:Performative Tail-Validity of Learned Planning Costs。

1. **科学层(无人占,仪器现成)**:定义并测量 ①+②——learned-vs-true
   low-tail 误排 + 它沿 CEM elite/refit 流的累积律(oracle bank + population
   recorder 直接可测;oracle ceiling 已是存在性证明);
2. **形式层**:performative 词汇移植到 planner↔内部 cost 回路(④ 的新应用;
   IMWM B.7 明示此区未审计 = 文献自己指路);
3. **方法层(窄门)**:环内 tail-rank 再校准,反馈只用**可部署信号**
   (已执行前缀残差,FCP-MPC 式延迟更新但作用于 rank 而非安全场;仿真
   实验可另配小 oracle 预算做上限);明确不做终端 selector(我们已证死)、
   不做冻结 verifier(ACID/WM-VAE 已占 + 我们的 selector 失败同理);
4. **⑤上执行**:LeWM+CEM PushT 开发,OGBench-Cube 第二域(= IMWM 同设定,
   正面对比 + 补它的未审计区)。

**开工前的一票否决实验(零/低 GPU,数据已在手)**:用现有 oracle bank +
replay 日志测——已执行前缀的真实残差能否预测(并在下一次 replan 修正)
low-tail 误排?相关 → 方法线开;不相关 → 方法线关,转问题定义论文
(①②测量 + oracle ceiling 做 headline,仍可发表)。

## 4. 对齐备注

- 与 `lewm_planning_status_20260721.md` §0 的岔路对齐:本判决把"方法空间
  是否存在"细化为"反馈通道是否可行",且给出了判决实验;
- ACID(2607.02403)必须进所有后续实验的 baseline 列;IMWM 的
  proposal-volume bound 和 G.5 诊断应被引用而非重做。

## 5. 否决实验回填(2026-07-23)

已在 A100 上按预锁定协议完成 `60/60` 个有效 next-replan state（前缀终止或
截断 `0`），判决为 **`CLOSE`**：

- honest prefix residual 对 baseline 下一次误排没有可泛化预测力：residual norm
  Spearman `0.102 [-0.175, 0.358]`；固定 5-fold OOF ridge
  `R²=-0.479`，MAE `0.121`，差于常数基线 `0.107`；
- 在同一个 baseline population 上，即使逐 state 用真值事后从
  `{alpha=0,.5,1}` 挑最好者，top30 recall 增益也只有
  `0.015 [0.009, 0.022]`，其 CI 上界仍低于预锁定 `.05` 否决线；
- 把修正放进全部 30 轮 CEM 后，held-out alpha 的 recursive recall 变化为
  `-0.006 [-0.018, 0.003]`；returned-mean true cost 变化为
  `+7.94 [-1.67, 25.33]`（越低越好），success 变化为 `0.000`。

因此关闭“单 action-block 真实残差作为持久加性偏置、直接校准下一次 CEM
tail-rank”的方法线，按原判决转向 **①②测量 + oracle ceiling 的问题定义线**。
这是否决预注册的乐观反馈族，不是“所有非线性或历史条件反馈都不可能”的证明。

复现实验协议见 `tail_validity_feedback_gate_protocol_20260723.md`；compact 结果、
输入及实现 SHA-256 见
`tail_validity_feedback_gate_20260723/REPORT.md` 与同目录 `report.json`。原始
四个 shard 留在 A100：
`/225010117/logs/tail_validity_feedback_gate_a100_20260723/`。
