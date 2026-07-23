# 2025H2–2026 世界模型认可趋势调研（2026-07-23）

> 口径：只用论文、官方会场/OpenReview 与作者项目页；“认可”优先主会接收，
> effect size 只抄一手来源。截至本日 [NeurIPS 2026 尚未出决定（9 月 24 日通知）](https://neurips.cc/Conferences/2026/CallForPapers)，
> 故不能列 “NeurIPS 2026 accepted”。主轴用 **(a)**，认可理由 **(b)**，范式/分布
> **(c)**，效果与卖点 **(d)**。

## 1. Latent / JEPA

- **[Temporal Straightening](https://openreview.net/forum?id=Ik1mKtUYlZ)**（[2603.12231](https://arxiv.org/abs/2603.12231)，ICML'26）：**(a)新目标+理论**，一项曲率正则让 latent 轨迹局部变直；**(b)** 简洁且把曲率连到测地距离/规划条件数；**(c)** 确定性 Euclidean JEPA latent；**(d)** 多任务显著提高 GD/CEM 成功率，统一 effect 未确认。
- **[Causal-JEPA](https://arxiv.org/abs/2602.11389)**（2602.11389，ICML'26）：**(a)新目标**，object-level latent masking 制造结构化部分可观测；**(b)** 简单干预式目标兼有形式分析；**(c)** object-slot 条件嵌入，不显式生成像素分布；**(d)** 反事实推理约 **+20pp**，仅用 patch-WM **1%** latent 特征即获可比控制。
- **[GRWM](https://openaccess.thecvf.com/content/CVPR2026/html/Xia_Cloning_Deterministic_Worlds_The_Critical_Role_of_Latent_Geometry_in_CVPR_2026_paper.html)**（[2510.26782](https://arxiv.org/abs/2510.26782)，CVPR'26）：**(a)新目标**，给 autoencoder 加时间邻近对比几何；**(b)** plug-and-play，并把长程失败定位到表示而非更大 dynamics；**(c)** 拓扑对齐 latent，生成 backbone 不限；**(d)** 长程 fidelity/stability 显著提升，统一数值未确认。
- **[GeoWorld](https://openaccess.thecvf.com/content/CVPR2026/html/Zhang_GeoWorld_Geometric_World_Models_CVPR_2026_paper.html)**（[2602.23058](https://arxiv.org/abs/2602.23058)，CVPR'26）：**(a)新模型**，Hyperbolic JEPA + geometric RL；**(b)** 几何选择直接服务层级/长程规划；**(c)** 双曲 latent 上的 energy-based prediction；**(d)** 对 V-JEPA 2，3/4-step SR 约 **+3/+2pp**。

## 2. Diffusion / Flow

- **[Navigation World Models](https://www.amirbar.net/nwm/)**（[2412.03572](https://arxiv.org/abs/2412.03572)，CVPR'25 Best Paper HM，窗口前沿基线）：**(a)新能力**，视频 WM 直接模拟、排序受约束导航轨迹；**(b)** 1B CDiT、代码/权重齐；**(c)** Gaussian-base conditional video diffusion；**(d)** 首次展示动态加约束/陌生场景想象，汇总 SR 未确认。
- **[Epona](https://openaccess.thecvf.com/content/ICCV2025/html/Zhang_Epona_Autoregressive_Diffusion_World_Model_for_Autonomous_Driving_ICCV_2025_paper.html)**（[2506.24113](https://arxiv.org/abs/2506.24113)，ICCV'25）：**(a)新模型**，局部时空分布逐步 AR，并用 chain-of-forward 抗累积误差；**(b)** 同时解决可变时长与实时规划；**(c)** 每步 conditional diffusion 串成 AR；**(d)** FVD 改善 **7.4%**、可预测分钟级并胜强 NAVSIM planner。
- **[DriveLaW](https://openaccess.thecvf.com/content/CVPR2026/html/Xia_DriveLaW_Unifying_Planning_and_Video_Generation_in_a_Latent_Driving_CVPR_2026_paper.html)**（[2512.23421](https://arxiv.org/abs/2512.23421)，CVPR'26）：**(a)新模型**，把 video latent 直接注入 diffusion planner；**(b)** 生成与规划不再“同框但解耦”；**(c)** video latent + Gaussian-base action diffusion；**(d)** FID/FVD 优于前 SOTA **33.3%/1.8%**，NAVSIM 新纪录。
- **[Motus](https://openaccess.thecvf.com/content/CVPR2026/html/Bi_Motus_A_Unified_Latent_Action_World_Model_CVPR_2026_paper.html)**（[2512.13030](https://arxiv.org/abs/2512.13030)，CVPR'26）：**(a)新模型**，MoT 联合理解/视频/动作专家；**(b)** 一套 UniDiffuser scheduler 覆盖五种生成模式并吃异构数据；**(c)** 多模态噪声路径 + optical-flow latent action；**(d)** 仿真较 X-VLA/π0.5 **+15%/+45%**，实机 **+11–48%**。
- **[Mask World Model](https://openreview.net/forum?id=CWerqtOXif)**（[2604.19683](https://arxiv.org/abs/2604.19683)，ICML'26）：**(a)新目标**，diffuse future semantic masks 而非 RGB；**(b)** 明确把 photometric realism 换成 control-relevant bottleneck；**(c)** 条件 mask diffusion + diffusion policy；**(d)** LIBERO/RLBench、实机与纹理扰动均胜 RGB-WM，精确汇总未确认。

**分支判读：**主会认可集中在 diffusion；本轮未找到 2026 主会已接收、且真正用于
decision/control 的 **flow-matching dynamics WM**。已接收 flow 工作更多是 action
policy，不能冒充 WM 证据。

## 3. AR-token

- **[RIG](https://openreview.net/forum?id=LQv9LU2Ufg)**（[2503.24388](https://arxiv.org/abs/2503.24388)，ICLR'26）：**(a)新能力**，端到端 AR 联合 reasoning→action→next-image 并可想象后自纠；**(b)** 把多模型 agent 压成单一生成接口；**(c)** 多模态 next-token 联合分布；**(d)** 样本效率 **>17×**，并支持 test-time scaling。
- **[DrivingGPT](https://openaccess.thecvf.com/content/ICCV2025/html/Chen_DrivingGPT_Unifying_Driving_World_Modeling_and_Planning_with_Multi-modal_Autoregressive_ICCV_2025_paper.html)**（[2412.18607](https://arxiv.org/abs/2412.18607)，ICCV'25）：**(a)新模型**，图像/动作离散 token 交错成 driving language；**(b)** 标准 next-token 同时做仿真与规划；**(c)** categorical AR joint；**(d)** nuPlan/NAVSIM 胜强基线，统一 effect 未确认。
- **[Chain of World](https://openaccess.thecvf.com/content/CVPR2026/html/Yang_Chain_of_World_World_Model_Thinking_in_Latent_Motion_CVPR_2026_paper.html)**（[2603.03195](https://arxiv.org/abs/2603.03195)，CVPR'26）：**(a)新目标**，structure/motion 解耦后只推理连续 motion chain；**(b)** 避开背景重建并与动作对齐；**(c)** continuous motion latent + sparse keyframe/action AR decoder；**(d)** 仿真胜 WM/VLA 基线，精确汇总未确认。
- **[PhyWM](https://openaccess.thecvf.com/content/CVPR2026/html/Venkatesh_Physical_Object_Understanding_with_a_Physically_Controllable_World_Model_CVPR_2026_paper.html)**（[2606.00439](https://arxiv.org/abs/2606.00439)，CVPR'26 Highlight）：**(a)新能力**，任意视觉变量条件概率查询；**(b)** 单一概率 WM 涌现对象/关节/物理关系；**(c)** probabilistic AR sequence model；**(d)** 首次由多未来运动相关性做到对象发现、3D 操作与 Visual Jenga，单一 effect 不适用。

## 4. 其他（SSM / scaling / policy optimization）

- **[Newt](https://openreview.net/forum?id=MPabX9LEds)**（[2511.19584](https://arxiv.org/abs/2511.19584)，ICLR'26）：**(a)新能力+基准**，一个语言条件 WM 在线学 200 个跨域/embodiment 任务；**(b)** foundation pretrain→light RL 配方、环境/演示/代码/200+ checkpoints 全开；**(c)** task-conditioned latent WM；**(d)** 多任务数据效率与 unseen adaptation 胜强基线，统一数值未确认。
- **[ScaleZero](https://openreview.net/forum?id=iU026Hr90y)**（[2509.07945](https://arxiv.org/abs/2509.07945)，ICLR'26）：**(a)新模型+理论**，MoE 降 task gradient conflict，DPS 按进展添 LoRA；**(b)** 一模型覆盖 Atari/DMC/Jericho；**(c)** UniZero-style task-conditioned latent + expert mixture；**(d)** 匹配单任务 agent，仅用 **71.5%** environment interactions。
- **[WestWorld](https://openreview.net/forum?id=ncRRCG4BfP)**（[2603.14392](https://arxiv.org/abs/2603.14392)，ICML'26 Spotlight）：**(a)新模型**，Sys-MoE + morphology structural embedding；**(b)** 物理先验让跨机器人扩展可解释；**(c)** system-conditioned trajectory latent mixture；**(d)** 89 环境预训，zero/few-shot、控制和 Unitree Go1 均改善，统一 effect 未确认。
- **[WMPO](https://openreview.net/forum?id=qE2FyvRvuF)**（[2511.09515](https://arxiv.org/abs/2511.09515)，ICLR'26）：**(a)新能力**，在 pixel-video WM 内对 VLA 做 on-policy GRPO；**(b)** WM 从 predictor 变成可反复采样的训练环境；**(c)** clip-level AR pixel-video generative distribution；**(d)** 无真实交互即提升样本效率并涌现 self-correction，精确汇总未确认。
- **[Drama](https://openreview.net/forum?id=7XIkRgYjK3)**（[2410.08893](https://arxiv.org/abs/2410.08893)，ICLR'25，SSM 最近主会锚点）：**(a)新模型**，Mamba-2 替换 WM sequence core；**(b)** 长记忆线性复杂度且 laptop 可训；**(c)** discrete VAE latent + deterministic SSM state；**(d)** 7M 参数 Atari100k mean 105，近参 DreamerV3 为 37。2026 主会新 SSM-WM 强信号未确认。

## 5. (e) 可复制的成功配方

1. **把 prediction 与 action/policy 放进同一个生成图，而非预测后外挂控制。**
   RIG 用同一 AR 分布联结 reasoning/action/image（>17×）；DriveLaW 共享 video
   latent 给 diffusion planner（双任务 SOTA）。Motus、DrivingGPT 是同构佐证。
2. **只建模 decision-relevant quotient，再给一个可解释的简单归纳偏置。**
   C-JEPA 的 object masking 带来约 +20pp；MWM 直接预测 mask 而非 RGB。
   Temporal Straightening/GRWM/GeoWorld 进一步说明：“一项几何选择 + 可说明为何利于
   planning + 实效”比泛化的低 loss 更容易被认可。
3. **认可来自新增可用闭环和可扩展资产，不只来自预测指标。**
   WMPO 把 WM 变成 on-policy 训练环境；Newt 把在线 WM 扩到 200 tasks 并释放
   benchmark/checkpoints；ScaleZero/WestWorld 用 task/system routing 扩容量。共同点是
   “第一次能做 X + 可复现入口”，而非仅在原 benchmark 小幅降误差。

**总判读：**2026 已接收信号不是“更准的一步 dynamics”，而是 **生成范式承载
action/learning 闭环**，或 **一个极简、任务相关的 latent 结构同时给机制解释和可见
能力增益**。这两类是后续嫁接现有 rate/不对称分配/耦合定律时应对齐的外壳；本调研
不重提已判死的 CI-GWM/BA-GWM。
