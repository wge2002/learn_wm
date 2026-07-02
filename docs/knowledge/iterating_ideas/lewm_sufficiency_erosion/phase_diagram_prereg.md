# 预注册:容量×步长相图(2026-07-02 启动,跑之前锁定)

> 背景:机制链已闭合(见 [dstar_decomposition.md](../../dstar_decomposition.md))——
> 多步共训教 encoder 收缩坐标系,信息量不变,planning 跟随复合稳定性。
> 但这一切发生在 D=192 对 ~7 维状态的极度宽裕容量下,理论(选择压力 toy theorem)
> 的前提"容量竞争"从未被激活。本实验把竞争逼出来。

## 目的(只有两个)

1. **H2 理论适用域**:容量紧张时,"要求可滚"(K=5)是否开始以任务信息为代价
   (planning 掉到 K=1 之下、角度探针在 K=5 侧掉得更狠)。
2. **H1 证书预测力**:6 格模型构成种群,检验"扰动增长@8 / refit 复合缺口"对
   planning 的排序预测力是否优于 self-drift、单步 MSE、探针 R²。
   (H1 不依赖 H2 的结局,种群本身即可检验。)

## 设计

- 网格:D_z ∈ {192(复用 iter2_baseline/iter2_multistep), 32, 8} × K ∈ {1, 5}。
- 新训练 4 个:`pd_d{32,8}_k{1,5}`,30 epochs,2 卡/个,与 iter2 协议一致
  (K=1 用 lewm.yaml,K=5 用 lewm_multistep.yaml)。
- 瓶颈实现:只掐表征接口——`projector.output_dim=D, predictor.input_dim/output_dim=D,
  action_encoder.emb_dim=D, pred_proj=D`;predictor 主干 hidden 保持 192
  (D=8 时 10.8M vs 11.7M),避免"f 容量缩水"的 confound。
- 每格五量:① planning(matched 3-frame history,50ep × 3 eval seeds);
  ② 自身 drift@8(归一化);③ 冻结-refit D*(单步/多步)→ 下界+复合缺口;
  ④ 扰动增长曲线(ε=0.01);⑤ 角度/位置线性探针 R²。

## 预注册预测(押注写死,准备被打脸)

- P1:D=192 复现 K=5 ≥ K=1(已知 88 vs 83)。
- P2:存在某个 D,使 K=5 的 planning < K=1 且角度 R² 差值扩大(理论押注:D=8 出现,
  D=32 不确定)。
- P3:跨全部格子,扰动增长@8 与 planning 的 Spearman 相关强于 drift@8、单步 MSE、
  角度 R² 与 planning 的相关。
- P4:所有 K=5 格子的扰动增长 < 同 D 的 K=1 格子(收缩性是 K=5 的稳定效应,与 D 无关)。

## 结局表(预注册解读)

| 结局 | 解读 |
| --- | --- |
| P2 成立(相边界存在) | 侵蚀现象以受控形式复活;论文 = 收缩坐标系的代价相图 + 提前预警的证书 |
| D=8 仍 K=5 ≥ K=1 | "收缩坐标系是免费午餐,即使容量竞争激烈";选择压力定理实用域被有界化 |
| D=8 两格全崩(任务下限 >8 维) | 边界在 (8,32) 内,加密 D=16(半天),不判负 |
| P3 证伪(证书与 planning 脱钩) | H1 死;收缩性叙事缺变量,回炉——与 Gate 0 同性质的自我检验 |

执行:`outputs/pd/run_pd.sh`(训练 → eval 目录 → planning 评测 → latent 导出;
refit/扰动/探针次日补)。
