# D* 分解:multistep 的收益住在哪里(2026-07-02)

> Gate 0 证伪"侵蚀现象"后的第一个机制实验。问题:同一个 K=5 多步目标,
> encoder+f 共训 → drift 0.177 + planning 88%;只给 f(sgmulti,双任务)→ 全坏。
> 那么收益到底是"f 学会了滚"还是"φ 学出了可滚的坐标系"?
> 方法 = v4 证书机器的第一次实战:冻结几何,从零 refit LeWM 真 predictor 阶段
> (Embedder+Predictor+pred_proj,~11.7M),单步 vs K=5 目标 × 3 refit seeds。
>
> 数据/脚本:`outputs/dstar/run_dstar.sh`(latent 导出 `regime_stepB_eval_data.py`
> 8000×11 帧窗口,同 drift@8 协议;refit `regime_lewm_predictor_stepB2.py --arm mono`)。

## 结果(normalized mse@8,除以各几何 latent var:base 0.968 / multi 0.872)

| 冻结几何 | refit 目标 | norm drift@8(3 seeds) | norm 单步 MSE |
| --- | --- | --- | --- |
| φ_baseline | K=5 多步 | **0.335 ± 0.008** | 0.0404 |
| φ_baseline | 单步 | 0.381 ± 0.014 | 0.0186 |
| φ_multistep | K=5 多步 | **0.158 ± 0.012** | 0.0282 |
| φ_multistep | 单步 | **0.194 ± 0.014** | 0.0192 |

参照(各自 co-trained predictor):baseline 0.315,multistep 0.177。

## 三个结论

1. **Drift 下界真实存在。** baseline 几何里,专职干净训练的 fresh predictor(K=5 目标,
   无 sgmulti 的单步/多步双任务)只能到 0.335,打不过 baseline 自己的 0.315。
   `inf_f drift(φ_base, f) ≈ 0.32-0.34`;sgmulti 的 0.358 已接近下界,它的失败 =
   下界(几何决定)+ 双任务干扰(k=1 翻倍)两者叠加。
2. **多步共训的 drift 收益 ≈100% 在几何侧。** multistep 几何里连单步训练的 fresh
   predictor 都到 0.194(比 baseline 几何的下界低 ~40%);几何项
   D̂*(φ_base)−D̂*(φ_multi) = 0.335−0.158 = 0.177,大于全部观测差距 0.138。
3. **几何变化的内容是"复合性/Markov 性",不是"单步可预测性"。**
   两个几何的单步 teacher-forced MSE 几乎相同(0.0186 vs 0.0192),
   但开环复合后差 2 倍(0.381 vs 0.194)。multistep encoder 学到的是
   误差复合更慢的坐标系 —— 即关闭了"复合缺口"(composed rollout vs 直接回归的差,
   round-2 审稿提出的 latent Markovianity certificate),而不是降低条件熵。

副产物:f 内的单步/多步目标冲突在两个几何都存在(多步 refit 使单步 MSE 恶化:
base 几何 2.2×,multi 几何 1.5×),但可组合几何显著缓解它 —— 与 sgmulti/multistep
的行为差异一致。

## 与 Gate 0 合并后的完整故事

> **多步训练不是教 predictor 怎么滚,而是教 encoder 一个可滚(更 Markov)的 latent。**
> 单步可预测性不变;复合缺口大幅关闭;控制不受损(planning 82→88%,matched history);
> 代价:历史接口脆弱化(1-frame 冷启动 22%)+ 角度线性可解码性下降(R² 0.80→0.68,
> 与 planning 解耦)。"可预测性压力吃掉控制信息"在 D=192/PushT 尺度上没有发生 ——
> 坐标重写是良性甚至有益的。

## 下一步(候选,未定)

1. **容量×步长相图**(宏观赌注):latent 维度 192→32→8 × unroll K=1/5(/10),
   问"买复合性什么时候开始付控制充分性的钱"。理论(选择压力 toy theorem)预测
   容量竞争下会出现相边界;找到 → 受控侵蚀现象+整套证书机器成一篇有相图的论文;
   找不到 → "Gaussian JEPA 预测目标在极端压缩下仍控制良性"同样是强结论。
   成本:粗网格 6 个从零训练 run,8 卡 1-2 天。
2. 复合缺口的直接测量:直接 k 步回归 head vs 复合 rollout,逐几何逐 k(便宜,半天)。
3. refit-f 的 planning 评测:φ_base 冻结 + refit multi-f 上 CEM,分离 sgmulti 的
   planning 损伤来源(f 质量 vs 共适应)。
