# 新 Idea 决策：Counterfactual Observer–Target Binding（COTB）

> 日期：2026-07-17  
> 定位：独立于 GazeSpot 转投稿的新问题 + 最小方法。Gazelle、冻结 DINOv3 和多层特征是技术底座，不作为 novelty。

## 0. 最终建议

推荐优先验证：

> **Does Gaze Following Follow the Person? Counterfactual Observer–Target Binding for Gaze Target Estimation**

一句话故事：**在完全相同的场景中只更换被查询人物，模型的 gaze prediction 是否真的切换到该人物的目标？** 现有 GTE 主要逐人用 AUC/L2/AP 评价，普通定位误差会掩盖 query–output correspondence 是否正确；COTB 把同帧 query switching 变成自然的受控对照，并直接监督正确 observer–target 配对优于交换配对。

这条路线比继续改 SASA/GGSF 更强，因为 novelty 位于**问题、协议和对应关系监督**，而不在“又一种多层融合”。

## 1. 原 idea 的早期致命缺陷审计

### 1.1 “GGSF + SASA + 多层 DINOv3”版本

- Paper type：Novel Method。
- Verdict：**Reject and Pivot**。

| # | Fatal flaw | Severity | 证据 |
|---|---|---|---|
| 1 | 核心机制已被本项目自己的 matched controls 否定 | CRITICAL | GGSF 不优于简单 CoordConv/no-prior；identity intervention 无稳定损失；SASA 的 fixed-mean/shuffle 干预几乎复现 learned 输出 |

按 data-refuted core mechanism 规则，一旦核心方法被简单控制匹配或击败，就应停止评分和修辞性挽救。因此 GGSF 不能再换名成为新论文贡献，SASA dynamicity 也不能继续迭代。

### 1.2 “DINOv3 + multilevel features”版本

多层特征在你的系统里确实有用，但只能作为 base design：近邻工作已经覆盖 frozen VFM、DINOv3 hierarchy、multi-scale/object-aware/geometric reasoning。安全表述是“采用多层特征作为实现基础”，不能声称“首次多层 DINOv3 gaze”。

## 2. COTB 的问题定义

同一帧图像为 `I`，其中人物查询为 `q_i, q_j`，对应 gaze targets 为 `y_i, y_j`。当 `y_i` 与 `y_j` 属于不同 target clusters 时，切换 `q_i → q_j` 保持以下因素不变：

- scene semantics；
- salient objects 与背景；
- 相机、光照和画面构图；
- 大部分潜在 confound。

唯一有意改变的是“问谁”。真正 person-conditioned 的模型应满足：

`P(y_i | I, q_i) + P(y_j | I, q_j) > P(y_j | I, q_i) + P(y_i | I, q_j)`。

这里的 `P(y | I,q)` 不用单个 argmax，而用 target 邻域内的 heatmap probability mass，减小标注点噪声。

### 2.1 为什么普通 L2 不够

若人物 A 的预测峰落到人物 B 的标注目标附近，L2 只把它记成一个较大的定位误差；它不会区分：

- 普通 heatmap 偏移；
- scene saliency shortcut；
- query cue 不足；
- observer–target correspondence 错配。

COTB 不是替代 L2，而是增加一个正交诊断维度，并要求在 matched-L2 或控制 L2 后仍能解释模型差异。

### 2.2 shared target 必须特殊处理

多人可能共同看同一对象。先按预注册半径把 gaze points 聚成 target clusters：

- 不同 cluster 才构成 hard swapped negatives；
- 同一 cluster 是 many-to-one positives，不能互相惩罚；
- annotation 不完整、out-of-frame 或不可靠 track 使用 ignore mask。

## 3. 方法：多层 DINOv3 是底座，binding 才是贡献

### 3.1 基础网络

1. **Shared scene encoder**：冻结 DINOv3，整幅图只编码一次。
2. **Static multilevel representation**：使用已经证实有效的 shallow/deep 或 four-layer feature；先用静态 concat/projection，避免把动态 fusion 混成第二条创新。
3. **Observer query**：从 head ROI appearance、bbox geometry 和 person token 形成 `q_i`。
4. **Query-conditioned decoder**：每个 observer query 从同一 scene feature 中解码 heatmap 与 in/out score。

第一版不要加入 GGSF、SASA、YOLO/SAM、FoV cone、MoE 或新的复杂 router。

### 3.2 Cluster-aware permutation contrastive loss

令 `S_ij` 为 query `q_i` 在 target cluster `c_j` 邻域内的 log probability mass。对 distinct-target pair `(i,j)`：

`L_bind = max(0, m - (S_ii + S_jj) + (S_ij + S_ji))`。

总损失：

`L = L_heatmap + λ_io L_io + λ_bind L_bind`。

多人时可扩展为 score matrix 的 row-wise ranking 或小规模 permutation objective；但 pilot 先做 pairwise，避免算法复杂度掩盖问题有效性。

### 3.3 推理成本

`L_bind` 只用于训练；若 backbone 本来就对 scene 编码一次，推理不增加新的视觉编码器。多人查询可批量 decode，因此潜在价值不只是准确率，也包括对 query-conditioned scene reuse 的规范化。

## 4. 新指标与统计协议

| 指标 | 定义 | 作用 |
|---|---|---|
| Pairwise Binding Accuracy | `Sii+Sjj > Sij+Sji` 的 pair 比例 | 最直接的正确配对率 |
| Swap Margin | `(Sii+Sjj)-(Sij+Sji)` | 连续强度，便于 paired CI |
| Own-Target Rank | 每行 score matrix 中 own cluster 的排名 | 支持 3+ distinct targets |
| Query-Switch Response | 切换 query 后预测分布/峰值是否朝对应 target 移动 | 检验输出对 query 的响应性 |
| X-TCR | 峰值最近 cluster 是否非 own cluster | 仅作 post-hoc error taxonomy，不能冒充因果 binding |

标准 AUC/L2/AP 必须保留。VAT 的统计以 video sequence 为 cluster bootstrap；主结果按 target separation、head size、target distance、people count 分层，并用 regression/matched analysis 说明 binding 不是 L2 的重命名。

## 5. 已有内部信号及其证据边界

当前 VAT raw-evaluation records 包含：

- 13,127 帧、31,978 person records；
- 5,545 帧存在至少两个不同 target clusters；
- 14,740 个 in-frame queries 可定义 X-TCR/association margin；
- 当前 GazeSpot 在这些 queries 上的 post-hoc X-TCR 为约 `19.61%`，5+ 人分桶约 `37.62%`。

这些数字只说明预测峰有时更接近同帧另一个 target cluster。现有报告明确写明 `counterfactual_query_swap: not implemented`，因此它们**不能**证明模型内部发生了人物绑定错误，也不能直接成为论文主结论。它们只足以支持“一天内值得做真正 query-switch pilot”。

## 6. 与最接近工作的碰撞边界

| 近邻工作 | 已覆盖内容 | COTB 必须守住的差异 |
|---|---|---|
| HGTTR (CVPR 2022) | end-to-end 同时检测 heads/targets，set matching | 不声称首次 multi-person/end-to-end；强调同场景 query-switch protocol 与 swapped-pair objective |
| Object-aware Gaze Target Detection (ICCV 2023) | 自动检测 heads/objects 并建立 head–object association | 不声称首次 association；区别是 pixel/cluster-level counterfactual pairing，而非 object detection pipeline |
| Sharingan (CVPR 2024) | controlled person token，多人共享 scene | 它是最重要强基线；COTB 要证明现有 controlled token 在 binding metric 上仍有缺口 |
| GazeHTA (2024 preprint) | head–target connection map | connection map 不等于同帧交换配对监督；但必须对照其 association 输出 |
| MTGS (NeurIPS 2024) | multi-person temporal GTE + social gaze | 不抢 temporal/social novelty；可作为视频外部验证 |
| Gaze-LLE (CVPR 2025) | frozen DINOv2、person positional prompt、scene once | 作为简洁强基线，验证问题不是旧 GazeSpot 特有 |
| GazeAnywhere (CVPR 2026) | 用文本/视觉 concept 指定 subject | promptable subject identification 已被覆盖；COTB 聚焦 query-output binding 的训练与评价 |
| Multi-scale Object-Aware (2026) | DINOv3 hierarchy、object representation、gaze geometry | COTB 不以多层/对象/FoV 为 novelty，且可把其方法视为潜在 base learner |

检索词包括 `counterfactual gaze following`、`query swap gaze target`、`observer-target binding gaze`、`permutation gaze target association`。截至 2026-07-17，**没有检索到直接同时采用同帧 query-switch、shared-target-aware swapped assignment objective 和独立 binding metric 的工作**。这只是检索结果，不是“已证明无人做过”；投稿前必须再做一次系统检索和引用追踪。

## 7. 一天 Go/No-Go pilot

### 7.1 先审计数据，不训练

1. 从 VAT 合并同帧全部可靠 person annotations；按 video/sequence 划分，禁止同帧泄漏。
2. 固定 target-cluster radius，并做 `0.05/0.075/0.10` 的敏感性分析，但主阈值提前注册。
3. 输出 video/frame/query/pair 数、shared/distinct cluster 分布、target separation 和 annotation completeness。
4. 实现真正的 counterfactual runner：同一 scene feature 下替换 observer bbox/head crop，保存每个 query 的 raw heatmap score matrix。

### 7.2 强基线

- Gaze-LLE；
- 当前 GazeSpot（只作诊断，不作机制基线）；
- Sharingan；
- 一个 scene-only 或 query-shuffled negative control。

### 7.3 冻结的 Go 条件

全部满足才进入训练：

1. 至少 `1,000` 个 `target separation ≥ 0.30` 的 distinct-target pairs，来自至少 `30` 个 video clusters；
2. 至少两个强模型的 far-target swap error ≥ `10%`，95% cluster-bootstrap CI 不包含低于 `5%` 的近饱和值；
3. 控制 L2、head size、target distance 和 people count 后，binding 指标仍提供非冗余信息；
4. query-shuffle control 显著更差，证明指标真的响应 observer query；
5. annotation audit 未发现大规模 missing-person/duplicate-track 使 negative pair 无效。

任一失败立即 No-Go。不要先训练 loss 再通过挑分桶寻找故事。

## 8. 两天最小方法实验

只比较：

1. 原始 heatmap/in-out loss；
2. naive individual hard negative；
3. pairwise permutation loss；
4. pairwise permutation + shared-target clustering。

`λ_bind` 只尝试两个预注册值，例如 `0.1` 和 `0.3`。先 1–2 epochs screen，再完整训练 winner。

晋级条件：

- binding error 相对下降至少 20%，sequence-cluster 95% CI 不跨 0；
- far-target 与 5+ 人子集方向一致；
- 标准 L2 退化不超过 `0.001`，AP 退化不超过 `0.003`；
- 改善不只是 heatmap 变尖：matched-L2 下 swap margin 仍提高。

## 9. Idea Evaluator 正式评分

### 9.1 First impression

- Paper type：**New Setting + Innovative Technique**。
- One-sentence story：主流 GTE 是否真的对“被查询的人”敏感，可用同帧 query switch 直接检验并监督。

### 9.2 Fatal flaws audit

| # | Flaw | Severity | Defense |
|---|---|---|---|
| 1 | 与 multi-person/head-target association 文献的边界可能被审稿人认为只是换 loss | MAJOR | 以新 evaluation contract、shared-target protocol、matched-L2 analysis 为第一贡献；Sharingan/GazeHTA/Object-aware 必须成为强基线 |
| 2 | 同帧 annotations 可能不完整，导致错误 hard negatives | MAJOR | 先做 completeness/track audit；只使用可靠 distinct in-frame clusters；缺失 query ignore；跨阈值敏感性分析 |

没有发现当前阶段的 CRITICAL flaw；但只有 pilot 通过后才能确认问题成立。

### 9.3 Lifecycle and capability match

| Aspect | User input / known context | Assessment |
|---|---|---|
| Idea category | New setting + innovative technique | 适合 2–3 个月完整周期，不适合 11 天赶 AAAI |
| Compute | 已有 Gazelle/DINOv3、VAT records、单卡 RTX 3090 工作流 | 绿色；第一版冻结 backbone、仅改 sampler/loss |
| Data | VAT 可形成同帧自然 pairs；外部数据待核 | 黄色；最大风险是 annotation completeness |
| Weekly effective hours | 用户未说明 | 未核验；若每周少于约 20 有效小时，优先只做 pilot 和旧稿转投，不并行完整训练 |
| Fit | 已有模型、脚本和失败分析基础 | 黄色偏绿；研究问题匹配，但数据协议需要严审 |

### 9.4 Five-dimension radar

| Dimension | Score | Evidence | Lift suggestion |
|---|---:|---|---|
| Higher | 7/10 | mechanism-based：直接惩罚 swapped pairing；已有 X-TCR 仅给出 failure signal，尚无训练收益 | 先证明 far-target binding error，再以跨两个 base models 的 gain 隔离贡献 |
| Faster | 8/10 | mechanism-based：scene 编码一次，binding loss 训练期使用，原则上零新增推理编码 | 报 N=1/3/5/10 people latency，和逐人重跑 encoder 对照 |
| Stronger | 8/10 | mechanism-based：同场景 hard negatives 抑制 saliency/query shortcut，直接针对 query fidelity | 增加 target separation、domain shift、bbox noise 分层和 matched-L2 分析 |
| Cheaper | 8/10 | 复用已有同帧标注，不依赖 YOLO/SAM/LVLM 或新人工标签 | 证明 annotation audit 可自动完成；若需大规模重标则下调 |
| Broader | 7/10 | mechanism-based：query-output binding 可迁移到 referring segmentation、human-object association 等 query-conditioned dense tasks | 论文先聚焦 gaze，discussion 再抽象，不做跨任务 scope creep |

最高上限是 **Faster、Stronger、Cheaper**。所有高分目前都以机制为主，不能当作实验事实。

### 9.5 Paradigm-shift probe

| Probe | Yes/No | 理由 |
|---|---|---|
| First Principles | Yes | 挑战“逐 query L2 足以证明模型在跟随指定人物”的默认假设 |
| Elephant in the Room | Yes | 同一场景多人看不同目标时，普通指标不区分对应关系错误 |
| Technology Cycle | No | DINOv3 让实现更强，但问题本身不依赖新模型周期 |
| Hamming's Rule | Yes | 若能可靠测量 binding，所有多人物/提示式 GTE 模型都需要报告它 |

Disruptive potential：**possible**。三个 probe 为 Yes，但是否真正重要取决于强模型在 far-target pairs 上是否仍有稳定缺口。

### 9.6 Feasibility

| Risk | Level | Mitigation |
|---|---|---|
| Compute | Low | 冻结 DINOv3，缓存 scene features，先 VAT pairwise loss |
| Data | Medium–High | 在训练前完成 completeness/track/cluster audit；无效立即 No-Go |
| Engineering | Medium | 先实现 raw score-matrix runner，再改 sampler/loss；禁止同时改 backbone/router |
| Timeline | Medium | pilot 1 天、最小方法 2 天、论文级 3-seed/跨模型约 4–8 周；不与 WACV 转投关键周过度并行 |

### 9.7 Verdict

**Accept with Revisions — worth pursuing, pending the validation experiment.**

Top three actions：

1. 实现真正的同帧 counterfactual query-switch runner，而不是重命名现有 X-TCR。
2. 在 Gaze-LLE、Sharingan、GazeSpot 上完成 far-target pair 的 data-validity + cluster-bootstrap pilot。
3. 只有 pilot 过门槛，才实现 shared-target-aware permutation loss；多层 DINOv3 保持为固定技术底座。

## 10. 备选方向及排序

| 方向 | 核心新意 | 风险 | 当前排序 |
|---|---|---|---:|
| COTB | 同帧 query-switch + binding objective/metric | 与多人 association 边界、annotation completeness | 1 |
| Counterfactual Target-Visibility Consistency | preserving crop 要求等变；target-removal 要求 in→out 且不 hallucinate；decoy edit 要求稳定 | crop 改变上下文/尺度，需自然 VAT transition 外证 | 2 |
| Visibility-Conditioned Amodal GTE | train-only multi-view teacher；测试单视图区分 visible surface、amodal intended target 与 visibility state | 完全遮挡时意图可能不可辨；需双层 GT/任务重定义 | 3 |
| Semantic Decoy Intervention | 保持 target 不变，干预非目标显著物体 | intervention 未必 label-preserving，HCLoRA 已覆盖 semantic shortcut | 4 |
| Fixation-state Temporal GTE | fixation/change-point，兼顾 jitter 与 switch delay | 与 MTGS 重叠、当前 transition 样本不足 | No-Go for now |
| Structured Selective GTE | outside token + spatial prediction set + abstention/risk–coverage；不以层间分歧为信号 | 纯 conformal 应用创新不足；ID coverage 不保证 OOD | 备选评测轴 |
| Hierarchy Disagreement | 用层间分歧估计失败风险 | 本项目现有 pilot 已是 No-Go | Stop |

## 参考近邻

Recasens et al. (2015); Chong et al. (2018, 2020); Tu et al. (2022); Tonini et al. (2023); Tafasca et al. (2024); Lin et al. (2024, GazeHTA); Gupta et al. (2024, MTGS); Ryan et al. (2025, Gaze-LLE); Cao et al. (2026, GazeAnywhere); Mi et al. (2026, Multi-scale Object-Aware Gaze Estimation).
