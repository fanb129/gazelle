# GazeSpot 转投决策与重构方案（2026-07-17）

> 适用对象：ACM MM 2026 被拒稿件 *Gaze in the Crowd: Frustum-Aware Feature Aggregation for Robust Gaze Target Estimation*。  
> 结论先行：不建议原故事换模板后直接转投，也不建议赶 AAAI 2027。首选是把它重构为 **WACV 2027 Evaluations & Datasets Track 的评测/失效分析论文**；若 2026-08-18 前未通过本文列出的硬门槛，则转为 **Image and Vision Computing** 的完整期刊稿。

## 1. 一句话判断

原稿中可保留的科学事实是：**在当前 Gaze-LLE-style 冻结 VFM 系统里，多层 DINOv3 特征比只用最后一层带来小幅而稳定的定位收益。** 不能继续作为核心结论的是：

- GGSF 是 physical frustum、gaze cone 或物理隔离；
- SASA 会随人物或样本动态选择层；
- DINOv3 普遍存在严重 over-smoothing；
- 人数更多等同于遮挡/跨人物干扰，且 GGSF 已解决这种干扰；
- 现有总增益可归因于 GGSF 和 SASA，而不是多层输入、分辨率、参数量或训练协议。

因此，旧稿的正确处理不是“润色”，而是**换研究问题、换证据结构、删掉被数据否定的机制主张**。

## 2. 审稿人式严重性审计

### CRITICAL：未解决前不可投稿

1. **定性图 provenance 不合格。** `scripts/eval_gazefollow.py` 中的可视化分支会主动模糊 baseline，并把 GT 混入 ours heatmap、GGSF mask 和 SASA weights。现有代码显示数值指标在这些操作前计算，因此不能据此断言数值表被污染；但所有相关定性图必须作废并从 raw inference 重生。
2. **“physical frustum”与实现不符。** GGSF 只用 `(x-cx, y-cy, w, h)` 经小型 MLP 生成 sigmoid gate，不含头部外观、眼睛、头姿、gaze direction、深度或相机几何。
3. **核心模块归因被内部对照否定。** matched spatial-prior 对照中，GGSF 的 AUC 与 CoordConv 几乎相同，L2 不优于 CoordConv，AP 还低于 no-prior；identity intervention 也未显示稳定损失。按 idea fatal-flaw 规则，GGSF 作为主贡献应判定为 **Reject and Pivot**。
4. **SASA 的“动态/person-adaptive”解释不成立。** fixed-mean 和 shuffle-weight 干预几乎复现 learned SASA；person-conditioned residual router 的 paired CI 也均跨 0。
5. **crowd 构念无效。** people count 是 density proxy，不是 occlusion、distractor interference 或错误绑定的直接测量；表中 AP 在人数 `=3`、`=4` 时反而低于 baseline，不能声称优势“稳定扩大”。
6. **主比较混入分辨率和容量。** baseline 默认 `448×448`、last-layer；GazeSpot 默认 `512×512`、four-layer 并增加参数与计算。没有 matched matrix 就无法归因。

### MAJOR：需要补齐的论文级证据

- 同分辨率、同 backbone、同 decoder、同训练预算的 layer/fusion/prior 因子实验；
- 至少 3 seeds 或按视频 sequence 的 cluster bootstrap CI；
- GOO-Real/ChildPlay 等外部域，或明确、可复现的 shift/stress-test；
- bbox jitter、目标距离、head size、target separation、in/out 的分层结果；
- total/trainable parameters、MACs/FLOPs、latency、FPS，且输入分辨率一致；
- 主稿与 appendix 中 scheduler/学习率等训练描述一致；
- crowd 子集 sample-ID manifest、构造脚本、分桶统计和混杂审计。

### MINOR：重写时一次清理

- 删除 “perfectly validates”“indisputable superiority”“overwhelming reversal”等结论强于证据的措辞；
- 删除 Woodstock/会议占位符，统一方程标点、表头和参数定义；
- 将 `GGSF` 改称 **bbox-conditioned relative-coordinate gate**，将 `SASA` 改称 **global layer reweighting**；
- 将 Crowd Subset 改称 **VAT density-stratified evaluation**。

## 3. 当前结果能安全支持什么

### 3.1 标准结果

| 对比 | GazeFollow | VideoAttentionTarget | 可支持的结论 |
|---|---|---|---|
| Gaze-LLE-L | `0.958 AUC / 0.099 Avg L2 / 0.041 Min L2` | `0.937 AUC / 0.103 L2 / 0.903 AP` | 强基线 |
| GazeSpot-L | `0.961 / 0.093 / 0.038` | `0.944 / 0.092 / 0.912` | 完整配置在当前协议下更好 |
| 差值 | `+0.003 / -0.006 / -0.003` | `+0.007 / -0.011 / +0.009` | 是有价值的系统结果，但未完成模块归因 |

GazeSpot-B 在 VAT 的 AP 为 `0.887`，低于 Gaze-LLE-B 的 `0.897`；所以“所有指标一致胜出”不成立。

### 3.2 渐进消融真正说明的事情

在 DINOv3-B 上，last-only → raw multilevel 已把 GF L2 从 `0.107` 降到 `0.103`、VAT L2 从 `0.106` 降到 `0.103`。之后 SASA 和 GGSF 的增益只有千分位量级。最稳妥结论是：

> 多层表示是主要增益来源；当前 global reweighting 和 relative-coordinate gate 最多是实现选择，尚未被证明是独立机制。

### 3.3 已有补充证据的正确用法

- GOO-Real：完整模型在 2,146 个样本上从 `0.8583/0.1997` 提升到 `0.8833/0.1761`，可报告为当前协议下的 OOD 结果，但仍不能归因给 GGSF。
- bbox jitter：20% 扰动时完整模型仍优于原 baseline，可说明系统没有立即崩溃；不能泛化为“geometry gate 对 detector error 鲁棒”。
- 复杂度：现测 baseline `53.61G MACs, 18.37ms, 54.45 FPS`，GazeSpot `70.54G MACs, 22.82ms, 43.82 FPS`，但两者分辨率不同；只能用作初步成本信号，必须重跑 matched resolution。

## 4. 推荐转投路线

### 路线 A：WACV 2027 E&D Track（首选，条件式）

WACV 2027 的 E&D Track 明确接收 benchmark failure analysis、evaluation stress test、evaluation protocol、audit、critical/negative result。Round 2 注册截止为 **2026-08-21 AoE**，论文截止为 **2026-08-28 AoE**，补充材料截止为 **2026-08-30 AoE**，且 Round 2 没有 rebuttal。官方信息见 WACV 2027 Call for Papers 与 Dates 页面。

建议题目：

> **When Deeper Is Not Better: Stress-Testing Frozen Vision Foundation Features for Gaze Target Estimation**

论文类型从“两个新模块的 SOTA 方法”改成：

1. 对 frozen DINOv2/DINOv3 的层、分辨率和融合方式做 matched evaluation；
2. 建立 density、head scale、target distance/separation、bbox noise、domain shift 的 stress-test；
3. 报告并解释负结果：global scalar routing 与 bbox-relative gate 并未稳定优于简单 controls；
4. 给出一个透明、可复现、低复杂度的 static multilevel baseline。

这条路线保留了绝大部分已有工程和结果，同时把审稿人最不接受的“机制夸大”变成评测贡献。

### 路线 B：Image and Vision Computing（最现实的保底）

IVC 是 rolling journal，scope 明确包含 image interpretation、human behavior understanding、data fusion，以及对方法的 quantitative comparison/performance evaluation。若不能在 8 月前完成 matched matrix、统计和 provenance 清理，就不要用不完整证据赶 WACV；转 IVC 后补足跨数据集、统计和完整误差分析。

期刊版仍应采用上述评测问题，而不是把 GGSF/SASA 包装回主要创新。

### 条件性备选

- **IEEE TMM**：只有在补强视频、多媒体/人类行为价值并完成完整实验后考虑。IEEE SPS 要求对曾被会议或期刊拒绝的稿件披露相关审稿意见及逐项回应；初投稿页数要求也更紧。
- **Pattern Recognition**：只有在形成真正的新问题/方法（例如本文配套 idea 报告中的 COTB）并有跨模型、跨数据集证据后考虑。单纯“DINOv3 + 多层融合”属于 routine application，scope 页面本身明确不鼓励。
- **AAAI 2027**：abstract/paper deadline 分别为 2026-07-21/07-28，无法在保证研究完整性的前提下完成，不建议赶。

## 5. WACV 版的新故事

### 5.1 六步逻辑链

1. Frozen VFM 简化了 GTE，但“最后一层就是最佳表示”通常被默认而非检验。
2. GTE 同时依赖精确位置、人物条件和场景语义，不同层/分辨率/融合设计可能改变结论。
3. 现有 leaderboard 平均指标无法说明收益来自 backbone、分辨率、多层信息，还是额外模块，也掩盖 density/scale/target ambiguity 下的失败。
4. 本文构建严格 matched 的 factorized evaluation，并给出可复现 stress-test protocol。
5. 结果显示多层特征确有定位价值，但复杂的动态性/几何叙事不被对照支持；一个简单 static multilevel baseline 已解释主要收益。
6. 贡献是更可靠的 evaluation practice、可复现 failure taxonomy 和对 frozen-VFM GTE 设计的经验结论。

### 5.2 贡献表述模板

可以写：

1. We conduct a controlled study that disentangles feature depth, image resolution, fusion, and spatial prompting in frozen-VFM gaze target estimation.
2. We introduce a reproducible stress-test protocol covering density, head scale, target geometry, box noise, and domain shift.
3. We show that multilevel features provide useful localization cues, while learned scalar routing and bbox-relative gating do not consistently outperform simple matched controls.

不要写：first multiscale DINOv3 GTE、physical frustum、person-adaptive routing、solves crowd interference、severe DINOv3 over-smoothing。

## 6. 最小决定性实验矩阵

| 研究问题 | 必跑配置 | 主指标 | 通过标准 |
|---|---|---|---|
| 层级信息是否有用 | last-only；L2+L11；four-layer static concat | GF/VAT AUC、L2、AP | matched setting 下多层定位改善且 CI 不跨 0 |
| 学习层权重是否必要 | equal；fixed learned mean；sample scalar；person-conditioned | 同上 + 权重方差 | dynamic 版本需胜过 fixed mean；否则报告负结果 |
| spatial prior 是否必要 | none；head map only；CoordConv；Gaussian；GGSF | 同上 | GGSF 若不胜，删除为方法贡献 |
| 收益是否来自分辨率 | 448/512 的 last-only 与 multilevel 2×2 | 同上 + FLOPs | 分离 resolution effect |
| density 是否真是特殊失效 | people count + head size + target distance/separation + video controls | cluster-bootstrap / robust regression | 只描述实际显著因素，不用人数替代机制 |
| 是否能外推 | GOO-Real；可得时 ChildPlay | 标准指标 | 至少一个外部域，不选择性只报优势项 |
| 是否可部署 | matched input 的 params/MACs/latency/FPS；N=1/3/5/10 | wall-clock + memory | 诚实报告成本，不再称“几乎无开销” |

统计要求：关键比较至少 3 seeds；VAT 以 sequence/video 为 cluster 重采样，不能把连续帧当独立样本。所有表同时报告 point estimate、95% CI、样本/cluster 数。

## 7. 定性图与复现硬规范

重新生成的每一张图必须：

- 直接来自 raw model output，不调用任何 `degrade_*`、`enhance_*_with_gt`、`correct_*_with_gt`；
- 固定 sample ID，报告选择规则，避免只挑成功案例；
- 保存 dataset/annotation hash、checkpoint SHA256、git commit、完整命令和随机种子；
- baseline/ours 使用完全相同的归一化、插值、colormap 和 alpha；
- GT 只能显示在独立 panel，不能进入模型输出或后处理；
- 同时提供随机样本页和 failure cases，而不仅是 curated positives。

建议把旧的 beautification 脚本隔离到 `non_scientific_demo/` 或彻底删除，并在科学绘图入口加入断言：可视化函数不得接收 GT 坐标用于修改预测。

## 8. 从现在到 WACV Round 2 的日程

| 日期 | 工作 | Go/No-Go 产物 |
|---|---|---|
| 07-17—07-23 | 清理定性路径；冻结 protocol；重跑 matched resolution smoke | provenance manifest；2×2 smoke 表 |
| 07-24—08-02 | 完成 layer/fusion/prior 主矩阵 | 关键配置 1 seed；删除无效模块的正式决定 |
| 08-03—08-10 | 3 seeds、GOO、bbox noise、density/confound、效率 | 主表 + CI + stress-test 图 |
| 08-11—08-17 | 按 E&D 逻辑重写全文和 appendix | 完整 8-page draft；artifact checklist |
| 08-18 | 硬 Go/No-Go | 全部 CRITICAL 清零才继续 |
| 08-21 | WACV Round 2 enrollment | 注册完成 |
| 08-22—08-27 | 内审、匿名材料、复现检查 | 无占位符、无不一致、无失真图 |
| 08-28 / 08-30 | paper / supplement deadline | 提交 |

### 08-18 的硬 No-Go 条件

以下任一成立就转 IVC，不赶 WACV：

- 仍有任何 GT-assisted qualitative；
- matched matrix 不能把多层收益与分辨率/容量分开；
- 没有 sequence-aware CI 或至少 3 seeds；
- 主结论仍依赖 GGSF/SASA 的旧机制措辞；
- stress-test 只是人数分桶，没有混杂控制；
- 论文仍靠“绝对 SOTA”而不是可靠评测贡献成立。

## 9. 与新 idea 的边界

新 idea 应单独立项。最推荐的是 **Counterfactual Observer–Target Binding（COTB）**：同一场景只切换 queried person，检验预测是否切换到对应 gaze target。DINOv3 多层特征在其中只作为技术底座，而不是 novelty。

不要把 COTB 临时加成旧稿的第四个模块。只有当 COTB 的一天 pilot 通过、配对数据有效、强基线存在稳定 query-binding failure，才启动一篇独立的 new-problem + method 论文。旧稿回答“怎样可靠评测 frozen VFM feature design”；新稿回答“模型是否真的跟随被查询的人”。

## 10. 最终推荐

**主决策：WACV 2027 E&D，条件式推进；8 月 18 日未通过硬门槛则转 IVC。**

这不是降低目标，而是把已有证据最强的部分放到合适的审稿标准下。若坚持按算法稿投稿，则当前旧方法不够：GGSF 已被数据否定，SASA dynamicity 不成立，多层 DINOv3 又已被近邻工作覆盖。算法路线应由独立的 COTB pilot 决定，而不是继续给旧框架加模块。

## 官方信息

1. WACV 2027, *Call for Papers*，含 E&D scope 与 Round 2 deadlines。
2. WACV 2027, *Dates and Deadlines*。
3. Elsevier, *Image and Vision Computing — Aims & Scope*。
4. IEEE Signal Processing Society, *Information for Authors*。
5. Elsevier, *Pattern Recognition — Aims & Scope*。

