# AAAI 2027：Hierarchy-Aware Selective Gaze Following 研究与实施方案

## 0. 文档状态

- 工作名称：**Knowing When Semantics Mislead: Hierarchy-Aware Selective Gaze Following under Domain Shift**
- 内部简称：`SelectiveGaze`
- 目标会议：AAAI 2027
- 当前阶段：方案冻结与 Go/No-Go pilot 设计
- 当前不执行：不修改 `gazelle/` 主干、不启动训练、不把下面的计划命令误认为已经实现
- 建议代码位置：仓库根目录新建的 `AAAISelectiveGaze/`
- 相关独立调研：[`../AAAI2027_new_problem_method_landscape.md`](../AAAI2027_new_problem_method_landscape.md)

本文中的“计划命令”定义后续代码应提供的命令行接口。除特别标为“现有命令”的部分外，它们要等对应脚本实现后才能运行。

---

## 1. 一句话概括

现有 Gazelle + DINOv3 多层特征能够提高平均 gaze target 定位精度，但模型即使在困难或陌生场景中也会强行给出答案。本方案保留当前 gaze predictor，通过比较 DINOv3 不同层对同一人物凝视目标的判断，估计本次预测是否可信；当中层几何证据与深层语义证据严重冲突时，系统给出较高风险并允许拒绝预测。

通俗地说，系统由两部分组成：

- **答题者**：现有 Gazelle，回答“这个人在看哪里”。
- **质检员**：新的 risk head，回答“这次预测有多大可能出错”。

---

## 2. 研究问题与可证伪假设

### 2.1 研究问题

**RQ1：** 当前多层融合主要使用深层特征时，中间层是否仍能提供最终层没有利用的错误预警信息？

**RQ2：** DINOv3 不同层的 gaze prediction 分歧，是否比最终 heatmap 自身的置信度更能识别 localization failure？

**RQ3：** 只用 GazeFollow 源域训练或标定的风险分数，能否迁移到 VAT 和 GOO-Real，仍然把错误样本排在前面？

### 2.2 三个可被实验推翻的假设

- **H1：层级分歧假设。** 分歧大的样本比分歧小的样本更容易产生较大的定位误差。
- **H2：增量信息假设。** 在已经知道最终 heatmap entropy/peak confidence 后，层级分歧仍能提供额外的错误识别能力。
- **H3：跨域稳定假设。** 在 GazeFollow 上得到的风险规律不会在 VAT、GOO-Real 上完全反转。

任何一个假设都不能仅靠可视化成立，必须用预先定义的指标检验。

---

## 3. 已知经验如何改变方案

此前 MM 动态特征融合实验显示，模型大多数时候更依赖深层特征。这个结果不直接否定层级分歧，但带来四个设计约束：

1. **不能把动态融合权重直接当成风险。** 融合权重只表示“为了降低平均训练误差，模型更愿意使用哪一层”，不表示该层在当前样本上一定可靠。
2. **risk 分支必须在融合前读取特征。** 如果先经过深层占优的融合，其他层的意见已经被压制，之后无法再研究真实的层间差异。
3. **每一层需要同等容量的小型预测器。** 这些小型预测器称为 `probe`，作用是把某一层特征单独翻译成 gaze heatmap。
4. **浅层始终很差属于 No-Go。** 如果浅层对所有样本都乱猜，那么层间分歧只是在测量“浅层能力不足”，不是在测量最终预测风险。

当前 DINOv3 ViT-B 实现已经抽取 `[2, 5, 8, 11]` 四层，见 `gazelle/backbone.py`；第一阶段固定使用这四层，不搜索层组合，避免把 pilot 变成大规模调参。

---

## 4. 文献地图与 novelty 边界

### 4.1 Gaze target estimation 中的直接相关工作

| 工作 | 已经解决的问题 | 与本方案的实质差异 |
|---|---|---|
| [Patch-Level Gaze Distribution Prediction, WACV 2023](https://openaccess.thecvf.com/content/WACV2023/papers/Miao_Patch-Level_Gaze_Distribution_Prediction_for_Gaze_Following_WACV_2023_paper.pdf) | 用 patch-level distribution 表达 gaze target 的多峰性和标注歧义。 | 研究“可能有多个合理位置”，不预测当前输出是否会错，也没有 selective prediction。其 heatmap entropy、peak、spread 必须作为本方案 baseline。 |
| [Suppressing Uncertainty in Gaze Estimation, AAAI 2024](https://ojs.aaai.org/index.php/AAAI/article/view/28368) | 根据标签、伪标签和邻域一致性识别低质量训练样本并修正标签。 | 任务是 appearance-based gaze estimation，重点是训练数据清洗，不是第三人称 scene gaze target 的测试时风险。 |
| [Gaze-LLE, CVPR 2025](https://openaccess.thecvf.com/content/CVPR2025/html/Ryan_Gaze-LLE_Gaze_Target_Estimation_via_Large-Scale_Learned_Encoders_CVPR_2025_paper.html) | frozen DINOv2 + person prompt + 轻量 decoder，并报告 GazeFollow 到 VAT/GOO-Real 等跨数据集精度。 | 证明 foundation feature 和跨域评测的重要性，但只报告平均 AUC/L2，没有 intermediate-layer risk、calibration 或 abstention。 |
| [Multi-view Gaze Target Estimation, ICCV 2025](https://openaccess.thecvf.com/content/ICCV2025/papers/Miao_Multi-view_Gaze_Target_Estimation_ICCV_2025_paper.pdf) | 给两个相机视角分别预测 gaze vector 和 aleatoric uncertainty，选择更可靠的视角。 | 是最接近的 gaze uncertainty 工作，但比较的是两个相机视角，不是一个 ViT 的不同层；目标是改善多视角预测，不是跨域 selective prediction。 |
| [RayGazeFM, CVPRW 2026](https://openaccess.thecvf.com/content/CVPR2026W/GAZE/papers/Ambati_RayGazeFM_Geometry-Grounded_Foundation_Adapters_for_Unified_3D_Gaze_Point-of-Regard_and_CVPRW_2026_paper.pdf) | 用概率 3D gaze ray 统一 gaze direction、point-of-regard 和 scene target，并将 uncertainty 用于 ray cone。 | 已经阻止我们声称“首次研究 gaze uncertainty”；但其 scene-target 评价仍是 AUC/L2/AP，没有 risk-coverage、abstention 或 DINO 层间分歧。 |
| [Enhancing Accuracy of Uncertainty Estimation in Appearance-based Gaze Tracking, CVPR 2026](https://openaccess.thecvf.com/content/CVPR2026/html/Zheng_Enhancing_Accuracy_of_Uncertainty_Estimation_in_Appearance-based_Gaze_Tracking_with_CVPR_2026_paper.html) | 研究 domain shift 下 gaze-angle uncertainty 的 post-hoc calibration。 | 校准思想和指标值得借鉴，但任务输入是脸/眼，输出是 gaze angle，不是 scene gaze target heatmap + in/out。 |
| [Enhancing Gaze Reasoning in Vision Foundation Models for Gaze Following, 2026](https://arxiv.org/abs/2605.22607) | 指出 VFM 容易依赖显眼物体而非真实 gaze cue，并用 local LoRA 和 out-of-cone penalty 修正。 | 直接支持“深层语义可能误导”的动机；该工作尝试修正平均预测，本方案检测这种冲突、估计风险并允许拒答。不能把 semantic-saliency bias 本身声称为首次发现。 |

谨慎结论：截至本方案调研，尚未检索到在 GazeFollow/VAT/GOO-Real 上从 DINO/ViT 多个深度分别解码 gaze target，并使用跨层空间分歧预测 localization failure、评价 risk-coverage 和跨域拒答的工作。这个结论是“本次系统检索未发现”，不是绝对不存在声明。

### 4.2 其他视觉领域的直接方法先例

| 工作 | 可迁移思想 | 对本方案的限制 |
|---|---|---|
| [Shallow-Deep Networks, ICML 2019](https://proceedings.mlr.press/v97/kaya19a/kaya19a.pdf) | 给中间层增加分类器，使用内部预测冲突识别可能的误分类。 | “层间冲突可以反映错误”不是新的一般原理。 |
| [MOOD, CVPR 2021](https://openaccess.thecvf.com/content/CVPR2021/papers/Lin_MOOD_Multi-Level_Out-of-Distribution_Detection_CVPR_2021_paper.pdf) | 利用不同深度的输出做 OOD detection 和 early exit。 | generic multi-level OOD 不能作为本方案的主要 novelty。 |
| [Layer Ensembles, MICCAI 2022](https://conferences.miccai.org/2022/papers/279-Paper1397.html) | 在不同层增加 segmentation heads，用单次前向中的层间预测差异估计 uncertainty。 | 与“多个层各出一张 heatmap，再计算分歧”的简单版本非常接近；仅把它搬到 gaze 不足以支撑 AAAI。 |
| [BLOOD, ICLR 2024](https://proceedings.iclr.cc/paper_files/paper/2024/hash/0a2f65c9d2313b71005e600bd23393fe-Abstract-Conference.html) | 用 Transformer 层间变换是否异常来检测 OOD。 | 表明“层间变化异常”已有方法先例，不能包装成全新 uncertainty 原理。 |
| [Intermediate Layer Classifiers for OOD Generalization, ICLR 2025](https://proceedings.iclr.cc/paper_files/paper/2025/hash/71c3451f6cd6a4f82bb822db25cea4fd-Abstract-Conference.html) | 中间层在部分 distribution shift 下比最后层更稳定。 | 支持保留中间层证据，但说明本方案必须提供 gaze-specific 机制。 |
| [Mysteries of the Deep, NeurIPS 2025](https://papers.nips.cc/paper_files/paper/2025/hash/a03037317560b8c5f2fb4b6466d4c439-Abstract-Conference.html) | foundation model 中间层包含互补的 OOD 信号，并可选择性聚合。 | “融合中间层做 OOD”已拥挤；不能把 entropy-based layer selection 当成贡献。 |

### 4.3 可以与不可以声称的贡献

**不可以声称：**

- 首个 gaze uncertainty 方法；
- 首个使用中间层进行 OOD detection 的方法；
- 首个 layer ensemble；
- 首次发现 VFM 的 semantic-saliency bias；
- 多层融合本身是主要 novelty。

**在实验证据成立后可以声称：**

1. 提出 single-view scene gaze target 的 selective prediction setting，同时处理 localization risk 和 in/out risk。
2. 揭示深层语义占优模型中的 gaze-specific geometry-semantics conflict 与预测失败之间的关系。
3. 提出在不改动主 gaze predictor 的条件下，利用预融合层级证据估计风险的方法。
4. 建立 GazeFollow、VAT、GOO-Real 的 ID、video-domain 和 retail OOD risk-coverage protocol。

---

## 5. Problem setting

对输入图像 `I` 和人物框 `b`，主模型输出：

- `H`：gaze heatmap，即目标可能位置的二维概率图；
- `p_in`：目标在画面内的概率；
- `r_loc`：定位风险，估计最终 gaze point 会错多远；
- `r_vis`：画内/画外判断出错的概率。

系统按风险从低到高保留预测。`coverage` 表示保留多少样本，例如 80% coverage 表示拒绝风险最高的 20%；`risk@80% coverage` 表示剩余 80% 样本的平均错误。

主任务不是把 OOD 判成一个类别，而是：**无论错误来自 ID 难样本还是未知域，风险分数能否把真实错误排在前面。**

---

## 6. 方法设计

### 6.1 保留现有主预测器

主预测器继续使用：

```text
image + head bbox
    -> frozen DINOv3 ViT-B layers [2, 5, 8, 11]
    -> multi-layer fusion（首轮沿用当前最强配置，融合形式不作为论文贡献）
    -> Gazelle decoder
    -> final heatmap + in/out
```

主预测器和 risk 分支解耦。训练 risk 时默认冻结主预测器，避免 risk loss 改变原本已经验证过的准确率。

### 6.2 Fixed layer probes

每个固定层增加相同结构的小型 `LayerGazeProbe`：

```text
layer feature + person head map
    -> 1x1 channel projection
    -> one lightweight head-conditioned block
    -> 64x64 probe heatmap
```

四个 probe 的结构和容量完全相同、参数独立、DINOv3 冻结。probe 只负责把各层信息变成可比较的 gaze prediction，不参与主 heatmap 的融合。

### 6.3 两级风险信号

**Level A：通用层间分歧。**

- probe peak 坐标的两两距离；
- probe heatmap 的平均 JS divergence。JS divergence 是衡量两张概率图差异大小的数值；越大表示两层意见越不一致；
- 各 probe heatmap 的 entropy、peak、top-1/top-2 margin；
- probe error correlation 的离线分析。

**Level B：gaze-specific geometry-semantics conflict。**

- 中层 probe 产生 head-conditioned direction/region evidence；
- 深层 probe 产生 scene semantic target evidence；
- 计算深层目标是否落在中层支持区域之外，以及二者 peak displacement；
- 该冲突作为单独特征输入 risk head。

Level B 是潜在的主要方法贡献；Level A 是必须保留的简单 baseline。不能事先把“中层=几何、深层=语义”当成事实，需要用分层可视化、非显眼目标分组和 counterexample 验证。

### 6.4 Risk head

`RiskHead` 是两层 MLP，输入只包含测试时可获得的信息：

- final heatmap 的 peak、entropy、spread、margin；
- in/out probability margin；
- layer disagreement；
- geometry-semantics conflict；
- 可选的 head box 面积和位置；
- 可选的 source feature-bank distance。

输出分开设计：

- `r_loc`：连续定位误差估计，只在 in-frame 样本上训练；
- `r_vis`：in/out 分类错误概率，只在存在 in/out 标注的数据上训练。

不把二者强行合成一个训练标签，因为 out-of-frame 样本没有合法的画内 gaze 坐标。

### 6.5 自动派生的 risk 标签

不需要新增人工标注。冻结主模型后，根据现有 GT 自动生成：

```text
localization target = 主模型预测点与真实 gaze point 的归一化 L2 距离
localization failure = 1[L2 > 0.15]
visibility failure = 1[(p_in >= 0.5) 与真实 in/out 不一致]
visibility soft target = (p_in - y_in)^2  # Brier error
```

建议 `r_loc` 主要回归连续 L2，`L2 > 0.15` 只用于错误检测指标和辅助二分类。这样结果不会完全依赖一个人为阈值。

---

## 7. 数据协议

### 7.1 GazeFollow

正式实验需要把官方 train 按图像身份固定划分为：

- 85% `base_train`：训练主 predictor 和 layer probes；
- 5% `base_val`：选择主 predictor/probe checkpoint，不训练 risk head；
- 5% `risk_train`：冻结 predictor 后生成真实错误，训练 risk head；
- 5% `risk_calibration`：选择风险阈值和做 calibration；
- 官方 test：最终 ID 评价，利用多标注降低单点标签歧义。

主 predictor、probe、risk head 和 calibration 不得使用官方 test 进行参数选择。

### 7.2 VAT

设计两条协议：

1. **GF -> VAT zero-shot：** 只使用 GazeFollow 训练的 predictor/probes/risk head，VAT 只测试；这是主要跨域结果。
2. **VAT-ID：** 从 GF checkpoint 初始化并在 VAT train 微调；VAT train 再按 video identity 划分 risk/calibration，官方 test 最终评价。

GF -> VAT zero-shot 主结果只评价 `r_loc`，因为 GazeFollow 没有足够的自然 out-of-frame 负例来训练可靠的 `r_vis`。VAT-ID 才同时评价 `r_loc` 和 `r_vis`。划分必须按视频序列，而不是随机按帧，否则相邻帧会泄漏。

### 7.3 GOO-Real

- 作为 retail OOD test；
- 主结果不用于训练 probe、risk head 或选择阈值；
- 主要评价 in-frame localization risk；若当前预处理没有可信的自然 out-of-frame 负例，不报告 visibility risk。

---

## 8. Baselines 与评价指标

### 8.1 必须包含的 baselines

从弱到强依次为：

1. Random rejection：随机拒绝相同数量样本；
2. Heatmap peak：最大热力图值；
3. Heatmap entropy/spread：热力图是否分散；
4. Top-1/top-2 margin：第一和第二候选峰值差距；
5. In/out margin；
6. SASA/fusion weight：作为负对照，验证“融合权重不等于风险”；
7. MC dropout：同一模型多次推理的预测变化；
8. Deep ensemble 或多 seed probes：更昂贵但可信的 uncertainty baseline；
9. Heteroscedastic head：直接学习每个样本的误差方差，接近 Multi-view GTE 的 uncertainty 设计；
10. Layer Ensemble：只用 probe heatmap variance，不使用 gaze-specific conflict；
11. Ours：final confidence + hierarchy disagreement + geometry-semantics conflict。

### 8.2 主指标

- **Spearman correlation：** 风险排序与真实 L2 排序是否一致；1 表示完全同序，0 表示没有单调关系。
- **Failure AUROC/AUPR：** 能否把 `L2 > 0.15` 的错误排在正确样本前面；AUROC 0.5 接近随机，1.0 为完美。
- **AURC：** coverage 从低到高时风险曲线下的面积，越低越好。
- **Risk@50/70/80/90% coverage：** 拒绝高风险样本后，保留样本的平均 L2 或错误率。
- **Calibration：** 预测“20% 出错”的样本是否真的约有 20% 出错；报告 ECE/Brier/CPE 中适用的指标。
- 保留常规 AUC、L2、in/out AP，证明 risk 分支没有损害主任务。

最关键的统计检验不是“disagreement 与 error 有相关性”，而是：

> 在控制 final heatmap entropy/peak 后，加入 disagreement 是否仍显著改善 failure AUROC/AURC。

---

## 9. Go/No-Go pilot：一天内完成

### 9.1 Pilot 目标

只回答一个问题：**层级分歧是否具有超出最终 heatmap confidence 的错误预警信息？**

Pilot 不训练完整 risk head，不搜索融合方式，不修改主 predictor。允许冻结 DINOv3 后训练四个小 probe 1--3 epoch。

### 9.2 Pilot 数据

- probe 训练：GazeFollow train；
- ID 诊断：预留的 GazeFollow subset；
- 跨域诊断：VAT train 的固定均匀子集、GOO-Real val；
- 不使用 VAT/GOO test 选择层、阈值或公式。

如果首日来不及建立严格 split，可以使用现有 GF checkpoint 做纯诊断，但结果只能决定是否继续，不能进入论文最终表格。

### 9.3 Go 条件

以下条件至少同时满足前三项：

1. disagreement 最高 20% 样本的 failure rate 至少是最低 20% 的 1.5 倍；
2. `final confidence + disagreement` 相比 `final confidence`，在至少两个数据域的 failure AUROC 提高 3--5 个百分点；
3. 控制 final entropy 后，disagreement 的增益仍大于 1 个百分点且 bootstrap 置信区间不覆盖明显负增益；
4. 各 probe error correlation 不全部高于 0.95，说明不是所有层在重复完全相同的错误；
5. 高分歧案例中能观察到稳定的 gaze-specific conflict，而非浅层随机噪声。

### 9.4 No-Go 条件

- 所有层 prediction/error 几乎相同；
- 浅层 probe 在所有样本上都很差，导致“处处分歧”；
- disagreement 与 final heatmap entropy 完全冗余；
- GazeFollow 有效但 VAT、GOO-Real 排序反转；
- 增益只来自排除 out-of-frame 样本，对 in-frame localization failure 无效。

No-Go 后仍可保留“selective gaze”问题，但应放弃 hierarchy 作为方法核心，转向 final confidence + representation novelty 或其他风险建模；其 AAAI 方法新颖性会相应降低。

---

## 10. 建议的新代码目录

后续代码全部放入新目录，不继续向历史 `gazelle/`、`AAAIModules/` 或 `AAAIAlchemyModels/` 堆叠实验逻辑：

```text
AAAISelectiveGaze/
├── SELECTIVE_GAZE_RESEARCH_AND_IMPLEMENTATION_PLAN.md
├── __init__.py
├── models/
│   ├── __init__.py
│   ├── layer_probe.py              # 四个固定层的轻量 gaze probes
│   ├── risk_head.py                # localization/visibility 双风险头
│   └── selective_gazelle.py        # wrapper；导入现有 gazelle，不复制 backbone
├── data/
│   ├── __init__.py
│   ├── split_builder.py            # GF 按图像、VAT 按视频建立无泄漏 split
│   └── prediction_cache.py         # 统一 per-sample 输出 schema
├── metrics/
│   ├── __init__.py
│   ├── disagreement.py             # JS、peak distance、geometry-semantic conflict
│   ├── selective.py                # AURC、risk@coverage、failure AUROC/AUPR
│   └── calibration.py              # ECE、Brier、可选 CPE
├── scripts/
│   ├── make_splits.py
│   ├── train_base.py               # 干净 split 的现有 Gazelle wrapper
│   ├── train_layer_probes.py
│   ├── cache_predictions.py
│   ├── evaluate_disagreement.py
│   ├── train_risk_head.py
│   └── evaluate_selective.py
└── tests/
    ├── test_split_leakage.py
    ├── test_risk_targets.py
    ├── test_disagreement_metrics.py
    └── test_selective_metrics.py
```

输出统一放在：

```text
AAAIResults/selective_gaze/
├── smoke/
├── splits/
├── probes/
├── prediction_cache/
├── pilot/
├── risk_heads/
└── final/
```

---

## 11. 实施步骤

### Step 0：定义数据与输出 schema

- GF 按 image path/group 切分，VAT 按 sequence id 切分；
- 为每个 person 保存唯一 `sample_id`；
- prediction cache 至少包含 final/probe heatmaps、in/out、GT、bbox、dataset、split、checkpoint hash；
- 测试 split 交集必须为零。

验收：`test_split_leakage.py` 和 synthetic smoke 通过。

### Step 1：复现主 predictor

- 使用现有 `gazelle.model.get_gazelle_model`；
- ViT-B 固定层 `[2,5,8,11]`；
- 首轮使用当前已知最强多层配置；
- 保存主任务 AUC/L2/AP，作为 risk 实验的不可退化基线。

验收：wrapper 与原脚本在同 checkpoint、同输入上的 heatmap 数值一致。

### Step 2：训练 fixed layer probes

- backbone 和主 predictor 全部冻结；
- 四个 probe 同结构、独立参数；
- 只用 `base_train`；
- 首轮 1--3 epoch，不做 layer search；
- 保存每层单独的 AUC/L2，排除“某层完全不可用”。

验收：至少中/深三层能产生非退化 heatmap；参数量、显存和速度记录完整。

### Step 3：无 risk head 的 disagreement pilot

- 在预留 ID、VAT、GOO-Real 上缓存输出；
- 计算 final confidence、layer disagreement 和真实 error；
- 只做固定公式和 logistic regression 分析；
- 按第 9 节 Go/No-Go 条件决定是否继续。

验收：生成 `per_sample.csv`、`summary.json`、coverage-risk 曲线和高/低分歧案例图。

### Step 4：训练双 risk head

- 在 `risk_train` 上根据冻结预测自动派生 L2/Brier target；
- `r_loc` 仅使用 in-frame 样本；
- `r_vis` 使用有 in/out GT 的样本；
- 在 `risk_calibration` 上校准数值或选择 coverage threshold；
- 不向主 predictor 反传梯度。

验收：risk head 明显优于 random，且至少不弱于 final heatmap entropy baseline。

### Step 5：正式三域评价

- GazeFollow-ID；
- GF -> VAT zero-shot；
- VAT-ID；
- GF -> GOO-Real zero-shot；
- 报告主任务、failure detection、risk-coverage、calibration 和 worst-domain 结果。

### Step 6：消融与 fatal-flaw audit

- 去掉 disagreement；
- 去掉 geometry-semantics conflict；
- 只用最后层；
- 使用 fusion weights；
- independent probes vs shared probe；
- 连续 L2 target vs `L2 > 0.15` 二分类；
- GF 单标注 train 与多标注 test 的标签歧义分析；
- 检查风险是否只是识别 small head、out-of-frame 或某个数据集。

---

## 12. 计划命令

以下命令均假设从仓库根目录运行。每条命令都直接写出服务器 Python
的绝对路径，不依赖 `PY`、`CUDA_VISIBLE_DEVICES` 等 shell 环境变量。
第一阶段支持 `--device` 的脚本显式使用 `cuda:0`：

```text
/home/fb/anaconda3/envs/py310/bin/python
```

一天 Pilot 固定复用以下已经训练完成的 GazeFollow SASA + GGSF checkpoint：

```text
/home/fb/src/paper/gazelleV1/experiments/train_gazefollow_sasa_ggsf/2026-02-26_15-51-06/epoch_14.pt
```

12.4 训练得到的 probe checkpoint 固定为：

```text
AAAIResults/selective_gaze/probes/gf_fixed_25811/layer_probes.pt
```

后续尚未实现阶段中的 `/path/to/vat_checkpoint.pt` 和 `/path/to/risk_checkpoint.pt`
仍然只是接口占位符。

### 12.0 当前实现状态与一天 Pilot 执行顺序

第一阶段只实现并允许运行以下顺序：

```text
12.1 单元测试与 synthetic smoke
  -> 12.2 构建无泄漏 splits
  -> 跳过 12.3，复用已有 GazeFollow checkpoint
  -> 12.4 训练 fixed probes
  -> 12.5 缓存三域预测
  -> 12.6 执行 Go/No-Go disagreement pilot
```

实现状态：

| 小节 | 状态 | 第一阶段是否运行 |
|---|---|---|
| 12.1 | 已实现 | 是 |
| 12.2 | 已实现 | 是 |
| 12.3 `train_base` | **未实现，属于 Formal 阶段** | **否，必须跳过** |
| 12.4 | 已实现 | 是 |
| 12.5 | 已实现 | 是 |
| 12.6 | 已实现 | 是 |
| 12.7 `train_risk_head` | **未实现，risk head 已明确延期** | 否 |
| 12.8 neural risk checkpoint evaluation | **未实现** | 否 |

如果执行未实现模块，Python 会报告 `No module named ...`。这不是环境问题，
也不应通过临时复制旧训练脚本来绕过数据协议。

### 12.1 单元测试与 synthetic smoke

```bash
/home/fb/anaconda3/envs/py310/bin/python -m pytest AAAISelectiveGaze/tests -q
```

```bash
/home/fb/anaconda3/envs/py310/bin/python -m AAAISelectiveGaze.scripts.train_layer_probes \
  --synthetic-smoke \
  --output-dir AAAIResults/selective_gaze/smoke/probes
```

```bash
/home/fb/anaconda3/envs/py310/bin/python -m AAAISelectiveGaze.scripts.evaluate_selective \
  --synthetic-smoke \
  --output-dir AAAIResults/selective_gaze/smoke/metrics
```

```bash
/home/fb/anaconda3/envs/py310/bin/python -m AAAISelectiveGaze.scripts.evaluate_disagreement \
  --synthetic-smoke \
  --bootstrap-iters 100 \
  --seed 3106 \
  --output-dir AAAIResults/selective_gaze/smoke/disagreement
```

### 12.2 构建无泄漏 splits

GazeFollow：

```bash
/home/fb/anaconda3/envs/py310/bin/python -m AAAISelectiveGaze.scripts.make_splits \
  --dataset gazefollow \
  --input-json /newhome/fb/dataset/gazefollow_extended/train_preprocessed.json \
  --output-dir AAAIResults/selective_gaze/splits/gazefollow \
  --base-train-ratio 0.85 \
  --base-val-ratio 0.05 \
  --risk-ratio 0.05 \
  --calibration-ratio 0.05 \
  --seed 3106
```

VAT，必须按 sequence 切分：

```bash
/home/fb/anaconda3/envs/py310/bin/python -m AAAISelectiveGaze.scripts.make_splits \
  --dataset vat \
  --input-json /newhome/fb/dataset/videoattentiontarget/train_preprocessed.json \
  --output-dir AAAIResults/selective_gaze/splits/vat \
  --group-key sequence_id \
  --base-train-ratio 0.85 \
  --base-val-ratio 0.05 \
  --risk-ratio 0.05 \
  --calibration-ratio 0.05 \
  --seed 3106
```

### 12.3 Formal：在干净 split 上训练主 predictor（未实现，Pilot 必须跳过）

**第一阶段不要运行本小节命令。** 一天 pilot 必须复用已有 GF checkpoint，
然后直接执行 12.4。正式阶段才实现 `AAAISelectiveGaze.scripts.train_base` 并重新训练，
确保 `risk_train` 和 `risk_calibration` 对主 predictor 未见。

以下仅保留为后续 Formal 阶段的接口设计，不是当前可执行命令：

```text
mkdir -p AAAIResults/selective_gaze/base
```

```text
nohup /home/fb/anaconda3/envs/py310/bin/python -u \
  -m AAAISelectiveGaze.scripts.train_base \
  --dataset gazefollow \
  --model gazelle_dinov3_vitb16 \
  --data-path /newhome/fb/dataset/gazefollow_extended \
  --train-json AAAIResults/selective_gaze/splits/gazefollow/base_train.json \
  --val-json AAAIResults/selective_gaze/splits/gazefollow/base_val.json \
  --layers 2 5 8 11 \
  --fusion raw_concat \
  --max-epochs 15 \
  --batch-size 60 \
  --lr 0.001 \
  --seed 3106 \
  --output-dir AAAIResults/selective_gaze/base/gf_raw_concat \
  > AAAIResults/selective_gaze/base/gf_raw_concat.log 2>&1 &
```

`raw_concat` 是首个可信 control，不表示最终一定采用它。若比较其他融合方式，只能根据 `base_val` 选择，不能查看 `risk_train`、VAT/GOO test 后再决定。

### 12.4 Pilot：训练四个 fixed probes

一天 Pilot 的主结果使用当前已经验证过的最强 GazeFollow predictor。如果现有
checkpoint 是 `DINOv3 ViT-B + SASA + GGSF`，则 `--fusion sasa`、
`--spatial-prior ggsf` 和 checkpoint 必须配套。probe 仍从 SASA 融合和 GGSF
门控之前读取四层原始特征；这些参数只用于正确重建并加载 base predictor。

先建立日志目录：

```bash
mkdir -p AAAIResults/selective_gaze/probes
```

```bash
nohup /home/fb/anaconda3/envs/py310/bin/python -u \
  -m AAAISelectiveGaze.scripts.train_layer_probes \
  --model gazelle_dinov3_vitb16 \
  --base-checkpoint /home/fb/src/paper/gazelleV1/experiments/train_gazefollow_sasa_ggsf/2026-02-26_15-51-06/epoch_14.pt \
  --fusion sasa \
  --spatial-prior ggsf \
  --data-path /newhome/fb/dataset/gazefollow_extended \
  --train-json AAAIResults/selective_gaze/splits/gazefollow/base_train.json \
  --val-json AAAIResults/selective_gaze/splits/gazefollow/base_val.json \
  --layers 2 5 8 11 \
  --epochs 3 \
  --batch-size 32 \
  --lr 0.001 \
  --freeze-backbone \
  --device cuda:0 \
  --seed 3106 \
  --output-dir AAAIResults/selective_gaze/probes/gf_fixed_25811 \
  > AAAIResults/selective_gaze/probes/gf_fixed_25811.log 2>&1 &
```

### 12.5 Pilot：缓存三域预测

GazeFollow 预留集：

```bash
/home/fb/anaconda3/envs/py310/bin/python \
  -m AAAISelectiveGaze.scripts.cache_predictions \
  --dataset gazefollow \
  --model gazelle_dinov3_vitb16 \
  --base-checkpoint /home/fb/src/paper/gazelleV1/experiments/train_gazefollow_sasa_ggsf/2026-02-26_15-51-06/epoch_14.pt \
  --probe-checkpoint AAAIResults/selective_gaze/probes/gf_fixed_25811/layer_probes.pt \
  --fusion sasa \
  --spatial-prior ggsf \
  --data-path /newhome/fb/dataset/gazefollow_extended \
  --json-path AAAIResults/selective_gaze/splits/gazefollow/risk_train.json \
  --batch-size 32 \
  --device cuda:0 \
  --output AAAIResults/selective_gaze/prediction_cache/gf_risk_train.parquet
```

VAT 跨域诊断：

```bash
/home/fb/anaconda3/envs/py310/bin/python \
  -m AAAISelectiveGaze.scripts.cache_predictions \
  --dataset vat \
  --model gazelle_dinov3_vitb16 \
  --base-checkpoint /home/fb/src/paper/gazelleV1/experiments/train_gazefollow_sasa_ggsf/2026-02-26_15-51-06/epoch_14.pt \
  --probe-checkpoint AAAIResults/selective_gaze/probes/gf_fixed_25811/layer_probes.pt \
  --fusion sasa \
  --spatial-prior ggsf \
  --data-path /newhome/fb/dataset/videoattentiontarget \
  --json-path AAAIResults/selective_gaze/splits/vat/risk_train.json \
  --batch-size 16 \
  --device cuda:0 \
  --output AAAIResults/selective_gaze/prediction_cache/vat_cross_domain.parquet
```

GOO-Real val：

如果 val JSON 尚未生成，先运行现有预处理脚本：

```bash
/home/fb/anaconda3/envs/py310/bin/python data_prep/preprocess_gooreal.py \
  --data_path /newhome/fb/dataset/gooreal_data \
  --split val \
  --output_json /newhome/fb/dataset/gooreal_data/gooreal_val_preprocessed.json
```

```bash
/home/fb/anaconda3/envs/py310/bin/python \
  -m AAAISelectiveGaze.scripts.cache_predictions \
  --dataset gooreal \
  --model gazelle_dinov3_vitb16 \
  --base-checkpoint /home/fb/src/paper/gazelleV1/experiments/train_gazefollow_sasa_ggsf/2026-02-26_15-51-06/epoch_14.pt \
  --probe-checkpoint AAAIResults/selective_gaze/probes/gf_fixed_25811/layer_probes.pt \
  --fusion sasa \
  --spatial-prior ggsf \
  --data-path /newhome/fb/dataset/gooreal_data \
  --json-path /newhome/fb/dataset/gooreal_data/gooreal_val_preprocessed.json \
  --zip-path /newhome/fb/dataset/gooreal_data/gooreal.zip \
  --zip-cache-dir /newhome/fb/dataset/gooreal_data/.gooreal_zip_cache \
  --batch-size 16 \
  --num-workers 0 \
  --device cuda:0 \
  --output AAAIResults/selective_gaze/prediction_cache/gooreal_val.parquet
```

GOO-Real 的 nested zip 中可能包含数据流不完整但仍可解码的 JPEG。缓存脚本默认先
严格解码，失败后仅对该图片启用 truncated-JPEG recovery，并在对应 per-sample
记录中写入 `image_decode_recovered=true`。如需完全禁止容错，可额外传入
`--strict-images`；一天 Pilot 默认保留可审计的容错行为。

### 12.6 Pilot：检验层级分歧是否真的预测错误

```bash
/home/fb/anaconda3/envs/py310/bin/python \
  -m AAAISelectiveGaze.scripts.evaluate_disagreement \
  --id-predictions AAAIResults/selective_gaze/prediction_cache/gf_risk_train.parquet \
  --shift-predictions AAAIResults/selective_gaze/prediction_cache/vat_cross_domain.parquet \
  --ood-predictions AAAIResults/selective_gaze/prediction_cache/gooreal_val.parquet \
  --failure-l2-threshold 0.15 \
  --bootstrap-iters 1000 \
  --seed 3106 \
  --output-dir AAAIResults/selective_gaze/pilot/hierarchy_disagreement
```

### 12.7 Full：训练 risk head（未实现，第一阶段不要运行）

本小节属于后续阶段。第一阶段没有实现完整 risk head，也不得启动该训练。
以下仅保留为接口设计：

```text
/home/fb/anaconda3/envs/py310/bin/python \
  -m AAAISelectiveGaze.scripts.train_risk_head \
  --train-predictions AAAIResults/selective_gaze/prediction_cache/gf_risk_train.parquet \
  --calibration-predictions AAAIResults/selective_gaze/prediction_cache/gf_risk_calibration.parquet \
  --targets localization \
  --localization-loss huber \
  --epochs 30 \
  --batch-size 512 \
  --lr 0.001 \
  --seed 3106 \
  --output-dir AAAIResults/selective_gaze/risk_heads/gf_source_only
```

`r_vis` 不能从 GazeFollow source-only 数据可靠训练。VAT-ID 时使用 `/path/to/vat_checkpoint.pt` 和 VAT 的 `risk_train/risk_calibration` prediction cache，另行运行同一脚本，并设置 `--targets localization visibility --visibility-loss bce`。

### 12.8 Full：三域 neural-risk selective evaluation（未实现，第一阶段不要运行）

第一阶段已实现的 `evaluate_selective --synthetic-smoke` 和基于固定 baseline 的评价
不加载 neural risk checkpoint。以下带 `--risk-checkpoint` 的命令属于后续阶段，
当前脚本会明确拒绝执行。它们仅保留为接口设计。

GazeFollow-ID：

```text
/home/fb/anaconda3/envs/py310/bin/python \
  -m AAAISelectiveGaze.scripts.evaluate_selective \
  --dataset gazefollow \
  --predictions AAAIResults/selective_gaze/prediction_cache/gf_test.parquet \
  --risk-checkpoint /path/to/risk_checkpoint.pt \
  --failure-l2-threshold 0.15 \
  --coverages 0.50 0.70 0.80 0.90 1.00 \
  --bootstrap-iters 1000 \
  --output-dir AAAIResults/selective_gaze/final/gf_id
```

GF -> VAT：

```text
/home/fb/anaconda3/envs/py310/bin/python \
  -m AAAISelectiveGaze.scripts.evaluate_selective \
  --dataset vat \
  --predictions AAAIResults/selective_gaze/prediction_cache/vat_test_from_gf.parquet \
  --risk-checkpoint /path/to/risk_checkpoint.pt \
  --localization-only \
  --failure-l2-threshold 0.15 \
  --coverages 0.50 0.70 0.80 0.90 1.00 \
  --bootstrap-iters 1000 \
  --output-dir AAAIResults/selective_gaze/final/gf_to_vat
```

GF -> GOO-Real：

```text
/home/fb/anaconda3/envs/py310/bin/python \
  -m AAAISelectiveGaze.scripts.evaluate_selective \
  --dataset gooreal \
  --predictions AAAIResults/selective_gaze/prediction_cache/gooreal_test_from_gf.parquet \
  --risk-checkpoint /path/to/risk_checkpoint.pt \
  --localization-only \
  --failure-l2-threshold 0.15 \
  --coverages 0.50 0.70 0.80 0.90 1.00 \
  --bootstrap-iters 1000 \
  --output-dir AAAIResults/selective_gaze/final/gf_to_gooreal
```

---

## 13. 预期输出与论文图表

每次正式评价至少输出：

- `run_manifest.json`：代码版本、checkpoint hash、数据 manifest、参数；
- `per_sample.parquet`：每个人物的 prediction、GT、risk 和分组属性；
- `metrics.json`：AUC/L2/AP、AUROC/AUPR、AURC、risk@coverage、calibration；
- `coverage_risk.csv`：绘图原始数据；
- `coverage_risk.png`：不同 baseline 的曲线；
- `failure_cases/`：高风险真错误、低风险真正确、false alarm、missed failure 四类案例。

论文最重要的结果图不是普通 AUC 柱状图，而是：

1. 三个数据域上的 coverage-risk 曲线；
2. 控制 final confidence 前后，hierarchy disagreement 的增量 AUROC；
3. geometry-semantics conflict 的典型成功和反例；
4. GazeFollow 训练、VAT/GOO-Real 测试时的 source-only risk 排序稳定性。

---

## 14. Fatal flaws 与停止条件

### Critical

1. **Layer Ensemble 直接迁移问题。** 如果方法只有“多个层各接一个 head，再算方差”，与已有 Layer Ensembles 的方法差异不足。
2. **深层主导导致虚假分歧。** 浅层普遍不具备 gaze 能力时，分歧不能解释为 uncertainty。
3. **无跨域增益。** 只在 GazeFollow 有效、在 VAT/GOO-Real 失效时，domain-shift 主故事不成立。
4. **置信度冗余。** 控制 final heatmap entropy 后分歧无增量，本方案的方法核心不成立。

### Major

5. GazeFollow train 单标注把人类歧义误记成模型错误；需要官方多标注 test 分析和 PDP 类 distribution baseline。
6. risk gain 只来自 small head、out-of-frame 或简单数据集识别，而不是真正 localization failure。
7. 使用 test 选择层、阈值或 risk 组合造成数据泄漏。

### 最终停止规则

如果一天 pilot 满足第 9.4 节任意两个核心 No-Go 条件，停止实现神经 risk head；先重新评估 selective gaze 是否还能以非层级方法形成足够贡献，不继续因为已有代码投入而追加训练。
