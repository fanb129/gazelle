# Gaze Target Estimation 深度文献地图（2015–2026）

> 检索冻结日期：2026-07-17  
> 范围：第三人称 gaze following / gaze target estimation，以及直接扩展到视频、多人、3D、对象、语义、像素级、VFM、VLM、promptable 和 multi-view 的代表性工作。  
> 数字约定：除“本项目”外均为论文作者报告值，未由本项目独立复现；不同输入、训练数据、backbone、标注转换和任务协议下的数字不能直接排名。

## 0. 执行摘要

这个领域的演化可以压缩为四代：

1. **2015–2020：head + scene 双流与 saliency。** 问题被定义成“给定一人的 head box，在图像/视频中回归一个 gaze heatmap，并判断是否出画”。
2. **2021–2023：geometry、depth、object 与 distribution。** 研究者发现纯 2D saliency 容易走捷径，于是引入 gaze direction、depth、point cloud、object proposals、patch distribution 和更细的 in/out 表示。
3. **2022–2024：Transformer、end-to-end、multi-person、temporal、semantic。** 重点从逐人双分支转向同时检测/关联所有 heads-targets、共享 scene computation，并开始预测目标对象/语义与社会 gaze。
4. **2025–2026：VFM/VLM 与新输入协议。** Gaze-LLE 证明冻结 DINOv2 + 轻量 decoder 足以成为强基线；随后路线迅速分化到 promptable subject、multi-view、object-aware DINOv3 hierarchy、local adaptation、MoE、VLM reasoning 和大规模 distillation。

对 Gazelle + DINOv3 + 多层特征最重要的结论是：

- 多层特征仍有工程价值，但已经不是可独立占据的 novelty；
- object-aware、FoV/gaze cone、MoE、VLM、promptable、多人物 scene reuse 都已有近邻；
- 当前更空的轴不是“怎样再融合一次 feature”，而是**怎样证明输出确实绑定到被查询人物、怎样评价真实可见性/遮挡状态、怎样用受控干预暴露 saliency shortcut**。

## 1. 研究问题与检索方法

### RQ1

GTE 从 2015 到 2026 的问题定义、模型结构和 benchmark 发生了什么变化？

### RQ2

哪些现有工作已经覆盖 DINOv3、多层、对象、几何、多人、VLM 和 prompt，因而不能再作为新论文的 headline novelty？

### RQ3

传统 CV、3D gaze、HOI/association、amodal perception、robustness/evaluation 中还有哪些机制可以迁移到 GTE，并形成目前检索未直接覆盖的新问题？

### 检索与核验

- 核心事实优先核对 CVF、NeurIPS Proceedings、OpenReview、出版社页面和 arXiv 正文；
- 先按 foundation、geometry/object、transformer/multi-person、VFM/new-setting 四个对立视角检索，再做引用前向/后向追踪；
- peer-reviewed、accepted、preprint/under-review 分开标记；
- “未检索到直接重合”只表示在给定关键词和语料下未发现，不等于证明无人做过。

## 2. 任务、数据集和指标：先把协议分清

### 2.1 任务族

| 任务 | 输入 | 输出 | 不能与什么直接横比 |
|---|---|---|---|
| Standard point-GTE | image/video + queried head box | heatmap/point + in/out | end-to-end head detection mAP、object AP、3D angular error |
| End-to-end HGT | image，无给定 head | 多个 head–target pairs | 给定 GT head 的 AUC/L2 |
| Gaze object prediction | image + head/query | gaze object box/class/mask | 只定位点的 AUC/L2 |
| Semantic/pixel GTE | image + query | point + class/mask | 普通 heatmap leaderboard |
| 3D gaze/target | face/head + geometry/depth | 3D direction/location | 2D normalized L2 |
| Multi-view GTE | 多个同步相机视图 | 某视图中的 target 或 cross-view target | 单视图公开 benchmark |
| Promptable GTE | text/point/visual concept 指定人物 | head + heatmap + in/out | 直接给 GT head bbox 的 standard GTE |

### 2.2 核心数据集

| 数据集 | 规模与标注 | 价值 | 主要缺点 |
|---|---|---|---|
| GazeFollow | 122,143 images、130,339 person instances；test 约 4,782 instances，通常每个有 10 个 gaze annotations | 最大静态标准集；支持 annotator ambiguity | test fixation 被刻意均匀化；train 单点/test 多点；同图多人记录需重组才能做 binding |
| VideoAttentionTarget (VAT) | 50 个 YouTube 节目、1,331 head tracks、164,541 person-frames；109,574 in-frame、54,967 out-of-frame | 视频、自然社会场景、标准 in/out AP | 连续帧非独立；transition/看镜头可能混入 out；必须按 track/show 聚类统计 |
| ChildPlay | 401 clips、120,549 frames、257,928 person-frame instances；每帧最多 3 人；七类 gaze state | 儿童、自然互动、状态更细 | 以室内/YouTube 为主；困难状态常被旧二元协议删去；需按 source video/channel 隔离 |
| GOO | 192,000 synthetic + 9,552 real（7,396/2,156 train/test）；24 类，平均约 80 objects/image | 密集零售对象与 object-level evaluation | 约 95% 为合成、仅 20 avatars；单人图；point/object 指标尚不统一 |
| VSGaze | 合并并统一多个 gaze following/social gaze 数据集，补 heads/tracks/social labels | 多人、时间、social gaze 联合学习 | 由异构来源组合，协议/缺失标签处理复杂 |
| MVGT | 2,281 独立 gaze events × 6 views = 13,686 images；28 subjects、4 scenes、6 calibrated cameras | 第一个专门 multi-view GTE benchmark | 68,430 view pairs 仍只有 2,281 独立事件；scene 与 subject 混杂；held-out scene 调参风险 |
| GazeSeg/PixelGaze | 基于 GazeFollow 的 72K images、270 target categories、pixel masks | 从点升级到 mask/class | pseudo/derived annotation 质量与长尾类别需审计；仍是衍生集 |

### 2.3 标准指标的准确含义

- **AUC**：把 GT gaze point/分布视为正区域，对预测 heatmap 做 ROC AUC；它奖励整体排序，但对峰值精确位置不敏感。GF 的稀疏点 GT、VAT 的 Gaussian/阈值区域、VideoCoAtt 的 proposal ROC 不是同一个 AUC。
- **Avg L2（GF 常用定义）**：预测峰到 10 个 annotator points 均值位置的归一化欧氏距离，即 `||p-mean(g_i)||`；均值可能落在两个真实物体之间。
- **Min L2**：预测峰到最近 annotator gaze point 的距离；更宽容，但可能掩盖标注分歧。
- **VAT L2/Dist**：通常为预测峰到单个 GT point 的归一化距离；与 GF Avg/Min 不是同一量。
- **In/Out AP**：对 target 是否在 frame 内/外的 binary score 做 average precision；类别比例、正类与 score 方向实现必须核对。VAT 原论文的 `α` 越大越倾向 in-frame，但文字常称 Out-of-Frame AP。
- **Angular error**：预测 2D/3D gaze direction 与 GT direction 的夹角；依赖 eye/head origin 定义。
- **mAP / wUoC / object AP / mIoU / Dice / recognition accuracy**：分别属于 end-to-end detection、gaze-object、segmentation/recognition，不能塞进 point-GTE 同一列。

### 2.4 最常见的不可比陷阱

1. 给 GT head box 与模型自己检测 head；
2. 单图逐人重跑 encoder 与 scene-once 多人解码；
3. GF-only 训练、GF→VAT fine-tune、GF+VAT+ChildPlay 混合监督、百万无标签蒸馏；
4. DINOv2/DINOv3 大小、输入 224/448/512、heatmap resolution 不同；
5. standard GF/VAT 与作者自建 consistent/inconsistent、crowd、semantic、PGE split；
6. 单次点估计与 3 seeds/按视频 cluster bootstrap；
7. trainable params 与包含 frozen backbone/外部 detector 的 total system cost。

### 2.5 数据集层面的隐藏偏差

| 数据/协议 | 必须写进论文的 caveat |
|---|---|
| GazeFollow | test gaze locations 被设计得近似均匀，削弱自然 center/saliency prior；train 单标注、test 多标注；官方 image-level split 不自动保证来源身份/近重复图完全隔离 |
| VAT | test 为 10 个未见节目、298 tracks、31,978 person-frames；统计单位应是 track/show，不是 frame；out 类混合真正图外、看镜头和潜在 transition |
| ChildPlay | `9.3%` frames 属 shift/occluded/uncertain/closed/not-annotated 等困难状态；只算 visible inside + binary in/out 会形成 easy-case conditional score |
| GOO | “201K”主要来自 192K synthetic；GOO-Real 单人且 retail-specific；正确 object 与小 L2 可能冲突，必须拆 object success 与 within-object localization |
| VideoGaze | 是 source frame + 5 candidate frames 的跨帧检索/定位；其 AP 是 candidate-frame selection，不是 VAT in/out AP |
| VideoCoAtt | 是 shared-attention target，约 71% negative；存在性 accuracy 易虚高，位置 L2 是固定分辨率 pixel，不是 normalized L2 |
| VSGaze | 由 VAT/ChildPlay/VideoCoAtt/UCO-LAEO 拼接，含 pseudo tracks/points 和极端 pairwise imbalance；应按 source dataset macro-average |
| Gaze-Co/GazeAnywhere | gaze 样本来自 GF/VAT/ChildPlay 的转换，不是独立泛化集；过滤清晰/中大 head 改变了 support |
| PixelGaze | 基于 GazeFollow 的衍生数据；只保留可分割对象会产生 selection bias；应同时报 object success、conditional IoU 与 point-in-mask |
| MVGT | 68,430 view pairs 来自 2,281 events；不能把 pair 当独立 n；leave-one-scene-out 中若按 held-out scene 调不同超参，需要 nested validation |
| 360° GTE | equirectangular pixel area 随纬度畸变，普通 planar L2/AUC 不成立；应使用 great-circle distance 和 spherical area weighting |

### 2.6 推荐的论文级 metric bundle

1. Point：normalized L2、success@`{0.05,0.10,0.20}`，并同时保留 GF Avg/Min。
2. Heatmap：准确描述 GT 构造的 AUC；有多标注时增加 NLL/energy score，而不只 argmax。
3. Visibility：AUPRC、AUROC、ECE/Brier，并拆分 outside、occluded、shift、closed-eyes、uncertain。
4. Object/semantic：object top-1、point-in-object、conditional IoU/mIoU 分开。
5. Association：pairwise binding accuracy、swap margin、own-target rank；同时保留 standard AUC/L2/AP。
6. Statistics：GF 按 image，VAT 按 track/show，ChildPlay 按 source video/channel，MVGT 按 gaze event/subject 做 paired cluster bootstrap。

## 3. 标准 point-GTE 的一条可比主线

下表复录常见协议下的代表数字，只用于理解历史趋势。2026 大模型/混合数据工作另表讨论。

| 方法 | 年份 | GF AUC / Avg L2 / Min L2 | VAT AUC / L2 / AP |
|---|---:|---|---|
| Chong et al. | 2018 | `.896 / .187 / .112` | `.833 / .171 / .712` |
| Lian et al. | 2018 | `.906 / .145 / .081` | `.837 / .165 / –` |
| Chong et al. | 2020 | `.921 / .137 / .077` | `.860 / .134 / .853` |
| Fang et al. | 2021 | `.922 / .124 / .067` | `.905 / .108 / .896` |
| ESCNet | 2022 | `.928 / .122 / –` | `.885 / .120 / .869` |
| Jin et al. | 2022 | `.920 / .118 / .063` | `.900 / .104 / .895` |
| PDP | 2023 | `.934 / .123 / .065` | `.917 / .109 / .908` |
| ChildPlay/3DFoV | 2023 | `.939 / .122 / .062` | `.914 / .109 / .834` |
| Sharingan | 2024 | `.944 / .113 / .057` | `– / .107 / .891` |
| ViTGaze | 2024 | `.949 / .105 / .047` | `.938 / .102 / .905` |
| Gaze-LLE-B | 2025 | `.956 / .104 / .045` | `.933 / .107 / .897` |
| Gaze-LLE-L | 2025 | `.958 / .099 / .041` | `.937 / .103 / .903` |

这些数字并不证明后来的结构一定更好：backbone、预训练、输入分辨率和训练数据同时变化。它们只显示 point-localization 已接近 `.95+ AUC / ~.10 L2`，所以新工作必须证明**新 setting、失败维度、效率或泛化**，而不只是千分位 AUC。

## 4. 逐篇深度卡片

### 4.1 Recasens et al., *Where Are They Looking?*（NeurIPS 2015，peer-reviewed）

**Motivation 与故事。** 人能把他人的头部朝向与场景候选物联系起来；只做 face gaze direction 不知道射线与哪个场景实体相交，只做 saliency 又不知道“谁在看”。论文据此把问题定义为给定 head location 的 gaze following，并创建 GazeFollow。

**方法。** scene/saliency pathway 与 head/gaze pathway 分别编码整图和 head/location，通过 shifted-grid gaze classification 与场景 saliency 融合生成最终预测。它确立了之后多年“head branch + scene branch + spatial output”的标准范式。

**结果与协议。** 原文 Table 1 报 `.878 AUC / .190 Avg L2 / .113 Min L2 / 24°`，Human 为 `.924/.096/.040/11°`。这些是原始 GazeFollow protocol，不应与后来扩展 label/实现的数字混用。更持久的贡献是 122K 级数据集与任务契约。

**可借鉴。** query 和 scene 必须联合；同场景不同人物天然构成受控 query 变化。**缺点。** head box 由外部给定；单帧、单点、场景 saliency shortcut 强；没有 in/out、对象语义、多人 correspondence 的独立评价。

### 4.2 Recasens et al., *Following Gaze in Video*（ICCV 2017，peer-reviewed）

**Motivation 与故事。** 单帧头姿常有歧义，但视频中人物运动、场景变化和跨帧可见目标提供互补证据。论文将静态 gaze following 扩展为：给一个人的 track/参考，跨帧寻找其 attention target。

**方法。** 联合学习 saliency、gaze cone、source→target frame 的 3D affine/几何关系与 candidate-frame probability；模型既选哪一帧包含目标，也在该 target view 中定位。

**结果与协议。** VideoGaze localization 报 `.890 AUC / .184 Dist / .123 MinDist / 7.76 KL`，candidate-frame selection AP `87.5`；Human localization `.901/.103/.063`。这是 source frame + 5 neighboring frames 的检索/定位，不是后来的 VAT frame-wise GTE。

**可借鉴。** temporal evidence 应在 target visibility/transition 层面建模，而不是简单平滑 heatmap。**缺点。** 数据和任务协议与现代 VAT 不统一；人物 track/参考依赖明显；未显式处理多人物共同/不同目标。

### 4.3 Chong et al., *Connecting Gaze, Scene, and Attention*（ECCV 2018，peer-reviewed）

**Motivation 与故事。** 早期方法常把 gaze direction 与 scene saliency 分开；论文提出 generalized attention estimation，认为人的外观、头部方向和 bottom-up scene saliency 应联合训练，并需要表示 out-of-frame gaze。

**方法。** 双 ResNet-50 scene/head branches 与 project-and-compare 融合；联合学习 within-frame heatmap、3D gaze angle 和 fixation likelihood，并通过多数据集 selective backprop 处理并非每个数据都具备的标签。

**结果与协议。** 常见统一表为 GF `.896/.187/.112`，VAT `.833/.171/.712`。它是重要历史基线，但数字低于后续方法也包含 backbone/训练年代差异。

**可借鉴。** in/out 不是附加标签，而是 attention distribution 的一部分；head-conditioned scene modulation 是核心。**缺点。** 多任务链条较手工；逐人计算；saliency 会成为 shortcut；没有对象/3D/时间与对应关系分析。

### 4.4 Lian et al., *Believe It or Not, We Know What You Are Looking At!*（ACCV 2018，peer-reviewed）

**Motivation 与故事。** gaze target 既取决于 head orientation，也取决于 scene 中哪些位置在几何上与语义上合理。论文强调 head cue 与 scene saliency 的融合应更直接。

**方法。** head crop + head position 先预测 2D direction；以 `γ={5,2,1}` 形成多尺度 direction fields，与 image feature 早期拼接后由 FPN-style decoder 输出 heatmap，并联合优化 angular 与 heatmap losses。

**结果与协议。** 统一表常报 GF `.906/.145/.081`，VAT `.837/.165`；相比 Chong 2018，距离指标改善明显，显示更强的 head–scene coupling 有价值。

**可借鉴。** 几何方向可作为 search-space bias，但应和不确定性结合。**缺点。** 仍是 CNN 双流与单人设置；没有 out-of-frame AP、对象级解释或多人共享计算；现代 VFM 下其模块 novelty 已过时。

### 4.5 Chong et al., *Detecting Attended Visual Targets in Video*（CVPR 2020，peer-reviewed）

**Motivation 与故事。** 真实 attention 会随时间变化，且目标可能出画；现有静态方法不能利用 temporal continuity，也缺少足够视频 benchmark。论文同时提出 VAT 数据集与视频模型。

**方法。** scene/head features 经 attention 融合后进入 ConvLSTM，联合输出 frame-wise gaze heatmap 与 in/out score；时间模块用于积累方向和目标证据。

**结果与协议。** GF `.921/.137/.077`，VAT `.860/.134/.853`；并在 social gaze behavior 分类中使用预测 attention map。VAT 的约 165K 连续帧成为之后主 benchmark。

**可借鉴。** 模型和 benchmark 的共同贡献比单个 block 更持久；视频统计必须按 clip/sequence 处理。**缺点。** ConvLSTM 可能平滑掉 target switch；VAT 的特殊状态主要压成 in/out；逐人双流成本较高。

### 4.6 Fang et al., *Dual Attention Guided Gaze Target Detection in the Wild*（CVPR 2021，peer-reviewed）

**Motivation 与故事。** 2D gaze heatmap 缺乏明确几何约束，深度又包含大量与 gaze 无关区域。论文用粗到细的 3D gaze orientation 与 depth-aware attention 缩小候选空间。

**方法。** 先预测 coarse gaze direction/FoV attention，再把估计深度投影到几何空间，使用 dual attention 联合头部方向和 scene depth，最后回归 2D target。

**结果与协议。** GF `.922/.124/.067`，VAT `.905/.108/.896`，相对 2020 baseline 的 VAT AUC/AP 和 L2 均明显改善。

**可借鉴。** 软几何场比只用 bbox 坐标更接近真正 gaze constraint；direction uncertainty 应被显式表示。**缺点。** monocular depth 和 gaze direction 错误会级联；计算重；fixed cone/FoV 容易过度排除真实 target。

### 4.7 Bao et al., *ESCNet: Gaze Target Detection with the Understanding of 3D Scenes*（CVPR 2022，peer-reviewed）

**Motivation 与故事。** 仅在图像平面上推断 gaze 会把前后深度不同但像素方向相近的区域混在一起；需要理解哪个表面在 gaze ray 上最先被看见。

**方法。** 由 monocular depth 构建 scene point cloud，估计 3D gaze direction，并将几何 gaze likelihood 与 scene semantic feature 融合；front-most/visible surface reasoning 用于处理遮挡。

**结果与协议。** GF `.928/.122`，VAT `.885/.120/.869`。GF 有提升，但 VAT AUC/AP 并未全面超过 Fang 2021，说明几何复杂度不自动转化为所有域收益。

**可借鉴。** “可见表面”比人数或 2D box 更接近 occlusion 构念；可发展 observability-aware GTE。**缺点。** 单目深度尺度/边界误差；3D 中间量缺少直接 GT；point-cloud pipeline 重且难归因。

### 4.8 Jin et al., *Depth-Aware Gaze-Following via Auxiliary Networks for Robotics*（EAAI 2022，peer-reviewed）

**Motivation 与故事。** 机器人需要在真实场景中分辨沿相近 2D 方向但深度不同的候选目标；主任务数据没有可靠深度 supervision。

**方法。** 使用辅助 depth/gaze networks 提供 scene geometry 与 head-direction cues，再融合到 gaze-following heatmap；强调模块化、可用于 robotics。

**结果与协议。** 常见统一表为 GF `.920/.118/.063`，VAT `.900/.104/.895`：定位距离较强，但 GF AUC 不一定最高。

**可借鉴。** auxiliary task 可在不重标 GTE 的情况下注入 geometry；应把 depth quality 与最终误差关联。**缺点。** 外部网络和预训练数据构成隐藏成本；模块误差相关；没有 end-to-end attribution、多人或 target semantics。

### 4.9 Tonini et al., *Multimodal Across Domains Gaze Target Detection*（ICMI 2022，peer-reviewed）

**Motivation 与故事。** RGB gaze models 对 domain shift、隐私限制和缺失模态脆弱；depth/pose 等模态可能在跨域与 privacy-sensitive settings 中更稳定。

**方法。** 模块化融合 head、scene、depth/pose 等模态，并研究 source→target domain adaptation 与模态缺失；重点不是追单一标准集 SOTA，而是 across-domain robustness。

**结果与协议。** 论文在多个 source/target 组合中报告 multimodal/domain-adaptation 优于只用 RGB 的 baselines；由于 split 与模态设置不同，不应把其数字放进标准 GF/VAT 排名。

**可借鉴。** 对你的新稿，真正的 generalization 应定义为可复现 shift，而不是只加一个外部集。**缺点。** 外部模态在部署时未必可用；融合提升可能来自额外信息量；domain adaptation 协议复杂。

### 4.10 Miao et al., *Patch-Level Gaze Distribution Prediction for Gaze Following*（WACV 2023，peer-reviewed）

**Motivation 与故事。** 单个 Gaussian heatmap + 独立 in/out classifier 无法表达多 annotator ambiguity，也割裂“frame 内分布”和“outside probability”。论文把输出改写成包含 outside 的 patch distribution。

**方法。** 将图像划分为 patches，直接预测 gaze probability distribution，并用一个 outside patch/token统一 in/out；distribution target 根据多标注方差调节，缓解 MSE 对不确定样本的错误惩罚。

**结果与协议。** GF `.934/.123/.065`，VAT `.917/.109/.908`；AUC/AP 强，但 GF Avg L2 不优于若干 depth 方法，体现 distribution calibration 与 argmax distance 的目标差异。

**可借鉴。** shared-target/ambiguous target 应用 distribution 而非硬 one-hot；COTB 可用 target-neighborhood mass。**缺点。** patch discretization 限制定位；distribution 质量依赖标注数量；仍逐 query 评价，没有 correspondence metric。

### 4.11 Tu et al., *End-to-End Human-Gaze-Target Detection with Transformers (HGTTR)*（CVPR 2022，peer-reviewed）

**Motivation 与故事。** 标准 GTE 依赖外部 head detector，且一次只问一人；这既有级联误差，也无法原生表示整幅图中多个 head–target instances。论文把任务重新表述成 DETR-style set prediction。

**方法。** Transformer 从整幅图直接输出多个 head boxes、gaze targets 和关联实例；Hungarian matching 负责预测集合与 GT 集合配对，不再把 head box 当输入。

**结果与协议。** R50 的 end-to-end “Real” setting：GF `.917 AUC/.133 Avg/.069 Min/.547 mAP`；VAT `.893/.137/.821 AP/.514 mAP`。headline 为 GF/VAT mAP `+6.4/+10.3` 个百分点。mAP 与 matched-detection distance 同时受 head detection/assignment 影响，不能与给定 GT head 的 AUC/L2 横比。

**可借鉴。** set prediction 暴露了“谁对应哪个 target”这一关系；COTB 可借鉴 permutation/assignment 思路。**缺点。** matching 发生在预测实例与 GT 间，不等于检验给定人物 query 的反事实响应；head detection error 会掩盖 gaze decoder 本身。

### 4.12 Wang et al., *GaTector: A Unified Framework for Gaze Object Prediction*（CVPR 2022，peer-reviewed）

**Motivation 与故事。** 一个点坐标不能告诉机器人/零售系统“人看的是哪个物体”；把独立 object detector 与 gaze heatmap 后处理拼起来又会产生不一致。论文定义 gaze object prediction 并引入 GOO。

**方法。** scene backbone 在 object detection 与 gaze estimation 间共享；Defocus 操作用低成本扩大 receptive field；energy aggregation 把 gaze heatmap 与候选 object boxes 结合；提出 wUoC，使不重叠预测框也能按 gaze-object 接近程度比较。

**结果与协议。** GOO-Synth 上 gaze `.957 AUC/.073 Dist/14.9°`，object detector `56.8 AP / 95.3 AP50 / 62.5 AP75`，最终 GOP `28.5 wUoC`。GT box 只把 wUoC 提到 `29.81`，GT heatmap 可到 `78.79`，表明主要瓶颈是 gaze 而非 detection。wUoC 是 GOP 专用指标，不能与 GF L2 直接排名。

**可借鉴。** 从 point 转向 target entity，能形成更有用的错误类型。**缺点。** 依赖 object boxes/categories；密集 retail 域偏；detector recall 给性能上限；object-level pipeline 与 2026 object-aware 工作已拥挤。

### 4.13 Tafasca et al., *ChildPlay: A New Benchmark for Understanding Children's Gaze Behaviour*（ICCV 2023，peer-reviewed）

**Motivation 与故事。** 主流 benchmark 几乎都是成人，难以支持儿童发展/临床场景；而且现实帧不总能强行标成 inside/outside。论文同时贡献儿童视频数据和 3DFoV 模型。

**方法。** ChildPlay 对每帧最多 3 人标 head、2D point、adult/child 与七类 gaze state；模型用 depth-preserving inference 和预测 3D gaze vector 构造 person-specific FoV，再与 scene feature 融合。

**结果与协议。** 标准外部表常复录 GF `.939/.122/.062`、VAT `.914/.109/.834`；ChildPlay full 上作者方法为 `.935 AUC/.107 Dist/.986 AP`，GF-only zero-shot 为 `.932/.115`，Human `.911/.048/.993`。更重要的发现是 children 的 face-looking prediction 明显弱于 adults。数据中 inside `85.3%`、outside `5.4%`，其他困难状态占 `9.3%`。

**可借鉴。** 把 occluded/shift/uncertain 从 out-of-frame 中拆出，是 observability-aware GTE 的直接依据。**缺点。** YouTube/室内域、下载可用性、儿童隐私与年龄偏差；3DFoV 仍受 depth/direction error 级联。

### 4.14 Tonini et al., *Object-aware Gaze Target Detection*（ICCV 2023，peer-reviewed）

**Motivation 与故事。** heatmap 只回答“哪里”，不解释“哪个对象/类别”，也没有显式 head–object relation。论文希望一次获得点、区域、类别、对象位置与 in/out 的可解释分析。

**方法。** Transformer 自动检测 heads 与 objects，并为每个 head 建立 gazed-head/object association；联合输出 gaze target area、pixel point、gazed-object class/location 和 out-of-frame probability。

**结果与协议。** 其 end-to-end association protocol 下，RGB-2D 在 GF `.922 AUC/.072 Avg/.033 Min`、VAT `.923/.102/.944 AP`；RGB+Depth 为 GF `.922/.069/.029`、VAT `.933/.104/.934`。摘要另报 AUC 最多 `+2.91%`、distance 最多下降 `50%`、out AP 最多 `+9%`，gazed-object AP `+11–13%`。这些数字不可与 GT-head standard GTE 粗体排名。

**可借鉴。** 显式关系与对象解释是强 baseline；COTB 不能声称首次 head–target association。**缺点。** object detection 标签/错误与 gaze 纠缠；对象边界并不总等于注视区域；end-to-end 多任务归因复杂。

### 4.15 Horanyi et al., *Where Are They Looking in the 3D Space?*（CVPRW 2023，peer-reviewed workshop）

**Motivation 与故事。** 2D image plane 会把不同深度上的候选压到相近像素；共同注意还需要综合多人的 FoV 与 subject attributes。论文把 joint attention target 放到 3D space 中。

**方法。** 使用 depth prior 构建 scene geometry，为多人生成 3D joint FoV probability map，并估计共同 attention target；也可退化到 single-person GTE。

**结果与协议。** 论文在 VideoCoAtt 的 joint-attention protocol 上优于既有方法，并报告在 GF/VAT single-attention 设置也有竞争力；其 3D/joint target 定义与 standard point-GTE 不是完全同一协议。

**可借鉴。** depth order 和 visibility 是比 2D crowd count 更真实的构念；supplement 还系统列出 human bias、GT ambiguity、head/target occlusion 与 physically impossible prediction。**缺点。** workshop 规模、monocular depth 误差、联合目标假设限制了普适性。

### 4.16 Tafasca et al., *Sharingan: A Transformer Architecture for Multi-Person Gaze Following*（CVPR 2024，peer-reviewed）

**Motivation 与故事。** 逐人重跑 CNN 在多人场景浪费 scene computation；无控制的 object-query 模型还需事后 matching，难以保证用户指定的人是谁。论文保留标准“给定 person”契约，同时一次处理多人。

**方法。** 把每人的位置/外观编码成 controlled person token，与共享 image tokens 一起经过 Transformer；multiscale decoder 为各 person 输出 heatmap/in-out。

**结果与协议。** GF `.944/.113/.057`，VAT L2 `.107`、AP `.891`，并在 ChildPlay 报告强结果。其突出价值是多人效率和可控 query，而不是只看单人点指标。

**可借鉴。** 它是 COTB 最重要 baseline：controlled token 已解决“输入接口可指定谁”，但未必证明输出对应关系在同帧 hard pairs 上可靠。**缺点。** 仍以逐 query 标准 loss/metric 为主；多人标注不完整；105M 级模型成本较高。

### 4.17 Song et al., *ViTGaze: Gaze Following with Interaction Features in Vision Transformers*（Visual Intelligence 2024，peer-reviewed）

**Motivation 与故事。** CNN 双流的融合发生较晚，难以在所有 image patches 与 head/query 间建立全局长程 interaction。ViT 自注意力天然提供 token-to-token 关系。

**方法。** 将 head/position guidance 注入 ViT interaction encoder，并通过多尺度/二维 guidance decoder 恢复精细 heatmap；重点是 encoder 内的 head–scene interaction，而不只是换 backbone。

**结果与协议。** GF `.949/.105/.047`，VAT `.938/.102/.905`，是 VFM 时代前较强的纯 Transformer 路线。

**可借鉴。** query-conditioned attention 应在 token interaction 中发生，而不是只在末端加一张 mask。**缺点。** encoder-heavy，scene 复用与多人边际成本不如冻结 encoder；没有对象/语义、binding 或明确 OOD protocol。

### 4.18 Lin et al., *GazeHTA: End-to-End Gaze Target Detection with Head-Target Association*（arXiv 2024，preprint）

**Motivation 与故事。** 多数 pipeline 依赖外部 head detector，并且即使同时检测 heads/targets，也缺少显式可视化的连接。论文把 association map 当作一等输出。

**方法。** 用预训练 diffusion model 提取丰富 scene feature，重新注入 head feature，加强 head prior，并预测 head–target connection map，直接输出多个人–目标实例。

**结果与协议。** 论文声称在两个 standard datasets 上优于既有 GTE 与两个 diffusion baselines；因其 end-to-end association protocol 与公开复现状态不同，本文不把绝对数字混入 GT-head 主表。

**可借鉴。** 显式 connection map 是关系建模强近邻。**缺点。** preprint、diffusion feature 成本高；connection map 的可解释性不等于因果/反事实 query fidelity；head detection 与 target localization 误差仍耦合。

### 4.19 Tafasca et al., *Toward Semantic Gaze Target Detection*（NeurIPS 2024，peer-reviewed）

**Motivation 与故事。** pixel coordinate 对应用价值有限，人更关心“看的是人、杯子还是屏幕”。论文把 semantic target label 与 localization 联合，建立新 benchmark/protocol。

**方法。** 为 GazeFollow 构建 pseudo semantic annotations 与 target dictionary；网络联合预测 gaze heatmap 和 semantic class，并通过共享表示让定位与识别互助。

**结果与协议。** 论文报告在 GazeFollow localization 上刷新当时结果，semantic recognition 与强 baseline 竞争，同时参数减少约 `40%`；semantic Accuracy@1/@3 与普通 AUC/L2 属不同评价轴。

**可借鉴。** “目标是什么”是独立贡献轴；可用于区分同类 decoy 与真正 target。**缺点。** pseudo-label noise、类别词表和长尾；定位正确不保证语义正确，反之亦然；与 PixelGaze/VLM 路线已形成竞争簇。

### 4.20 Gupta et al., *MTGS: A Novel Framework for Multi-Person Temporal Gaze Following and Social Gaze Prediction*（NeurIPS 2024，peer-reviewed）

**Motivation 与故事。** GTE、looking-at-human、mutual gaze、shared attention 往往被不同模型分别处理；大多数 GTE 还静态、逐人。论文希望统一多人、时间和 social gaze。

**方法。** temporal Transformer 同时处理 frame tokens 与 person-specific tokens；VSGaze 统一多个 gaze/social datasets，补 head detections/tracks 和异构标签；多任务训练共同预测 target 与 social relations。

**结果与协议。** 论文在 GF、VAT、ChildPlay 与 UCO-LAEO 等数据上报告 multi-person gaze following/social gaze 的 SOTA；同时使用 AUC/Dist/Min Dist、AP@10/F1 等多套指标，不能压成单一 leaderboard 数字。

**可借鉴。** temporal/person tokens 与 shared-target/social labels 可成为 COTB 外部验证；事件级 evaluation 应分 fixation 与 switch。**缺点。** 统一数据带来异构 supervision、missing labels 和 domain composition confound；“时间 Transformer”本身已无新意。

### 4.21 Ryan et al., *Gaze-LLE: Gaze Target Estimation via Large-Scale Learned Encoders*（CVPR 2025，peer-reviewed）

**Motivation 与故事。** 过去方法不断增加 head encoder、scene encoder、depth/pose auxiliaries 和复杂 fusion；通用自监督 VFM 也许已把所需空间/语义信息编码在 scene feature 中。论文用最小 decoder 检验“复杂 pipeline 是否仍必要”。

**方法。** 冻结 DINOv2 ViT-B/L，整图编码一次；把 queried head bbox 变成 positional prompt 注入 scene tokens；3 层轻量 Transformer decoder 输出 heatmap/in-out。supplement 还比较 head token/cross-attention 与多人扩展。

**结果与协议。** ViT-B：GF `.956/.104/.045`、VAT `.933/.107/.897`；ViT-L：GF `.958/.099/.041`、VAT `.937/.103/.903`。scene encoder 占 >95% computation，supplement 报 1→10 people latency 约 `15ms→19ms`。

**可借鉴。** 它应是所有新 Gazelle 工作的默认强基线；scene-once 和轻 decoder 已被占据。**缺点。** 只用 final DINOv2 layer；head appearance 很弱；标准指标不检验 query binding；冻结 VFM 的 shortcut/shift 仍未充分暴露。

### 4.22 Dai et al., *GazeTarget360: Towards Gaze Target Estimation in 360-Degree for Robot Perception*（IROS 2025，peer-reviewed）

**Motivation 与故事。** 机器人/360° 场景中目标可能跨越常规前视视野，且 eye contact/robot interaction 改变 gaze interpretation；普通平面 GTE 的输入与应用边界太窄。

**方法。** 使用预训练视觉 encoder、head/person cues 与 multiscale decoder，同时处理 target localization 和机器人相关/360° 条件；强调小型可训练头与跨场景部署。

**结果与协议。** 常见标准复录为 GF `.957/.101`，VAT `.934/.103/.887`，trainable params 约 `1.9M`。其 360/robot setting 才是主要贡献，标准点指标未全面超过 Gaze-LLE。

**可借鉴。** 新 setting 能比千分位模块更有论文价值；机器人评测需要明确 input contract。**缺点。** 标准集收益有限；360/eye-contact 条件与通用 GTE 数据规模、协议可比性有限。

### 4.23 Liu/Guo et al., *Towards Pixel-Level Prediction for Gaze Following / PixelGaze*（arXiv 2024；ICLR 2026 submission，preprint/under review）

**Motivation 与故事。** 点和 heatmap 无法描述被注视对象的完整边界与类别；自然场景中 gaze target 应同时被定位、分割和识别。

**方法。** prompt-based VFM 以 head box 为初始 prompt，依次预测 FoV map、heatmap、segmentation mask 和 recognition；构建 72K images、270 categories 的 pixel-level 衍生 benchmark。后续版本名为 PixelGaze，指标/作者表述有更新。

**结果与协议。** 早期 GazeSeg 版本报告 Dice `.325`、top-5 recognition `71.7%`、GF AUC `.953`；PixelGaze OpenReview 版本报告 mIoU `34.9%`、recognition accuracy `45.1%`，并称 point-GTE SOTA。版本和指标定义不同，引用时必须锁定版本。

**可借鉴。** mask/semantic target 可辅助处理同类实例和 object extent。**缺点。** 衍生/pseudo annotations、版本漂移、长尾类别；FoV→heatmap→mask 的级联很重；object/segmentation 轴已不能作为无人做过的新意。

### 4.24 Miao et al., *Multi-view Gaze Target Estimation*（ICCV 2025，peer-reviewed）

**Motivation 与故事。** 单视图中 face/head 或 target 可能遮挡、出画、角度不佳；同步第二视图可提供更清晰 head appearance 和 scene geometry，但传统 3D reconstruction 每次推理太重。

**方法。** Head Information Aggregation (HIA) 融合两视图的 face appearance/几何；Uncertainty-based Gaze Selection (UGS) 选择更可靠输出；Epipolar-based Scene Attention (ESA) 沿极线聚合参考视图背景。MVGT 用 6 台标定相机采集 28 subjects、4 scenes。

**结果与协议。** 在 13,686 images 的 MVGT 上，多视图相对单视图显著改善，尤其当参考视图脸部清晰；还实现只用第二视图中的 person appearance 预测第一视图 target 的 cross-view task。主指标为该数据集下 AUC/L2 等，不能与 GF/VAT 单视图表直接混排。

**可借鉴。** uncertainty-based evidence selection 与 epipolar constraint 比“多层融合”更有机制。**缺点。** 标定多相机条件、场景/被试规模小；对普通单摄像头 GTE 不适用；它已经占据 multi-view 这一新 setting。

### 4.25 Yang & Lu, *GazeLLM: A Plug-and-Play Zero-Shot LLM Reasoning Framework for Boosting Gaze Target Detection*（Visual Intelligence 2026 online，peer-reviewed）

**Motivation 与故事。** 视觉模型定位准但高层语义弱，LLM/VLM 会推理“人可能在看什么”却难以精确落点；论文把二者拆成结构化 scene extraction、语言推理和视觉定位。

**方法。** off-the-shelf grounding detector 与 Depth Anything V2 把 heads/objects/category/depth/gaze direction 转为结构化 3D scene description；LLM 零样本推理候选 target，再把结果作为 plug-in 提升基础 gaze detector。

**结果与协议。** 论文报告在多个基础模型/benchmark 上零样本提升 gaze target detection；其核心指标是 plug-in 前后 AUC/Distance 的增量，而不是一个可独立横比的端到端模型。具体数字应按选定 base model 复录。

**可借鉴。** semantic reasoner 与 continuous localizer 分工；可用 LLM 生成困难语义候选。**缺点。** detector/depth/LLM 的成本与误差链很长；zero-shot reasoning 难以复现；隐私/延迟不适合实时多人。

### 4.26 Dai et al., *GazeMoE: Perception of Gaze Target with Mixture-of-Experts*（ICRA 2026，peer-reviewed）

**Motivation 与故事。** 不同图像需要不同 cue：有时 head orientation 主导，有时 scene semantics/geometry 主导；固定 decoder 容量与单一路径难以适配长尾条件。

**方法。** 冻结 DINOv2-L，使用 shared/routed Mixture-of-Experts decoder、class-balanced loss 和 task-specific augmentations；只训练约 `3.4M` 参数。

**结果与协议。** 作者报告 GF `.959/.101`，VAT `.939/.097/.917`，ChildPlay `.945 AUC/.106 L2/.994 AP`。它在 VAT AP/L2 上强，但不是所有 GF 指标的绝对最好。

**可借鉴。** input-adaptive routing 已有正式先例，任何新 router 都必须证明 per-person 而非 image-global 差异。**缺点。** expert specialization 是否有真实语义常难解释；MoE gain 可能来自容量/augmentation/class balance；没有 observer-binding evaluation。

### 4.27 Wang et al., *Enhancing Gaze Reasoning in Vision Foundation Models for Gaze Following*（arXiv 2026，preprint；HCLoRA）

**Motivation 与故事。** VFM 可能依赖显著对象/scene semantics，而忽略 head-specific direction，在 target 不符合语义先验时失败。论文把这种现象称为 semantic shortcut。

**方法。** 以 Gaze-LLE/DINOv2 为基础，只在 head-related tokens 上施加 gated LoRA；训练时加入 out-of-cone (OOC) penalty，抑制与 GT direction 明显不一致区域；构建 consistent/inconsistent split 分析 shortcut。

**结果与协议。** VAT overall：ViT-B `.9352 AUC/.1006 L2/.8988 AP`，ViT-L `.9387/.0951/.9068`。GF inconsistent subset 上，Gaze-LLE→HCLoRA→HCLoRA+OOC 的 Avg L2 为 `.1519→.1383→.1353`；该 subset 不能与 standard GF 主表横比。

**可借鉴。** 机制应在特意构造的 conflict split 上验证；local adapter 比全模型微调更可控。**缺点。** cone 用 GT 构造训练惩罚，测试时未显式输出 calibrated field；consistent/inconsistent split 的构造可能引入选择偏差；semantic shortcut 轴已被占据。

### 4.28 Cao et al., *Gaze Target Estimation Anywhere with Concepts*（CVPR 2026，peer-reviewed）

**Motivation 与故事。** 传统系统先检测 head 再把 box 交给 GTE；crowd/遮挡下前级错误会级联，用户也难以用“穿红衣的男孩”直接指定人物。论文把 GTE 变成 promptable subject-conditioned task。

**方法。** 文本 noun phrase 或视觉 point prompt 指定 subject；冻结/预训练 visual encoder 后接 transformer detector/projection，同时预测 head box、in/out 与 heatmap；构建从 GF/VAT/ChildPlay 转换并人工核验的 Gaze-Co/PGE 数据。

**结果与协议。** PGE 下 DINOv3-L text-all：GF `.958/.099/.050`，VAT `.928/.123/.879`，ChildPlay L2/AP `.098/.906`；CLIP-L 版本略弱。这里包含 subject identification error，不能与 GT-head Gaze-LLE 直接排名。

**可借鉴。** concept prompt 是更实用的 query contract。**缺点。** 约 870M、4×H100 训练；文本描述质量/歧义；promptable subject novelty 已被占据，COTB 不能只改成文本 query。

### 4.29 Miao et al., *OmniGF: A Dual-Branch Vision-Language Framework for Unified Gaze Following*（arXiv 2026，preprint）

**Motivation 与故事。** 文本分支擅长语义/social reasoning，却受离散坐标限制；连续视觉分支定位准却不会回答高层 gaze 语义。论文用双分支在一个 4B VLM 中统一多人空间、语义和社会 gaze。

**方法。** 结构化 language branch 为所有人生成 reasoning states/person anchors；continuous spatial branch 从 VLM dense hidden states 解码高分辨率 heatmap；cropped head embeddings 同时 grounding 多人。

**结果与协议。** 作者报告 GF Avg/Min L2 `.091/.040`，VAT L2/AP `.096/.923`，ChildPlay `.090/.996`；使用 Qwen3-VL-4B、LoRA 和 H100 80GB，不能与轻量 frozen-DINO head 做纯架构归因。

**可借鉴。** language 与 continuous localization 应分支；同场景一次处理多人已被明确覆盖。**缺点。** 大模型成本、训练数据与 prompt recipe；目前 preprint；标准指标无法隔离语言推理是否真正带来 binding/semantics。

### 4.30 Mi et al., *Multi-scale Object-Aware Gaze Estimation via Geometric Reasoning*（2026，reported ECCV acceptance / concurrent paper）

**Motivation 与故事。** pixel heatmap 忽略对象结构；深层 VFM 语义强但精细位置不足；相似对象会竞争。论文组合 object tokens、多尺度 DINOv3 与 gaze geometry。

**方法。** offline YOLO11x + SAM2 产生 object masks/tokens；head crop 预测 2D gaze direction并生成固定 `120°` FoV；冻结 DINOv3-L，使用 shallow/middle/deep static residual fusion，再做 object-aware cross-token fusion。

**结果与协议。** 作者报告 GF `.961/.094/.038`，VAT `.948/.095/.923`，ChildPlay `.987/.084/.990`，GOO-Real `.977/.092`。这些数字未由本项目复现；当前公开材料中组件/骨干消融存在若干完全重复数值且缺少部分单项对照，宜标为 paper-reported。

**可借鉴。** object representation + predicted direction 是比 bbox gate 更强的 baseline。**缺点。** 外部 detector/segmenter 成本未完整计入；static multilevel、object、FoV 三轴都很拥挤；近似 fixed cone 处理 uncertainty 有限。

### 4.31 Ye et al., *PaGE: Towards Practical Human-Level Gaze Target Estimation*（arXiv 2026，preprint）

**Motivation 与故事。** 单一 scene encoder/轻 decoder 仍缺局部 head/eye spatial reasoning；只换 DINOv3 不足。论文恢复强 head branch、双向 head-scene interaction，并把大模型能力蒸馏到学生。

**方法。** scene/head 双 DINOv3 branches、统一 2D coordinate encoding、bidirectional cross-attention；混合 GF/VAT/ChildPlay 监督，全模型 SFT；ViT-H+ teacher 再用 1.17M unlabeled images 蒸馏 ViT-S/B students。

**结果与协议。** ViT-B distilled：GF `.9660/.0814/.0295`，VAT `.9688/.0677/.9450`，ChildPlay `.0697/.9969`；ViT-H+：GF `.9659/.0804/.0288`，VAT `.9719/.0643/.9509`。但 GF-only 版本仅 `.9595/.0958/.0391`，显示 headline 大幅收益同时来自混合数据与 distillation recipe。

**可借鉴。** head appearance 和 training data 是强上限因素；排行榜必须分 dataset-only 与 mixed/distilled。**缺点。** H100、最高约 `2373.6 GFLOPs`、百万无标签数据；与单卡轻量路线不公平；方法归因不能只看最终表。

### 4.32 Fan et al., *Inferring Shared Attention in Social Scene Videos*（CVPR 2018，peer-reviewed，相关任务）

**Motivation 与故事。** 社会场景中更高层的问题不是每人分别看哪里，而是一群人是否共享同一 attention target。论文建立 VideoCoAtt 并直接预测 shared target 的存在与位置。

**方法。** 汇聚多人的 gaze/head cues 与 scene proposals，判断是否存在 shared attention，并在候选区域中定位共同 target。

**结果与协议。** VideoCoAtt 含 380 sequences、约 492K frames；约 349K/139K/3K frames 分别有 0/1/>1 shared targets。指标含 existence accuracy、proposal ROC/AUC 和 raw-pixel L2，均不能与 standard person-specific GTE 直接横比。

**可借鉴。** shared-target clustering 是多人 GTE 中必须保留的正例结构。**缺点。** shared attention ≠ 每人的完整 gaze；负例约 71%，accuracy 易虚高；图外/遮挡共同目标可能被当成“不存在”。

### 4.33 Li et al., *Looking Here or There? Gaze Following in 360-Degree Images*（ICCV 2021，peer-reviewed）

**Motivation 与故事。** 常规窄 FoV 把大量 target 标为 out-of-frame，无法区分“看向画面边界外”与真正不可见；360° 全景可以覆盖完整方向空间。

**方法。** 在 equirectangular/spherical 表示上建模 head direction 与 panoramic scene，构建约 10K gaze pairs 的 GazeFollow360，并以球面位置评价。

**结果与协议。** 论文报告 360° gaze following 相对 planar/adapted baselines 的改善；项目协议还给出 64×64 heatmap AUC。关键指标应是 spherical/great-circle distance，不能把 equirectangular pixel L2 与 GF normalized L2 相加。

**可借鉴。** 任务几何必须匹配成像空间；“无 out-of-frame”可以把可见性和 FoV 拆开。**缺点。** 全景畸变、数据量小、head resolution 低；360 setting 已被占据，且仍未解决遮挡/不可判 gaze。

## 5. 设计空间占位图：哪些 headline 已经不能再用

| 设计轴 | 代表工作 | 对新论文的含义 |
|---|---|---|
| head + scene 双流 | Recasens 2015；Chong 2018/2020 | 基础范式，不是创新 |
| temporal GTE | Recasens 2017；Chong 2020；MTGS 2024 | “加 temporal Transformer”不够 |
| depth / 3D / FoV | Fang 2021；ESCNet 2022；ChildPlay 2023；Horanyi 2023 | “加 gaze cone/depth”已拥挤，必须是新可见性/不确定性问题 |
| end-to-end multi-person | HGTTR 2022；GazeHTA 2024 | 不能声称首次多人 head–target detection/association |
| scene-once controlled person query | Sharingan 2024；Gaze-LLE 2025；OmniGF 2026 | 多人共享 scene computation 已被占据 |
| gaze object/semantic/mask | GaTector 2022；Object-aware 2023；Semantic Gaze 2024；PixelGaze | object/semantic/pixel target 已成独立簇 |
| frozen VFM + tiny decoder | Gaze-LLE 2025 | Gazelle 基座本身不是 novelty |
| DINOv3 + multiscale hierarchy | Multi-scale Object-Aware 2026；当前 GazeSpot | “DINOv3 多层提分”只能当技术底座 |
| VFM local adaptation/shortcut | HCLoRA 2026 | semantic shortcut + local LoRA 已有近邻 |
| adaptive expert routing | GazeMoE 2026 | input-adaptive MoE 已被占据 |
| promptable subject | GazeAnywhere 2026 | text/point 指定人物已被占据 |
| VLM/LLM reasoning | GazeLLM；OmniGF | “接一个 LLM”不是新问题 |
| multi-view | MVGT 2025 | multi-camera GTE 已被占据 |
| large-data distillation | PaGE 2026 | 轻学生 + 大 teacher/data recipe 已有强基线 |

## 6. 从其他 CV / gaze 方向可迁移的机制

### 6.1 Multi-person pose / keypoint grouping → observer–target binding

CenterGroup、Group Pose 等方法不只检测 keypoints，还直接学习“哪个 joint 属于哪个 person center”。迁移到 GTE 时，scene targets 类似共享 keypoints，person queries 类似 centers；关键不是再加 attention，而是建立 query–target assignment matrix，并用同帧交换配对作 hard negatives。

**最有希望的落点：COTB。** 本轮对 `counterfactual gaze following`、`query swap gaze target`、`observer-target binding`、`permutation gaze association` 的检索，没有发现同时采用**同帧 query switch + shared-target-aware swapped objective + 独立 binding metric**的直接重合工作。Sharingan、HGTTR、GazeHTA、Object-aware 是必须对照的局部近邻。结论只能写“no directly overlapping work retrieved under these keywords”。

### 6.2 HOI / relation detection → 同类目标与人物–对象对应

HOI detection 的难点不是有没有 person/object features，而是同图大量合法但错误的 person–object pairs。它常用 pairwise scoring、hard-negative mining、bipartite matching 和关系 token。GTE 可以把同帧其他人的 target、同类 object instances 作为 hard negatives，直接优化正确 observer–target pair。

**边界。** Object-aware GTE 已建立 head–object association；因此新意必须是反事实 query fidelity/交换矩阵，而非“首次使用关系”。

### 6.3 Amodal segmentation / depth ordering → observability-aware GTE

传统 occlusion reasoning 区分 visible mask、amodal object、front/back ordering。ChildPlay 已证明现实 gaze 包含 inside-visible、outside、shift、occluded、closed-eyes、uncertain 等状态；现有标准二元 in/out 把它们压扁。

**可形成的新问题。** 同时预测 `visible target / occluded in-frame / outside-FoV / indeterminate`，并在 occluded 情况输出 visible-surface 与 amodal-intended 两个 target distributions。一个更具体的方案是用 MVGT 第二视角只在训练时作 privileged teacher，测试仍单视图。MVGT 本身是 test-time 双视图 view selection，因而 setting 不同。本轮未检索到把四类 observability contract、amodal target 与 train-only cross-view supervision 统一的直接工作；但完全遮挡时意图可能本来就不可辨，且标注成本高。

### 6.4 Causal robustness / counterfactual augmentation → semantic-decoy intervention

鲁棒 CV 不只测自然 accuracy，而是保持 label 不变地编辑背景/shortcut，观察输出是否变化。GTE 可在保持 head 与 GT target 不变时编辑非目标显著物、同类 decoy 或他人 target，测 heatmap JS divergence、peak shift 与 GT mass。

**边界。** HCLoRA 已明确研究 semantic shortcut，Multi-scale Object-Aware 也研究 object competition。可守住的只能是**受控成对 intervention protocol**，而不是首次发现 distractor。最大风险是编辑未必真正 label-preserving。

### 6.5 Geometric equivariance / crop consistency → visibility transition contract

检测/分割常用几何等变性：输入坐标变换后输出应同步变换。对 GTE，可以构造 target-preserving crop 与 target-excluding crop：前者要求 heatmap 等变，后者要求 in→out 并避免在剩余 salient object 上 hallucinate。

**边界。** random crop 与 in/out distribution 已被 PDP/GazeMoE 等使用。潜在新意是三类成对响应：target-preserving 时坐标等变，target-removal 时 in→out 且画内 mass 消失，irrelevant-decoy edit 时预测稳定。首轮只用可精确映射的 crop；synthetic crop 可能泄漏边界/尺度 shortcut，必须在自然 VAT transitions 验证。

### 6.6 Calibration / selective prediction → 何时不应给出单点

医学影像与自动驾驶常报告 calibration、risk–coverage、conformal set，而 GTE 多数只给 AUC/L2。多标注、occluded/uncertain、out-of-frame 天然需要分布或集合预测。这里不能声称“首个 gaze uncertainty”：MVGT 已用 aleatoric uncertainty 选视角，RayGazeFM 输出 probabilistic gaze ray，appearance-based gaze 也已有 uncertainty calibration。相对更空的是 **single-view scene GTE 的 abstention、structured spatial prediction set 与跨域 coverage protocol**。

**本项目证据。** final heatmap confidence 的 failure AUROC 已约为 GF-ID `.808`、VAT shift `.832`、GOO-Real `.659`，50% coverage 的 L2 可从全量约 `.095/.124/.187` 降至 `.048/.060/.159`；但 hierarchy-disagreement 的增量 AUROC 为 `-.0007/-.0012/+.0013`，没有额外价值。因此“用多层分歧做 uncertainty”已经 No-Go。若重启，只能做 outside token + spatial set + abstention，并以 ECE/Brier/NLL/AURC/worst-group coverage 为主。

### 6.7 Change-point / event detection → fixation-state temporal GTE

传统 tracking 和时序分割把序列建模为稳定状态 + change points，而非全程平滑。Gaze video 可显式输出 fixation/saccade/target switch，分别优化 stable jitter 与 switch delay。

**边界。** MTGS 已覆盖 temporal multi-person；当前项目 transition event 数和标注有效性不足，普通 EMA/Kalman 还会以 lag 换平滑。现阶段不优先。

## 7. 仍值得做的三个 gap（按优先级）

### Gap A：Counterfactual Observer–Target Binding（首选）

- **问题：** 同一 scene 切换 queried person，输出是否随正确 observer 切换？
- **为什么不是已有多人工作：** controlled token、end-to-end association 已有，但主流 loss/metric 仍未把 diagonal-vs-swapped correspondence 单独评价。
- **最小方法：** shared-target clustering + pairwise permutation contrastive loss；DINOv3 multilevel 只作 base。
- **一天证伪：** 强 baselines 在 far-separated targets 上若 swap error <5%，或控制 L2 后现象消失，立即停止。

### Gap B：Observability-aware / Amodal GTE（第二）

- **问题：** visible、occluded、outside-FoV、shift/uncertain 不应被压成同一个 binary label。
- **方法候选：** state head + visible/amodal dual distributions + depth ordering + calibrated abstention；MVGT 第二视角仅作 train-time privileged teacher。
- **主要成本：** 需要 ChildPlay 状态或新标注；若没有可靠 occluded target GT，论文会退化成分类器。

### Gap C：Counterfactual Visibility Transition（第三）

- **问题：** target 被 crop 出画后，模型是否错误转向剩余 saliency？
- **方法候选：** preserving/removal/irrelevant-decoy 三类 pairs + equivariance/visibility-flip/hallucination-mass/invariance losses。
- **主要风险：** synthetic boundary artifacts；必须用自然 video transitions 外证。

不优先：再做 multiscale/FPN、另一个 gaze cone、object tokens、MoE/router、VLM prompt、generic temporal Transformer、hierarchy-disagreement uncertainty。

## 8. 对 Gazelle + DINOv3 + 多层特征的具体落地

建议把现有技术栈冻结为：

1. frozen DINOv3 scene encoder，scene 只算一次；
2. 先用可复现的 static multilevel concat 或 shallow+deep projection；
3. head ROI appearance + bbox geometry 形成 person query；
4. 轻量 query-conditioned decoder 输出 heatmap/in-out；
5. novelty 仅来自 COTB protocol/objective/metrics。

不要在第一版同时引入新 router、GGSF、FoV、object detector、VLM。这样无论 COTB 是否成功，都能清楚归因：

- 若 binding 提升且标准指标不退化，证明关系监督有效；
- 若只改善自定义 metric，论文最多是 evaluation contribution；
- 若强 baseline 没有 binding gap，立即 No-Go，不继续堆模块。

## 9. RQ 最终回答

**RQ1。** 领域已从双流 heatmap 回归演化到 geometry/object、Transformer 多人、VFM/VLM 与新输入/输出协议；标准点指标正在饱和，真正进展越来越来自 task contract、数据与评测，而非单一 fusion block。

**RQ2。** DINOv3、多层、object/FoV、多人共享 scene、MoE、promptable subject、VLM reasoning、multi-view 和 distillation 均已有直接近邻。GazeSpot 不能再以 GGSF/SASA 机制和“首个多层 DINOv3”转投。

**RQ3。** 最值得迁移的是 pose/HOI 中的 assignment 与 hard-negative 思路，形成 COTB；其次是 amodal/occlusion 的 observability contract。两者都应先做数据有效性与强 baseline 证伪，不能把“没搜到”写成绝对首创。

## 10. 参考文献索引

1. Recasens et al. *Where Are They Looking?* NeurIPS, 2015.
2. Recasens et al. *Following Gaze in Video.* ICCV, 2017.
3. Chong et al. *Connecting Gaze, Scene, and Attention.* ECCV, 2018.
4. Lian et al. *Believe It or Not, We Know What You Are Looking At!* ACCV, 2018.
5. Fan et al. *Inferring Shared Attention in Social Scene Videos.* CVPR, 2018.
6. Chong et al. *Detecting Attended Visual Targets in Video.* CVPR, 2020.
7. Fang et al. *Dual Attention Guided Gaze Target Detection in the Wild.* CVPR, 2021.
8. Li et al. *Looking Here or There? Gaze Following in 360-Degree Images.* ICCV, 2021.
9. Bao et al. *ESCNet: Gaze Target Detection with the Understanding of 3D Scenes.* CVPR, 2022.
10. Jin et al. *Depth-Aware Gaze-Following via Auxiliary Networks for Robotics.* EAAI, 2022.
11. Tu et al. *End-to-End Human-Gaze-Target Detection with Transformers.* CVPR, 2022.
12. Wang et al. *GaTector: A Unified Framework for Gaze Object Prediction.* CVPR, 2022.
13. Tonini et al. *Multimodal Across Domains Gaze Target Detection.* ICMI, 2022.
14. Miao et al. *Patch-Level Gaze Distribution Prediction for Gaze Following.* WACV, 2023.
15. Tafasca et al. *ChildPlay: A New Benchmark for Understanding Children's Gaze Behaviour.* ICCV, 2023.
16. Tonini et al. *Object-aware Gaze Target Detection.* ICCV, 2023.
17. Horanyi et al. *Where Are They Looking in the 3D Space?* CVPRW, 2023.
18. Tafasca et al. *Sharingan: A Transformer Architecture for Multi-Person Gaze Following.* CVPR, 2024.
19. Song et al. *ViTGaze: Gaze Following with Interaction Features in Vision Transformers.* Visual Intelligence, 2024.
20. Lin et al. *GazeHTA: End-to-End Gaze Target Detection with Head-Target Association.* arXiv:2404.10718, 2024.
21. Tafasca et al. *Toward Semantic Gaze Target Detection.* NeurIPS, 2024.
22. Gupta et al. *MTGS: A Novel Framework for Multi-Person Temporal Gaze Following and Social Gaze Prediction.* NeurIPS, 2024.
23. Ryan et al. *Gaze-LLE: Gaze Target Estimation via Large-Scale Learned Encoders.* CVPR, 2025; arXiv:2412.09586.
24. Dai et al. *GazeTarget360: Towards Gaze Target Estimation in 360-Degree for Robot Perception.* IROS, 2025.
25. Miao et al. *Multi-view Gaze Target Estimation.* ICCV, 2025; arXiv:2508.05857.
26. Liu/Guo et al. *Towards Pixel-Level Prediction for Gaze Following / PixelGaze.* arXiv:2412.00309 / OpenReview, 2024–2026.
27. Yang & Lu. *GazeLLM: A Plug-and-Play Zero-Shot LLM Reasoning Framework for Boosting Gaze Target Detection.* Visual Intelligence, 2026 online.
28. Dai et al. *GazeMoE: Perception of Gaze Target with Mixture-of-Experts.* ICRA, 2026; arXiv:2603.06256.
29. Wang et al. *Enhancing Gaze Reasoning in Vision Foundation Models for Gaze Following.* arXiv:2605.22607, 2026.
30. Cao et al. *Gaze Target Estimation Anywhere with Concepts.* CVPR, 2026.
31. Miao et al. *OmniGF: A Dual-Branch Vision-Language Framework for Unified Gaze Following.* arXiv:2605.26399, 2026.
32. Mi et al. *Multi-scale Object-Aware Gaze Estimation via Geometric Reasoning.* arXiv:2606.29334, concurrent paper, 2026.
33. Ye et al. *PaGE: Towards Practical Human-Level Gaze Target Estimation.* arXiv:2607.04860, 2026.
