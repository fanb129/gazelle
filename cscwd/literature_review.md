# 视频注视目标估计相关论文解读

整理日期：2026-11-10。阅读范围：论文正文、可访问的补充材料，以及注明的作者官方模型发布。

2026-11-17 补充本轮方向讨论已核验的机制与实现边界，见文末；当前研究方案改为 [因果 heatmap/In/Out 联合修正](temporal_joint_refinement.md)，执行按 `implementation_plan.md` 的 R1–R6。本次文档更新不新增文献结果数值或本项目实验。

本文以第三人称注视目标估计为主，另纳入一篇与因果记忆设计直接相关的第一人称论文。每篇均区分推理输入、模型输出和训练监督；表中结果是文献报告值，不是本项目复现值，也不是统一协议下重新跑出的排行榜。

共同的数据集标注与指标解释见[数据集与评价指标速查](datasets_and_metrics.md)。

<a id="p01-detecting-attended-visual-targets-in-video"></a>
## P01 Detecting Attended Visual Targets in Video

- 作者：Eunji Chong, Yongxin Wang, Nataniel Ruiz, James M. Rehg。
- 发表：CVPR 2020。
- [论文主页](https://openaccess.thecvf.com/content_CVPR_2020/html/Chong_Detecting_Attended_Visual_Targets_in_Video_CVPR_2020_paper.html)
- [论文 PDF](https://openaccess.thecvf.com/content_CVPR_2020/papers/Chong_Detecting_Attended_Visual_Targets_in_Video_CVPR_2020_paper.pdf)
- [官方代码及数据入口](https://github.com/ejcgt/attention-target-detection)
- 重点位置：第4节、图3、表1—4。

### 要解决的问题与方法

给定一个人在视频中的头部轨迹，逐帧确定其注视落点，并识别视线落在画外的情况。

场景 CNN 与头部 CNN 分别编码图像和头部裁剪。头部特征与头部位置生成空间注意力，调节场景特征；融合后的特征进入 ConvLSTM，解码得到注视热图。画内分数用于调节热图响应。先在 GazeFollow 学习空间定位，再在 VAT 上训练视频模型。

### 输入、输出与监督

| 类别 | 内容 |
|---|---|
| 推理输入 | 连续 RGB 帧；指定人物每帧的头部框。头部裁剪和位置掩码由图像及框派生 |
| 输入单位 | 一个人物轨迹；场景中多人的注视需分别处理 |
| 论文配置 | 场景/头部图像缩放到 224×224；时序模块为两层 ConvLSTM |
| 预测输出 | 每帧注视热图；每帧画内概率 α。热图 argmax 给出二维落点 |
| 训练监督 | 画内落点对应的高斯热图；画内/画外二值标签。热图使用 MSE，状态使用分类损失 |
| 不属于推理输入 | 真实注视点、真实画内外标签 |

### 数据集、指标与结果

| 数据集/任务 | 标注或输出对象 | 指标 | 代表性结果与来源 |
|---|---|---|---|
| GazeFollow，静态定位 | 同一人物的多个目标点标注 | AUC ↑、Avg Dist ↓、Min Dist ↓ | 空间模型：0.921 / 0.137 / 0.077，表1 |
| VAT，视频定位与状态 | 逐帧目标点、画内外标签 | AUC ↑、归一化 L2 ↓、In/Out AP ↑ | 完整模型：0.860 / 0.134 / 0.853，表2 |
| VAT，时序消融 | 同上 | 同上 | 去掉时序：0.854 / 0.147 / 0.848，表2 |
| VideoCoAtt，共同注视 | 帧级共同注视与目标区域 | 共同注视判定指标 ↑、像素 L2 ↓ | 83.3% / 57，表4；距离不是 VAT 的归一化 L2 |
| 幼儿社会注视扩展实验 | 从玩具转向检查者面部的事件 | Precision ↑、Recall ↑ | RGB+预测热图的 ResNet/随机森林方案：0.708 / 0.759，表3 |

幼儿实验是独立的下游事件识别实验，不是 VAT 的分类结果。该私有样本不作为当前四天实验的必需资源。

### 对当前选题的解读

普通循环记忆与画内外分类已有清楚的基线。新方法应与同一基础模型上的普通时序模块比较，说明选择性历史使用的额外作用。该论文的社会注视扩展也表明，逐帧热图之外可以有事件层面的评价，但不能把不同任务的数值混为同一指标。

<a id="p02-patch-level-gaze-distribution-prediction-for-gaze-following"></a>
## P02 Patch-Level Gaze Distribution Prediction for Gaze Following

- 简称：PDP；作者：Qiaomu Miao, Minh Hoai, Dimitris Samaras。
- 发表：WACV 2023。
- [论文 PDF](https://openaccess.thecvf.com/content/WACV2023/papers/Miao_Patch-Level_Gaze_Distribution_Prediction_for_Gaze_Following_WACV_2023_paper.pdf)
- [作者提供的 PDF](https://www3.cs.stonybrook.edu/~minhhoai/papers/GazeFollowing-WACV23.pdf)
- [arXiv 入口](https://arxiv.org/abs/2211.11062)
- 重点位置：第3节、图3—5、表1—3。

### 要解决的问题与方法

单一高斯热图的像素回归不能充分表达标注分歧，独立的画内外分支也割裂了两项任务。

PDP 将特征网格变为 inside tokens，增加一个 outside token；先做空间注意力，视频版本再进行时间注意力。网络保留精细热图头，并预测含画外类别的粗粒度分布。高斯热图经 patch 聚合、归一化构造分布监督；联合热图 MSE 和分布 KL 损失。

### 输入、输出与监督

| 类别 | 内容 |
|---|---|
| 推理输入 | 场景 RGB、指定人物头部框/位置掩码及头部裁剪；完整配置还输入估计深度 |
| 输入单位 | 一个人物对应的单帧或视频片段；不同时联合预测场景中所有人物 |
| 精细输出 | 二维注视热图，argmax 得到目标点 |
| 粗粒度输出 | K 个画内 patch 与一个画外类别构成的 K+1 维分布，K 为特征网格大小 |
| 状态输出 | 由粗分布中的画内/画外质量得到，而非另设完全独立的二分类头 |
| 训练监督 | 注视点生成的高斯热图、画内外标签，以及由其派生的 patch 分布 |
| 时序边界 | 单图推理移除时间注意力；官方发布视频实现的时间注意力没有 causal mask，窗口各帧互相访问，不能作为严格因果配置 |

### 数据集、指标与结果

| 数据集 | 配置 | AUC ↑ | Avg Dist / L2 ↓ | Min Dist ↓ | In/Out AP ↑ |
|---|---|---:|---:|---:|---:|
| GazeFollow | 完整空间模型，有深度 | 0.934 | 0.123 | 0.065 | 不作为该测试集的标准报告项 |
| GazeFollow | 无深度 | 0.928 | 0.131 | 0.072 | — |
| VAT | 完整视频模型，有深度 | 0.917 | 0.109 | — | 0.908 |
| VAT | 无深度 | 0.907 | 0.116 | — | 0.881 |
| VAT | 去掉时间注意力 | 0.914 | 0.111 | — | 0.906 |

来源：原文表1—3。GazeFollow 的距离列为 Avg Dist，VAT 为单标注归一化 L2。

### 对当前选题的解读

主要贡献是分布表示及监督方式，时间注意力增益较小。outside token、画内位置与画外状态联合分布均不宜单独当成新贡献。完整配置使用额外深度，主表必须与纯 RGB 配置分开。

实现边界依据：[官方时间注意力调用](https://github.com/qiaomu-miao/gazefollowing_pdp/blob/master/pdp_model.py#L385)。

<a id="p03-mtgs"></a>
## P03 MTGS

- 全名：MTGS: A Novel Framework for Multi-Person Temporal Gaze Following and Social Gaze Prediction。
- 作者：Anshul Gupta, Samy Tafasca, Arya Farkhondeh, Pierre Vuillecard, Jean-Marc Odobez。
- 发表：NeurIPS 2024。
- [正式论文入口](https://proceedings.neurips.cc/paper_files/paper/2024/hash/1caf09c9f4e6b0150b06a07e77f2710c-Abstract-Conference.html)
- [正式论文 PDF，含补充材料](https://www.proceedings.com/content/079/079017-0500open.pdf)
- [作者 PDF](https://www.idiap.ch/~odobez/publications/GuptaTafascaFarkhondehVuillecardOdobez-NEURIPS-2024-MTGazeSocial.pdf)
- [官方代码与后续 DINO 更新](https://github.com/idiap/MTGS)
- 重点位置：第3节、第4.3节、表2—3、补充材料 D。

### 要解决的问题与方法

统一多人的注视目标估计与社会注视预测。头部外观先经 Temporal Gaze Processor 建模，再与框位置组成 person token；Interaction Module 进行人物—场景双向交互，以及人物之间和同一人物跨时间的交互。预测模块输出热图、状态与人物对关系。训练包含热图、方向、状态和社会注视损失。

### 输入、输出与监督

| 类别 | 内容 |
|---|---|
| 推理输入 | T 帧场景；每个人的头部框轨迹与头部裁剪；需要跨帧稳定人物身份 |
| 论文时序设置 | T=5，时间步长3；训练最多采样4个人，测试处理实际人数 |
| 人物级输出 | 每人每帧的注视热图、画内外分数；辅助二维注视向量 |
| 关系级输出 | 有向 LAH 分数、由其推导的 LAEO，以及人物对 SA 分数 |
| 训练监督 | 注视点、高斯热图、由头部与目标点派生的方向、画内外标签、人物对关系标签 |
| 时序边界 | 官方发布中心窗口实现包含当前帧之前和之后的帧，时间注意力无 causal mask；不将其直接作为严格在线配置 |

### 数据集对应的指标与结果

| 数据集 | 实验角色 | 报告指标 | 结果例子 |
|---|---|---|---|
| GazeFollow | 静态模型训练与初始化 | AUC、Avg/Min Dist | 原文表3：0.929 / 0.116 / 0.059 |
| VAT | 视频定位与画内外判断 | Dist、In/Out AP | 原文表3，单数据集微调：0.105 / 0.869 |
| ChildPlay | 儿童交互域微调 | Dist、In/Out AP | 表3：GF 初始化后微调为0.117 / 0.994；VSGaze 初始化后微调为0.113 / 0.993 |
| VSGaze | 多数据集、多任务联合训练 | Dist、In/Out AP、LAH F1、LAEO F1、SA AP | 原文表2直接关系解码：0.107 / 0.940 / 0.795 / 0.590 / 0.576 |
| UCO-LAEO | 相互注视微调评价 | 论文表3报告 Dist、LAEO AP | VSGaze 初始化后微调：0.019 / 0.974 |
| VideoCoAtt，作为 VSGaze 组成部分 | 共同注意及关系评价 | 人物对 SA AP；相关人物级指标见补充表 | 不能用旧任务的帧级共同注视判定数值代替 SA AP |

VSGaze 整合、扩展 VAT、ChildPlay、VideoCoAtt、UCO-LAEO 的人物轨迹和任务标注；它不是一套完全独立拍摄的数据。GF 是另外的静态预训练来源。

LAH 的 F1 要求正确识别目标人物，不能只判断“是否看着某个人”；SA AP 的评价单位是人物对。原文中直接关系解码与后处理结果分开报告，不能取各自最好的列拼成一个配置。

### 后续官方 DINO 版本

官方仓库还发布了冻结 DINOv2 的后续模型。以下数值来自仓库，不是原始论文表3：

| VAT 配置，448×448 | Dist ↓ | In/Out AP ↑ |
|---|---:|---:|
| MTGS-DINO-static | 0.096 | 0.878 |
| MTGS-DINO-temporal | 0.096 | 0.888 |

仓库还列出原模型的另一组数值，与论文表3存在差异。后续正式比较应固定论文版或官方发布版，并记录具体配置、数据和 checkpoint；不能默认所有称作 MTGS 的结果可互换。

### 对当前选题的解读

人物级时序表示是最直接的参考。官方 DINO 配置的例子说明，时间信息可能主要改善状态判断，而非整体定位误差。原文补充材料也讨论了视频冗余及关键变化片段占比较小的问题，支持进一步做按事件分组的分析。

实现边界依据：[官方中心窗口采样](https://github.com/idiap/MTGS/blob/main/mtgs/datasets/videoattentiontarget_temporal.py#L178)、[Attention 实现](https://github.com/idiap/MTGS/blob/main/mtgs/networks/mtgs_net.py#L777)。该说明针对此发布配置，不泛指所有可能的 MTGS 变体。

<a id="p04-probabilistic-spatiotemporal-transformer"></a>
## P04 Probabilistic Spatiotemporal Transformer

- 中文题目：基于时空 Transformer 的注视目标检测概率模型。
- 作者：王子千、高旭杰、张俊晨、李顼、孙永宣。
- 发表：《微电子学与计算机》2026，43(2)：72—83。
- DOI：10.19304/J.ISSN1000-7180.2024.0851。
- [期刊论文主页](https://mc.spacejournal.cn/article/doi/10.19304/J.ISSN1000-7180.2024.0851)
- [期刊 PDF](https://mc.spacejournal.cn/cn/article/pdf/preview/10.19304/J.ISSN1000-7180.2024.0851.pdf)
- 重点位置：第3节、图1、表1—5。本文未核验可直接使用的作者代码入口。

### 要解决的问题与方法

学习视频注视关系与过去—未来语义的连续性。以 ResNet-50 提取帧特征，空间 Transformer 产生人物与注视关系查询；时空可变形注意力融合历史和当前特征。通过未来特征重建及概率分布约束进一步学习连续变化，联合基础预测、时间变化及重建/分布损失。

### 输入、输出与监督

| 类别 | 内容 |
|---|---|
| 输入 | 视频帧；原文第3节写明当前帧、前4帧及后一帧，图1有 t+1 分支 |
| 人物条件 | 属于同时处理人物检测和注视关系的框架；不能默认等同于输入真实人物头部框的设置 |
| 论文配置 | 帧输入224×224；ResNet-50；固定数量的注视关系查询，论文设为20 |
| 输出 | 查询对应的头部标签、头部热图、画内外标签及注视热图；目标热图为64×64 |
| 训练监督 | 人物头部与注视标注；时间变化监督；未来特征重建及分布约束 |
| 待核实边界 | 未来帧分支究竟仅用于训练，还是测试预测也依赖未来帧，须在复现时进一步确认 |

论文表4又以“4个参考帧、序列长度5”描述主设置。整理时保留这个与第3节未来帧描述之间的核验项，不据此断言其已实现严格在线推理。

### 数据集、指标与结果

| 数据集 | 标注与任务 | 指标 | 报告值与条件 |
|---|---|---|---|
| VAT | 人物头部、注视点、画内外 | AUC ↑、归一化 L2 ↓、AP ↑ | 表1 Ours 位于 ExHead 列：0.937 / 0.095 / 0.946 |
| VideoCoAtt | 共同注视人物及目标框 | 共同注视指标 A ↑、像素 L2 ↓ | 表2：98.7 / 32；A 不是热图 AUC |

表1区分 HeadGT 与 ExHead。与现有 Gaze-LLE/GazeSpot 比较时，应写清人物框来源与匹配方式。论文未在上述实验中报告 GazeFollow 或 ChildPlay 的新模型结果，不补填这些数据集的数值。

### 对当前选题的解读

“时间一致性”“变化路径”“概率性过去—未来建模”已有具体工作。可能的差异应落在严格因果条件、选择性历史更新和切换延迟，并通过实验确认。额外人物检测与未来信息也使它不适合作为不加说明的同设置对照。

<a id="p05-gaze-lle"></a>
## P05 Gaze-LLE

- 全名：Gaze-LLE: Gaze Target Estimation via Large-Scale Learned Encoders。
- 作者：Fiona Ryan, Ajay Bati, Sangmin Lee, Daniel Bolya, Judy Hoffman, James M. Rehg。
- 发表：CVPR 2025。
- [论文 PDF](https://openaccess.thecvf.com/content/CVPR2025/papers/Ryan_Gaze-LLE_Gaze_Target_Estimation_via_Large-Scale_Learned_Encoders_CVPR_2025_paper.pdf)
- [arXiv 入口](https://arxiv.org/abs/2412.09586)
- [官方代码](https://github.com/fkryan/gazelle)
- 重点位置：第3节、第4节、表3—5。

### 要解决的问题与方法

利用冻结的 DINOv2 场景编码器替代复杂多分支网络。头部位置作为人物提示加入场景 tokens，轻量 Transformer 解码注视热图；附加状态 token 预测画内概率。原模型没有跨帧模块，在 VAT 上也是单帧预测。

### 输入、输出与监督

| 类别 | 内容 |
|---|---|
| 推理输入 | 单帧 RGB；指定人物的头部框，可为同一图像的多个人物分别解码 |
| 不需要的外部输入 | 独立头部图像编码分支、深度图、姿态图、目标分割 |
| 论文配置 | RGB 448×448，DINOv2 patch size14，特征网格32×32；解码维度256、3层 Transformer |
| 输出 | 每人64×64注视热图；适用配置的画内概率；二维点由热图 argmax 得到 |
| 训练监督 | 画内目标高斯热图，及 VAT/ChildPlay 的画内外标签；联合 BCE 损失 |
| 与本项目区别 | 原文骨干是 DINOv2；本项目的 DINOv3、多层融合等属于后续实现，结果不能直接冠以原始 Gaze-LLE 配置 |

### 数据集指标与原论文结果

| 数据集/设置 | 模型 | 指标顺序 | 原文报告值 |
|---|---|---|---|
| GazeFollow，表3 | ViT-B | AUC / Avg L2 / Min L2 | 0.956 / 0.104 / 0.045 |
| GazeFollow，表3 | ViT-L | 同上 | 0.958 / 0.099 / 0.041 |
| VAT，数据集内微调，表3 | ViT-B | AUC / L2 / In/Out AP | 0.933 / 0.107 / 0.897 |
| VAT，数据集内微调，表3 | ViT-L | 同上 | 0.937 / 0.103 / 0.903 |
| ChildPlay，微调，表4 | ViT-B | AUC / L2 / AP / P.Head | 0.949 / 0.106 / 0.994 / 0.715 |
| ChildPlay，微调，表4 | ViT-L | 同上 | 0.951 / 0.101 / 0.994 / 0.662 |
| VAT，仅 GF 训练，表5 | ViT-B / ViT-L | AUC / L2 | 0.932 / 0.105；0.937 / 0.100 |
| GOO-Real，仅 GF 训练，表5 | ViT-B / ViT-L | AUC / L2 | 0.901 / 0.174；0.898 / 0.175 |
| ChildPlay，仅 GF 训练，表5 | ViT-B / ViT-L | AUC / L2 | 0.946 / 0.114；0.951 / 0.101 |

GOO-Real 是零样本跨域定位，不是视频时序评价；无对应画内外 AP。GF 测试不按 VAT 的方式报告画内外 AP。原文部分对照结果含重新计算标记，尤其 VAT AUC，引用时应保留脚注含义。

### 对当前选题的解读

适合冻结模型、缓存特征，只训练时序增量。应以相同 checkpoint、框输入与评估脚本对比“逐帧、固定平滑、普通时序模块、拟议模块”。原文 ViT-B 的 GF AUC 为0.956；本项目旧表中的其他数值需要注明是复现还是不同配置。

<a id="p06-sharingan"></a>
## P06 Sharingan

- 全名：Sharingan: A Transformer Architecture for Multi-Person Gaze Following。
- 作者：Samy Tafasca, Anshul Gupta, Jean-Marc Odobez。
- 发表：CVPR 2024。
- [论文主页](https://openaccess.thecvf.com/content/CVPR2024/html/Tafasca_Sharingan_A_Transformer_Architecture_for_Multi-Person_Gaze_Following_CVPR_2024_paper.html)
- [论文 PDF](https://openaccess.thecvf.com/content/CVPR2024/papers/Tafasca_Sharingan_A_Transformer_Architecture_for_Multi-Person_Gaze_Following_CVPR_2024_paper.pdf)
- [官方代码](https://github.com/idiap/sharingan)
- 重点位置：第3节、图3—4、表1—2。

### 要解决的问题与方法

高效地同时预测多人的注视。场景产生 image tokens，头部外观和框坐标产生 person-specific gaze tokens。联合 Transformer 编码后，Conditional DPT 用不同深度的场景特征及人物 tokens 解码每个人的热图；另预测画内外概率，辅以二维注视方向监督。

### 输入、输出与监督

| 类别 | 内容 |
|---|---|
| 推理输入 | 单帧 RGB、场景中待估计人物的头部框与头部裁剪 |
| 论文配置 | 场景与头部裁剪224×224；ViT-B 场景编码器、Gaze360 预训练的 ResNet-18 头部编码器 |
| 输出 | N个人各自的64×64热图、N个画内外分数；辅助二维单位注视向量 |
| 训练监督 | 注视热图 MSE、画内外 BCE、由头部和目标位置派生的二维方向余弦损失 |
| 多人机制 | 共享场景特征并联合处理人物 tokens；依然需要人物框，不是该网络自行完成头部检测 |
| 时间机制 | 核心网络为单帧；视频 demo 外接跟踪不等于模型进行了跨帧注意力建模 |

### 数据集、指标与结果

| 数据集 | 报告指标 | 代表性结果 |
|---|---|---|
| GazeFollow，表1 | AUC / Avg Dist / Min Dist / PLAH | 0.944 / 0.113 / 0.057 / 0.667 |
| VAT，表1，微调 | Dist / PLAH / In/Out AP | 0.107 / 0.738 / 0.891；该主表未报告 VAT AUC |
| ChildPlay，表2，微调 | Dist / PLAH / In/Out AP | 0.106 / 0.600 / 0.990 |

PLAH 是人物头部相关的语义指标。本篇要求预测点与真实点落入同一个头部框，较 ChildPlay 最初 P.Head 的二分类定义更严格，不能把不同定义下的列直接混排。

### 对当前选题的解读

可参考多人共享场景表示和每人独立状态，但人物 token 的时序扩展已有 MTGS。若当前预算只够研究单人轨迹记忆，不必同时引入多人社会关系任务。

<a id="p07-childplay"></a>
## P07 ChildPlay

- 全名：ChildPlay: A New Benchmark for Understanding Children's Gaze Behaviour。
- 作者：Samy Tafasca, Anshul Gupta, Jean-Marc Odobez。
- 发表：ICCV 2023。
- [论文主页](https://openaccess.thecvf.com/content/ICCV2023/html/Tafasca_ChildPlay_A_New_Benchmark_for_Understanding_Childrens_Gaze_Behaviour_ICCV_2023_paper.html)
- [论文 PDF](https://openaccess.thecvf.com/content/ICCV2023/papers/Tafasca_ChildPlay_A_New_Benchmark_for_Understanding_Childrens_Gaze_Behaviour_ICCV_2023_paper.pdf)
- [作者发布的数据集](https://zenodo.org/records/8252535)
- 重点位置：第3节、第4节、第5.2节、表3—4。

### 要解决的问题与方法

提供儿童与成人自然交互场景的密集注视标注，并提出三维几何基线。由估计深度和相机参数重建点云，从头部预测三维注视方向，生成3D Field-of-View热图；再融合图像、头部位置和方向信息，预测目标热图与画内外状态。

### 输入、输出与监督

| 类别 | 内容 |
|---|---|
| 推理输入 | 场景 RGB、指定人物头部框/头部裁剪；深度和相机参数由额外模型推断 |
| 论文配置 | 头部裁剪224×224；场景训练512×288，测试保持宽高比并将长边缩放到512 |
| 中间预测 | 相机坐标点云、三维注视方向、3DFoV热图；这些不是真实三维传感器测量 |
| 最终输出 | 注视目标热图、画内概率；由热图得到二维落点 |
| 训练监督 | 二维注视点、画内外标签；结合估计点云派生三维方向伪标签，使用方向、热图和状态损失 |
| 注意区别 | 数据集有七种状态，不表示该基线模型输出七分类状态；论文主要预测画内外和目标位置 |

### 数据集和标注

ChildPlay 包含95个视频中的401段片段、120,549帧、257,928个人物帧实例。七类状态为 inside-frame、outside-frame、gaze-shift、occluded、eyes-closed、uncertain、not-annotated。

这些标签可支持按状态分析，但不能假设 gaze-shift/occluded 等帧都有可靠的二维落点。occluded 也不能未经定义核验就解释成“被注视物体受到遮挡”。

### 各数据集指标与原文结果

| 数据集 | 配置 | 指标顺序 | 原文报告值 |
|---|---|---|---|
| GazeFollow，表4A | 完整3DFoV模型 | AUC / Avg Dist / Min Dist / P.Head | 0.936 / 0.125 / 0.064 / 0.760 |
| VAT，表4B | 完整3DFoV模型，微调 | AUC / Dist / AP / P.Head | 0.914 / 0.109 / 0.834 / 0.902 |
| ChildPlay，表3 | 完整模型，微调，全体 | AUC / Dist / AP / P.Head | 0.935 / 0.107 / 0.986 / 0.663 |
| ChildPlay，表3 | 同上，仅儿童 | 同上 | 0.939 / 0.098 / 0.989 / 0.604 |
| ChildPlay，表3 | 同上，仅成人 | 同上 | 0.928 / 0.121 / 0.983 / 0.704 |

上述选取完整3DFoV变体，不与同文2D cone或未微调变体的最好列拼接。不同后续论文可能引用其中不同变体，出现数值差异时应回查原表。

### 对当前选题的解读

它提供可复用的切换及不确定状态标注，适合验证状态相关行为而不新增采集。儿童定位距离较好、看向头部的语义指标却未必更好，也说明“几何距离改善”不等价于“交互语义改善”。

<a id="p08-argaze"></a>
## P08 ARGaze

- 全名：ARGaze: Autoregressive Transformers for Online Egocentric Gaze Estimation。
- 作者：Jia Li 等。
- 状态：2026年预印本；本文依据 arXiv:2602.05132v1，不将其写成已确认录用的会议论文。
- [arXiv 入口](https://arxiv.org/abs/2602.05132)
- [HTML全文及附录](https://arxiv.org/html/2602.05132v1)
- [PDF](https://arxiv.org/pdf/2602.05132)
- 重点位置：第3节、第4.1节、表1、表4—5、附录B/C。本文未核验可直接使用的作者代码入口。

### 要解决的问题与方法

根据第一人称视频估计相机佩戴者当前看哪里。冻结 DINOv3 提取视觉特征，将历史预测热图编码成 tokens，并围绕历史预测位置获取局部目标模板；因果 Transformer 融合历史与当前视觉信息，预测当前热图。使用 KL 散度训练，并以 scheduled sampling 缓解真实历史与预测历史之间的差异。

### 输入、输出与监督

| 类别 | 内容 |
|---|---|
| 推理输入 | 第一人称当前 RGB 帧、固定窗口内的历史预测热图、从历史帧派生的目标模板 |
| 不需要的输入 | 第三人称指定人物的头部框或头部裁剪；佩戴者自身头部通常不在画面内 |
| 论文配置 | 冻结 DINOv3 ViT-S/16；图像/模板224×224；3层因果解码器；默认3个历史热图 |
| 输出 | 当前帧64×64注视热图；不是多人的热图集合，也不直接输出 VAT 式人物画内外分类 |
| 训练监督 | 第一人称注视坐标构造的热图；训练混用真实历史热图和模型历史预测 |
| 测试历史 | 来自模型自己的历史预测，不能输入真实历史注视标签 |
| 时间机制 | 显式严格因果，只使用当前及过去信息；固定历史窗口限制长期流式内存 |

### 数据集、指标与结果

| 数据集 | 任务与数据来源 | 主指标：F1 / Precision / Recall，% |
|---|---|---|
| EGTEA Gaze+ | 第一人称烹饪活动；使用官方训练/测试划分 | 44.48 / 35.22 / 60.37 |
| Ego4D gaze subset | 大规模第一人称视频的注视标注子集 | 41.82 / 33.47 / 55.71 |
| EgoExo4D egocentric subset | 使用第一人称视角；将注视信息投影到二维；另划分参与者/地点/任务域外测试 | IID：70.58 / 75.66 / 66.14 |

来源：原文表1。这里的F1是论文使用的 Adaptive F1，不是画内外分类F1，也不是 LAH/LAEO F1。正文说明沿用 GLC 协议，但没有在已核验的指标段完整展开二值化阈值、容忍区域和汇总细节；复现前应检查其评估实现，不能自行补一个阈值后宣称同协议。

效率还报告 FPS、平均/P95延迟、CUDA分配/保留显存；原文使用 RTX A6000。不同硬件、输入及缓存边界下不能直接与3090测量值比较。

### 对当前选题的解读

历史热图、目标模板和因果记忆都已有明显重合，不能单独作为新贡献。值得验证的差异是第三人称人物条件、画内外状态转换及错误历史控制。其数据集和监督来源与 VAT 不同，适合作方法参考，不适合直接把其 F1 数值放入 VAT 排名表。

## 跨论文比较和选题判断

| 设计或结论 | 已有工作 | 后续需要补足的证据 |
|---|---|---|
| 普通时序特征融合 | P01、P02 | 同一基础模型上的普通时序模块对照 |
| 画内位置与画外状态联合分布 | P02 | 状态切换时的记忆更新策略，而非只增加 outside token |
| 人物级跨帧表示 | P03 | 当前观测错误与真实切换的区分能力 |
| 时间变化、概率连续性 | P04 | 因果信息约束、切换响应及可比设置 |
| 冻结基础编码器 | P05及P03官方更新 | 时序增量的实际收益、开销和可复现协议 |
| 历史热图、历史模板 | P08 | 可信写入、失效记忆释放及自回归误差积累控制 |
| 状态相关分析 | P07数据标注 | 状态定义、有效坐标掩码及完整测试覆盖 |

“选择性记忆可能有帮助”是需要验证的假设。不能仅凭平均定位误差下降就声称解决遮挡、切换和不确定性；相应结论需要对应片段、状态或校准实验支撑。

## 2026-11-17 补充：其他视频任务与自适应滤波

以下为此前方向讨论中已核验原始资料的整理，服务于本轮方案定位；不构造跨任务数值排行榜。

| 工作 | 视频信息的作用与边界 | 对当前方案的含义 |
|---|---|---|
| [One Euro Filter，CHI 2012](https://gery.casiez.net/publications/CHI2012-casiez.pdf) | 速度自适应一阶低通，低速去抖、高速减滞后 | 当前位移驱动 EMA 的直接思想近邻；普通自适应滤波应进入强简单对照 |
| [FGFA，ICCV 2017](https://arxiv.org/html/1703.10025) | 先运动对齐，再融合跨帧特征；原版邻域含未来 | 可以借鉴先建立对应的原则，不能照搬非因果窗口或混称同条件基线 |
| [Cutie，CVPR 2024](https://arxiv.org/html/2310.12982) | 像素与目标级记忆支持新帧分割；给定首帧目标 mask | 视觉目标表示比旧坐标更丰富，但注视需要允许主动换目标，不能固定目标身份 |
| [LSTR，NeurIPS 2021](https://proceedings.neurips.cc/paper/2021/hash/08b255a5d42b89b0585260b6f2360bdd-Abstract.html) | 远期粗上下文与近期细节分工，用于在线动作检测 | 背景记忆与快速响应可分开表示；原输入/特征提取条件须与纯 RGB 严格因果方案区分 |
| [MoViNets，CVPR 2021](https://arxiv.org/abs/2103.11511) | 流式特征缓冲和因果运算，以受控内存处理视频 | 因果缓存与轻量时间建模是成熟手段，本身不作为新贡献 |

Chong 的[官方模型](https://github.com/ejcgt/attention-target-detection/blob/master/model.py#L242)默认使用单向 ConvLSTM，但[官方数据加载](https://github.com/ejcgt/attention-target-detection/blob/master/dataset.py#L218)含居中头框平滑。循环层单向不等于整个预处理/推理流水线严格因果。本项目 T1 已保留当前帧原始真实框。

## P09 EyeTAG：轨迹变化与切换偏差的相邻工作

- [EyeTAG: Eye Trajectory-Aware Gaze Estimation](https://arxiv.org/abs/2610.00922)，2026 年 10 月预印本，页面注明 BMVC 2026 accepted。
- 输入脸部/眼部视觉证据，输出三维注视方向；不是 VAT 的第三人称画面内目标估计。
- 从自身过去预测计算差分轨迹作为因果运动 token，融合当前视觉证据，并用控制骨干和架构的消融分析快速眼动偏差。
- 启发：显式预测历史、因果轨迹和稳定/响应分析已有先例；本项目需要用同输入普通时序头及历史失效机制的消融说明额外作用。其角度误差不放入 VAT 排名表。

## 应用锚点：团队活动与注视

[Where are they looking in the operating room?](https://arxiv.org/html/2604.20574v1)，2026 年预印本，扩展手术室注视标注，并利用注视热图研究角色、阶段及团队交流事件。它为“注视 → 团队活动理解”提供具体任务链条，不证明我们的规则或新模块已改善协作。其标注采样和任务指标也不能直接替代 VAT 连续帧切换评价。

当前研究假设和对照要求汇总于 [temporal_joint_refinement.md](temporal_joint_refinement.md)。联合输出、门控、视觉对应或历史模板的单纯组合，不预先写成已成立的创新点。
