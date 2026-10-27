# GazeSpot：面向鲁棒注视目标估计的头部条件空间门控与多层特征融合

Bin Fan，Tongjie Pan，Huiyu Zhai，Tao Zhou，Yalan Ye

电子科技大学计算机科学与工程学院，中国成都

## 摘要

基于冻结视觉基础模型的注视目标估计具有吸引力，因为它避免了沉重的任务专用编码器；然而，现有大多数设计只解码最后一层特征。对于区分相邻人物和物体所需的局部几何信息，这种表示可能不够敏感。我们提出 GazeSpot，一种将头部条件空间门控与自适应多层特征融合相结合的单分支模型。空间门控把相对于头部的坐标映射为软掩码，以抑制无关区域；融合模块则为来自骨干网络不同深度的特征分配随图像变化的权重。GazeSpot 既不使用深度和姿态输入，也不依赖外部目标检测和分割。采用冻结的 DINOv3 ViT-L 骨干网络时，GazeSpot 在 GazeFollow 上取得 0.961 AUC 和 0.093 平均 $L_2$，在 VideoAttentionTarget 上取得 0.944 AUC 和 0.092 $L_2$。在至少包含四个人的拥挤 VideoAttentionTarget 帧上，相比仅使用最后一层特征的基线，GazeSpot 将 $L_2$ 从 0.134 降低到 0.124。

**关键词：** 注视目标估计，视觉基础模型，空间门控，多层特征，拥挤场景

## 1. 引言

注视目标估计根据一幅图像和一个头部边界框，预测该人物正在注视的位置 [1–3]。该任务可支持意图理解、人机交互和社会场景分析 [4–6]。经典系统使用独立分支分别编码裁剪后的头部与场景 [2, 7]；后续方法进一步引入深度、姿态、交互或物体线索 [8–12]。这些设计虽有效，却需要多个编码器或额外预处理。ViTGaze [13] 和 Gaze-LLE [14] 通过将冻结的视觉基础模型（VFM）与轻量注视解码器结合，简化了整个流程。

**图 1：动机。** 仅解码 VFM 的最后一层特征，可能会使模型响应被拥挤场景中的显著干扰物吸引。GazeSpot 将多个特征层级与头部条件空间门控相结合，使响应更集中于标注目标附近。

冻结 VFM 的简洁性很有吸引力，但现有单分支模型通常只解码其最后一层特征图。深层特征具有较强语义，而点级定位还依赖相对位置和局部边界 [15]。层次化特征在密集预测中十分有效 [16]，自监督 Transformer 也会在任务专用训练之前显现出物体定位线索 [17, 18]。如图 1 所示，当邻近人物和物体竞争解码器响应时，这一区别尤为重要。头部位置通常在解码阶段作为掩码或嵌入加入，但它并不会直接调节场景特征的空间响应。

我们在一组受控的骨干网络比较中观察到了同样的问题。在保持最后一层解码器其余部分不变时，将 DINOv2 [19] 替换为 DINOv3 [20]，会使 GazeFollow 上的 $L_2$ 从 0.104 变为 0.107，并使拥挤 VAT 帧上的 $L_2$ 从 0.133 变为 0.139。该结果不能证明特征发生了过平滑，但它说明，更强的最后一层表示并不会自动带来更准确的注视定位。这促使我们保留中间层特征，而不是把最后一层视为唯一有用的表示。

近期的几何方法提供了另一条路径。Mi 等人 [21] 将 DINOv3 特征与来自 YOLO11x 和 SAM2 的物体掩码结合，监督一个二维注视方向，并将其转换为角度视场先验。其物体级建模非常适合在候选目标之间进行推理。我们的目标不同：保留紧凑的单分支设定，并且只要求注视跟随基准本就提供的头部边界框。我们从相对坐标中学习稠密空间先验，并针对每幅图像自适应调整不同特征层级的权重，从而避免外部目标候选、分割掩码和方向标签。

为此，我们提出 **GazeSpot**。几何引导空间聚焦（Geometry-Guided Spatial Focus，GGSF）将头部相对坐标映射为软门控，并在解码前将其施加到 VFM 特征图上。尺度感知语义聚合（Scale-Aware Semantic Aggregation，SASA）随后为浅层、中间层和深层特征分配输入相关的权重。不同于固定求和，这种融合能够随场景变化，同时保留深层语义特征。融合后的表示由一个轻量 Transformer 解码器处理。

我们的贡献包括：一种头部条件空间门控；一种融合局部结构与深层语义的自适应多层特征机制；以及分别考察拥挤场景、跨数据集迁移和头框噪声的实验评估。

**图 2：GazeSpot 总体结构。** GazeSpot 从冻结的 VFM 中提取四个层级。GGSF 对每一层施加头部条件软空间门控，SASA 使用随图像变化的权重融合门控后的特征，轻量解码器则预测注视热图和画内/画外得分。

## 2. 方法

### 2.1 概述

给定图像 $I\in\mathbb R^{3\times H_I\times W_I}$ 和目标头部边界框 $\mathbf b=[x_{\min},y_{\min},x_{\max},y_{\max}]$，GazeSpot 预测热图 $\hat{\mathbf M}\in[0,1]^{H_o\times W_o}$，并在需要时预测画内概率 $\hat p$。如图 2 所示，冻结的 DINOv3 骨干网络在深度方向均匀采样，提供 $K=4$ 个特征图 $\{\mathbf F_k\}_{k=1}^{K}$。线性投影在保持共同 patch 网格的同时对齐各层的通道维度。因此，同一场景特征可以被复用于每个已标注人物；对于不同查询，仅头部边界框及其产生的空间门控发生变化。

模型包含三个阶段。首先，GGSF 将头部边界框转换为稠密的头部相对坐标场，并在聚合前对每个特征层级进行门控。随后，SASA 为门控后的各层特征估计随图像变化的权重，并将拼接结果投影为 $D$ 维特征图。最后，轻量 Transformer 联合解码注视热图与画内/画外预测。在训练和推理期间，模型均不使用目标检测器、分割模型、深度估计器或注视方向估计器。

### 2.2 头部条件空间门控

令 $\mathbf c=(c_x,c_y)$、$w$ 和 $h$ 分别表示归一化头部边界框的中心和尺寸。在特征位置 $\mathbf p=(x,y)$ 处，GGSF 构造四通道几何张量：

$$
\mathbf G(\mathbf p)=[x-c_x,\;y-c_y,\;w,\;h],
\qquad \mathbf G\in\mathbb R^{4\times H\times W}. \tag{1}
$$

一个由 $1\times1$ 卷积、归一化和 ReLU 实现的两层逐点 MLP，将该坐标场映射为软掩码。掩码沿通道维广播，并施加到所有选定层级：

$$
\mathbf M_g=\sigma(\operatorname{MLP}(\mathbf G)),\qquad
\mathbf F_k^g=\mathbf F_k\odot\mathbf M_g,\quad k=1,\ldots,K. \tag{2}
$$

在各层之间共享同一门控，可为整个特征层次提供一致的、相对于头部的空间偏置，同时把层级选择留给 SASA。该门控不编码注视向量，也不单独决定最终方向；方向由门控后的图像证据推断。门控不依赖掩码监督，而是通过热图和画内/画外目标端到端学习。我们将最终 MLP 的偏置初始化为 2，使 $\mathbf M_g$ 在训练早期接近 1，从而减少对预训练特征的扰动。因此，我们把 GGSF 解释为学习得到的空间聚焦，而不是经过标定的三维视锥或硬视场边界。

### 2.3 自适应多层融合

经过空间门控后，SASA 选择当前场景所需的特征粒度。全局平均池化将每个门控后特征概括为：

$$
\mathbf v_k=\operatorname{GAP}(\mathbf F_k^g)
=\frac{1}{HW}\sum_{i=1}^{H}\sum_{j=1}^{W}\mathbf F_k^g(i,j). \tag{3}
$$

共享的两层打分网络 $s(\cdot)$ 对这些描述子进行评估，并在层级维度执行 Softmax，得到归一化权重：

$$
\alpha_k=\frac{\exp s(\mathbf v_k)}{\sum_{j=1}^{K}\exp s(\mathbf v_j)},
\qquad \sum_{k=1}^{K}\alpha_k=1. \tag{4}
$$

门控后的特征经过加权、拼接，并投影至解码器维度 $D=256$：

$$
\mathbf X=\operatorname{Conv}_{1\times1}\!\left(
\mathop{\Vert}_{k=1}^{K}\alpha_k\mathbf F_k^g\right). \tag{5}
$$

其中，$\Vert$ 表示通道拼接。不同于固定平均，这些权重会随图像及目标条件特征而变化。较早层级可以保留拥挤区域中的边界和局部结构，较深层级则提供区分合理候选目标所需的语义。拼接会在 $1\times1$ 投影混合各层之前，保留层级特有的通道。

### 2.4 注视解码

我们将 $\mathbf X$ 展平为序列 $\mathbf S\in\mathbb R^{HW\times D}$。学习得到的二维位置嵌入 $\mathbf E_{\mathrm{pos}}$ 用于保留 patch 布局。我们还将查询头部边界框栅格化到 patch 网格，并把二值头部区域映射为学习得到的嵌入 $\mathbf E_{\mathrm{head}}$。解码器输入为：

$$
\mathbf S_{\mathrm{in}}=
[\mathbf T_{\mathrm{in/out}};\,\mathbf S+\mathbf E_{\mathrm{pos}}+\mathbf E_{\mathrm{head}}], \tag{6}
$$

其中，$\mathbf T_{\mathrm{in/out}}\in\mathbb R^{1\times D}$ 是学习得到的状态 token。三个 Transformer 块输出 $[\hat{\mathbf T}_{\mathrm{in/out}};\hat{\mathbf S}]$。状态 token 被送入 MLP 进行画内分类；patch token 则被恢复为二维形状，并通过反卷积头上采样：

$$
\hat p=\sigma(\operatorname{MLP}(\hat{\mathbf T}_{\mathrm{in/out}})),\qquad
\hat{\mathbf M}=\sigma(\operatorname{Deconv}(\operatorname{Reshape}(\hat{\mathbf S}))). \tag{7}
$$

### 2.5 训练目标

通过在标注注视点处放置高斯分布来构造目标热图 $\mathbf M_{\mathrm{gt}}$。像素级二元交叉熵监督 $\hat{\mathbf M}$，另一项二元交叉熵则根据画内标签 $p_{\mathrm{gt}}$ 监督 $\hat p$。总损失为：

$$
\mathcal L=\mathcal L_{\mathrm{heat}}+\lambda\mathcal L_{\mathrm{in/out}}. \tag{8}
$$

GazeFollow 只包含画内标注，因此 $\lambda=0$。对于 VideoAttentionTarget（VAT），$\lambda=1$，两个分支联合优化。

## 3. 实验

### 3.1 数据集与评估指标

我们在 GazeFollow [1] 和 VideoAttentionTarget（VAT）[2] 上进行评估。AUC 和欧氏 $L_2$ 误差用于衡量定位性能。GazeFollow 还报告多个标注中的最小 $L_2$，VAT 则报告画内/画外预测的平均精度（AP）。对于拥挤场景分析，我们根据已标注头部数量将 VAT 测试帧划分为：少于三个头部（7,945 帧）、三个头部（3,015 帧）、四个头部（1,628 帧）和多于四个头部（539 帧）。此外，我们使用 GOO [12] 的真实图像划分，对在 GazeFollow 上训练的检查点进行分布外（OOD）测试。

### 3.2 实现细节

图像被缩放至 $512\times512$，热图尺寸为 $64\times64$。DINOv3 骨干网络保持冻结，解码器维度为 256，模型在一张 RTX 3090 上使用 Adam 和余弦学习率衰减进行训练。主表结果均采用完整的 GGSF+SASA 配置；ViT-B 和 ViT-L 两行仅骨干网络规模不同。拥挤场景和扰动实验使用 ViT-B 检查点，并且扰动仅在测试时施加。

### 3.3 主要结果

**表 1：GazeFollow 与 VideoAttentionTarget 上的最新方法比较。** “参数量”表示各方法所报告的可训练参数量。最优和次优结果分别以粗体和下划线标出。

| 类别 | 方法 | 发表场所 | 参数量 | GF AUC ↑ | GF 平均 $L_2$ ↓ | GF 最小 $L_2$ ↓ | VAT AUC ↑ | VAT $L_2$ ↓ | 画内/画外 AP ↑ |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 多分支/多模态 | Chong 等 [22] | ECCV 2018 | 51M | 0.896 | 0.187 | 0.112 | 0.833 | 0.171 | 0.712 |
| 多分支/多模态 | Lian 等 [23] | ACCV 2018 | 55M | 0.906 | 0.145 | 0.081 | 0.837 | 0.165 | — |
| 多分支/多模态 | Chong 等 [2] | CVPR 2020 | 61M | 0.921 | 0.137 | 0.077 | 0.860 | 0.134 | 0.853 |
| 多分支/多模态 | Fang 等 [7] | CVPR 2021 | 68M | 0.922 | 0.124 | 0.067 | 0.905 | 0.108 | 0.896 |
| 多分支/多模态 | Bao 等 [8] | CVPR 2022 | 29M | 0.928 | 0.122 | — | 0.885 | 0.120 | 0.869 |
| 多分支/多模态 | Jin 等 [24] | EAAI 2022 | 52M | 0.920 | 0.118 | 0.063 | 0.900 | 0.104 | 0.895 |
| 多分支/多模态 | Miao 等 [25] | WACV 2023 | 61M | 0.934 | 0.123 | 0.065 | 0.917 | 0.109 | 0.908 |
| 多分支/多模态 | Tafasca 等 [26] | ICCV 2023 | 25M | 0.939 | 0.122 | 0.062 | 0.914 | 0.109 | 0.834 |
| 多分支/多模态 | Sharingan [27] | CVPR 2024 | 105M | 0.944 | 0.113 | 0.057 | — | 0.107 | 0.891 |
| VFM/单分支 | ViTGaze [13] | VI 2024 | 22M | 0.949 | 0.105 | 0.047 | 0.938 | 0.102 | 0.905 |
| VFM/单分支 | GT360 [28] | IROS 2025 | 1.9M | 0.957 | 0.101 | — | 0.934 | 0.103 | 0.887 |
| VFM/单分支 | Gaze-LLE（ViT-B）[14] | CVPR 2025 | 2.8M | 0.958 | 0.104 | 0.045 | 0.933 | 0.107 | 0.897 |
| VFM/单分支 | Gaze-LLE（ViT-L）[14] | CVPR 2025 | 2.9M | 0.958 | 0.099 | 0.041 | 0.937 | 0.103 | 0.903 |
| 物体感知 | Mi 等 [21] | ECCV 2026 | 7.1M | **0.961** | 0.094 | **0.038** | **0.948** | 0.095 | **0.923** |
| 本文方法 | GazeSpot（ViT-B） | — | 3.5M | 0.958 | 0.101 | 0.043 | 0.940 | 0.099 | 0.887 |
| 本文方法 | GazeSpot（ViT-L） | — | 3.8M | **0.961** | **0.093** | **0.038** | 0.944 | **0.092** | 0.912 |

表 1 给出了包含 Mi 等人 [21] 在内的更新比较。在 GazeFollow 上，GazeSpot-ViT-L 的 AUC 和最小 $L_2$ 与该方法持平，同时将平均 $L_2$ 从 0.094 降至 0.093。在 VAT 上，Mi 等人的 AUC 和画内/画外 AP 更高，而 GazeSpot 的定位误差更低（0.092 对 0.095）。GazeSpot 使用 3.8M 可训练参数，并且不需要 [21] 中用于构建物体 token 的外部检测与分割模型。这些结果体现的是互补的权衡，而不是某一种方法在所有指标上的一致优势。

为了进行跨数据集检验，我们将 GazeFollow 上训练的 ViT-B 检查点直接用于 GOO-real，不进行微调。复现的 Gaze-LLE 基线与 GazeSpot 使用相同的数据划分、预处理、检查点选择规则和评估脚本。如表 2 所示，GazeSpot 将 AUC 从 0.8583 提升至 0.8833，并将 $L_2$ 从 0.1997 降至 0.1761。GOO-real 每个样本只有一个目标标注，因此平均 $L_2$ 与最小 $L_2$ 相同。

**表 2：使用 GazeFollow 训练检查点且不微调时，在 GOO-real 上的 OOD 评估。**

| 方法 | AUC ↑ | 平均 $L_2$ ↓ | 最小 $L_2$ ↓ |
|---|---:|---:|---:|
| 复现的 Gaze-LLE | 0.8583 | 0.1997 | 0.1997 |
| GazeSpot | **0.8833** | **0.1761** | **0.1761** |

**表 3：VAT 上按拥挤密度划分的定位结果。**

| 头部数量 | 帧数 | 仅最后一层基线 AUC | 仅最后一层基线 $L_2$ | GazeSpot AUC | GazeSpot $L_2$ |
|---|---:|---:|---:|---:|---:|
| $<3$ | 7,945 | 0.958 | 0.078 | 0.958 | **0.077** |
| $=3$ | 3,015 | 0.932 | 0.100 | **0.934** | **0.099** |
| $=4$ | 1,628 | 0.934 | 0.111 | **0.942** | **0.101** |
| $>4$ | 539 | 0.875 | 0.182 | **0.890** | **0.171** |

表 3 的密度分组结果显示，在稀疏帧上两种方法差异很小，但随着人物数量增加，差距变得更大。当恰好有四个头部时，GazeSpot 将 $L_2$ 降低 9.0%；当头部数量超过四个时，$L_2$ 降低 6.0%。在至少包含四个头部的合并子集上，仅使用最后一层的基线取得 0.915 AUC/0.134 $L_2$，而 GazeSpot 为 0.925/0.124。将每个头部边界框扰动最多 20% 后，GazeSpot 变为 0.923/0.124，说明学习得到的门控对中等程度的边界框噪声并不过度敏感。

**图 3：定性分析。** （a）空间门控抑制干扰物，并生成比最后一层基线更紧凑的热图。（b）在相同拥挤图像上，DINOv3 的 PCA 区域比 DINOv2 更宽泛。

**表 4：采用 DINOv3 ViT-B/16 骨干网络的组件消融。**

| 多层特征 | SASA | GGSF | GF AUC | GF $L_2$ | VAT AUC | VAT $L_2$ |
|---:|---:|---:|---:|---:|---:|---:|
|  |  |  | 0.955 | 0.107 | 0.933 | 0.106 |
| ✓ |  |  | 0.957 | 0.103 | 0.937 | 0.103 |
| ✓ | ✓ |  | 0.958 | 0.102 | 0.939 | 0.101 |
| ✓ | ✓ | ✓ | **0.958** | **0.101** | **0.940** | **0.099** |

### 3.4 消融实验

表 4 分别考察多层特征、自适应融合和空间门控的作用。直接进行多层融合，相比只使用最后一层特征的基线，在两个数据集上均有提升。SASA 进一步降低了 $L_2$，加入 GGSF 后，在不改变骨干网络的情况下获得最佳定位结果。对于 ViT-B 模型，GazeSpot 具有 3.5M 可训练参数和 89.2M 总参数，而基线为 2.8M 可训练参数和 88.5M 总参数。在 $512\times512$ 分辨率下，GazeSpot 需要 141.1 GFLOPs，在 RTX 3090 上每幅图像耗时 22.8 ms；对应基线为 107.2 GFLOPs 和 18.4 ms。因此，额外鲁棒性带来了可测量但适中的计算开销。

该消融还区分了“访问多层特征”和“选择特征层级”的作用。向模型暴露全部四个层级解释了相对于最后一层模型的大部分增益，而 SASA 和 GGSF 带来了更小但一致的进一步提升。这一结果支持如下观点：对于精确定位，最后一层并不总是最佳的唯一表示。但这并不意味着浅层特征始终更优，因为 SASA 保留了深层特征，并与其他层级联合选择其贡献。

**特征可视化。** 图 3 补充了拥挤密度分析和最后一层特征比较。子图（a）显示门控使预测集中在合理目标周围，子图（b）显示 DINOv3 产生了更宽泛的区域。我们不把这些可视化视为因果证据；表 4 中的受控结果表明，在加入自适应加权或空间门控之前，中间层特征本身就能降低 $L_2$。

## 4. 结论

GazeSpot 将头部条件空间门控与自适应多层 VFM 特征相结合，用于注视目标估计。两个组件都能改善 GazeFollow 和 VAT 上的定位性能，并且在拥挤 VAT 帧上的增益更大。相比近期的物体感知几何推理方法，GazeSpot 不需要外部物体掩码和方向监督，并使用更少的可训练参数；但其 VAT AUC 和画内/画外 AP 仍然较低。这一权衡说明，轻量空间条件化与物体级推理是提升鲁棒注视估计的互补方向。

## 5. 参考文献

1. Adria Recasens, Aditya Khosla, Carl Vondrick, Antonio Torralba. “他们在看哪里？”NeurIPS, 2015。
2. Eunji Chong, Yongxin Wang, Nataniel Ruiz, James M. Rehg. “视频中的受关注视觉目标检测。”CVPR, 2020。
3. Adria Recasens, Carl Vondrick, Aditya Khosla, Antonio Torralba. “视频中的注视跟随。”ICCV, 2017。
4. Alireza Fathi, Yin Li, James M. Rehg. “利用注视学习识别日常行为。”ECCV, 2012。
5. Thorsten O. Zander, Matti Gaertner, Christian Kothe, Roman Vilimek. “将眼动输入与脑机接口结合，实现非接触式人机交互。”International Journal of Human-Computer Interaction, 2011。
6. Nathan J. Emery. “眼睛揭示一切：社会注视的神经行为学、功能与演化。”Neuroscience and Biobehavioral Reviews, 2000。
7. Yi Fang, Jiapeng Tang, Wang Shen, Wei Shen, Xiao Gu, Li Song, Guangtao Zhai. “野外场景中的双重注意力引导注视目标检测。”CVPR, 2021。
8. Jun Bao, Buyu Liu, Jun Yu. “ESCNet：借助三维场景理解进行注视目标检测。”CVPR, 2022。
9. Anshul Gupta, Samy Tafasca, Jean-Marc Odobez. “用于注视目标预测的模块化多模态架构：面向隐私敏感场景的应用。”CVPR, 2022。
10. Francesco Tonini, Cigdem Beyan, Elisa Ricci. “跨域多模态注视目标检测。”ICMI, 2022。
11. Francesco Tonini, Nicola Dall’Asen, Cigdem Beyan, Elisa Ricci. “物体感知注视目标检测。”ICCV, 2023。
12. Binglu Wang, Tao Hu, Baoshan Li, Xiaojuan Chen, Zhijie Zhang. “GaTector：统一的注视物体预测框架。”CVPR, 2022。
13. Yuehao Song, Xinggang Wang, Jingfeng Yao, Wenyu Liu, Jinglin Zhang, Xiangmin Xu. “ViTGaze：利用视觉 Transformer 交互特征进行注视跟随。”Visual Intelligence, 2024。
14. Fiona Ryan, Ajay Bati, Sangmin Lee, Daniel Bolya, Judy Hoffman, James M. Rehg. “Gaze-LLE：通过大规模学习编码器进行注视目标估计。”CVPR, 2025。
15. Petr Kellnhofer, Adria Recasens, Simon Stent, Wojciech Matusik, Antonio Torralba. “Gaze360：自然场景中的无物理约束注视估计。”ICCV, 2019。
16. Tsung-Yi Lin, Piotr Dollár, Ross Girshick, Kaiming He, Bharath Hariharan, Serge Belongie. “用于目标检测的特征金字塔网络。”CVPR, 2017。
17. Mathilde Caron, Hugo Touvron, Ishan Misra, Hervé Jégou, Julien Mairal, Piotr Bojanowski, Armand Joulin. “自监督视觉 Transformer 中涌现的性质。”ICCV, 2021。
18. Oriane Siméoni, Gilles Puy, Huy V. Vo, Simon Roburin, Spyros Gidaris, Andrei Bursuc, Patrick Pérez, Renaud Marlet, Jean Ponce. “使用自监督 Transformer 且无需标签进行物体定位。”BMVC, 2021。
19. Maxime Oquab 等. “DINOv2：无监督学习鲁棒视觉特征。”Transactions on Machine Learning Research, 2024。
20. Oriane Siméoni 等. “DINOv3。”arXiv:2508.10104, 2025。
21. Jiajie Mi, Xinyu Liu, Mengke Song, Chenglizhao Chen. “通过几何推理进行多尺度物体感知注视估计。”ECCV, 2026。
22. Eunji Chong, Nataniel Ruiz, Yongxin Wang, Yun Zhang, Agata Rozga, James M. Rehg. “连接注视、场景与注意力：通过联合建模注视和场景显著性进行广义注意力估计。”ECCV, 2018。
23. Dongze Lian, Zehao Yu, Shenghua Gao. “信不信由你，我们知道你在看什么！”ACCV, 2018。
24. Tianlei Jin, Qizhi Yu, Shiqiang Zhu, Zheyuan Lin, Jie Ren, Yuanhai Zhou, Wei Song. “面向机器人的辅助网络深度感知注视跟随。”Engineering Applications of Artificial Intelligence, 2022。
25. Qiaomu Miao, Minh Hoai, Dimitris Samaras. “用于注视跟随的 patch 级注视分布预测。”WACV, 2023。
26. Samy Tafasca, Anshul Gupta, Jean-Marc Odobez. “ChildPlay：理解儿童注视行为的新基准。”ICCV, 2023。
27. Samy Tafasca, Anshul Gupta, Jean-Marc Odobez. “Sharingan：用于多人注视跟随的 Transformer 架构。”CVPR, 2024。
28. Zhuangzhuang Dai, Vincent Gbouna Zakka, Luis J. Manso, Chen Li. “GazeTarget360：面向机器人感知的 360 度注视目标估计。”IROS, 2025。

> 说明：本译文忠实对应当前 `main.pdf` 的正文、公式、表格、图注和参考文献。表中数值按当前稿件保留；其中需要修正或复核的内容见配套审核报告。
