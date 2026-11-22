# CSCWD Router：问题定义、方法边界与行文逻辑

更新：2026-11-22。写作任务 W1。主方法按本轮讨论采用 **DINOv3 ViT-B/16 + 多层表示 + gaze-conditioned Router**。DINOv2 单层适配保留为已有工程与对照路线。本文件确定论文怎么论证；不把尚未完成的跨模型测速、VAT test 或消融写成结果。

## 1. 一句话定位

**利用人物条件决定哪些图像位置值得继续执行深层视觉计算，在注视定位精度和实际推理成本之间取得更好的权衡。**

工作标题：**Gaze-Conditioned Spatial Routing for Efficient Gaze Target Estimation**。

完整方法可以由 DINOv3、多层解码和 Router 组成；新增贡献集中在条件化计算分配。DINOv3 是已有编码器，多层表示是沿用的基础配置。最终目标是相对官方 GazeLLE 同时获得有竞争力的精度与更低延迟；是否达到由统一实验决定，不预写为“双优”。

## 2. 从 GazeLLE 自然引出问题

行文依次回答六个问题：

| 段落 | 要建立的事实或问题 | 如何接到下一步 |
|---|---|---|
| 1 | GazeLLE 等方法用冻结视觉编码器和轻量解码器，降低了任务训练的参数量 | 可训练参数少，不等于推理时编码器计算少 |
| 2 | 冻结编码器仍对所有图像位置执行完整深度的前向；人物条件主要在注视解码时使用 | 能否把人物条件提前引入编码过程？ |
| 3 | 同图不同人物需要的证据不同；头框确定查询身份和相对几何，人物外观及场景提供可学习的注视线索 | 需要预测与查询相关、具有高目标召回率的 support，而非仅按通用显著性选 token |
| 4 | 在编码器中层做出决策，仅让选中位置继续执行深层 blocks | Router 直接改变实际执行的 token 数 |
| 5 | decoder 仍需完整空间网格，多人还需共享图像计算 | 用中层特征回填未选位置；多人 support 聚合后执行一次稀疏 suffix |
| 6 | 完整系统与机制贡献是两个不同问题 | 与官方 GazeLLE 比系统表现；与相同表示的 matched dense 比 Router 增量 |

“全图完整计算”是执行方式，不等于已经证明未选区域全部无用。VAT 旧结果提示，目标以外的关系上下文也可能影响定位。

GazeLLE 已支持一张图只编码一次、供多个人物查询复用。本文不能把“从 N 次 backbone 变为 1 次”作为相对 GazeLLE 的改进。新增多人机制是：**在共享编码的前提下，把多个查询的需求合并为同一条有限预算的稀疏深层路径**。[官方多人推理说明](https://github.com/fkryan/gazelle#usage)

## 3. DINOv3 与多层特征怎样引入

### 3.1 DINOv3：选择实现载体

在提出“人物条件化计算分配”之后，再说明实现采用冻结的 DINOv3 ViT-B/16。它提供可复用的稠密视觉表示；本轮直接复用仓库中已有的 DINOv3 路由实现和训练基础。DINOv3 的来源是第三方预训练模型，不列为本文贡献。[DINOv3 技术报告](https://arxiv.org/abs/2508.10104)

不使用“DINOv3 更先进，所以本文天然更准”的推导，也不使用“DINOv3 严重过平滑，所以必须改成多层”的旧动机来承担新论文的核心论证。已有结果没有隔离 backbone、层级表示和训练配置各自的作用。

可直接采用的表述：

> 我们在冻结的 DINOv3 编码器上实现人物条件化空间路由，并沿用多层特征解码作为基础配置。研究重点是在给定表示框架内，将有限的深层计算预算分配给与人物查询相关的位置。

采用 DINOv3 不意味着直接加载官方 GazeLLE 的 DINOv2 decoder checkpoint。官方 checkpoint 用于外部基线；DINOv3 多层分支使用它自己的 dense decoder 初始化，并报告初始化工件与训练预算。两者共享冻结编码器加轻量解码器的设计思路，不能写成权重完全相同的底座。

### 3.2 必须区分的两种“使用中间层”

| 用途 | 当前实现 | 论文中的身份 |
|---|---|---|
| Router 的决策输入 | block 5 后的场景网格、从同一网格池化的头部外观、bbox-relative geometry | 计算决策所需的中层证据 |
| Decoder 的多层表示 | blocks 2、5、8、11 的四组特征；后两组按稀疏执行结果重建 | 沿用的表示配置及其稀疏化适配 |

block 编号从 0 开始，分别对应第 3、6、9、12 层。四组特征具有相同的 32×32 patch 网格，是不同深度的表示，不是四种输入分辨率，也不是执行四遍 backbone。

Router **不需要先取得第 9、12 层特征再决定是否执行这些层**。它只使用路由点之前已计算的特征，避免“先算完再剪枝”。当前 Router 没有额外把四层 decoder 特征全部作为输入。

头框提供位置与尺寸，不直接提供 gaze direction。头部外观和中层场景特征可供模型学习人物与场景的关系，但当前没有显式头姿或方向预测分支，也没有方向监督。因此使用“人物外观与相对空间线索”，不把浅层特征定义为“真实空间方向”。

### 3.3 多层表示与 sparse execution 的联系

浅层两组特征保留完整网格；深层两组只对选中 patch 继续计算，未选位置使用路由点的中层激活回填，再按实现归一化。这样 decoder 始终可以接收完整空间布局，沿用现有通道拼接和投影接口。

这给出一个可检验的设计动机：保留完整的早期表示，可能有助于承载不均匀的计算深度。**“多层提高剪枝耐受性”仍需同一 DINOv3 backbone 下的单层/多层 × dense/Router 对照**。DINOv2 单层与 DINOv3 多层之间的差异不能单独证明它。

## 4. 方法的固定描述

输入为单帧 RGB 与该帧真实头框集合。没有未来帧、深度、检测器、分割器或跨帧记忆。

```text
RGB, 512×512
  -> frozen DINOv3 blocks 0–5, dense
       -> 保存 block-2 / block-5 完整特征
       -> block-5 场景 + 头部外观 + 相对几何
          -> 每人 support -> per-image max-union -> 固定预算 Top-K
  -> gather selected patches + special tokens + 对应 RoPE
  -> frozen DINOv3 blocks 6–11, sparse
       -> block-8 / block-11 selected features
       -> 未选位置用 block-5 激活回填
  -> 四层 raw concatenation + 1×1 projection
  -> 共享场景特征上的逐人 gaze decoder
  -> heatmap，以及 VAT 的 in/out score
```

对人物 i，support 概率由 `R(F_mid, head_appearance_i, geometry_i)` 得到；对同图人物取逐位置最大值，再在共同预算中取 Top-K。头部与少量 escape 位置按现有实现优先排序，仍计入总预算。special tokens 保留，patch 与其 RoPE 索引一致。

冻结 backbone 仍允许训练 Router 和 decoder；冻结不会免除推理前向。训练配方与阶段预算另行报告，不能把方法写成训练所有 backbone 参数。

原实现的实际可核对位置：`gazelle/routing/router.py`、`gazelle/routing/backbone.py`、`gazelle/routing/model.py`。主方法关闭 SASA/GGSF，使用 `raw_concat`；本轮没有改模型代码。

K50 表示后六个 blocks 保留一半 patch；前六个 blocks、Router、回填、融合和 decoder 仍有成本。不要将它写成“整网计算减半”或“2× 实际加速”。

## 5. 与 MM / rebuttal / ICASSP 的实质边界

三份 PDF 的逐页定位与素材结论见 [PAPER_REVIEW.md](PAPER_REVIEW.md)。本地 ICASSP 已经把多层特征、自适应融合和头部条件门控写进标题、摘要、贡献、方法与消融；raw multi-level concat 也已作为消融行出现。

| 项目 | MM / ICASSP 已覆盖 | CSCWD 应承担的新增内容 |
|---|---|---|
| 研究问题 | 拥挤场景的表示、定位和鲁棒性 | 人物条件化的计算预算分配 |
| DINOv3、多层访问 | 已使用并分析 | 说明复用范围与来源，作为基础配置 |
| head-relative conditioning | 已用于软空间门控 | 联合图像证据预测用于计算选择的 support |
| routing 含义 | SASA 给已计算的不同层特征分配权重 | Top-K 决定哪些位置实际继续执行深层 blocks |
| mask 的作用 | 完整 backbone 之后乘特征 | 编码器中途改变 suffix 序列长度 |
| 多人 | 已描述共享场景特征 | 多查询 support 聚合与共享稀疏 suffix 的预算和精度 |
| 主证据 | 原主表、拥挤子集、OOD、头框噪声 | 新的计算图检查、统一测速、路由消融和预算权衡 |

建议贡献表述：

1. 基于场景、头部外观与相对几何的高召回 gaze support，使深层计算选择依赖人物查询。
2. 将多人需求聚合到共同 token 预算，并通过稀疏 suffix 与中层回填保留完整解码网格的执行机制。
3. 以相同表示的 dense 对照和官方 GazeLLE 为两类基线，实测定位、覆盖率及模型前向效率，建立计算收益与精度代价之间的关系。

这些是本文要验证和论证的贡献，不表示所有组件在整个领域内首次出现。通用 token sparsification 已有 DynamicViT 等工作；需要说明任务条件与多人执行的区别，并用相同预算的通用/无条件路由对照支撑它。[DynamicViT](https://proceedings.neurips.cc/paper_files/paper/2021/hash/747d3443e319a22747fbb873e8b2f9f2-Abstract.html)

面向 dense prediction 的通用动态空间稀疏化也已有快/慢路径与结构化网格保持设计，因此不把“浅层回填”单独写成首次提出。本文要论证的是它与人物条件、目标覆盖监督及多人共同预算的结合。[Dynamic Spatial Sparsification](https://arxiv.org/abs/2207.01580)

MM 已拒稿不等于其进入在审 ICASSP 的内容可再次主张为新贡献。保留并按投稿规则披露相关在审稿、复用内容和新增内容；正文引用并不自动解决实质重叠。没有“换标题、换 backbone 或低于某个文字查重比例即可安全”的保证。[ICASSP 2027 editorial policies](https://2027.ieeeicassp.org/about/editorial-policies/)、[SPS policy and procedures](https://signalprocessingsociety.org/volunteers/policy-and-procedures)

可用于相关工作/投稿说明的差异表述，按实际稿件和会议要求填写引用：

> Our related GazeSpot manuscript studies head-conditioned spatial gating and adaptive multi-level feature fusion for gaze localization. The present work reuses the multi-level representation as a baseline configuration and studies a different computation mechanism: intermediate person-conditioned token selection, a shared sparse encoder suffix, and early-exit reconstruction. The new evaluation focuses on routing causality and measured accuracy–efficiency trade-offs.

## 6. Introduction 五段草稿

### 中文逻辑稿

**第一段，应用与已有进展。** 注视目标估计根据图像和人物头框推断注视位置，为人机交互和协作场景中的注意理解提供基础。冻结视觉编码器与轻量解码器的结合简化了任务训练，GazeLLE 是代表性方法。计算效率应同时关注可训练参数和实际推理执行。

**第二段，计算缺口。** 冻结仅改变哪些参数被更新；在推理时，视觉编码器仍对所有图像位置执行完整深度计算。即使最终查询只涉及某个人物，人物条件通常也在场景编码之后进入解码。因此需要研究：能否在完整场景编码结束前，利用查询人物决定哪些区域继续获得深层计算？

**第三段，任务条件。** 通用显著性不能单独确定一个人物正在看哪里。头框提供查询身份和相对几何，头部外观与场景特征提供可学习的关系线索。我们将这些信息结合为中层 support 预测，以固定预算保留与该人物查询相关的位置；同图多人的 support 聚合为一条共享路由。

**第四段，实现选择。** 我们在冻结的 DINOv3 ViT-B/16 上实现该机制，并沿用多层表示解码。Router 位于编码器中部，之后仅对选中 patch 执行深层 blocks。完整早期特征和中层回填维持 decoder 所需的空间网格。DINOv3 与多层访问是采用的表示基础，新增机制是实际稀疏执行及其人物条件化决策。

**第五段，验证。** 实验分别比较相同表示的 matched dense 与官方 GazeLLE，前者量化路由贡献，后者评价完整系统。评价同时报告定位、support 覆盖率、实际延迟和吞吐，并通过预算、人物条件和多人查询实验检验设计。结尾的具体数值及“优于”“保持”用语在统一实测后填写。

### 英文工作稿

Gaze target estimation predicts a person's visual target from an image and a head bounding box, supporting attention understanding in human–computer interaction and collaborative settings. Frozen visual encoders coupled with lightweight gaze decoders, as exemplified by Gaze-LLE, substantially reduce the parameters that require task-specific training. Their inference cost, however, also depends on the computation performed by the visual encoder.

Freezing the encoder does not remove its forward computation. In a dense frozen-encoder pipeline, all image locations pass through the full encoder before the gaze decoder uses the person query. This motivates a question: can person-specific evidence guide the allocation of deep visual computation before scene encoding is complete?

This decision requires information about the queried person as well as the scene. A head box identifies the query and defines relative spatial coordinates, while head appearance and intermediate scene features provide evidence for learning query-relevant regions. We predict a person-conditioned support at an intermediate encoder layer and select locations under a fixed token budget. For multiple queries in one image, their supports are aggregated into a shared route.

We instantiate this mechanism with a frozen DINOv3 ViT-B/16 encoder and an existing multi-level decoding configuration. Only selected patch tokens continue through the encoder suffix. Dense early features and an intermediate-feature fallback preserve the spatial grids required for gaze decoding. Our study concerns person-conditioned token selection and sparse execution; the pretrained encoder and multi-level representation form the adopted baseline configuration.

We evaluate routing against a matched dense model with the same representation and compare the complete system with official Gaze-LLE. The evaluation measures localization, support coverage, model-forward latency, and throughput, together with controls for token budget, person conditioning, and multi-person queries. Quantitative outcome statements will be completed after the corresponding measurements.

最后一句是写作占位说明，正式投稿时替换为真实结果；不得把未来实验写为已经完成。

## 7. MM 图 2 的处理决定与新图安排

MM 图 2 位于 PDF 第 2 页：五个样例的输入、DINOv2 最后一层 PCA、DINOv3 最后一层 PCA。候选源文件是 `icassp2027/2026ACMMM/Figure/dinov2VSdinov3_compressed.pdf`，本轮已与 PDF 渲染图目视核对。

**保留源素材，当前不直接复用为 CSCWD 主动机图。** 理由与用途：

- 图中没有浅层行、头部朝向测量、support 或 token 删除结果，不能支撑“浅层代表注视方向”或“这些位置可以不算”。
- PCA 是投影展示；不同模型的拟合、颜色映射、输入分辨率和特征网格需要统一说明。颜色更平滑不能单独建立过平滑或可安全剪枝的因果结论。
- “最后一层不足、多层恢复细节”已是 ICASSP 的主要论证。把同一图换成 Router 名称会延续相同动机，不能形成新方法证据。
- 原始输入图、绘图代码和排版元素可以作为新图的制作基础；新的 support、token mask 和热图必须来自新方法真实输出，并标清数据与 checkpoint 来源。若涉及已有授权或第三方素材，按实际使用条件处理，不把已拒稿自动视为任何素材均可自由复制的证明。

新图安排：

| 图 | 内容 | 在论证中的作用 |
|---|---|---|
| Fig. 1 | 同图两个不同人物查询；dense 完整执行与 query-conditioned support / token 选择示意 | 建立“不同查询需要不同计算分配”的问题；执行示意与实测输出分别标注 |
| Fig. 2 | dense prefix → scene/head/geometry → support union → Top-K → sparse suffix → fill → decoder | 明确计算被省在哪里，和 ICASSP 的 post-backbone soft gate 区分 |
| Fig. 3 | 真实预测的 support、保留位置、GT、dense/Router 热图；含成功与定位退化样例 | 将覆盖率、上下文和定位联系起来 |
| Fig. 4（篇幅允许） | 实测 accuracy–latency Pareto | 展示完整模型权衡，不能用 token 比例代替延迟 |

MM 图 2 若最终保留，只作为注明来源的辅助特征观察；其旧标题“DINOv3 严重过平滑”的结论不直接沿用。当前没有生成新的预测可视化。

## 8. 实验表格必须回答的不同问题

### 8.1 完整系统与机制贡献

| 行 | 表示与执行 | 用途与状态 |
|---|---|---|
| A | 官方 GazeLLE，DINOv2-B/14，单层 dense，448 输入 | 外部系统基线；列明官方 checkpoint |
| B | DINOv3-B/16，最后一层 dense，512 输入 | 同 backbone 的层级表示对照；新协议结果待补，不能直接搬 ICASSP 行充当本轮结果 |
| C | DINOv3-B/16，四层 raw-concat matched dense，512 输入 | Router 的主要机制对照；已有旧协议 GF 结果 |
| D | DINOv3-B/16，四层 raw-concat + Router K50，512 输入 | 候选完整方法；已有旧协议 GF 与 VAT holdout 结果 |
| E（用于层级耐受性主张） | DINOv3-B/16，最后一层 + 相同 Router | 与 B/C/D 共同检验多层是否降低剪枝损失；尚未完成 |

C 与 D 使用相同表示、dense 初始化和 decoder 适配预算，独立登记 Router support 训练成本。K100 是同路径无剪枝 control，不替代 C 的原生 dense 执行时间。

A 与 D 比较完整系统，不能把其差异全部归因于 Router。若 D 只在 AUC 略高而 Avg L2 更差，按指标分别报告或描述为相近精度。若 D 仅快于 C，结论就是加速自身 dense；只有同条件直接测速优于 A 后，才能写相对 GazeLLE 加速。B/14 与 B/16 的区别、448/512 输入均列入表格；不能将 B/14 的结果泛称所有 GazeLLE，包括 L/14。

### 8.2 效率与多人

统一同一 GPU、精度、计时范围、batch 和输入样本来源，保留各模型被评估的输入尺寸。数据加载、预处理和 H2D 在计时外时，表头写 model-forward latency。forward 包含 Router、gather/scatter、融合和 decoder。报告 B1/B4/B8、正反序、warmup、重复次数和显存；准确率使用哪个 checkpoint，新的主表效率就测哪个 checkpoint。

多人主对照是 **GazeLLE 的共享 dense 编码、DINOv3 的共享 dense 编码、共享 sparse union**。per-person repeat 只作为查询独立执行的控制组；比它更快不能单独证明优于原生 GazeLLE。按人数报告定位与 coverage，避免把“图上有多人”和“同时查询多个人”混为一谈。

GazeFollow 官方测试的旧记录为 4,782 个 single-head 图像查询，因此多人主张需要另有真实多头查询子集。读取同帧多个真实头框仍是单帧推理，不引入时序模块。

### 8.3 最小消融

- 同预算 learned Router 对 untrained / person-independent score，验证人物条件与学习是否起作用。
- 固定其他配置的预算比较与真实 Pareto，不从测试集上不断挑新的 K。
- 若正文主张多层提高剪枝耐受性，再完成 B/C/D/E；否则将多层明确列为实现选择。
- 若主张上下文保留有益，需要固定总预算的明确修改与对照，不能把 K75 相对 K50 的回升直接当成已分离的因果证据。

### 8.4 2026-11-22 确认的新训练/选模口径

用户选择后续对齐公开 train/test 训练流程：使用全部官方 train，逐 epoch 评测 test，不另切 val。GazeFollow 以 Min L2、VAT 以 L2 作为预先确定的 checkpoint 主选择指标，选定同一个 epoch 后报告该模型全部指标；并列情况在启动前固定。VAT 训练中稀疏采样的 test 指标与最终逐帧 test 分开记录。原官方 GF 脚本确实跟踪 test Min L2 最好的 epoch；原官方 VAT 脚本逐 epoch 评测，但没有自动定义最佳 checkpoint 选择器。[GF 脚本](https://github.com/fkryan/gazelle/blob/main/scripts/train_gazefollow.py)、[VAT 脚本](https://github.com/fkryan/gazelle/blob/main/scripts/train_vat.py)

新 run 明确登记为 `test-selected`，dense 与 Router 使用相同选择规则。已有 `full_train_no_eval / fixed_final` 和 `source_video_holdout / validation_selected` 记录保留原标签；不改名、不与新 run 混拼成一个协议。官方已发布 checkpoint 的选择信息与自行重训模型的选择信息分别列明。

这项是本轮文档决定，**现有训练/评测入口尚未改为新协议**。本轮未启动训练、推理或测速。

## 9. 已有证据与目前能写的结果

原始小报告的指标、来源 SHA、PDF SHA 和旧效率汇总已登记到 [WRITING_EVIDENCE_20261122.json](../results/WRITING_EVIDENCE_20261122.json)，本轮只读取和汇总旧文件。

### 9.1 GazeFollow：旧 fixed-final 实验

| DINOv3 多层模型 | AUC ↑ | Avg L2 ↓ | Min L2 ↓ |
|---|---:|---:|---:|
| Matched dense | 0.957264 | 0.102592 | 0.043748 |
| Router K50 | 0.956462 | 0.104375 | 0.045044 |
| K50 − matched dense | −0.000802 | +0.001783 | +0.001296 |

正式 GF 准确率工件是 full-train 后固定 final checkpoint。旧效率矩阵则使用另一份 split-clean checkpoint、FP16 dummy 输入和开启的 autocast cache，不能把两者直接当成同一最终 checkpoint 的联合测量。

旧效率相对自身 DINOv3 多层 dense 的正反序比值几何均值：B1 `1.001525×`、B4 `1.060676×`、B8 `1.148545×`。没有显存收益，B1 基本持平。它们不是相对官方 DINOv2 GazeLLE 的加速。

会话此前只读核对的服务器官方 GazeLLE-B/14 GF 结果为 AUC `0.956141`、Avg L2 `0.104136`、Min L2 `0.045332`。据此 DINOv3 K50 的 AUC 略高、Avg L2 略差；不能写成各指标均胜出。对应 T4 原始 JSON 尚未经 Git 同步到本地，本轮本地证据包没有将这三个会话数值冒充已归档的原始报告。完整跨模型表在报告回传与统一测速后形成。

### 9.2 VAT：旧 source-video holdout

| 模型 | AUC ↑ | L2 ↓ | In-out AP ↑ |
|---|---:|---:|---:|
| Dense base | 0.948849 | 0.100565 | 0.920883 |
| Matched dense | 0.946727 | 0.103769 | 0.938995 |
| K50 | 0.925232 | 0.108179 | 0.940644 |
| K75 | 0.936652 | 0.107026 | 0.941260 |

四个验证源视频、2,051 帧、4,034 个人物查询，其中 2,969 个画内。K50/K75 的注视点保留率分别为 96.40%/99.02%，但定位仍退化。扩大 K 同时改变上下文和深层表示，当前支持“仅保住目标位置并不足够”的观察，尚未分离各退化原因。

尚无这条 Router 的 VAT full-train official-test 与正式效率结果。旧 holdout 结论保留；新全量 train/test 实验可以检验不同数据预算与选模口径下的结果，不能预设它会消除相对 dense 的差距。

## 10. 论文组织与下一步

正文按问题顺序组织：Introduction（训练轻量与推理计算的区别）→ Related Work（冻结编码器、通用 token 稀疏化、相关 GazeSpot 的边界）→ Method（support、union、sparse suffix、fill、decoder）→ Experiments（完整系统、matched dense、效率、条件化与人数）→ Conclusion（仅总结已验证的权衡）。

下一步先完成同一最终 checkpoint 的统一跨模型测速入口，并把旧正式小报告纳入可复现证据；再实现已选定的新 full-train/test-selected 协议。服务器脚本届时给完整手动 nohup 命令，本轮不生成执行命令、不启动服务器任务。多人和消融按依赖逐项推进，不同时铺开半成品。
