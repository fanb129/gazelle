# CSCWD Router：问题定义、方法边界与行文逻辑

SOTA-1 更新（2026-11-24）：按用户决定，主目标转为近期强方法的精度—成本比较，保留GazeLLE参照。公开指标/权重与比较范围见 [近期方法核对](../results/SOTA_LANDSCAPE_20261124.md)；不以全面超过GazeLLE作为推进前提，不改变train/test、seed0或ICASSP贡献边界。

更新：2026-11-23，E2；写作始于 W1。主方法采用 **DINOv3 ViT-B/16、ViT-L/16 + 多层表示 + gaze-conditioned Router**。DINOv2 单层适配保留为已有工程与对照路线。本文件确定论文怎么论证；VAT full-test 结果见第 9.3 节，不把尚未完成的跨模型测速或消融写成结果。

E0 最新执行：GF/VAT 各跑 ViT-B/16 与 ViT-L/16，所有新训练 base seed 0。四组入口已实现，从兼容旧 decoder 新训 support 3 + sparse decoder 5 epochs；不继承 seed3106 Router、不新增 dense。完整命令见 `../RUN_DINOV3_TRAIN_TEST.md`。

W2 历史写作决定：限时实验以 [EXPERIMENT_PLAN.md](../EXPERIMENT_PLAN.md) 为当前执行清单；不新增 dense/matched dense 训练，只训练 Router 模型。已有同表示 dense 证据按旧协议单独保留，当前新主表评价完整系统。

## 1. 一句话定位

**利用人物条件决定哪些图像位置值得继续执行深层视觉计算，在注视定位精度和实际推理成本之间取得更好的权衡。**

工作标题：**Gaze-Conditioned Spatial Routing for Efficient Gaze Target Estimation**。

完整方法可以由 DINOv3、多层解码和 Router 组成；新增贡献集中在条件化计算分配。DINOv3 是已有编码器，多层表示是沿用的基础配置。目标是在纯RGB、真实头框、单帧设置下，取得接近近期高精度方法的定位表现，并验证人物条件化计算分配能否带来更好的完整推理成本。Mi et al.的object-aware方法、TextGaze和OmniGF进入重点比较，GazeLLE保留为直接相关基线；速度结论使用统一硬件实测。

## 2. 从近期方法的计算需求引出问题

行文依次回答六个问题：

| 段落 | 要建立的事实或问题 | 如何接到下一步 |
|---|---|---|
| 1 | 近期方法通过对象语义、多层特征或视觉语言推理提高精度；GazeLLE提供冻结编码器加轻量解码器的直接基础 | 比较重点是完整推理成本及精度，可训练参数少不等于编码器计算少 |
| 2 | 冻结编码器仍对所有图像位置执行完整深度的前向；人物条件主要在注视解码时使用 | 能否把人物条件提前引入编码过程？ |
| 3 | 同图不同人物需要的证据不同；头框确定查询身份和相对几何，人物外观及场景提供可学习的注视线索 | 需要预测与查询相关、具有高目标召回率的 support，而非仅按通用显著性选 token |
| 4 | 在编码器中层做出决策，仅让选中位置继续执行深层 blocks | Router 直接改变实际执行的 token 数 |
| 5 | decoder 仍需完整空间网格，多人还需共享图像计算 | 用中层特征回填未选位置；多人 support 聚合后执行一次稀疏 suffix |
| 6 | 完整系统与机制贡献是两个不同问题 | 主表比较近期强方法并保留GazeLLE；统一硬件验证成本，同表示控制/预算响应/条件化消融解释Router增量，旧matched dense保留原协议 |

“全图完整计算”是执行方式，不等于已经证明未选区域全部无用。VAT 旧结果提示，目标以外的关系上下文也可能影响定位。

GazeLLE 已支持一张图只编码一次、供多个人物查询复用。本文不能把“从 N 次 backbone 变为 1 次”作为相对 GazeLLE 的改进。新增多人机制是：**在共享编码的前提下，把多个查询的需求合并为同一条有限预算的稀疏深层路径**。[官方多人推理说明](https://github.com/fkryan/gazelle#usage)

## 3. DINOv3 与多层特征怎样引入

### 3.1 DINOv3：选择实现载体

在提出“人物条件化计算分配”之后，再说明实现采用冻结的 DINOv3 ViT-B/16 和 ViT-L/16。它提供可复用的稠密视觉表示；本轮直接复用仓库中已有的 DINOv3 路由实现和训练基础。DINOv3 的来源是第三方预训练模型，不列为本文贡献。[DINOv3 技术报告](https://arxiv.org/abs/2508.10104)

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

上述是 ViT-B。ViT-L 使用 blocks 5/11/17/23，在 block 11 后路由，前后各 12 blocks；输入仍是 512，网格仍是 32×32，通道由 B 的 768 变为 L 的 1024。两个规模各自使用匹配的初始化。

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
3. 以官方 GazeLLE 的实测和公开文献为系统对照，报告定位、覆盖率及模型前向效率；已有相同表示 dense 结果按其原协议讨论计算收益与精度代价。

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

**第四段，实现选择。** 我们在冻结的 DINOv3 ViT-B/16 和 ViT-L/16 上实现该机制，并沿用多层表示解码。Router 位于编码器中部，之后仅对选中 patch 执行深层 blocks。完整早期特征和中层回填维持 decoder 所需的空间网格。DINOv3 与多层访问是采用的表示基础，新增机制是实际稀疏执行及其人物条件化决策。

**第五段，验证。** 新主表与官方 GazeLLE 比较完整系统，评价同时报告定位、support 覆盖率、实际延迟和吞吐，并在同一已训练模型上分析预算与多人查询。旧 matched dense 按其原协议单独讨论；本轮不新增 dense 训练，不把新主表差异全部归因于 Router。若余量允许，再补 scene-only Router 的同预算训练。结尾的具体数值及“优于”“保持”用语在统一实测后填写。

### 英文工作稿

Gaze target estimation predicts a person's visual target from an image and a head bounding box, supporting attention understanding in human–computer interaction and collaborative settings. Frozen visual encoders coupled with lightweight gaze decoders, as exemplified by Gaze-LLE, substantially reduce the parameters that require task-specific training. Their inference cost, however, also depends on the computation performed by the visual encoder.

Freezing the encoder does not remove its forward computation. In a dense frozen-encoder pipeline, all image locations pass through the full encoder before the gaze decoder uses the person query. This motivates a question: can person-specific evidence guide the allocation of deep visual computation before scene encoding is complete?

This decision requires information about the queried person as well as the scene. A head box identifies the query and defines relative spatial coordinates, while head appearance and intermediate scene features provide evidence for learning query-relevant regions. We predict a person-conditioned support at an intermediate encoder layer and select locations under a fixed token budget. For multiple queries in one image, their supports are aggregated into a shared route.

We instantiate this mechanism with frozen DINOv3 ViT-B/16 and ViT-L/16 encoders and an existing multi-level decoding configuration. Only selected patch tokens continue through the encoder suffix. Dense early features and an intermediate-feature fallback preserve the spatial grids required for gaze decoding. Our study concerns person-conditioned token selection and sparse execution; the pretrained encoder and multi-level representation form the adopted baseline configuration.

We compare the complete system with official Gaze-LLE and report localization, support coverage, model-forward latency, and throughput. Additional analyses examine token budget and multi-person queries. Existing matched-dense results are discussed under their original protocol, separately from the new test-selected experiments. Quantitative outcome statements will be completed after the corresponding measurements.

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
| C | DINOv3-B/16，四层 raw-concat matched dense，512 输入 | 已有旧协议 GF 机制对照；本轮不新增训练，与新表分开 |
| D | DINOv3-B/16，四层 raw-concat + Router K50，512 输入 | 完整方法；旧 GF/VAT holdout 保留，新协议 VAT B/L 见第 9.3 节 |
| E（用于层级耐受性主张） | DINOv3-B/16，最后一层 + 相同 Router | 与 B/C/D 共同检验多层是否降低剪枝损失；尚未完成 |

严格归因 Router 收益需要 C 与 D 使用相同表示、dense 初始化、decoder 适配预算和选模规则，并独立登记 Router support 训练成本。本轮不新增 C，旧 C/D 配对按旧协议保留；新 D 与 A 的差异作为完整系统比较。K100 是同路径无剪枝 control，不替代 C 的原生 dense 执行时间。

A 与 D 比较完整系统，不能把其差异全部归因于 Router。若 D 只在 AUC 略高而 Avg L2 更差，按指标分别报告或描述为相近精度。若 D 仅快于 C，结论就是加速自身 dense；只有同条件直接测速优于 A 后，才能写相对 GazeLLE 加速。B/14 与 B/16 的区别、448/512 输入均列入表格；不能将 B/14 的结果泛称所有 GazeLLE，包括 L/14。

### 8.2 效率与多人

统一同一 GPU、精度、计时范围、batch 和输入样本来源，保留各模型被评估的输入尺寸。数据加载、预处理和 H2D 在计时外时，表头写 model-forward latency。forward 包含 Router、gather/scatter、融合和 decoder。按 W2 限时清单必做 B1/B8，B4 可选，报告正反序、warmup、重复次数和显存；准确率使用哪个 checkpoint，新的主表效率就测哪个 checkpoint。

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

该协议已在 E0 独立入口实现；E2 的 VAT B/L 已完成训练与逐帧 test，并按该规则选模。原历史入口和旧结果保持原协议。

## 9. 已有证据与目前能写的结果

原始小报告的指标、来源 SHA、PDF SHA 和旧效率汇总已登记到 [WRITING_EVIDENCE_20261122.json](../results/WRITING_EVIDENCE_20261122.json)，该文件保留旧证据；新 E2 原始结果与核验见第 9.3 节。

### 9.1 GazeFollow：旧 fixed-final 实验

| DINOv3 多层模型 | AUC ↑ | Avg L2 ↓ | Min L2 ↓ |
|---|---:|---:|---:|
| Matched dense | 0.957264 | 0.102592 | 0.043748 |
| Router K50 | 0.956462 | 0.104375 | 0.045044 |
| K50 − matched dense | −0.000802 | +0.001783 | +0.001296 |

正式 GF 准确率工件是 full-train 后固定 final checkpoint。旧效率矩阵则使用另一份 split-clean checkpoint、FP16 dummy 输入和开启的 autocast cache，不能把两者直接当成同一最终 checkpoint 的联合测量。

旧效率相对自身 DINOv3 多层 dense 的正反序比值几何均值：B1 `1.001525×`、B4 `1.060676×`、B8 `1.148545×`。没有显存收益，B1 基本持平。它们不是相对官方 DINOv2 GazeLLE 的加速。

会话此前只读核对的服务器官方 GazeLLE-B/14 GF 结果为 AUC `0.956141`、Avg L2 `0.104136`、Min L2 `0.045332`。据此 DINOv3 K50 的 AUC 略高、Avg L2 略差；不能写成各指标均胜出。对应 `t4_gf_fulltrain_seed3106/accuracy/dense_official.json` 已随 `8f2ae9c` 回传，并核对 decoder 与官方权重一致；本次只归档这一份官方精度报告，不视为 T4 总验收。新的 GF-B 比较见第 9.4 节，统一测速尚未完成。

### 9.2 VAT：旧 source-video holdout

| 模型 | AUC ↑ | L2 ↓ | In-out AP ↑ |
|---|---:|---:|---:|
| Dense base | 0.948849 | 0.100565 | 0.920883 |
| Matched dense | 0.946727 | 0.103769 | 0.938995 |
| K50 | 0.925232 | 0.108179 | 0.940644 |
| K75 | 0.936652 | 0.107026 | 0.941260 |

四个验证源视频、2,051 帧、4,034 个人物查询，其中 2,969 个画内。K50/K75 的注视点保留率分别为 96.40%/99.02%，但定位仍退化。扩大 K 同时改变上下文和深层表示，当前支持“仅保住目标位置并不足够”的观察，尚未分离各退化原因。

旧 holdout 结论单独保留。新的官方 train/test 结果如下，正式效率仍待测量。

### 9.3 VAT：seed0 full-train / test-selected（2026-11-23）

GPU 1 的 B/L 队列均完成，各 support 3 + sparse decoder/inout 5 epochs。结果提交 `0fc2844` 已经 Git 回传本机，详见 [E2 审查](../results/E2_VAT_SEED0_REVIEW.md)。

| K50 模型 | AUC ↑ | L2 ↓ | In-out AP ↑ | 所选 decoder epoch（从 1 开始） |
|---|---:|---:|---:|---:|
| ViT-B raw-concat + Router | 0.913548 | 0.110971 | 0.901738 | 4 |
| ViT-L raw-concat + Router | 0.923034 | 0.097970 | 0.916599 | 2 |

逐轮 test rate 6，以 L2 最小选定一个模型，最终 rate-1 test 各 31,978 个查询、20,655 画内。14 个大工件 SHA、选模/冻结 state 和两组逐查询明细核验通过；导出指标重聚合与原报告差值均 0，没有新增模型 forward 或测速。

相对计划中同规模 GazeLLE 文献参照，B 的 AP 较高但 AUC/L2 较差；L 的 L2/AP 较好但 AUC 较差。不能写成 VAT 全面优于 GazeLLE。L 的选择来自 K75 训练轮，实际评测仍为 K50；不将后续轮更高 AUC 与较早轮较低 L2 混拼。尚未做同口径官方 checkpoint 推理和效率比较，差距也未被分解为 backbone、表示和 Router 的各自作用。

### 9.4 GF-B 与 VAT 续训（2026-11-23）

结果提交 `8f2ae9c` 已回传，详见 [联合审查](../results/E1_GFB_E2C_REVIEW.md)。GF-B 按 test Min L2 选第 5 轮，AUC/Avg/Min L2 为 `0.956236/0.101550/0.043420`；已有官方 GazeLLE-B 报告为 `0.956141/0.104136/0.045332`。两者 test 标注和 4,782 个单头查询相同、均 FP32，Avg/Min L2 相对降低 2.48%/4.22%，AUC 基本持平；这属于完整系统比较，不把差异全部归因于 Router，也没有速度结论。

VAT B/L 各新增 8 个低 LR K50 epochs，decoder 确有更新，但新轮次没有改善 L2，仍保留旧 best，最终逐帧指标不变。原始 VAT dense 初始化/routed K100 的 rate-6 AUC 为 B `0.939097`、L `0.944592`，三项精度路径差均为 0；同口径选中 K50 的 AUC 为 `0.913401/0.922349`，下降仍在。当前不再用原样追加 decoder epochs 作为提升依据，接续预算响应与完整 GF Router→VAT 迁移；GF-L 继续训练，GF-B 可先测 E3。

### 9.5 完整 GF-B Router→VAT-B（2026-11-23）

结果提交 `170ace8` 已回传，详见 [E2-T 审查](../results/E2T_GF_TO_VAT_REVIEW.md)。继承 GF-B 的完整 Router 与热图 decoder，仅新增 in/out，K50、seed 0、冻结 backbone；8 epochs 中首轮固定 Router、后 7 轮联合适配。按 rate-6 test L2 选第 1 轮，最终 rate-1 AUC/L2/AP `0.940033/0.100131/0.883183`。相对旧 VAT-B 差 `+0.026485/-0.010840/-0.018555`，L2 相对降低 9.77%。选中模型的 Router 与 GF 来源逐值相同，因此证据支持完整 GF 迁移恢复定位，不能写成联合 VAT Router 更新改善定位。

后续联合阶段提高 soft coverage 和 GT point coverage，但 hard Gaussian mass coverage 下降，L2 没有改善。Router、decoder 和 in/out token 同时更新，尚不能单独归因；AP 仍较低，主表应保留同一选中模型的三项指标。下一项优先仅适配分类头、保护定位；尚无同硬件正式效率结果，也不能将完整系统改善全部归因于人物条件化机制。

### 9.6 GF-L 中断后以实际 3+4 收口（2026-11-23）

服务器关机中断最后一轮，用户决定不补跑。实际完成 support 3 + decoder 4，按预定 test Min L2 选第 2 轮，K50 AUC/Avg/Min L2 `0.960007/0.094061/0.037721`。第 4 轮为 `0.960678/0.093370/0.037724`，保留完整行，不因较高 AUC 改变选模规则。结果提交 `459651a` 已回传；从 last 内嵌 best 导出的 52 tensors 逐值一致，原恢复文件不变，详见 [E1-L 审查](../results/E1L_GF_INTERRUPTED_REVIEW.md)。

报告保留原计划 3+5 与实际 3+4，标记 user-closed，指标来自完整 epoch test，没有独立复评或逐查询导出。相对已登记 GazeLLE-L 文献参照，当前 AUC/定位有竞争力；仍需同硬件效率，不能提前写精度与速度双优。历史 initializer 的 warm-start 标签保留，当前运行的四层配置由 manifest 明确登记。当前先接效率与完整 GF-L→VAT-L，不新增 GF-L dense。

### 9.7 GF-B / VAT-B 正式效率（2026-11-24）

两组各 B1/B8 正反序矩阵已回传，详见 [E3-B 审查](../results/E3_B_EFFICIENCY_REVIEW.md)。相对同规模官方 GazeLLE 的 latency speedup，GF-B 为 B1 `0.7822×`、B8 `0.9918×`，VAT-B 为 `0.8255×/1.0251×`。B1 更慢，B8 GF 基本持平、VAT 仅小幅更快，B8峰值 allocated memory 高约59.5%。当前不能写成完整系统明显更快或更省显存。

同一 checkpoint 下，K50 相对同路由路径 K100 在 B8 为 `1.2519×/1.2559×`，suffix token 实际从1029降至517；这一结果说明当前实现能减少自身全token路径成本，不替代原生DINOv3 dense对照。完整系统仍额外承担四层表示、回填、人物条件和融合成本。当前可保留定位收益与批量成本接近的权衡，待阶段profile与推理等价优化补齐后再更新效率主张。GF-L和VAT-L后续结果独立报告。

## 10. 论文组织与下一步

用户接受当前VAT-B，分类头适配后移。GF-L按实际3+4收口，效率与完整GF-L→VAT-L按原版本接续。SOTA-1后优先准备TextGaze官方B/L权重推理比较，再检查OmniGF；Mi暂无公开权重，先保留论文精度参照。阶段profile、同权重native full-token控制及none全一门/高维repeat等价快路径保留为后续优化，不新增dense训练。预算响应、人物条件化对照与真实多人union接续。

正文按问题顺序组织：Introduction（训练轻量与推理计算的区别）→ Related Work（冻结编码器、通用 token 稀疏化、相关 GazeSpot 的边界）→ Method（support、union、sparse suffix、fill、decoder）→ Experiments（完整系统、matched dense、效率、条件化与人数）→ Conclusion（仅总结已验证的权衡）。

E0 执行见 `../EXPERIMENT_PLAN.md` 与 `../RUN_DINOV3_TRAIN_TEST.md`：四组均 seed0，GF-L 实际 3+4，其余 3+5，不新增 dense 训练。VAT B/L 及额外 8 轮续训、GF-B、GF-L 收口均已回传核验；旧 GPU 0/1 命令保留供复现，不重启已收口任务。对实际选中的同一 checkpoint 补统一推理测速与多人分析。旧正式报告保留原协议。
