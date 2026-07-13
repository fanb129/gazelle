# AAAI 2027 Venue Fit 调研：Gaze 领域接收论文、Idea 类型与写作风格

> 核验日期：2026-07-13  
> 用途：在决定 GazeSpot 的方法修改之前，先判断 AAAI-27 的正式评审标准、gaze 相关接收论文的贡献类型，以及可以迁移的叙事与实验模式。  
> 边界：本文区分“AAAI 官方明文标准”“从有限接收样本归纳的观察”和“对 GazeSpot 的条件性推论”。接收论文样本不能用于反推出完整的录用概率或隐含偏好。

## 摘要

AAAI-27 官方没有规定论文必须采用复杂架构、必须刷新全部 SOTA，或必须属于某一种 idea 类型。主会明确接受 methodological、algorithmic、empirical、integrative 与 critical contributions；评审维度是 significance/novelty、soundness、AAAI relevance、clarity 与 reproducibility。更关键的是，AAAI-27 官方明确表示，相比只在狭窄子领域做增量 SOTA，更偏好探索新 territory、提出研究问题或对更广 AI 社区有意义的扎实技术论文。

在 gaze 领域，AAAI 已正式接收与本项目高度相关的论文：AAAI-24 主会的 *Gaze Target Detection by Merging Human Attention and Activity Cues* 直接使用 GazeFollow 和 VideoAttentionTarget；2022-2026 年还连续接收了 PureGaze、Gaze-Consistent Feature、Gaze Frontalization、SUGE、CLIP-Gaze、Gaze Label Alignment、MoEGaze 等 gaze generalization 工作。它们的方法复杂度差异很大，但共享一个稳定模式：先把 task-specific failure 抽象为更一般的学习问题，再让方法结构对应这个问题。AAAI-24 的直接 gaze-target 论文只有两个主要模块和四张实验表，说明“模块简单”本身并非拒稿理由；但其相对 baseline 的关键提升达到约 14%-21%，并补齐模块替代、定性证据与运行成本。相比之下，小幅增益、缺少公平融合对照、模块机理描述超过实际实现，仍会直接触发 AAAI 的 soundness 和 significance 风险。

因此，本轮不应先锁定新的 SASA/GGSF 形式。下一步应先选择 GazeSpot 要遵循的 AAAI paper archetype，并用训练成本较低的 VAT 做现象验证。用户实测训练成本为：GazeFollow 每次约 36 小时，VAT 每次约 7 小时；在单卡 3090 和 AAAI-27 截止前约两周的条件下，VAT 应承担 idea discovery/ablation，GazeFollow 只用于少数已经筛选后的决定性实验。

## 1. Research Brief

### 1.1 Research Questions

- **RQ1：** AAAI-27 官方如何定义可接受贡献与评审标准？它是否明确偏好某一种 idea 或写作风格？
- **RQ2：** AAAI 近年是否有 gaze target、gaze prediction 或 human visual attention 的正式接收论文？这些论文用什么问题抽象、方法结构和实验规模获得完整贡献？
- **RQ3：** 与 GazeSpot 最接近的接收论文，在 Introduction、Figure 1/2、challenge-module mapping、claim 和实验组织上有哪些可迁移模式？

### 1.2 检索与纳入方法

- 时间范围：主要覆盖 AAAI 2024-2026，必要时回溯更早工作；AAAI-27 使用当前官方 CFP。
- 一手来源优先：AAAI 官方 CFP、submission instructions、AAAI proceedings 页面与官方 PDF。
- 直接领域：gaze target detection/following、egocentric gaze prediction、appearance-based gaze estimation。
- 邻近方法：foundation-model adaptation、query-conditioned dense prediction、multi-scale fusion、robustness/generalization。
- 只有正式 proceedings 页面可核验的论文才被称为“AAAI accepted paper”；arXiv、workshop 与其他会议不用于推断 AAAI 风格。

## 2. AAAI-27 官方标准：允许多种贡献，但明确反对狭窄增量

### 2.1 官方接受的贡献类型

[AAAI-27 Main Technical Track CFP](https://aaai.org/conference/aaai/aaai-27/main-technical-track-call/) 明确列出以下贡献类型：

- theoretical；
- methodological；
- algorithmic；
- empirical；
- integrative，即连接不同 AI 子领域的方法与思想；
- critical，即对既有目标、假设或方法作原则性分析。

这意味着 AAAI 不存在“只有复杂新网络才能投稿”的官方规则。一个可靠的 empirical finding、跨领域连接或有普适意义的 critical analysis 都可以成为贡献，但必须达到 substantive contribution 的标准。

### 2.2 核心评审维度

官方列出的评分维度是：

1. 贡献的 significance 与 novelty，包括问题、方法、实验和分析；
2. claims 的理论和/或实证 soundness；
3. 对 AAAI community 的 relevance；
4. exposition clarity；
5. responsible research 与 reproducibility。

对 GazeSpot 最相关的是：**novelty 不只来自模块，也可以来自研究问题或分析；但论文中每个强 claim 都必须被实验直接支持。** 因此，ACM MM 审稿中指出的 frustum overclaim、融合对照不公平和 crowded subset 构造不清，在 AAAI 语境下仍是核心风险，不是换一种写法即可绕过的问题。

### 2.3 AAAI 明确写出的“偏好”

AAAI-27 CFP 唯一可以直接称为 venue preference 的表述是：探索新 territory、指出新方向、提出新问题、回答研究问题，或提出对单一 AI 子领域之外也有意义的方法，比只在狭窄子领域做 incremental SOTA 更受偏好。

这一条不能被曲解为“不需要性能”或“故事比实验重要”。官方同时要求 empirical soundness；更准确的理解是：**性能提升需要服务于一个更有意义的问题或机制结论，而不是成为论文唯一内容。**

### 2.4 Main Track 与 AI for Social Impact 不应混淆

AAAI-27 Main Track 与 AI for Social Impact (AISI) 使用不同 rubric。AISI 更强调社会问题的重要性、AI 方法与应用问题的适配、部署/后续工作和社会影响，而不单纯奖励 technical novelty。

现有 GazeSpot 使用公开通用 benchmark，并未提出特定医疗/教育/社会应用、部署路径或新用户群体，因此目前更适合 Main Technical Track。AAAI-26 的 autistic-children gaze paper 属于 AISI，其“新应用 + 新数据集 + 临床意义”故事不能直接移植成 GazeSpot 的 main-track 叙事。

### 2.5 两阶段评审和七页正文的写作影响

- AAAI-27 Main Track 采用两阶段评审，Phase 1 先由两位 human reviewers 评估；未过 Phase 1 就没有 author response。
- 正文只有 7 页，最大总长 9 页，超过第 7 页只能放 references。
- Supplementary material 不是 reviewers 必看内容；所有决定 accept/reject 的核心证据必须进入正文。

条件性推论：第一阶段必须让 reviewer 在 Abstract、Introduction、Figure 1 与主要实验表中迅速读懂“现象是什么、为什么现有方法不足、我们的 idea 为什么自然、关键证据在哪里”。依赖 appendix 才能解释公平性或核心消融的论文风险很高。

## 3. AAAI 中与 GazeSpot 最相关的正式论文

### 3.1 总览：直接领域并不空白

| 论文 | Track | Paper archetype | 核心问题抽象 | 方法规模 | 主要证据 | 对 GazeSpot 的参考价值 |
|---|---|---|---|---|---|---|
| Gaze Target Detection by Merging Human Attention and Activity Cues, AAAI-24 | Main/CV | 新 cue relationship + technique | 复杂背景中的目标不仅由 saliency/geometry 决定，还与人物活动交互相关 | soft gaze attention + interaction branch | GF/VAT；主表、2 组模块消融、定性、运行时间 | 最直接的 task/benchmark/写作参照 |
| Ego-PMOVE, AAAI-26 | Main/CV | 新 setting + multi-view technique | exo-view 提供补充信息，但严格对齐会引入 spatial/semantic interference | prompt disentanglement + 3 experts + router + decoder/loss | 3 datasets；7 张表；替代融合、缺失 view、failure cases | challenge-module mapping 与完整实验参照 |
| MoEGaze, AAAI-26 | Main/CV | empirical insight + generalization method | 用户外观差异导致 cross-domain failure；相似训练外观更有效 | appearance routing + specialized experts + prototype router | motivation experiment；4 cross-domain tasks；data efficiency；消融 | “先验证现象、再提出方法”的最佳参照 |
| Toward Gaze Target Detection of Young Autistic Children, AAAI-26 | AISI | new application/problem + dataset + method | neurotypical data 无法覆盖 autistic children，且 face-directed gaze 严重少数 | 新 AGT dataset + social/non-social experts + context gate | 16,582 frames；类别分析；domain-specific metrics | 说明 application paper 如何成立，但不等价于 Main Track |
| Mind the Gap: Human-AI Visual Attention for Accident Anticipation, AAAI-26 | AI Alignment | empirical/benchmark analysis | 时间关键任务中的 human-AI attention alignment 缺少定量研究 | dataset/analysis + VLM evaluation | gaze annotations；动态时间分析；attention-guided evaluation | 说明 empirical contribution 可接收，但属于不同 special track |

### 3.2 AAAI-24 直接 gaze-target 论文：方法不复杂，但证据闭环

该文是最重要的直接参照，因为它与 GazeSpot 使用相同的 GazeFollow/VideoAttentionTarget 任务，而不是邻近 gaze direction 或 egocentric saliency。

它的 Introduction 逻辑是：

1. gaze target 比 gaze direction 更适合理解现实人类注意；
2. 既有方法主要沿 gaze direction 做 saliency 或引入 scene geometry；
3. 复杂背景下仍会失败，因为没有利用 attention 与 activity cues 的关系；
4. Figure 1 用一个人物看 frisbee 的例子对比 existing method 与 ours；
5. Figure 2 展示 body-part/object interaction 与 gaze target 的关系，同时主动展示 non-interactive target 和 full-body interaction 不一定等于 gaze target 的反例；
6. 方法因此采用 soft gaze attention 和 body-part/object interaction 两条分支。

实验正文包含：

- GF 与 VAT 主结果；
- soft gaze attention 对照；
- no-HOI、full-body HOI、是否 gaze-guided 等 interaction alternatives；
- 模块级可视化；
- 与 depth/3D 方法的推理时间比较。

它的完整方法只包含两个主要思想，并不比 GazeSpot 天然复杂。关键差异是：其主模块相对 baseline 在 GF Avg L2 上报告约 13.9% 相对改善、Min L2 约 20.8%，而且每个 claim 均有替代模块或反例支持。该样本说明“简单模块也能被 AAAI 接收”，但不支持“微小增益和不完整对照也没关系”。

### 3.3 MoEGaze：先做 motivation experiment，再定义模块

MoEGaze 不是从“MoE 很流行”起步，而是先报告一个 empirical observation：测试对象与训练对象的 appearance divergence 越大，gaze error 越高；用 appearance-similar 子集训练反而可能优于 generic full-data training。Figure 2 专门验证 self/in-group/random training data 的差异。

在现象成立后，MoE 才成为自然方法：不同 experts 学习不同 appearance subsets，prototype router 为 unseen users 选择合适 expert。其贡献顺序也是“现象发现 → 基于发现的方法 → router → cross-domain/data-efficiency 结果”。

对本项目的可迁移点不是 MoE，而是研究顺序：在决定 SASA/GGSF 如何改之前，应先证明具体 failure phenomenon，例如不同层是否真的对不同人物/场景互补、错误是否真的来自 query confusion、当前 spatial prior 是否真的压制干扰。没有这个现象层，任何新模块都容易继续被评为 arbitrary design。

### 3.4 Ego-PMOVE：新信息既有 complement 也有 interference

Ego-PMOVE 的故事不是“多一个 exocentric branch 会更好”，而是同时陈述两面：exo-view 有互补信息，但 viewpoint shift 与 semantic granularity gap 会使强行融合引入噪声。Figure 1 直接标出 complement 与 interference，随后三个 expert 分别对应 shared information、view discrepancy 和 missing exocentric compensation。

这种写法的价值在于 challenge-module 一一对应；但它也带来较大的实验负担：论文使用多数据集、常规融合替代、每个 expert、router、decoder、loss、view drop 和 failure cases 共七张实验表。对于单卡 3090 和两周期限，不能无计划复制这种多模块故事。

### 3.5 AISI autistic-gaze 论文：强在新 population，不是通用 benchmark 小切片

该文的 Figure 1 先展示 autistic children 的 target distribution 与 neurotypical data 明显不同，然后说明 off-the-shelf/fine-tuned models 为什么失败；论文同时贡献首个 AGT dataset、具体社会/临床问题和 social/non-social expert method。

因此，它不能被用来论证“只要定义一个 Crowd subset 就符合 AAAI”。它成立的原因是 population、应用问题、数据分布、数据集与 method 全部一致，而不仅是从旧 test set 中切一个困难 subset。

### 3.6 更广的 AAAI gaze 谱系：共同点是把局部任务提升为一般学习问题

直接同任务的 Main Track 样本目前主要是 AAAI-24 的 attention-activity cues；因此，不能据此声称 AAAI “偏爱 gaze target detection”。但把范围扩展到 gaze estimation，可以观察到一条连续且与本项目很相关的研究谱系：

| 论文 | 年份 | 被抽象的一般问题 | 核心 idea | 对 GazeSpot 的启示 |
|---|---:|---|---|---|
| [PureGaze](https://ojs.aaai.org/index.php/AAAI/article/view/19921) | 2022 | identity、illumination、expression 等 nuisance 破坏跨域泛化 | self-adversarial feature purification | 最值得参考的问题抽象：不是“难样本”，而是可命名的无关因素 |
| [Gaze-Consistent Feature](https://ojs.aaai.org/index.php/AAAI/article/view/25406) | 2023 | gaze-relevant representation 应对 nuisance perturbation 保持一致 | nuisance perturbation + consistency learning | 用干预/一致性检验机制，而非只比较最终误差 |
| [Gaze from Origin / Gaze Frontalization](https://ojs.aaai.org/index.php/AAAI/article/view/28452) | 2024 | 连续、近乎无限的 gaze labels 使分类式 domain generalization 难以直接适用 | gaze frontalization auxiliary task | task-specific 现象可以上升为一般 continuous-label generalization 问题 |
| [Suppressing Uncertainty in Gaze Estimation](https://ojs.aaai.org/index.php/AAAI/article/view/28368) | 2024 | 野外图像同时有 input uncertainty 与 label uncertainty | 估计不确定性后 sample weighting / label correction | 两个模块来自两类可区分噪声，而不是经验拼接 |
| [CLIP-Gaze](https://ojs.aaai.org/index.php/AAAI/article/view/28496) | 2024 | gaze 数据覆盖有限，直接迁移视觉语言模型又会引入 gaze-irrelevant semantics | prompt tuning + sample-relation refinement | “使用 foundation model”本身不是理由，必须说明保留和抑制哪些知识 |
| [Test-Time Personalization with Meta Prompt](https://ojs.aaai.org/index.php/AAAI/article/view/28151) | 2024 | 个体化需要标签和大规模参数更新，测试时成本过高 | unlabeled meta prompt、少量可训练参数 | 实用约束也能成为 AAAI 问题，但须有清晰 cost axis |
| [Gaze Label Alignment](https://ojs.aaai.org/index.php/AAAI/article/view/33060) | 2025 | domain gap 不只来自 image shift，还来自采集系统与生理差异造成的 label shift | 先模拟/测量 label deviation，再做 alignment | 最典型的“先发现遗漏变量，再提出 plug-in 方法” |
| [MoEGaze](https://ojs.aaai.org/index.php/AAAI/article/view/37683) | 2026 | train/test user appearance divergence 与 error 相关 | appearance-aware experts + prototype router | 最适合资源有限项目模仿的“motivation experiment → method”路线 |
| [Privacy-Protected Generalized Gaze Estimation](https://ojs.aaai.org/index.php/AAAI/article/view/38021) | 2026 | gaze 数据多样性与隐私采集限制相互冲突 | synthetic data + domain stability adaptation | broader constraint 可支撑 data/method co-design，但实验负担较大 |

该谱系反复出现四类 paper archetype：

1. **failure diagnosis → task-specific mechanism**：PureGaze、Gaze-Consistent Feature、Gaze Frontalization、SUGE、Gaze Label Alignment；
2. **跨领域知识引入 → 控制迁移副作用**：CLIP-Gaze；
3. **empirical phenomenon → adaptive architecture**：MoEGaze；
4. **现实约束 → data/method co-design**：test-time personalization、privacy-protected gaze。

这比“AAAI 喜欢哪种网络”更有解释力：接收样本往往把 gaze 中的局部困难表述成 generalization、uncertainty、personalization、privacy、cue complementarity 等更广的 AI 问题，但仍用 gaze-specific 证据证明它，而不是只在 Introduction 中拔高。

## 4. 从接收样本归纳的写作与实验模式

以下是样本观察，不是 AAAI 官方规则。

### 4.1 稳定模式一：先命名现象或缺失关系，再命名模块

- AAAI-24 gaze target：attention-activity relationship；
- MoEGaze：appearance divergence 与 cross-subject error；
- Ego-PMOVE：cross-view complement 与 interference；
- autistic gaze：population distribution shift 与 minority social gaze。

它们都没有把“用了某个 backbone/attention/MoE”作为第一层故事。模型组件是对前述现象的响应。

### 4.2 稳定模式二：Figure 1 负责让 failure 与 key idea 可视化

Figure 1 常见两种形态：

- existing vs ours 的 motivated example，同时展示缺失 cue；
- problem structure + high-level solution，同时标出 complement/interference。

完整 architecture 通常放 Figure 2；若论文依赖 empirical insight，Figure 2 也可能先给 motivation experiment，架构后移。

对当前论文的直接含义是：单层与多层融合的定量对比若用于证明 hierarchical complementarity，更适合成为 Figure 1 的分析面板或单独 motivation figure；Figure 2 应保留为完整 data flow 与 challenge-module mapping。不能只把原 Figure 2 换成一组柱状图，而让读者失去方法总览。

### 4.3 稳定模式三：不是只做 component on/off，而是比较 naive alternatives

样本论文普遍包含普通融合、单分支、full-body interaction、无 guidance、标准 self-attention、co-training 等 alternative controls。仅给 `baseline → +module A → +module B` 很难说明选择该机制的必要性。

### 4.4 稳定模式四：结果必须匹配所声称的价值轴

- 声称复杂背景：给 difficult examples 和 interaction alternatives；
- 声称 generalization：给 cross-domain tasks；
- 声称 missing-view robustness：给 view-drop curve；
- 声称 data efficiency：给 training-ratio curve；
- 声称 lightweight：给 inference cost，而不只报 trainable parameters。

这与 AAAI 官方 soundness 标准一致：claim 的强度不能超过实验实际覆盖范围。

### 4.5 稳定模式五：先建立公平主结论，再做机制诊断

相近接收样本通常把与 claim 相关的条件变量显式放入表格：backbone、训练数据、source/target availability、参数量、in-domain/cross-domain、minority class 等。消融也不止 `Base + A + B`，而是包含：

- 组件 on/off；
- naive alternatives，例如 concat、sum、单层/多层、普通 attention、full-body interaction；
- 机制变量，例如层数、query 数、rank 或数据比例；
- 必要时提供 oracle/upper bound，说明方法真正的剩余瓶颈。

这对昂贵的 gaze target 训练尤其重要：一个基于已有 checkpoint 的 oracle 或替换实验，可能比再训练一个结构更复杂的模型更快否证错误 idea。

### 4.6 Hard subset 如何获得合理地位

接收样本中的 hard/minority setting 并非仅用“更难、提升更大”来成立。Autistic-gaze paper 有独立 population、数据分布、新数据集、minority-class 指标和 routing upper bound；野外 gaze estimation 工作则用明确采集条件解释 domain shift。

因此，Crowded subset 更稳妥的角色是 **general mechanism 的 stress test**，而不是当前论文的独立新任务。若要提高它的叙事地位，至少需要：固定且可复现的构造规则；报告整体与分组性能；控制 head size、target size、occlusion、in/out-frame ratio 等混杂因素；证明退化确实来自所声称的 multi-person interference，而不只是样本总体更难。

### 4.7 不应模仿的表面风格

- 不能因为接收论文使用 “novel/first/SOTA” 就机械复制强词；这些词的安全性取决于当时的文献和证据。
- 不能从某篇接收论文模块较多，推断 AAAI 偏好堆模块；AAAI-24 直接 gaze paper 反而只有两个主要模块。
- 不能把 AISI 的真实社会应用和新数据集故事，替换成旧 benchmark 的一个自建 subset。
- 不能把“接收样本做了 SOTA”误读成“AAAI 只看 SOTA”；官方 CFP 明确反对只有狭窄增量的工作。

## 5. 对 GazeSpot 的 Venue-Fit 约束：此处暂不决定具体方法

### 5.1 最可能匹配的 paper archetype

当前材料最适合 **empirical insight-driven technique paper**，而不是：

- new benchmark paper：没有新采集数据；
- social-impact application paper：没有新 population、部署或领域合作；
- pure SOTA paper：2026 新作和有限算力使 headline competition 风险过高；
- pure efficiency paper：当前单人 FLOPs 并不天然优于 PaGE distilled student。

但“empirical insight”究竟是什么仍未确定。候选现象必须先通过低成本实验，而不能由写作反推：

- hierarchical feature complement 是否真实、是否随 person query 改变；
- crowded failures 是否来自 query-target confusion，而不只是 head size/occlusion；
- head-conditioned spatial prior 是否抑制 distractors，还是仅等价于 CoordConv；
- DINOv3 的优势究竟来自哪一层级或哪类样本。

这里的判断不是要求论文“必须提出别人从未碰过的组件”，也不是要求为了 AAAI 改成大型架构。更准确的门槛是：原有两个模块能否被同一个机制性命题解释，并通过 matched controls 被证伪或证实。若可以，保留并重解释原模块比推倒重做更符合当前时间与算力；若不可以，才需要替换其中一个模块。

### 5.2 Idea 讨论前必须满足的证据顺序

1. 用已有模型和 VAT 做 failure taxonomy；
2. 选择一个可重复、可量化、对更广 query-conditioned prediction 有意义的现象；
3. 再根据现象设计最小方法；
4. 最后决定是否保留 SASA/GGSF 名称和模块结构。

这个顺序是本轮调研最重要的流程修正。

## 6. 单卡 3090 与截止时间下的研究预算

已知真实成本：

- GazeFollow 单次完整训练约 36 小时；
- VAT 单次完整训练约 7 小时；
- 当前为单卡 RTX 3090；
- AAAI-27 abstract/full paper deadlines 分别为 2026-07-21/07-28 AoE。

因此，在 idea 尚未确定时直接跑 GF 是高风险决策。建议将后续预算分成：

| 阶段 | 目的 | 建议资源 |
|---|---|---|
| Venue/phenomenon research | 决定故事原型与待验证现象 | 当前调研，不占 GPU |
| VAT diagnosis | failure taxonomy、layer/fusion/spatial controls | 先使用已有 checkpoint 做分析；需要训练时每个 variant 约 7h |
| VAT pilot | 筛掉 arbitrary modules | 约 4-6 个精心选择的 runs，即 28-42 GPU-hours |
| GazeFollow confirmation | 只验证 VAT 已通过的最终候选 | 原则上控制在 2-3 个决定性 runs，即 72-108 GPU-hours |
| Reliability | 评估方差 | 优先在 VAT 对关键配置多 seed；GF 不应大规模铺 seeds |

该表不是最终实验计划。只有在本报告完成、确定 paper archetype 和现象之后，才进入方法设计与 run allocation。

## 7. 对 Research Questions 的回答

**RQ1。** AAAI-27 不偏好某个固定架构类型；官方允许多种贡献。它明确偏好能探索新 territory、回答研究问题或产生跨子领域意义的扎实工作，而不是只有狭窄增量 SOTA。写作上的硬要求来自两阶段评审、7 页正文和 core evidence 不能依赖 supplementary。

**RQ2。** AAAI 存在直接 gaze-target 和一系列高度相关 gaze papers。直接同任务 Main Track 样本较少，最直接的是 AAAI-24 的 attention-activity cue paper；更广的 gaze estimation 谱系持续围绕 nuisance purification、uncertainty、domain/label shift、personalization、foundation-model transfer 与 privacy 展开。它们共同说明：更有说服力的故事不是“针对一个特殊子集加模块”，而是从该任务中识别一个更一般、可验证的学习问题。AAAI-26 的 autistic gaze 属于 AISI，不能与 Main Track 直接类比。

**RQ3。** 最稳定的可迁移模式是“现象/关系 → naive solution 为什么失败 → challenge → 对应模块 → 与 claim 一致的实验”。Figure 1/2 和 alternative controls 比模块命名更关键。对 GazeSpot 而言，应先用 VAT 找到可复现的现象，再决定如何修改原方法。

## References

[1] AAAI, “AAAI-27 Main Technical Track Call,” 2026.

[2] Yaokun Yang, Yihan Yin, Feng Lu, “Gaze Target Detection by Merging Human Attention and Activity Cues,” AAAI, 2024.

[3] Heqian Qiu et al., “Ego-PMOVE: Prompt-aware Mixture of View Experts Network for Egocentric Gaze Prediction,” AAAI, 2026.

[4] Zheng Liu, Feng Lu, “MoEGaze: A Mixture of Experts Approach for Generalizable Gaze Estimation,” AAAI, 2026.

[5] Shijian Deng et al., “Toward Gaze Target Detection of Young Autistic Children,” AAAI, 2026.

[6] Hoe Sung Ryu, Christian Wallraven, “Mind the Gap: Quantifying and Aligning Human-AI Visual Attention for Accident Anticipation,” AAAI, 2026.

[7] Zhiwei Jiang et al., “PureGaze: Purifying Gaze Feature for Generalizable Gaze Estimation,” AAAI, 2022.

[8] Yihua Cheng et al., “Learning a Generalized Gaze Estimator from Gaze-Consistent Feature,” AAAI, 2023.

[9] Yihua Cheng, Feng Lu, “Gaze from Origin: Learning Gaze Estimation by Gaze Frontalization,” AAAI, 2024.

[10] Jiawei Bao et al., “Suppressing Uncertainty in Gaze Estimation,” AAAI, 2024.

[11] Zhaokang Chen et al., “CLIP-Gaze: Towards General Gaze Estimation via Visual-Linguistic Model,” AAAI, 2024.

[12] Zheng Qin et al., “Gaze Label Alignment: Alleviating Domain Shift for Gaze Estimation,” AAAI, 2025.
