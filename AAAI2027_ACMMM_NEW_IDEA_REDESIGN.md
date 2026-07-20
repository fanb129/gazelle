# 从 GazeSpot 到新 Idea：证据审计、竞争边界与唯一推荐路线

> 核验日期：2026-07-15  
> 结论性质：研究决策报告，不是可直接投稿的论文文本。外部工作的数字均为作者报告值；内部探索性统计只有在明确标注时才能作为 pilot 线索，不能直接写入论文主表。

## 执行结论

原 ACMMM 工作中可以保留的，不是 `SASA + GGSF` 这两个模块名，而是一个更朴素的事实：**冻结 VFM 的单层表示不是 gaze target estimation 的唯一有效输入，多层特征在现有系统中确实带来了一部分定位收益。** 但当前证据不支持原来的机制解释：

- GGSF 不是物理 frustum，也没有可靠的独立贡献；它实质上是 bbox 相对坐标生成的近 identity 乘性 gate。
- SASA 不是 person-adaptive routing；它学到的是一个以最后一层为主的任务级固定缩放配方。
- “DINOv3 严重 over-smoothing”“crowd 中发生 hierarchy disagreement”“模型通过 GGSF 隔离了其他人”等主张都超出了现有证据。
- GazeSpot 与 baseline 的主比较还混合了 `448 → 512` 输入分辨率、多层输入和额外参数，不能把全部提升归因于 SASA/GGSF。

ECCV 2026 的 [Multi-scale Object-Aware Gaze Estimation via Geometric Reasoning](https://arxiv.org/abs/2606.29334v1) 已经覆盖 frozen DINOv3、多尺度浅/中/深特征、对象级表征、预测 gaze direction 和 120° FoV。因而“多尺度 + gaze cone/geometry + distractor suppression”不再是可守住的 novelty。

本报告只推荐一条新主线：

> **Counterfactual Observer–Target Binding（COTB）：把同一场景中“更换被查询人物”视为自然反事实，在场景语义和候选目标保持不变时，直接训练和评价输出是否随 observer query 正确切换。**

它不是新的 fusion block，也不是另一个 object/FoV pipeline。核心方法是一个 shared-target-aware 的 permutation contrastive objective：正确的 observer–target 配对应优于交换配对。该方向与原 crowd/distractor 观察连续，但把含混的“crowd 更难”改写成可证伪的“observer–target binding 是否失败”。

---

## 1. 审计范围与证据等级

### 1.1 已阅读材料

本轮覆盖：

- `ACMMM2026rebuttal/` 下的 ACMMM 主稿、Appendix、OpenReview 评审汇总、Gaze-LLE 论文、rebuttal strategy、P0 命令与全部 P0 结果；
- `gazelle/`、`scripts/`、`data_prep/` 中与 backbone、模型、训练、评价、Crowd 构造、bbox noise、复杂度、fusion/spatial-prior ablation 和可视化有关的代码；
- `AAAIResults/P0`、`P05`、`P1a`、`P2`、`P3` 与 `selective_gaze` 的结果，以及相应研究/实施文档；
- Multi-scale Object-Aware 的 arXiv v1 正文与实验；
- 为判断新方向碰撞，核对了 Gaze-LLE、Sharingan、GazeHTA、Object-aware Gaze Target Detection、ESCNet、MTGS、GazeMoE、HCLoRA 和 RayGazeFM 的公开论文页面或正文。

### 1.2 证据等级

- **A：直接代码或完整实验输出。** 可用于内部判定；进入论文仍需协议公平、统计可靠。
- **B：原论文表格或单次实验。** 可描述现象，不足以证明机制。
- **C：探索性诊断。** 只用于决定 pilot，不可当最终 claim。
- **外部 paper-reported。** 可用于 Related Work/SOTA 表，但不是本项目独立复现。

---

## 2. 原论文的核心故事：论文声称什么，代码实际做了什么

### 2.1 原故事的六步逻辑

1. Gaze target estimation 既需要目标语义，又需要精确空间定位。
2. 在 crowded scenes 中，深层 DINOv3 feature 被主张存在空间细节弱化和 distractor hijacking。
3. 因此不能只用最后一层，需要联合多个中间层。
4. SASA 按图像内容给不同层动态加权，兼顾浅层空间和深层语义。
5. GGSF 根据 head bbox 构造“frustum-like”空间 mask，抑制与被查询人物无关的区域。
6. 两者结合，在 crowd 中减少 distractor interference，同时保持轻量。

这个故事的直觉是连贯的，但第 2、4、5、6 步均比实现和实验能支持的范围更强。

### 2.2 真实实现

#### GGSF

[`gazelle/model.py`](/Users/fanb/src/gazelle/gazelle/model.py:26) 显示 GGSF 的输入只有：

`[x-cx, y-cy, bbox_width, bbox_height]`。

这些量经过 `4→32→1` 的 `1×1 Conv` MLP 和 sigmoid，得到一个 `[0,1]` mask；同一个 mask 乘到四层 feature 上。它没有 head crop、眼睛、头姿、2D/3D gaze direction、深度、相机参数或对象信息。decoder 随后又在 [`model.py`](/Users/fanb/src/gazelle/gazelle/model.py:301) 注入一次二值 head map，因此 GGSF 重复编码了 bbox 位置信息。

准确名称只能是 **bbox-conditioned relative-coordinate gate**，不能叫 physical frustum、gaze cone 或 physical isolation。

#### SASA

[`ScaleAwareSemanticAggregator`](/Users/fanb/src/gazelle/gazelle/model.py:96) 对每层做全局平均池化，以共享 MLP 输出四个 layer scalars，softmax 后缩放各层，最后仍是 concat，再用 `1×1 Conv` 投影。代码在 [`model.py`](/Users/fanb/src/gazelle/gazelle/model.py:129) 明确忽略 `head_token`。

因此，在没有前置 bbox gate 的情况下，同一图像里的不同人物必然得到相同 layer weights；即使有 GGSF，人物差异也只来自一个近 identity 的 bbox mask。它不是 gaze/person-specific layer routing。

#### Decoder 与比较协议

后端基本继承 Gaze-LLE：head map positional prompt、3 个 transformer blocks、deconvolution heatmap head，以及 VAT 的 in/out token。真正发生改变的是：

- baseline 用最后一层 DINOv3，默认输入 `448×448`；见 [`model_v0.py`](/Users/fanb/src/gazelle/gazelle/model_v0.py:12)；
- GazeSpot 用四层、默认输入 `512×512`；见 [`model.py`](/Users/fanb/src/gazelle/gazelle/model.py:157)；
- [`compute_flops.py`](/Users/fanb/src/gazelle/scripts/compute_flops.py:10) 也明确按不同分辨率 profile 两者。

所以原主表是“单层 448 baseline”对“多层 512 + SASA + GGSF”，不是只改变一个机制的 matched comparison。

### 2.3 原故事的准确重建

经过代码和实验审计，最诚实的版本是：

> GazeSpot 在 Gaze-LLE-style frozen-VFM decoder 上引入四层 DINOv3 feature。一个以深层为主的学习缩放配方先调整各层幅度，再 concat 投影；一个 bbox 相对坐标 gate 在进入 decoder 前轻微缩放空间 feature。整个配置在若干 benchmark 上优于单层、较低分辨率 baseline，但现有证据不能把收益归因于 person-adaptive routing 或物理 geometry。

---

## 3. 真正有效的部分与已经成立的实验事实

### 3.1 可以保留的事实

| 事实 | 证据 | 可以说什么 | 不能说什么 |
|---|---|---|---|
| 四层输入优于 last-layer 配置 | 主稿消融：GF Avg L2 `0.107→0.103`，VAT L2 `0.106→0.103` | 多层信息在当前系统中有用 | SASA 动态选择导致全部收益 |
| SASA 配置在主稿消融中再带来小幅增益 | GF `0.103→0.102`，VAT `0.103→0.101` | 学习的层幅度配方可能有用 | 样本级/person级动态 routing 有效 |
| 完整配置标准指标较强 | ViT-L：GF `0.961/0.093/0.038`；VAT `0.944/0.092/0.912` | 系统总体有竞争力 | 每个模块分别被验证 |
| GOO-Real 迁移优于 baseline | 2,146 samples：AUC `0.8583→0.8833`，L2 `0.1997→0.1761` | 完整系统在该协议下迁移更好 | GGSF 或 SASA 单独产生 OOD 增益 |
| bbox jitter 下整体差距保持 | VAT crowd 0–20% jitter 多数 AUC/L2 更好 | 完整系统对这种合成 bbox noise 没有明显崩溃 | GGSF 使系统对 detector error 具有普适鲁棒性 |
| dense scenes 的 localization 更难 | VAT 中人数增多时 L2 退化；回归中 people count 仍有关联 | crowd 可作为 stress test | 已证明存在跨人物内部干扰或 hierarchy disagreement |

### 3.2 真正有效模块的判定

#### 多层 representation：保留，但降级为技术基础

P0 hierarchy proxy 显示不同层并不完全冗余；主稿 progressive ablation 也显示多层 concat 有收益。P0.5 中把 learned weights 固定为全局均值，几乎复现原输出；把权重跨帧 shuffle 也几乎不变。这说明有效部分更像：

> **四层 feature + 以最后一层为主的稳定深度配方**，而不是 dynamic routing。

Multi-scale Object-Aware 已经使“使用 DINOv3 多层特征”失去 novelty，因此它可以作为 backbone design，不应再做第一贡献。

#### GGSF：删除为贡献

内部行为审计：

- mask mean 约 `0.9482`；
- normalized spatial entropy 约 `0.99997`；
- inter-person L1 约 `0.0093`；
- 把 mask 强制设为 1，AUC/L2/AP/X-TCR/margin 的 paired CI 均跨 0；
- 在 matched spatial-prior control 中，GGSF 与 CoordConv 的 AUC 几乎完全相同，L2/AP 还不是最好；无 spatial prior 的 AP 更高。

结论是 data-refuted core mechanism：**不能继续修辞性挽救 GGSF。**

#### SASA dynamicity：删除为贡献

内部行为审计：

- 平均 weights 约为 `L2=.034, L5=.097, L8=.201, L11=.668`；
- inter-person L1 约 `0.0009`；
- fixed mean 与 learned SASA 统计不可区分；
- shuffle weights 也几乎逐点复现；
- 后续 person-conditioned residual router 把 inter-person L1 提高到 `0.0141`，但整体 AUC/L2/AP/X-TCR/margin 的 CI 全跨 0。

所以“让 routing 真正随人物变化”本身也已做过 pilot，结果 No-Go。不要从 hierarchy routing 继续迭代。

### 3.3 需要清零的定性证据

[`scripts/eval_gazefollow.py`](/Users/fanb/src/gazelle/scripts/eval_gazefollow.py:40) 和 [`scripts/eval_vat.py`](/Users/fanb/src/gazelle/scripts/eval_vat.py:40) 包含：主动模糊 baseline、向 ours heatmap 混入 GT、用 GT 修正 GGSF mask、用 GT 重写 SASA weights 的函数。VAT 脚本区分 RAW/Tricked，而 GazeFollow 可视化路径直接使用这些处理。

从代码路径看，表格指标在可视化处理前计算，不能据此断言数值被污染；但**所有由这些分支产生的定性图、mask 图和 weight 图都不能作为科学证据**，必须从 raw checkpoint 重新生成，并保留 image ID、checkpoint hash 和命令 provenance。

---

## 4. Multi-scale Object-Aware：核心方法、结果与证据边界

### 4.1 它的核心故事

该工作认为 pixel heatmap regression 缺少对象结构，gaze 应被分成：

1. 在对象层面辨别可能被注视的实体；
2. 在被选实体内部定位精确 gaze point；
3. 用 head appearance 推断 gaze direction，以 FoV 限制几何可达区域；
4. 用 frozen DINOv3 的浅/中/深层特征同时保留空间和语义信息。

### 4.2 方法签名

- YOLO11x + SAM2-hiera-large 离线产生 object masks；
- image tokens 与 masked object tokens 做三层 cross-token fusion，重建 object-aware response；
- head crop + eye position 经 MLP 预测归一化 2D gaze direction，并用 head-to-target GT direction 做 cosine supervision；
- 以预测 direction 构造 120° FoV cone，再 Gaussian smoothing；
- frozen DINOv3 ViT-L/16，输入 `512×512`，输出 `64×64` heatmap；
- shallow/middle/deep 固定残差融合：`Fdeep + αmid Fmid + αshallow Fshallow`。

它的 FoV 与 GGSF 有本质区别：前者在测试时由 head appearance 预测 gaze direction；后者只有 bbox 相对坐标。

### 4.3 主要结果与消融

作者报告：

| Dataset | AUC ↑ | L2/Avg L2 ↓ | Min L2 ↓ | AP ↑ |
|---|---:|---:|---:|---:|
| GazeFollow | 0.961 | 0.094 | 0.038 | - |
| VAT | 0.948 | 0.095 | - | 0.923 |
| ChildPlay | 0.987 | 0.084 | - | 0.990 |
| GOO-Real | 0.977 | 0.092 | - | - |

与 GazeSpot ViT-L 相比：GF AUC/Min L2 持平，GazeSpot GF Avg L2 和 VAT L2 略好；竞争工作 VAT AUC/AP、ChildPlay 和 GOO-Real 作者报告值更强。因此不能说对方全面支配 GazeSpot，也不能忽略其更完整的 benchmark 覆盖。

它的完整模型相对无组件 baseline 的 GF 为 `0.920/0.188/0.118 → 0.961/0.094/0.038`，VAT 为 `0.886/0.162/0.865 → 0.948/0.095/0.923`。Object-only、多尺度-only、两两组合均有作者报告增益。注意这些数字尚未独立复现；论文没有公开可用代码，且 backbone ablation 的数值与 component table 中四行逐项相同，组件表也缺 FOV-only。应把它们标成 paper-reported，而不是无条件接受机制归因。

### 4.4 与原工作的重叠

| 维度 | GazeSpot | Multi-scale Object-Aware | 结论 |
|---|---|---|---|
| Frozen VFM | DINOv3 B/L | DINOv3-L | 高度重叠 |
| 输入/输出 | 512 / 64 heatmap | 512 / 64 heatmap | 高度重叠 |
| 多层表示 | 四个中间层，学习缩放后 concat | shallow/mid/deep 固定残差 | “多尺度本身”已无 novelty |
| 空间约束 | bbox-relative near-identity gate | predicted direction + 120° FoV | 对方机制更接近真实 gaze geometry |
| 对象语义 | 无显式 object representation | YOLO+SAM object tokens | 对方独有部分 |
| 失败动机 | crowd/distractor hijacking | object ambiguity/semantic competition | 叙事高度重叠 |
| 人物绑定 | 每人独立 decode，无直接 binding loss | 每人条件化，但仍逐 query 训练/评价 | 留下新的问题空间 |

### 4.5 已经不能再声称的 novelty

1. 首个把 DINOv3 hierarchical features 用于 gaze target estimation；
2. 首个用多尺度解决语义—空间冲突；
3. 首个 geometry/FoV-guided frozen-VFM gaze 方法；
4. GGSF 是 physical frustum 或 gaze cone；
5. SASA 是 person-adaptive/dynamic scale selection；
6. 原方法首次解决 crowded distractor interference；
7. scene encoding once / multi-person reuse 本身是创新；Gaze-LLE 和 Sharingan 已有相关设计，[Sharingan](https://openaccess.thecvf.com/content/CVPR2024/html/Tafasca_Sharingan_A_Transformer_Architecture_for_Multi-Person_Gaze_Following_CVPR_2024_paper.html) 还显式以 controlled person token 做多人预测；
8. head–target association 本身是创新；[GazeHTA](https://arxiv.org/abs/2404.10718) 已显式学习 connection maps，ICCV 2023 [Object-aware Gaze Target Detection](https://openaccess.thecvf.com/content/ICCV2023/html/Tonini_Object-aware_Gaze_Target_Detection_ICCV_2023_paper.html) 也建立 head–object association；
9. “轻量且无显著开销”：实测相对 448 baseline，GazeSpot 的 MACs 约 `+31.6%`，latency 约 `+24–27%`，FPS 约 `-20%`；最多说 trainable head 较小且 batch-1 仍可实时；
10. “DINOv3 严重 over-smoothing”：现有 PCA 和 proxy 不足以支持 backbone-level general claim。

---

## 5. SelectiveGaze No-Go：只作为排除项

Hierarchy disagreement pilot 已给出明确 No-Go：

- ID 上 `final confidence + disagreement` 相对 final confidence 的 failure AUROC 增量约 `-0.00069`；
- shift 上约 `-0.00124`；
- OOD 上约 `+0.00129`，CI 跨 0；
- 高 disagreement 20% 样本的 failure rate 反而低于低 disagreement 20%。

因此本报告不延续 selective prediction，也不把 hierarchy disagreement 作为任何新方向的动机、模块或辅助贡献。它只说明：**层间分歧没有提供超出 final heatmap confidence 的风险信息。**

---

## 6. 四个 gaze-specific 新方向

## 6.1 方向 A：Counterfactual Observer–Target Binding（唯一推荐）

### 核心思想

同一场景 `I` 中，人物 `q_i` 和 `q_j` 分别看向 `y_i`、`y_j`。把 query 从 `q_i` 换成 `q_j` 是一个天然的受控变化：scene semantics、salient objects、背景和相机完全不变，只有“谁在看”改变。一个真正 observer-conditioned 的模型应使输出从 `y_i` 切换到 `y_j`。

定义目标邻域得分：

`S_ij = log sum_{x in N(y_j)} P(x | I, q_i)`。

对 distinct-target pair，加入交换配对 margin：

`L_bind = max(0, m - (S_ii + S_jj) + (S_ij + S_ji))`。

多人情形先按 gaze point 距离聚成 shared-target clusters。同一 cluster 是 many-to-one positives，不互相当负例；没有完整标注的 query 使用 ignore mask。方法不需要 YOLO/SAM、方向头、FoV 或新多尺度模块，可以直接加到 Gaze-LLE/GazeSpot heatmap decoder 上。

### Novelty

高于一般“head–target association”，但必须谨慎命名。Sharingan 已有 controlled person tokens，GazeHTA 已有显式 connection maps；本次检索未发现它们把**同场景 query switching 作为自然反事实，使用 diagonal-vs-swapped assignment objective，并以 target-cluster-aware binding metric 独立评价**。安全 claim 是“提出 counterfactual binding protocol/objective”，不能说“首次研究多人关联”。

Sharingan 的公开正文还说明：GazeFollow 训练时加入 detected context people，但 loss 只从 annotated target person 反传。因此“同时看到多人”不等于本方向的 cross-query binding supervision。

### 与 Multi-scale Object-Aware 的实质差异

- 对方问：场景中有哪些对象、几何上哪里可达、如何在对象内定位；
- 本方向问：在相同对象和场景条件下，输出是否属于正确 observer；
- 对方增强 `scene/target representation`；本方向约束 `query-output correspondence`；
- 本方向可以把对方模型当 backbone/base learner，不与 object tokens、FoV 或 multi-scale claim 竞争。

### 已有探索性信号（C 级）

基于现有 VAT records 的一次只读 sanity check：

- 11,960 个同帧 distinct-target pairs；
- baseline 正确 diagonal assignment 约 `81.96%`，GazeSpot 约 `82.98%`；
- 5+ 人时 baseline 约 `70.08%`，GazeSpot约 `69.59%`；
- 5+ 人且 target separation ≥0.30 时，GazeSpot 仍只有约 `79.51%`。

这里的 diagonal/swap 代价由 records 中的预测峰值到两个 GT target 的欧氏距离计算，并非上文方法最终使用的 heatmap-neighborhood mass。该统计尚未做 sequence-cluster bootstrap、标注完整性审计或 matched-confound regression，不能写进论文；但它表明原方法没有解决该现象，且问题不完全由相邻 target 造成。

### 可证伪假设

> H-A：控制 target separation、head size、target distance、in/out 和 video 后，现有模型在同场景 distinct-target pairs 上仍有显著 observer–target swap error；cluster-aware permutation loss 能降低 swap error，并在相同或更好的 L2 下提高 binding margin。

以下任一成立即推翻方向：

1. 在 `target separation ≥0.30` 的干净 pairs 上，Sharingan/PaGE/Gaze-LLE 已接近饱和，错误率低于 5%；
2. people count 的效应在控制混杂后消失，错误只是普通 localization noise；
3. 加 `L_bind` 只能锐化 heatmap，却不能在 matched-L2 下提高 binding；
4. VAT 同帧 annotations 不完整，无法形成可信 negatives。

### 一天 pilot

1. 固定 frame grouping、shared-target threshold、video-level split；审计同帧标注完整性和重复 track。
2. 对现有 baseline、GazeSpot、可用的 Gaze-LLE/Sharingan checkpoint 保存 raw heatmaps。
3. 计算 pairwise binding accuracy、swap margin、row-wise target ranking；按 target separation 和 people count 分层。
4. 用 video sequence 做 2,000 次 cluster bootstrap；用 mixed-effects/logistic regression 控制 head size、target distance、target separation。
5. Go 条件：至少两个模型在 far-target pairs 上仍有 ≥10% swap error；问题覆盖 ≥1,000 pairs/≥30 videos；GazeSpot 没有稳定改善；标准 L2 不能完全解释该错误。

第一天不训练新模型。诊断不过，不实现 loss。

### Fatal flaw

最危险的问题不是模型效果，而是**“association 已有”导致 novelty 被否定**。防御只能来自三件事同时成立：自然反事实协议揭示标准 L2/AUC 看不到的 failure；permutation objective 显著改善该 failure；该改善在 Gaze-LLE 和至少一个多人模型上可泛化。只有一个 margin loss，不足以支撑论文。

### 投稿潜力

- **AAAI：8/10，Conditional Accept。** 适合“受控现象 → 新评价契约 → 原则性训练目标”的 empirical-insight paper。前提是混杂控制和跨模型验证完整。
- **ACM MM：7.5/10。** 多人视频、同场景反事实可视化和 query-conditioned media understanding 契合；最好同时带来 VAT 标准 L2/AP 改善。

## 6.2 方向 B：Counterfactual Target-Visibility Transition

### 核心思想

对包含 queried head 和 in-frame target 的图像构造两类严格配对 crop：

- **target-preserving crop**：head 和 target 都保留，heatmap 应按 crop transform 等变；
- **target-excluding crop**：head 保留、target 被裁出，新标签应从 in-frame 变为 out-of-frame，模型不应在剩余画面中 hallucinate 一个显著替代目标。

训练统一 outside token/heatmap distribution，并使用 equivariance loss、visibility-flip loss 和 inside hallucination-mass loss。核心是 label-aware paired intervention，不是普通 random crop augmentation。

### Novelty

中高。PDP 类工作已有 outside distribution，GazeMoE 也使用 region-specific/random cropping；本次检索未发现把“target-preserving 与 target-excluding crop”成对构造成 gaze-specific counterfactual transition，并联合评价等变性、in/out flip 和 hallucination mass 的工作。不能声称首次用 crop 或首次做 out-of-frame。

### 与 Multi-scale Object-Aware 的差异

对方在固定 frame 内用对象和 FoV缩小搜索；本方向改变 frame boundary，研究 gaze target 的可见性契约和反事实稳定性，不依赖对象、多尺度或 direction cone。

### 可证伪假设

> H-B：当前模型在 target-excluding crop 中会高置信度转向剩余 salient distractor；paired counterfactual training 能降低 hallucination，并在 target-preserving crop 与自然 VAT in/out 边界上保持定位和可见性一致。

### 一天 pilot

从 GazeFollow/VAT 各采 1,000 个 in-frame 样本，为每个样本生成一个 preserving crop、一个 excluding crop 和一个相同尺度的 random control crop；运行现有 checkpoints，报告：transform-aligned L2、in/out flip accuracy、inside peak/mass、original accuracy retention。无需训练。

### Fatal flaw

crop 会改变画面组成、尺度和上下文，模型可能学习边界/缩放 shortcut；synthetic out-of-frame 不等同于自然出画。若无法在自然 VAT boundary 或另一数据集验证，容易被认为是 augmentation paper。GazeMoE 的 crop augmentation 也会压缩 novelty。

### 投稿潜力

- **AAAI：7/10。** 需要把 intervention validity 和自然数据外部验证做得很严。
- **ACM MM：7.5/10。** 鲁棒性 benchmark、视频边界和可视化更契合，但不能只有 synthetic 指标。

## 6.3 方向 C：Semantic-Decoy Intervention Learning

### 核心思想

从原“distractor hijacking”出发，不再尝试用多层或 FoV 抑制所有干扰，而是直接问：**非目标显著对象的外观变化，是否会不合理地改变 observer-conditioned prediction？**

构造 target-preserving decoy interventions：对与 GT target 足够远的高显著区域、其他人物 target 或 detector proposal 做 blur/color attenuation/patch replacement；head 与真目标保持不变。模型应对 decoy edit 保持输出稳定，同时对真目标区域的合法几何变换保持等变。训练使用 decoy-invariance contrast，而非新的 object fusion module。

### Novelty

中等。HCLoRA 已经明确提出 VFM semantic shortcut，并用 out-of-cone penalty 改善 non-salient target；Multi-scale Object-Aware 也以 object competition 为核心。差异只能是**用成对干预测量和训练 decoy causal sensitivity**，不能再声称首次发现 semantic distractor。

### 与 Multi-scale Object-Aware 的差异

对方增加 object semantics，可能仍偏向视觉更强的同类实例；其 failure case 也承认 close, semantically similar objects 会选错。本方向不增强 object representation，而是要求无关 decoy 的外观变化不应改变 observer–target relation。

### 可证伪假设

> H-C：现有模型对 non-target decoy edits 的输出变化显著大于与空间扰动匹配的 background edits；decoy-invariance training 能降低这种敏感性，并改善真实 non-salient/semantic-conflict cases。

### 一天 pilot

在 500–1,000 个样本上用三种廉价 intervention：其他已标注 target blur、显著区域 blur、随机同面积 background blur。测 heatmap JS divergence、peak shift 和 GT mass change；只在 decoy edit 明显高于 random control 时 Go。

### Fatal flaw

“无关 decoy”很难保证真正 label-preserving；编辑可能改变对人类 gaze intent 的合理解释。若使用 detector/inpainting，又会引入与竞争工作类似的外部 pipeline。缺少真实 semantic-conflict benchmark 时，方法很难证明不是数据增强。

### 投稿潜力

- **AAAI：6/10。** 因果措辞风险高，需人类或自然 paired validation。
- **ACM MM：6.5/10。** 定性效果直观，但最容易被判为 augmentation/robustness 增量。

## 6.4 方向 D：Fixation-State Causal Temporal Gaze

### 核心思想

把视频 gaze 表示为 piecewise-stationary latent state：多数帧处于 fixation，少数时刻发生 saccade/target change。因果模型只使用当前和过去帧，同时输出 target state 与 change-point probability；稳定期抑制 jitter，change point 时快速释放旧 target，避免普通 temporal smoothing 的 lag。

### Novelty

中等。MTGS 已做 multi-person temporal gaze following，因此“加入 temporal transformer”没有 novelty。可守住的差异是 fixation/change-point state、causal protocol、stable jitter 与 switch delay 的事件级目标。

### 与 Multi-scale Object-Aware 的差异

竞争工作完全静态，正文也把 temporal reasoning 列为 future work。本方向利用 gaze 的 fixation/saccade动力学，不涉及 object-aware 或多尺度。

### 可证伪假设

> H-D：GT target trajectories 具有足够强的分段稳定性；显式 change-point gate 能同时降低 stable jitter 和 switch delay，而非只用惯性换取平均 L2。

### 一天 pilot

复用 VAT cached predictions，做 stable/switch 分层、EMA/Kalman/causal-change gate 三个无训练 baseline，评价 jitter–delay Pareto。现有 P2 已观察到 switch degradation，但 in→out/out→in 只有 `238/251`，低于预注册的 `500/方向` 门槛；一天内必须先做 transition validity QC 或引入 VSGaze/ChildPlay 的足量事件。

### Fatal flaw

当前数据证据不足，且静态模型的 boundary-lag-like pattern 可能来自标注切换规则、遮挡或视觉证据，而非 temporal memory。普通 smoothing 会天然恶化 change delay。MTGS 的直接重叠也很强。

### 投稿潜力

- **AAAI：当前 5/10；获得足量有效事件后可到 7/10。**
- **ACM MM：6/10。** 视频 setting 合适，但本项目现阶段不应优先投入。

---

## 7. 方向比较与唯一选择

| 方向 | Gaze-specific 性 | 与竞争工作差异 | 一天可证伪 | 数据风险 | 方法非堆叠性 | AAAI | ACM MM | 判定 |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| A. Counterfactual observer binding | 5/5 | 4/5 | 5/5 | 中 | 5/5 | 8/10 | 7.5/10 | **唯一推荐** |
| B. Visibility transition | 5/5 | 4/5 | 5/5 | 中高 | 5/5 | 7/10 | 7.5/10 | 备选 |
| C. Semantic decoy intervention | 4/5 | 3/5 | 5/5 | 高 | 4/5 | 6/10 | 6.5/10 | 不优先 |
| D. Fixation-state temporal | 5/5 | 3/5 | 3/5 | 当前很高 | 4/5 | 5/10 | 6/10 | 当前 No-Go |

选择 A 的原因不是它一定能涨最多，而是：

1. 它直接继承原 crowd/distractor 故事中尚未被证伪的部分，但把含混机制改成可操作的 query-switch intervention；
2. 它与 Multi-scale Object-Aware 的对象/几何/多尺度轴正交；
3. 它不需要继续挽救 GGSF、SASA 或 hierarchy；
4. 现有 records 已出现可观测缺口，且 GazeSpot 没有改善 5+ 人 binding；
5. 一天内可以用纯评价决定是否停止，不会再次陷入数天模块搜索。

---

## 8. 推荐论文的完整逻辑链

### 8.1 Paper type

**Empirical insight-driven technique paper / new evaluation setting + minimal method。** 不是 pure SOTA paper，也不是新 benchmark collection paper。

### 8.2 六段 Introduction 逻辑

1. **背景与 running example。** Gaze target estimation 是 query-conditioned dense prediction：同一场景中问不同的人，正确答案可以完全不同。
2. **现有进展与局限。** Frozen VFMs、multi-scale、object-aware semantics、gaze geometry 已显著增强 scene/target representation；多人模型也能一次处理多个 person tokens。但主流 loss/metric 仍主要逐 query 评价，不能隔离模型是否真正响应 observer identity。
3. **问题本质。** 在同一帧切换 queried observer 是自然反事实：它固定 scene semantics 和候选目标，只改变 observer。若模型跟到另一个人的目标，普通 L2 只把它记成定位误差，无法区分 query binding failure。
4. **挑战。** 需要处理 shared attention、标注不完整、相邻 targets 和视频重复；还要证明 binding metric 不是 L2/people count 的重命名。
5. **方法。** 构造 frame-level counterfactual query sets，按 target clustering 定义 many-to-one positives，用 permutation contrastive objective 使正确 observer–target assignment 优于 swapped assignment；不引入 object detector 或 FoV pipeline。
6. **贡献。** 新诊断协议；cluster-aware binding objective；跨单人/多人 base models、标准指标和混杂控制的系统验证。

### 8.3 Challenge–module/experiment 对齐

| Challenge | 方法/协议 | 决定性证据 |
|---|---|---|
| 场景 saliency 与 observer cue 混合 | 同场景 query-switch natural counterfactual | same-frame binding vs shuffled-scene/random-pair controls |
| shared attention 不能互为负例 | target clustering + many-to-one positives | no clustering / hard exclusive / proposed 对比 |
| 普通 localization 混淆 binding | diagonal-vs-swapped score、matched-L2 分析 | 按 target separation 分层 + regression/sequence bootstrap |
| 不想依赖对象/几何外部模块 | pixel-neighborhood mass objective | no detector/SAM，参数/FLOPs 不变 |

### 8.4 建议贡献表述

可以写：

1. We expose observer–target binding as a hidden failure mode through natural counterfactual query switches within the same scene.
2. We introduce a shared-target-aware permutation contrastive objective that directly supervises the correspondence between person queries and gaze outputs.
3. We establish a controlled evaluation protocol that separates binding from target proximity, head scale, crowd density, and ordinary localization error.

不要写：

- first multi-person gaze model；
- first head-target association；
- first counterfactual learning in gaze（除非投稿前再做系统检索并有足够证据）；
- solves crowd interference；
- causal identification（自然 query switch 具有受控性，但 observational annotations 仍不是完整因果实验）。

---

## 9. 实施计划与冻结的 Go/No-Go

### Phase 0：0.5 天，协议和数据有效性

1. 以原视频/clip ID 做 split；不同 person tracks 不能把同一 frame 泄漏到 train/val。
2. 合并同一 frame 的所有可靠 annotations；检查 duplicate track、missing query、out-of-frame。
3. 预注册 target cluster threshold；建议从 heatmap GT sigma 推导，而不是看结果调阈值。
4. 只在 distinct in-frame target clusters 上计算主 binding；shared targets 单列。

验收：输出 manifest、frame count、query count、pair count、video count、cluster-size distribution 和 hash。

### Phase 1：0.5 天，一天 pilot 的决定性诊断

模型：baseline、GazeSpot、Gaze-LLE、Sharingan；PaGE 若 checkpoint 可直接运行则加入。

指标：

- Pairwise Binding Accuracy：`Sii+Sjj > Sij+Sji`；
- Swap Margin；
- row-wise own-target rank；
- heatmap similarity / peak separation 只作辅助；
- 标准 L2/AUC/AP。

统计：video-cluster bootstrap 2,000 次；按 people count、target separation、head size、target distance 分层；用 mixed-effects/logistic regression 控制混杂。

**Go：**

1. ≥1,000 far-target pairs、≥30 videos；
2. 至少两个强模型在 `target separation ≥0.30` 上 swap error ≥10%；
3. binding error 与 L2 相关但不等价，matched-L2 或回归残差中仍有模型差异；
4. 原 GazeSpot 不稳定改善，说明新问题不是旧模块已解决。

**No-Go：** 任一核心数据有效性失败，或强 baselines far-target error <5%，或控制混杂后现象消失。

### Phase 2：1–2 天，最小方法

不改 backbone，不增加 object/direction/head branch。只改 batch sampler 和 loss：

`L = L_heatmap + λ_io L_io + λ_bind L_bind`。

先在 VAT train-derived sequence split 做：

- control：原 loss；
- individual hard-negative loss；
- proposed pairwise permutation loss；
- proposed + shared-target clustering。

`λ_bind` 只允许两个预注册值，例如 `0.1/0.3`。先短训 1–2 epochs；winner 再完整训练。不要同时搜索 router、FoV、object tokens 或新 fusion。

晋级条件：binding error 相对下降 ≥20%，95% CI 不跨 0；far-target 与 5+ 人方向一致；标准 L2 不退化超过 `0.001`，AP 不退化超过 `0.003`。

### Phase 3：3–5 天，论文级证据

1. VAT 关键配置 3 seeds；
2. 在 Gaze-LLE 和 Sharingan/GazeSpot 两类 base model 上验证 plug-in generality；
3. GazeFollow 若能可靠合并同图多 query，则做外部验证；不能就只报标准指标，不制造不可信 pairs；
4. GOO-Real 报标准/OOD 性能；如果没有多 query annotations，不强行报 binding；
5. 比较 GazeHTA/Object-aware 的概念边界；Multi-scale Object-Aware 无代码时只报作者结果，不伪复现；
6. 报总/可训练参数、`N=1/3/5/10` latency；方法本身原则上不增加推理成本。

### Phase 4：论文图表

- Figure 1：同一 scene 切换两个人，baseline/GazeSpot 的 swapped/collapsed output 与 proposed output；
- Figure 2：frame-level query set → target clustering → score matrix → diagonal-vs-swap loss；
- Table 1：标准 GF/VAT/GOO performance；
- Table 2：counterfactual binding 主表，含 far-target 和 shared/distinct split；
- Table 3：random negatives、individual margin、exclusive assignment、cluster-aware permutation；
- Table 4：混杂控制与效率；
- qualitative 全部来自 raw inference，禁止任何 GT-based visualization correction。

---

## 10. 最终审稿人式判定

### 推荐方向 A

**Verdict：Conditional Accept for immediate pilot。**

- Higher：7/10。已有探索性 gap，但尚无训练收益。
- Faster：9/10。训练增加 pair loss，推理原则上零额外成本。
- Stronger：8/10。若成立，直接针对 observer-specific failure，而不是平均 feature quality。
- Cheaper：9/10。不需要 YOLO/SAM、第二个 DINO branch 或 LVLM。
- Broader：8/10。可抽象到 query-conditioned dense prediction 的 query-output binding，但论文仍应以 gaze 证据为中心。

### 最终 fatal-flaw rule

如果一天 pilot 不能证明 far-target observer swap 是独立于普通 L2 的稳定 failure，立即停止。不要把现有 `X-TCR` 换名为 counterfactual binding，也不要先实现 loss 再寻找支持故事的分桶。

如果 pilot 通过但方法只改善自定义 binding metric、不改善或保持标准性能，则最多是一项诊断/benchmark contribution；要达到 AAAI/ACM MM 主会强度，还需要跨模型一致性、shared-target 正确处理和至少一个标准/OOD 价值轴。

---

## References

- Jiajie Mi et al., [Multi-scale Object-Aware Gaze Estimation via Geometric Reasoning](https://arxiv.org/abs/2606.29334v1), ECCV 2026.
- Fiona Ryan et al., [Gaze-LLE: Gaze Target Estimation via Large-Scale Learned Encoders](https://openaccess.thecvf.com/content/CVPR2025/html/Ryan_Gaze-LLE_Gaze_Target_Estimation_via_Large-Scale_Learned_Encoders_CVPR_2025_paper.html), CVPR 2025.
- Samy Tafasca et al., [Sharingan: A Transformer Architecture for Multi-Person Gaze Following](https://openaccess.thecvf.com/content/CVPR2024/html/Tafasca_Sharingan_A_Transformer_Architecture_for_Multi-Person_Gaze_Following_CVPR_2024_paper.html), CVPR 2024.
- Zhi-Yi Lin et al., [GazeHTA: End-to-end Gaze Target Detection with Head-Target Association](https://arxiv.org/abs/2404.10718), 2024.
- Francesco Tonini et al., [Object-aware Gaze Target Detection](https://openaccess.thecvf.com/content/ICCV2023/html/Tonini_Object-aware_Gaze_Target_Detection_ICCV_2023_paper.html), ICCV 2023.
- Weidong Bao et al., [ESCNet: Gaze Target Detection with the Understanding of 3D Scenes](https://openaccess.thecvf.com/content/CVPR2022/html/Bao_ESCNet_Gaze_Target_Detection_With_the_Understanding_of_3D_Scenes_CVPR_2022_paper.html), CVPR 2022.
- Anshul Gupta et al., [MTGS: A Novel Framework for Multi-Person Temporal Gaze Following and Social Gaze Prediction](https://papers.nips.cc/paper_files/paper/2024/hash/1caf09c9f4e6b0150b06a07e77f2710c-Abstract-Conference.html), NeurIPS 2024.
- Zhuangzhuang Dai et al., [GazeMoE: Perception of Gaze Target with Mixture-of-Experts](https://arxiv.org/abs/2603.06256), ICRA 2026.
- Shijing Wang et al., [Enhancing Gaze Reasoning in Vision Foundation Models for Gaze Following](https://arxiv.org/abs/2605.22607), 2026.
- Murari Ambati, [RayGazeFM: Geometry-Grounded Foundation Adapters for Unified 3D Gaze, Point-of-Regard, and Gaze Target Estimation](https://openaccess.thecvf.com/content/CVPR2026W/GAZE/html/Ambati_RayGazeFM_Geometry-Grounded_Foundation_Adapters_for_Unified_3D_Gaze_Point-of-Regard_and_CVPRW_2026_paper.html), CVPRW 2026.
