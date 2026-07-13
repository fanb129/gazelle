# GazeSpot → AAAI 2027：文献调研、Idea 判定与重构方案

> 日期：2026-07-13  
> 目标：基于 ACM MM 稿件、附件、评审意见、现有代码与实验日志，判断哪些内容应保留、删除或重做，并形成 AAAI 2027 的可证伪研究路线。  
> 截止时间：[AAAI-27 官方时间表](https://aaai.org/conference/aaai/aaai-27/)为 7 月 21 日摘要、7 月 28 日全文、7 月 31 日补充材料与代码（AoE）。主稿只有 7 页技术内容；关键证据不能放到可选读的补充材料中。

## 1. 执行结论

这篇工作不适合按“补几个 baseline、改写动机”的方式直接转投。建议做一次**核心问题重构**：

- **删除 GGSF 作为主要贡献。** 它不是物理 frustum，也没有独立实验证据支持其有效性；当前日志中，它不优于无空间先验，并在 L2/AP 上略输 CoordConv。
- **SASA 可以保留为技术起点，但不能原样保留为主要贡献。** 当前实现只是每层全局池化后预测一个整层标量，且 `head_token` 被明确忽略；它没有真正解决“不同人物需要不同层级证据”的问题。
- **不能把“恢复 DINOv3 双分支 + cross-attention”当作新贡献。** 2026 年 7 月公开的 PaGE 已经完整占据这条路线；Sharingan、MTGS、GazeHTA、GazeAnywhere 也分别覆盖多人 token、联合多人推理、头—目标关联和主体 prompt。
- **也不能把“DINOv3 多层融合 + 更真实的 FOV”作为主要新意。** 2026 年 6 月的 Multi-scale Object-Aware Gaze Estimation 已经组合了 frozen DINOv3 hierarchical features、object tokens、head/gaze direction supervision 与 FOV prior。
- **Crowd 子集应从“人数多的困难样本”升级为“人物查询是否保持正确、是否误跟其他人的目标”的诊断协议。** 人数只能作为一个分层变量，不能作为问题定义本身。
- 最值得做 P0 验证的候选主线是：**单次场景编码下的 person-conditioned multi-layer routing + query/target disambiguation**。它应同时回答两个问题：模型是否选对了 DINO 层级证据，以及模型是否跟对了被查询人物。

按 AAAI-27 的官方审稿标准，优先考虑“新问题、研究问题、原则性分析或超越狭窄子领域的技术”，而非窄幅 SOTA 增量。因此，新稿应从“两个轻量模块”改成“一个明确失败模式 + 一套诊断协议 + 一个针对性方法”。参见 [AAAI-27 Main Technical Track CFP](https://aaai.org/conference/aaai/aaai-27/main-technical-track-call/)。

## 2. 当前证据审计

### 2.1 GGSF 的真实能力边界

代码中的 GGSF 只接收 `(x-cx, y-cy, w, h)`，经两层 `1×1 Conv` 和 sigmoid 生成 mask（`gazelle/model.py:26-91`）。它不包含 head appearance、眼睛或头部朝向、深度、相机参数、3D gaze direction 或物体语义。同一 bbox 几何在不同图像上会产生相同 mask，因此更准确的名称是：

> bbox-conditioned relative-position gate

而不是 physical frustum、gaze cone 或几何视锥。它还把同一个 `[0,1]` mask 乘到所有 DINO 层，只能抑制证据，不能提供新的方向或人物信息（`gazelle/model.py:243-255`）。与此同时，decoder 后面本来就再次注入了 bbox head map（`gazelle/model.py:301-303`），所以 GGSF 很可能只是重复编码同一信号。

现有 VAT Crowd `>=4`、seed 3106 的空间先验日志为：

| 空间先验（均配 SASA） | AUC ↑ | L2 ↓ | In/Out AP ↑ |
|---|---:|---:|---:|
| None | 0.925876 | 0.125472 | **0.933709** |
| Fixed Gaussian | 0.836038 | 0.212179 | 0.918661 |
| CoordConv | 0.925957 | **0.123550** | 0.931160 |
| GGSF | **0.925958** | 0.124042 | 0.926080 |

GGSF 的 AUC 与 CoordConv 只差约 `0.000001`，L2 和 AP 都不是最好；相对无空间先验也没有稳定优势。按照 idea-evaluator 的致命边界，**GGSF 作为核心机制已经被现有数据否定，应 pivot，而不是换名继续包装。**

### 2.2 SASA 的证据目前不成立

SASA 对四层特征分别做 global average pooling，经共享 MLP 产生四个标量，再对整层特征加权后 concat。实现中 `head_token` 参数被明确忽略（`gazelle/model.py:118-151`）。因此：

- 对同一图像中的不同人物，如果没有前置 GGSF，它们得到相同层权重；
- 权重没有 spatial、channel 或 token 级选择能力；
- 当前名字中的 “scale-aware” 实际是“image-global layer weighting”。

历史 fusion 结果表面上显示 SASA 优于 raw concat/equal/FPN：

| Fusion | AUC ↑ | L2 ↓ | In/Out AP ↑ |
|---|---:|---:|---:|
| Raw concat | 0.8099 | 0.2358 | 0.9008 |
| Equal weight | 0.9041 | 0.1405 | 0.9256 |
| FPN | 0.7131 | 0.2671 | 0.8868 |
| SASA | 0.9258 | 0.1239 | 0.9266 |

但这些数值**不能用于论文结论**：所有 VAT variant 都从完整的 `GazeSpot SASA+GGSF` GazeFollow checkpoint 初始化（日志中反复出现 `Initializing from ...train_gazefollow_sasa_ggsf...epoch_14.pt`），SASA 加载了已经训练过的权重，而 raw concat/FPN 的新增模块缺失并保持随机初始化；随后只用小学习率做 VAT 微调。公平实验必须让每个 variant 从相同 backbone 出发，在 GazeFollow 上各自完整训练，再按相同方式转移到 VAT。

### 2.3 还有两个重要混杂因素

1. 当前 baseline 默认输入是 `448×448`（`gazelle/model_v0.py:12`），GazeSpot 是 `512×512`（`gazelle/model.py:157`）。主结果中的提升没有完全隔离分辨率影响。PaGE 也明确指出 DINOv3 patch size 为 16，因此使用 512 来保持与 DINOv2/448 相同的 patch 数；这意味着 512 本身是一个必须控制的设计变量。
2. Crowd 构造脚本当前条件、注释和输出名不完全一致：例如代码条件是 `len(heads) > 4`，注释/打印曾写成 `>=4`；历史记录又多次手工切换 `==3`、`==4`、`<3`、`>4`。新稿必须用一个参数化脚本和 manifest 固化规则、样本 ID、版本与哈希。

### 2.4 定性图证据链必须清零重做

`scripts/eval_gazefollow.py` 和 `scripts/eval_vat.py` 含有以下函数：

- `degrade_baseline_heatmap`：主动模糊 baseline；
- `enhance_gazespot_with_gt`：把 GT Gaussian 混入 ours heatmap；
- `correct_spotlight_with_gt`：把 GT 混入 GGSF mask；
- `correct_sasa_weights_with_gt`：根据 GT 距离重写层权重。

在 GazeFollow 脚本中这些函数被直接用于可视化；VAT 同时输出 RAW 和 `Tricked` 两行。指标是在进入可视化分支前用原始输出计算，因此**目前看到的这两个脚本不会用 trick 修改表格指标**，但任何由这些分支生成的定性图都不能作为科学证据。AAAI 版本必须删除所有 GT 后处理和 baseline degradation，从原始 checkpoint 重跑，并保留 image ID、checkpoint、命令和 git commit 的 provenance。

## 3. 文献版图与被占据的路线

### 3.1 DINOv3 的正确定位

[DINOv3](https://arxiv.org/abs/2508.10104) 本身通过 Gram anchoring 专门缓解长训练下 dense feature map degradation，并把高质量 dense features 作为主要结果之一。因此，当前稿件用少量 PCA 图直接宣称 “DINOv3 severe over-smoothing” 会与 backbone 论文的核心证据正面冲突。

更稳妥的表述是：

> DINOv3 是当前强 dense VFM，但 gaze following 的任务瓶颈不只在 scene semantics，还在 query-person conditioning、gaze reasoning 与 decoder adaptation；backbone 替换本身只带来有限收益。

这也被 [PaGE](https://arxiv.org/abs/2607.04860) 的消融直接支持：在其 Table 1 中，DINOv2→DINOv3 是小幅提升，而 token-concat head branch、SIM cross-attention、2D RoPE 和分阶段 fine-tuning 带来更明显提升。PaGE 还在 DINOv2、CLIP、TIPSv2、DINOv3 上做了统一 backbone 对比，DINOv3 最好但方法具有跨 backbone 可移植性。

### 3.2 Closest work

| 工作 | 已经覆盖的设计空间 | 对本项目的约束 |
|---|---|---|
| [Gaze-LLE, CVPR 2025](https://openaccess.thecvf.com/content/CVPR2025/html/Ryan_Gaze-LLE_Gaze_Target_Estimation_via_Large-Scale_Learned_Encoders_CVPR_2025_paper.html) | 单 scene VFM + person-specific positional prompt；分析多人 prompt | 新稿必须证明比“简单 bbox prompt”多解决了什么 |
| [Sharingan, CVPR 2024](https://openaccess.thecvf.com/content/CVPR2024/html/Tafasca_Sharingan_A_Transformer_Architecture_for_Multi-Person_Gaze_Following_CVPR_2024_paper.html) | 单次 scene 编码、controlled person token、多人联合 transformer、multiscale decoder | “多人 token + shared scene + multiscale”本身不是 novelty |
| [MTGS, NeurIPS 2024](https://papers.nips.cc/paper_files/paper/2024/file/1caf09c9f4e6b0150b06a07e77f2710c-Paper-Conference.pdf) | 多人时序 gaze following + social gaze，person tokens 和新社会注视 metrics | 不能把多人联合推理或 shared attention 当首次提出 |
| [GazeHTA](https://arxiv.org/abs/2404.10718) | 端到端多人物 head-target instances、显式 connection map、association mAP | “显式头—目标关联”必须比 connection map 更具体 |
| [Object-aware Gaze Target Detection, ICCV 2023](https://openaccess.thecvf.com/content/ICCV2023/html/Tonini_Object-aware_Gaze_Target_Detection_ICCV_2023_paper.html) | 检测 heads/objects 并建立 head-object association | object association 也已有强先例 |
| [PaGE, 2026](https://arxiv.org/abs/2607.04860) | DINOv3 scene/head 双分支、双向 cross-attention、统一坐标 2D RoPE、SFT 和 distillation | 直接封住“恢复双分支 + 普通 cross-attention” |
| [Enhancing Gaze Reasoning / HCLoRA, 2026](https://arxiv.org/abs/2605.22607) | head-conditioned local LoRA、out-of-cone penalty、语义 shortcut 诊断 | 直接封住“head-conditioned adapter + cone penalty” |
| [GazeAnywhere, CVPR 2026](https://openaccess.thecvf.com/content/CVPR2026/papers/Cao_Gaze_Target_Estimation_Anywhere_with_Concepts_CVPR_2026_paper.pdf) | DINOv3 + 文本/视觉主体 prompt，联合主体定位、in/out 与 gaze target | 主体识别/promptable gaze 也不是空白 |
| [HGTTR, CVPR 2022](https://openaccess.thecvf.com/content/CVPR2022/html/Tu_End-to-End_Human-Gaze-Target_Detection_With_Transformers_CVPR_2022_paper.html) | 一次预测所有 head-target pairs | 端到端多人 set prediction 已存在 |
| [Multi-scale Object-Aware Gaze Estimation, 2026](https://arxiv.org/abs/2606.29334) | frozen DINOv3 浅/中/深层残差融合、object tokens、head/gaze direction 与 120° FOV | 直接封住“多层 DINOv3 + object semantics + 更真实 gaze cone”的组合 |

因此，以下方案不够支撑 AAAI 主贡献：普通 head crop branch、普通 head-scene cross-attention、普通 person tokens、固定 gaze cone、generic MoE/FPN、仅换 DINOv3、仅按人数做 harder subset。

这篇最新的 multi-scale/object-aware 工作也说明：新方法的可辩护差异不能只是“比 SASA 更强的多层融合”。必须落在**每个 query person 的条件路由**与**多人干扰下的关系辨析/评估**，并用直接实验与其静态浅/中/深残差融合区分。

## 4. 对三个修改想法的直接回答

### 4.1 SASA：需要重做 Figure 2，也需要重做模块

你的判断正确：现有 Figure 2 的 DINOv2/DINOv3 最后一层 PCA 既不能证明 DINOv3 特有 over-smoothing，也不能证明多层融合必要。

建议替换为三部分证据：

1. **单层—多层性能曲线。** 同一 DINOv3、同一 512 输入、同一 decoder 与训练预算，对比每个单层、两层组合、四层，以及 raw concat、equal/static learned sum、FPN、当前 SASA、person-conditioned router。
2. **层级能力诊断。** 对各层分别测 head/people boundary separability、局部 head-pose/方向 linear probe、目标语义/区域定位 probe；再按 sparse/crowded 分层。只有这样才能回答“为什么需要浅层/中层”。
3. **同图多人物案例。** 展示对同一 scene 更换 head query 后，不同层权重和热图是否发生合理变化。当前 SASA 对人物不敏感，这会自然引出新方法。

建议把当前 SASA 重构成 **person-conditioned multi-layer routing**：

- scene DINOv3 只编码一次，抽取多层 tokens；
- 用 bbox RoIAlign 或轻量 head crop encoder 构造 person query；
- query 对每层产生 person-specific 权重，最好是 token/spatial 或 channel 级，而不是整层单标量；
- 所有人物可并行路由，共享 scene 特征，避免 PaGE 为每个分支重复重型 backbone 的成本；
- 不要预设四层一定更好，让 layer subset 与 routing 粒度的消融决定设计。

### 4.2 GGSF：不要升级成更复杂 mask；应被关系建模替代

“回到分支架构”作为 baseline 非常有必要，但作为主方法已经晚于 PaGE。推荐的区别不是“我们的 cross-attention 更复杂”，而是：

> 用轻量 person query 从共享的多层 scene features 中路由信息，并显式优化 query-person 与 target 的对应关系。

候选训练目标可包括：

- 当前人物预测峰值到自己的 target 比到其他人物 target 更近的 margin；
- 同帧不同 head query 的条件特征对自己的 target 为正、其他相互分离 target 为 hard negative；
- 对 shared-attention（多人看同一目标）采用 many-to-one positives，不能做互斥 Hungarian assignment；
- 对缺失标注的人/目标使用 ignore mask，避免 false negatives；
- gaze direction 只能作为带不确定性的 soft residual/bias，不能硬 mask 掉 cone 外信息。

必须对比：no relation、仅 context person tokens、随机 negatives、crowd-aware hard negatives、GazeHTA connection supervision、Sharingan、PaGE-style head branch。若最后只是双分支 cross-attention，就应停止该路线。

### 4.3 Crowd：从人数桶升级为“查询忠实度与关联歧义”协议

当前 `<3 / =3 / =4 / >4` 只说明样本中有几个人，不能证明错误来自 crowd interference。人数同时混杂：head size、遮挡、目标距离、in/out 比例、视频/场景类型、人与目标间距离、可见眼睛/后脑、target saliency，以及同一视频的连续重复帧。

建议把新协议暂称为 **Crowd Query-Fidelity Protocol**（名称以后再定），而不是宣称创建了新数据集。对 VAT 中同一 frame 的所有 annotated heads 一次性评估，至少报告：

- **Target Assignment Accuracy / Target Confusion Rate。** 对 query `i` 的预测点 `p_i`，在同帧所有 in-frame targets 中找最近的 `y_j`；若最近目标不属于 `i`，记为 query-target confusion。
- **Association margin。** `min_{j≠i} d(p_i,y_j) - d(p_i,y_i)`；正值越大说明越能保持人物—目标对应。
- **Query swap sensitivity。** 在 scene 不变时交换 head query，预测是否按目标身份发生相应变化，而不是输出近似相同热图。
- **Heatmap collapse。** 比较不同 head query 的 heatmap 相似度与 GT target 相似度；目标不同却热图近似相同是 collapse。

必须先把距离小于阈值的 GT targets 聚成 shared-target cluster，否则会错误惩罚共同注意。结果不只按人数，还要按以下变量画曲线：

- 人数；
- 最小 head-head 距离与 head size；
- target-target separation；
- gaze ray 附近 competing heads/objects 数量；
- in/out 比例；
- target-is-person / target-is-object；
- shared attention / distinct targets；
- 遮挡与 profile/back-head proxy。

统计上以视频 sequence 为 cluster 做 bootstrap/置信区间，不能把连续帧当作独立样本。可进一步用分层匹配或 mixed-effects regression 检查：控制 head size、target distance、in/out 等变量后，density/interference 是否仍显著影响错误率。发布脚本、阈值、manifest、样本 ID 和哈希。

## 5. 三条路线的 Idea 评价

评分为 1（弱）到 5（强）；“Stronger”在没有新实验前只给保守分。

| 路线 | Higher | Faster | Stronger | Cheaper | Broader | 新颖性 | 可行性 | 判定 |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| A. 原稿补 baseline、重写 GGSF/crowd 动机 | 2 | 3 | 1 | 4 | 1 | 1 | 5 | **Reject direct transfer** |
| B. DINOv3 完整双分支 + cross-attention | 3 | 1 | 4 | 1 | 2 | 1 | 2 | **Reject：PaGE 高度重合** |
| C. 共享场景编码 + person-conditioned multi-layer routing + query-fidelity 协议/损失 | 4 | 4 | 3 | 4 | 4 | 3 | 3 | **Conditional accept for pilot** |

路线 C 仍有三项 fatal risk：

1. Sharingan 已有 controlled person tokens 和 multiscale decoder；若方法差异只剩一个 attention block，仍然不够。
2. Multi-scale Object-Aware Gaze Estimation 已有 DINOv3 hierarchical fusion、object semantics 与 gaze FOV；必须证明 person-conditioned routing 不是其静态融合的轻微变体。
3. GazeHTA 已有 association mAP 和 connection map；必须证明 query-confusion 指标揭示了标准 AUC/L2/mAP 看不到的失败，并且方法针对的是“错误跟随另一个人”，而非泛化的关联学习。
4. VAT 的标注若不完整，跨人物 negatives 会产生错误监督；必须先审计 annotation completeness，并允许 shared target 和 unknown target。

## 6. 公平实验矩阵

### 6.1 Backbone 与层融合

- DINOv2 / DINOv3，至少一个相同模型规模；可将 CLIP 或 TIPSv2 放附录，主稿只保留能回答 RQ 的行。
- DINOv3 每个候选单层；代表性两层；全部四层。
- Last-layer、raw concat、equal sum、static learned weights、FPN、SASA、person-conditioned router。
- 全部使用相同 512 输入、decoder width/depth、训练 epoch、augmentation、optimizer；报告 trainable/total params、FLOPs、scene latency 和每增加一人的边际 latency。

### 6.2 人物条件与关系建模

- bbox prompt only（Gaze-LLE style）；
- bbox CoordConv；
- lightweight head query；
- full head branch / PaGE-style cross-attention；
- Sharingan-style multi-person token；
- proposed routing，无 relational loss；
- proposed routing + relational loss。

关键行至少 3 seeds；每个 variant 独立从 GazeFollow 训练，再用同一策略转 VAT，不能共享某一 variant 的完整 checkpoint。

### 6.3 数据与外部有效性

- GazeFollow：标准性能与单层/多层主消融；
- VAT：标准指标 + query-fidelity/crowd protocol，作为核心诊断集；
- GooReal：已有数据，作为 OOD/真实场景证据；
- ChildPlay：PaGE、Sharingan、GazeAnywhere 都报告该集。若仍无法获取，必须明确限制，不能写“comprehensive benchmarks”；同时把方法主张限定在可验证范围。

## 7. 截止期驱动的 Go / No-Go 计划

### P0：24 小时内，不训练或只做极小训练

1. 从原始 baseline、GazeSpot checkpoint 生成 VAT 同帧多人物预测，计算 Target Confusion Rate、association margin、query-swap/collapse 指标。
2. 画指标随人数、target separation、head proximity 的曲线，并做 sequence-level bootstrap。
3. 检查错误样本：模型是否真的经常“跟了另一个人的目标”，还是主要由小头、遮挡、out-of-frame、标注噪声造成。
4. 统计 GGSF mask 的均值、方差、熵和跨样本差异，判断它是否近似 identity/常数。
5. 删除可视化 trick，确认任何论文图均来自 raw inference。

**Go 条件：** query-target confusion 随干扰强度显著增加，且标准 AUC/L2 没有充分反映这一现象。  
**No-Go 条件：** crowd 退化主要由 head size/occlusion/in-out 等混杂解释，或 Sharingan/PaGE 已经在拟议指标上完全解决。

### P1：接下来 48–72 小时

1. 先实现最小 person-conditioned layer router：共享 DINOv3 多层 scene features + bbox/RoI person query，不先上完整双分支。
2. 公平训练 last-layer、equal、SASA、router 四行；先看 VAT 标准指标与 query-fidelity 指标。
3. 若 router 有信号，再加入 joint multi-person 与 relational loss；若没有，立即停止方法扩展，转向纯评估/分析型论文也来不及充分完成时，应重新评估是否适合本轮 AAAI。

### P2：成稿前

- 主表统一分辨率与训练流程；
- 对最近工作 PaGE、HCLoRA、GazeAnywhere 做直接讨论与尽可能的可运行比较；
- 公布 protocol 构造代码、manifest 和评估脚本；
- 主稿只写证据支持的 claim：不用 “physical”“fundamentally”“indisputable”“unfathomable”等措辞；
- Figure 1 用同图两人物的 query-confusion motivated example；Figure 2 用单层/多层诊断与 query-conditioned routing 证据；方法图单独展示共享 scene 编码和 per-person routing。

## 8. 建议的新论文逻辑链

1. 视觉 foundation models 提供强场景语义，但 gaze following 仍要求把预测绑定到被查询人物。
2. 现有指标把每个人独立评价，难以区分“定位误差”和“跟错人物的目标”；crowded scenes 放大这种关联歧义。
3. 通过同帧多人物 counterfactual query 评估，我们定义并量化 query-target confusion，并控制人数之外的混杂因素。
4. 单一深层特征或 image-global layer fusion 对所有人物使用相同层级证据，无法适配不同人物的局部几何与场景语义需求。
5. 我们用共享 DINOv3 多层 scene encoding 和 person-conditioned routing，在保持多人扩展效率的同时生成 query-specific features；关系监督进一步降低 target confusion。
6. 实验分别证明：新指标揭示标准指标遗漏的失败；路由机制选取了人物相关的层级证据；收益在公平分辨率/参数/训练控制和 OOD 条件下成立。

这是一个待 P0 数据验证的逻辑链，不应先写成既定事实。

## 9. Primary sources

- [AAAI-27 Main Technical Track CFP](https://aaai.org/conference/aaai/aaai-27/main-technical-track-call/)
- [DINOv3](https://arxiv.org/abs/2508.10104)
- [Gaze-LLE, CVPR 2025](https://openaccess.thecvf.com/content/CVPR2025/html/Ryan_Gaze-LLE_Gaze_Target_Estimation_via_Large-Scale_Learned_Encoders_CVPR_2025_paper.html)
- [PaGE, 2026](https://arxiv.org/abs/2607.04860)
- [Enhancing Gaze Reasoning in Vision Foundation Models, 2026](https://arxiv.org/abs/2605.22607)
- [Sharingan, CVPR 2024](https://openaccess.thecvf.com/content/CVPR2024/html/Tafasca_Sharingan_A_Transformer_Architecture_for_Multi-Person_Gaze_Following_CVPR_2024_paper.html)
- [MTGS, NeurIPS 2024](https://papers.nips.cc/paper_files/paper/2024/file/1caf09c9f4e6b0150b06a07e77f2710c-Paper-Conference.pdf)
- [GazeHTA](https://arxiv.org/abs/2404.10718)
- [GazeAnywhere, CVPR 2026](https://openaccess.thecvf.com/content/CVPR2026/papers/Cao_Gaze_Target_Estimation_Anywhere_with_Concepts_CVPR_2026_paper.pdf)
- [Object-aware Gaze Target Detection, ICCV 2023](https://openaccess.thecvf.com/content/ICCV2023/html/Tonini_Object-aware_Gaze_Target_Detection_ICCV_2023_paper.html)
- [HGTTR, CVPR 2022](https://openaccess.thecvf.com/content/CVPR2022/html/Tu_End-to-End_Human-Gaze-Target_Detection_With_Transformers_CVPR_2022_paper.html)
- [Multi-scale Object-Aware Gaze Estimation via Geometric Reasoning, 2026](https://arxiv.org/abs/2606.29334)
- [Head-Local-Global Coordination, ECCV 2024](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/03933.pdf)
- [ESCNet, CVPR 2022](https://openaccess.thecvf.com/content/CVPR2022/html/Bao_ESCNet_Gaze_Target_Detection_With_the_Understanding_of_3D_Scenes_CVPR_2022_paper.html)
- [Dual Attention Guided Gaze Target Detection, CVPR 2021](https://openaccess.thecvf.com/content/CVPR2021/html/Fang_Dual_Attention_Guided_Gaze_Target_Detection_in_the_Wild_CVPR_2021_paper.html)
