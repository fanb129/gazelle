# GazeSpot 转投 AAAI 2027：2026 近邻工作数据、比较策略与方法重构

> 核验日期：2026-07-13  
> 用途：内部研究决策文档，不是可直接粘贴进论文的 Related Work。所有外部数值均为论文作者报告值；除明确说明外，尚未由本项目独立复现。

## 摘要

本轮调研回答三个问题：

- **RQ1：** 2025-2026 年最接近 GazeSpot 的工作实际报告了什么结果，它们的训练条件和故事分别是什么？
- **RQ2：** 哪些工作必须引用、必须进入 SOTA 表、必须在本地运行；预印本是否需要处理？
- **RQ3：** 在单卡 RTX 3090 和约两周时间内，如何把现有 SASA 与 GGSF 重构为更有机制依据、也更可能带来增益的方法？

主要结论如下。第一，GazeSpot 并未被所有 2026 方法全面碾压：其 ViT-L 在 GazeFollow 上与 ECCV 2026 Multi-scale Object-Aware 基本持平，在 VAT 的 L2 上还略好；PaGE 的 headline 结果明显更强，但其中混合了 GF、VAT、ChildPlay 监督和 1.17M 无标签蒸馏数据，其 GF-only 结果与当前 GazeSpot 仍基本同级。第二，原始 GGSF 已被内部对照实验否定，不能继续作为核心贡献；SASA 的历史对照又存在初始化不公平，现有增益不能作为可信证据。第三，最值得尝试的不是完整复制 PaGE 的双 DINOv3 分支，而是把两个模块统一为一个 **amortized person-conditioned hierarchical adapter**：场景只编码一次，由轻量 person query 同时控制“用哪些层/通道”和“在什么空间范围内检索”，再用同帧其他 query/target 作为关联 hard negatives；crowd 是 query binding 的压力测试，而不是问题定义本身。

## 1. 调研方法与可比性约定

### 1.1 纳入范围

核心语料限定为截至 2026-07-13 已公开、且与第三人称 gaze target estimation 直接相关的工作：Gaze-LLE、GazeSpot、PaGE、OmniGF、Multi-scale Object-Aware、HCLoRA 和 GazeAnywhere。Sharingan、MTGS、GazeHTA、ViTGaze、FGI-Gaze 等用于定位设计空间，但不在本文件逐表复录全部旧结果。

### 1.2 三种数字不能混为一谈

- **同一标准任务、同一官方 test split 的作者报告值：** 可以进入 SOTA 表，但仍需标注不同 training data、backbone 和输入模态。
- **论文自建 subset 或转换后的 task：** 只能放专门的分析表。例如 HCLoRA 的 consistent/inconsistent split、GazeAnywhere 的 Gaze-Co/PGE 不能与标准主表直接排名。
- **本项目内部结果：** 必须与文献报告值分栏，不得用粗体暗示严格 apples-to-apples，除非训练和评估协议已经统一。

## 2. 当前性能版图：我们到底落后多少

### 2.1 标准 benchmark 的近邻结果

表中 `-` 表示论文未报告。GazeSpot 数字来自 ACM MM 稿件；其余来自对应论文主表。PaGE、OmniGF 与 ECCV 2026 方法使用了明显不同的训练数据或计算预算，因此该表用于判断竞争压力，不用于归因方法优劣。

| 方法 | 状态 | 主要计算/训练条件 | GF AUC ↑ | GF Avg L2 ↓ | GF Min L2 ↓ | VAT AUC ↑ | VAT L2 ↓ | VAT AP ↑ | Child L2 ↓ | Child AP ↑ |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Gaze-LLE ViT-L [1] | CVPR 2025 | frozen DINOv2 ViT-L | 0.958 | 0.099 | 0.041 | 0.937 | 0.103 | 0.903 | 0.101 | 0.994 |
| GazeMoE [10] | ICRA 2026 | frozen DINOv2-L；shared/routed MoE decoder；3.4M learnable params | 0.959 | 0.101 | - | 0.939 | 0.097 | 0.917 | 0.106 | 0.994 |
| **GazeSpot ViT-L（当前稿）** | ACM MM 2026 rejected | frozen DINOv3 ViT-L；512；单卡 3090 | 0.961 | 0.093 | 0.038 | 0.944 | 0.092 | 0.912 | - | - |
| HCLoRA ViT-B [2] | arXiv v1, 2026-05-21 | DINOv2/Gaze-LLE；标准 VAT overall；GF 只报自建子集 | - | - | - | 0.9352 | 0.1006 | 0.8988 | - | - |
| HCLoRA ViT-L [2] | arXiv v1, 2026-05-21 | 同上 | - | - | - | 0.9387 | 0.0951 | 0.9068 | - | - |
| OmniGF [3] | arXiv v1, 2026-05-26 | Qwen3-VL-4B；LoRA；H100 80GB；一次处理多人 | - | 0.091 | 0.040 | - | 0.096 | 0.923 | 0.090 | 0.996 |
| Multi-scale Object-Aware [4] | **ECCV 2026 accepted** | frozen DINOv3-L；offline YOLO11x+SAM2；单卡 5090 | 0.961 | 0.094 | 0.038 | 0.948 | 0.095 | 0.923 | 0.084 | 0.990 |
| PaGE ViT-S Distill [5] | arXiv v1, 2026-07-06 | 96.9 GFLOPs；H+ teacher；1.17M unlabeled distillation | 0.9643 | 0.0858 | 0.0327 | 0.9640 | 0.0739 | 0.9371 | 0.0751 | 0.9966 |
| PaGE ViT-B Distill [5] | arXiv v1, 2026-07-06 | 283.1 GFLOPs；同上 | 0.9660 | 0.0814 | 0.0295 | 0.9688 | 0.0677 | 0.9450 | 0.0697 | 0.9969 |
| PaGE ViT-H+ [5] | arXiv v1, 2026-07-06 | 2373.6 GFLOPs；单卡 H100；scene+head 双分支 | 0.9659 | 0.0804 | 0.0288 | 0.9719 | 0.0643 | 0.9509 | 0.0687 | 0.9954 |

### 2.2 对 GazeSpot 的定量判断

- 对 Multi-scale Object-Aware：GazeFollow 几乎持平；VAT 上 GazeSpot 的 L2 更低 `0.092 vs. 0.095`，但 AUC/AP 更低 `0.944/0.912 vs. 0.948/0.923`。所以不能说“所有 2026 工作都全面超过我们”。
- 对 OmniGF：GazeSpot 的 GF Min L2 和 VAT L2 更好，但 OmniGF 的 GF Avg L2、VAT AP、ChildPlay 更强；而且 OmniGF 是 4B VLM，计算条件完全不同。
- 对 PaGE：headline 差距真实且明显。以 PaGE ViT-H+ 为例，GazeSpot ViT-L 在 GF Avg L2 落后 `0.0126`，VAT L2 落后 `0.0277`。但这不是纯架构公平比较：PaGE 还报告了只使用 GazeFollow 监督的模型为 `0.9595 AUC / 0.0958 Avg L2 / 0.0391 Min L2`，当前 GazeSpot 的 `0.961/0.093/0.038` 略优。真正拉开 headline 的是混合数据训练、全模型 SFT 和 1.17M 无标签蒸馏；仅靠改写故事当然不能填平，但也不应错误地把差距全归因于双分支。
- 不能把现有 GazeSpot 直接包装成“更高效的 PaGE”。当前 GF 配置实测为 `141.08 GFLOPs / 22.82 ms / 43.82 FPS`（RTX 3090，512 输入），而 PaGE 的蒸馏 ViT-S 报告 `96.9 GFLOPs`。两者 profiler、硬件和数据流尚未统一，但至少说明 **单人物 FLOPs 更低** 不是现成结论。真正值得验证的是 scene feature 只算一次后，`N=3/5/10` 人时的总成本和每新增一人的边际成本。
- 当前 GazeSpot 在 GOO-Real 的内部结果为 `0.8833 AUC / 0.1761 L2`；PaGE 零样本为 `0.930 / 0.140`，Multi-scale Object-Aware 报告 `0.977 / 0.092`。但三者训练数据和预处理尚未统一，这个差距只能当强烈预警，不能直接归因。

投稿时不要只做一个混合排行榜。建议把主表分成两个 block：**dataset-only / no external gaze supervision** 与 **mixed-data / external distillation**，并增加 `Training Data`、`Unlabeled Data`、`Scene Encoder × People` 三列。我们的首要数值目标应是在 dataset-only block 稳定提高当前 GazeSpot，而不是用一张 3090 去复制 PaGE 的 H100+1.17M 蒸馏条件。

## 3. 逐篇方法、故事与关键实验数据

### 3.1 PaGE：把性能提升归因于 head-scene interaction 与训练 recipe，而非 DINOv3 本身

**状态与资源。** PaGE 于 2026-07-06 发布 arXiv v1，论文声称项目页已提供代码和 checkpoint。全部训练在单张 H100 上完成；ViT-L/H+ 需要梯度累积以达到 effective batch size 60。

**一句话故事。** Gaze target estimation 同时需要全局场景语义和局部 head/eye 空间推理；简单换成更强 VFM 不够，因此 PaGE 恢复 scene/head 双分支，用双向 cross-attention、统一 2D 坐标和两阶段训练把模型推向 human-level，再用大规模无标注数据蒸馏为轻量学生。

**方法签名。** 512×512 scene DINOv3 与 256×256 head DINOv3；5 层 Scene-head Interaction Module；scene/head 双向 cross-attention；2D RoPE；先冻 backbone 训练 decoder 15 epochs，再全模型 SFT 5 epochs；H+ teacher 在 1.17M 无标注图片上蒸馏学生，最后再 SFT 3 epochs。

#### “Road to PaGE” 消融

| 修改 | GF AUC | GF Avg/Min L2 | VAT AUC/L2/AP | Child AUC/L2/AP |
|---|---:|---:|---:|---:|
| Gaze-LLE ViT-B reevaluated | 0.9543 | 0.1104 / 0.0492 | 0.9324 / 0.1074 / 0.8970 | 0.9478 / 0.1051 / 0.9920 |
| 简化混合数据训练 | 0.9552 | 0.1074 / 0.0482 | 0.9370 / 0.1078 / 0.8926 | 0.9499 / 0.1103 / 0.9907 |
| DINOv2 → DINOv3 | 0.9554 | 0.1044 / 0.0450 | 0.9383 / 0.1118 / 0.9053 | 0.9496 / 0.1062 / 0.9927 |
| token-concat head branch | 0.9581 | 0.1003 / 0.0439 | 0.9462 / 0.1000 / 0.9138 | 0.9585 / 0.0955 / 0.9942 |
| decoder 3 → 5 layers | 0.9583 | 0.0985 / 0.0419 | 0.9516 / 0.0928 / 0.9150 | 0.9604 / 0.0907 / 0.9949 |
| DINO feature dropout | 0.9591 | 0.0979 / 0.0412 | 0.9502 / 0.0912 / 0.9235 | 0.9596 / 0.0911 / 0.9945 |
| SIM head branch | 0.9596 | 0.0961 / 0.0392 | 0.9543 / 0.0911 / 0.9304 | 0.9618 / 0.0905 / 0.9949 |
| 2D RoPE | 0.9599 | 0.0939 / 0.0380 | 0.9557 / 0.0886 / 0.9421 | 0.9652 / 0.0883 / 0.9961 |
| decoder → SFT（ViT-B） | 0.9625 | 0.0892 / 0.0345 | 0.9618 / 0.0844 / 0.9491 | 0.9690 / 0.0810 / 0.9971 |
| PaGE ViT-L | 0.9648 | 0.0823 / 0.0306 | 0.9690 / 0.0736 / 0.9460 | 0.9728 / 0.0724 / 0.9957 |
| PaGE ViT-H+ | 0.9659 | 0.0804 / 0.0288 | 0.9719 / 0.0643 / 0.9509 | 0.9746 / 0.0687 / 0.9954 |

**对我们的启示。** PaGE 的数据直接反驳“用了 DINOv3 就自然会大幅提升”：drop-in 替换只带来小幅或不稳定变化；head cue、坐标编码、decoder 与训练 recipe 才是主要增益来源。恢复完整双分支可以作为强 baseline，但不能再作为我们的 novelty。

### 3.2 OmniGF：4B VLM 用语言分支产生 person anchor，再由连续分支做精确定位

**状态与资源。** 2026-05-26 arXiv v1。论文给出 GitHub，但截至核验日仓库只有 README，并写明 “code will be released soon”，所以不能视为已可复现。模型为 Qwen3-VL-4B-Instruct + LoRA，scene/head 输入为 448/224，batch 8、梯度累积 2，训练使用单张 H100 80GB，耗时 2-30 小时。

**一句话故事。** 文本生成擅长语义推理但不擅长连续空间定位；OmniGF 因此让结构化语言分支为每个人生成 gaze reasoning state 和 person anchor，再从 VLM hidden states 接一个连续 heatmap 分支，实现一次前向的多人物空间、语义和社会注视推理。

**方法签名。** 所有人 head box 以结构化 prompt 输入；Gaze360 预训练 ResNet-18 编码 head crop 并替换 `<gaze_pad>` token；语言分支生成 JSON；空间分支将 person hidden token 与 image tokens 相乘后解码 heatmap；可同时预测 in/out 与社会注视。

#### 主结果与消融

| 方法 | GF Avg/Min L2 | VAT L2/AP | Child L2/AP |
|---|---:|---:|---:|
| Gaze-LLE ViT-L（其表中） | 0.099 / 0.041 | 0.103 / 0.903 | 0.101 / 0.994 |
| Sharingan | 0.113 / 0.057 | 0.107 / 0.891 | 0.106 / 0.990 |
| OmniGF | **0.091 / 0.040** | **0.096 / 0.923** | **0.090 / 0.996** |

| OmniGF 组件 | GF Avg L2 | GF Min L2 |
|---|---:|---:|
| Qwen3-VL zero-shot 文本坐标 | 0.340 | 0.258 |
| LoRA language branch | 0.114 | 0.055 |
| spatial decoder only | 0.167 | 0.103 |
| language + spatial dual branch | 0.104 | 0.047 |
| + dynamic head-token injection | **0.091** | **0.040** |

**对我们的启示。** “同一场景一次处理多人”已经被明确提出，不能单独作为 novelty；但 OmniGF 的 4B/H100 成本给轻量模型留下了空间。我们的可辩护目标应是：用 frozen VFM 的多层缓存与轻量 person query 获得接近的 query binding，同时报告每增加一个人的边际耗时。

### 3.3 Multi-scale Object-Aware：对象候选 + 方向 FOV + 静态多层残差融合

**状态与资源。** arXiv v1 发布于 2026-06-28，arXiv 明确标注 **Accepted by ECCV 2026**。使用单张 RTX 5090；DINOv3 ViT-L/16 完全冻结；YOLO11x 和 SAM2-hiera-large 离线生成 object masks；512×512 输入、64×64 heatmap、Adam、初始学习率 1e-3；论文未给可用代码/ckpt 链接，且 batch/epoch、具体层号、若干模块细节不足。

**一句话故事。** gaze target 不应被视为无结构的像素回归，而应先在对象层面产生候选，再用 head/gaze geometry 约束可达区域，最后用 DINOv3 浅/中/深层特征做语义到空间的精确定位。

**方法签名。** object masks → object tokens 与 image tokens cross-modal fusion → head appearance + eye position 预测 2D gaze direction → 120° FOV cone → frozen DINOv3 浅/中/深层残差融合：`Fdeep + αmid Fmid + αshallow Fshallow`。

#### 主结果

| 数据集 | AUC ↑ | Avg L2/L2 ↓ | Min L2 ↓ | AP ↑ |
|---|---:|---:|---:|---:|
| GazeFollow | 0.961 | 0.094 | 0.038 | - |
| VAT | 0.948 | 0.095 | - | 0.923 |
| ChildPlay | 0.987 | 0.084 | - | 0.990 |
| GOO-Real | 0.977 | 0.092 | - | - |

#### 组件、层级和几何消融

| Object | Multi-scale | FOV | GF AUC/Avg/Min L2 | VAT AUC/L2/AP |
|---:|---:|---:|---:|---:|
| × | × | × | 0.920 / 0.188 / 0.118 | 0.886 / 0.162 / 0.865 |
| ✓ | × | × | 0.932 / 0.166 / 0.101 | 0.903 / 0.140 / 0.887 |
| × | ✓ | × | 0.937 / 0.148 / 0.087 | 0.906 / 0.134 / 0.891 |
| ✓ | ✓ | × | 0.954 / 0.111 / 0.049 | 0.937 / 0.108 / 0.912 |
| ✓ | × | ✓ | 0.945 / 0.132 / 0.069 | 0.922 / 0.122 / 0.903 |
| × | ✓ | ✓ | 0.946 / 0.125 / 0.063 | 0.919 / 0.124 / 0.904 |
| ✓ | ✓ | ✓ | **0.961 / 0.094 / 0.038** | **0.948 / 0.095 / 0.923** |

| 层级 | GF AUC | GF Avg L2 | GF Min L2 |
|---|---:|---:|---:|
| shallow | 0.931 | 0.139 | 0.068 |
| middle | 0.936 | 0.132 | 0.063 |
| deep | 0.943 | 0.122 | 0.056 |
| shallow + middle | 0.940 | 0.126 | 0.059 |
| shallow + deep | 0.952 | 0.108 | 0.047 |
| shallow + middle + deep | **0.961** | **0.094** | **0.038** |

FOV 角度 `60°/90°/120°/150°` 的 GF 结果分别为 `0.940/0.125/0.059`、`0.951/0.109/0.048`、`0.961/0.094/0.038`、`0.953/0.107/0.046`。方向损失权重 `λdir=0.5` 最好。

**对我们的启示。** 单纯“DINOv3 多层 + 更真实 gaze cone”已经不是空白。可以保留多层问题，但必须把静态全局系数升级为 **person-conditioned routing**；若做空间先验，必须与其固定 120° FOV 区分，例如建模方向不确定性并使用 additive/residual bias，而不是 hard/multiplicative suppression。

**证据可信度注意。** 该文仍有几处需要谨慎对待：它所称 `7.1M parameters` 没有明确说明是否只统计可训练模块，显然也没有把离线 YOLO11x/SAM2 与 frozen DINOv3 的总成本完整计入；其 backbone ablation 的四行数字又与组件消融的四个不同配置逐项完全相同，正文没有解释这一巧合；组件表也缺少 FOV-only，层级表缺少 middle+deep。这个异常不足以证明结果有误，但在代码、supplement 和训练细节均不可用时，主表只能标为 paper-reported，不能把消融结论当成已独立验证。

### 3.4 HCLoRA：把失败定义为 semantic shortcut，并只局部适配 head tokens

**状态与资源。** 2026-05-21 arXiv v1，无接收信息。正文写明 “accepted 后发布代码”。论文没有完整公开 LoRA rank、适配层、OOC 层数/角度/权重以及 optimizer/lr/epoch/batch，现阶段无法按正文独立复现。

**一句话故事。** VFM 强在场景语义，却可能用显著物体代替真实 gaze reasoning；全量 fine-tuning 又会破坏 scene representation。因此只在 head 位置附近做 local LoRA，并用 out-of-cone penalty 抑制与 gaze direction 不一致的中间证据。

**方法签名。** 用 bbox Gaussian 生成 head-conditioned LoRA gate，只调制 head 附近 token 的 low-rank update；训练时用 GT head-to-target direction 构造 soft cone，对 cone 外的辅助 heatmap 概率质量施加惩罚。注意：它在测试时并不预测一个新的方向分支，cone 主要是训练正则。

#### GazeFollow consistent/inconsistent split

该 split 由 Qwen3-VL-32B-Instruct 先预测“语义上可能吸引注意”的区域，再按 GT gaze 是否落入这些框来分组，不是官方 split。

| 方法 | Consistent AUC/Avg/Min L2 | Inconsistent AUC/Avg/Min L2 |
|---|---:|---:|
| Gaze-LLE | 0.9689 / 0.0834 / 0.0373 | 0.9270 / 0.1519 / 0.0664 |
| HCLoRA ViT-B | 0.9696 / 0.0777 / 0.0348 | 0.9340 / 0.1353 / 0.0547 |
| HCLoRA ViT-L | 0.9719 / 0.0709 / 0.0301 | 0.9371 / 0.1271 / 0.0498 |

#### VAT overall 与组件消融

| 方法 | VAT AUC ↑ | L2 ↓ | Angular ↓ | AP ↑ |
|---|---:|---:|---:|---:|
| Gaze-LLE | 0.9347 | 0.1071 | 15.02 | 0.8979 |
| HCLoRA ViT-B | 0.9352 | 0.1006 | 12.26 | 0.8988 |
| HCLoRA ViT-L | 0.9387 | 0.0951 | 11.75 | 0.9068 |

在 GF inconsistent subset 上，Gaze-LLE → HCLoRA → HCLoRA+OOC 的 Avg L2 为 `0.1519 → 0.1383 → 0.1353`，Min angular error 为 `7.22° → 5.72° → 4.85°`。不同辅助监督中，OOC 的 inconsistent AUC/Avg L2 最好为 `0.9340/0.1353`，而 rigid gaze-cone 为 `0.9304/0.1385`。

**对我们的启示。** “head-conditioned adapter”与“cone 外抑制”都已有直接先例。我们若预测方向，必须强调 **测试时不确定性建模、soft additive bias 和共享 scene encoding**，不能只换一个 cone 公式。

### 3.5 GazeAnywhere：它解决的是 promptable subject identification，不是同一个输入协议

**状态与资源。** CVPR 2026 正式发表，代码和 DINOv3-L checkpoint 已公开。模型约 870M，训练使用 4×H100，推理使用 L40S；输入 512，训练集为从 GazeFollow/VAT/ChildPlay 转换并人工核验的 120K Gaze-Co。公开 checkpoint 当前主要支持 text concept，论文中的 visual prompt 接口尚未完整暴露。

**一句话故事。** 传统 gaze pipeline 先检测 head 再做 gaze，crowd 中前级检测错误会级联；GazeAnywhere 将任务改成 Promptable Gaze Estimation，用外观/位置/动作/姿态文本直接指定人物，同时联合预测 head box、in/out 和 gaze heatmap。

**为什么不能直接排进标准主表。** 它接收 human-verified concept prompt，且 test split 经过 Gaze-Co 转换/过滤；这不是 GazeSpot 的 GT bbox 输入协议。应在新任务/setting 表中比较，不应把数值与标准 GF/VAT 粗体排名。

#### PGE 主结果

| 方法/提示 | GF-Concept AUC/Avg/Min L2 | VAT-Concept AUC/L2/AP | Child-Concept L2/AP |
|---|---:|---:|---:|
| GazeAnywhere CLIP-L（text all） | 0.953 / 0.105 / 0.056 | 0.913 / 0.137 / 0.874 | 0.104 / 0.915 |
| GazeAnywhere DINOv3-L（text all） | 0.958 / 0.099 / 0.050 | 0.928 / 0.123 / 0.879 | 0.098 / 0.906 |
| no prompt | 0.944 / 0.144 / 0.090 | 0.875 / 0.210 / 0.796 | - |
| visual prompt | 0.958 / 0.100 / 0.050 | 0.914 / 0.131 / 0.894 | - |
| text prompt（all fields） | 0.958 / 0.099 / 0.050 | 0.928 / 0.123 / 0.879 | - |

#### Backbone 与 loss 消融

| Encoder | GF AUC/Avg/Min L2 | VAT AUC/L2/AP |
|---|---:|---:|
| CLIP-B | 0.942 / 0.123 / 0.070 | 0.908 / 0.145 / 0.837 |
| CLIP-L | 0.953 / 0.105 / 0.056 | 0.913 / 0.137 / 0.874 |
| SigLIP2-L | 0.953 / 0.105 / 0.056 | 0.910 / 0.147 / 0.858 |
| DINOv3-L | **0.958 / 0.099 / 0.050** | **0.928 / 0.123 / 0.879** |
| MetaCLIP2-H | 0.951 / 0.109 / 0.059 | 0.912 / 0.150 / 0.857 |

只用 gaze loss 的 GF/VAT 为 `0.956/0.102/0.052` 与 `0.925/0.135`；加入 head localization 后为 `0.958/0.099/0.051` 与 `0.924/0.128`；完整 gaze+presence+head 为 `0.958/0.099/0.050` 与 `0.928/0.123/0.879`。

### 3.6 GazeMoE：自适应 cue routing 已经有正式先例，但没有覆盖 per-person layer routing

**状态与资源。** 2026-03-06 发布，论文标注 ICRA 2026；官方 Hugging Face 已提供推理代码与权重。使用 frozen DINOv2-L、shared/routed MoE decoder、class-balancing auxiliary loss 和区域裁剪/光度增强，报告 3.4M learnable parameters。

**一句话故事。** eyes、head pose、gesture 与 scene context 对不同样本的重要性不同，因此用 mixture-of-experts 从 frozen VFM 中自适应选择 gaze-related cues，而不是让单一 decoder 在所有情形使用相同组合。

| 数据集 | AUC ↑ | Avg L2/L2 ↓ | AP ↑ |
|---|---:|---:|---:|
| GazeFollow | 0.959 | 0.101 | - |
| VAT | 0.939 | 0.097 | 0.917 |
| ChildPlay | 0.945 | 0.106 | 0.994 |

**对我们的启示。** “input-adaptive routing”不能再笼统声称首创。PCLR 必须把差异落到：同一 scene 内对 **不同 person query** 产生不同的 layer/channel routing，并共享 scene cache；实验上直接比较 GazeMoE 的 official checkpoint、普通 MoE、image-global SASA 与 person-conditioned routing。GazeMoE 的标准定位数字没有全面超过当前 GazeSpot，因此它更像 novelty/机制 baseline，而非不可逾越的 SOTA。

## 4. 预印本要不要加：分层处理，而不是一刀切

AAAI-27 的专门 Citation and Comparison 页面在本次核验时尚未检索到。[AAAI-26 官方投稿规则](https://aaai.org/conference/aaai/aaai-26/submission-instructions/)说明：必须引用最相关的 refereed publications；作者不必知道所有未审稿 arXiv；投稿截止前两个月内公开的工作视为 contemporaneous，不强制处理。不过同一规则也明确指出，广为人知的 arXiv 仍可能影响 novelty 判断。[AAAI-27 官网](https://aaai.org/conference/aaai/aaai-27/)目前给出的 full-paper deadline 为 2026-07-28 AoE，因此本文件中的 5-7 月新作大多属于 concurrent work。战略上，最安全的做法如下。

| 工作 | 引用 | 报告值进入论文表格 | 本地重跑 | 本项目建议 |
|---|---:|---:|---:|---|
| PaGE | **必须** | **必须**，标 `concurrent preprint` 与训练数据 | **优先**；直接跑其轻量 checkpoint，不重新训练 | 最接近、代码/ckpt 声称可用；还应跑 crowd/query-fidelity protocol |
| Multi-scale Object-Aware | **必须** | **必须**；它是 ECCV 2026 accepted，不是普通未审稿稿件 | 暂不重实现；无代码且细节缺失 | 主表用作者报告值；对比其 static multi-scale/FOV 的最小控制即可 |
| OmniGF | **必须** | 建议放 `concurrent/preprint` 独立 block | 不重跑；仓库目前是 placeholder，4B/H100 | 它威胁多人/统一任务的 novelty，但不是 3090 下公平 baseline |
| HCLoRA | **必须** | VAT overall 可进；GF 自建 split 单独放 | 不重实现完整方法；无代码且细节不足 | 将 OOC/cone supervision 作为概念 baseline 即可，不宣称复现 HCLoRA |
| GazeAnywhere | **必须** | 放 PGE/不同 setting 表，不放标准 GF/VAT 排名 | 可选；text checkpoint 可用，但输入协议不同 | 用于说明 subject identification 与 cascade error，不是主 baseline |
| GazeMoE | **必须** | **必须**；正式发表、标准协议 | **必须**；官方权重可用 | 直接检验 adaptive cue routing 是否已解决 SASA 所声称的问题 |
| Gaze-LLE / Sharingan | **必须** | **必须** | **必须优先使用官方 checkpoint/代码** | 这是公平、可运行、同协议的核心 baseline |

核心原则：**预印本不等于必须重实现。** 是否运行取决于代码/ckpt 是否可用、协议是否相同、是否直接挑战核心 claim。对无代码预印本，引用并报告作者数值已经足够；仓促照论文重写一个不完整版本，既不公平也浪费时间。

## 5. 两个创新点如何真正升级

### 5.1 先统一问题：不是为 crowd 做两个补丁，而是做 query-conditioned VFM adaptation

建议把新稿的问题抽象为：

> A frozen scene VFM produces one generic representation, but gaze target estimation is a query-conditioned dense prediction problem: different people in the same scene require different representation depths and different spatial evidence. How can one adapt a shared scene representation to many person queries accurately and efficiently?

这个抽象把原来的两个模块统一起来：

- SASA 回答 **what evidence**：当前人物应该用哪些层和通道；
- GGSF 的替代物回答 **where to search**：当前人物的空间证据如何作为软偏置进入模型；
- crowded scene 不是特定场景假设，而是同一图像有多个 competing queries 时，检验 query conditioning 是否真的生效的压力测试；
- 场景只做一次 DINOv3 encoding，所有人物复用多层特征，形成 accuracy-efficiency 约束。注意 Gazelle/Gaze-LLE 本来就先编码 scene 再复制特征，因此 **scene-once 不能单独写成创新**；新意必须是不同 person queries 如何条件化同一份缓存的多层表示，以及这种条件化是否比重型 head interaction 更省边际成本。

### 5.2 SASA 替代：Person-Conditioned Residual Layer Router（PCLR）

当前 SASA 对每层全局池化后只输出一个 layer scalar，而且显式忽略 `head_token`。在启用 GGSF 时它会通过 bbox mask 间接随人物变化，但仍没有 head appearance，也没有显式的 person query。建议改为以下轻量结构。

1. 从共享的多层特征 `F_l` 中，对 bbox 做 RoIAlign/GAP，并拼接 bbox geometry，得到基础 person query：

   `q_i = MLP([ROI(F_1,b_i), ..., ROI(F_L,b_i), e_box(b_i)])`。

2. 每层先投影到统一的较小通道 `C=128/256`；由 `q_i` 预测 **per-layer, per-channel** 权重，而不是一个层标量：

   `a_i,l,c = softmax_l(MLP_l(q_i))`。

3. 采用 deep-dominant residual fusion，而不是四层全 concat：

   `F_i = P_D(F_D) + Σ_{l<D} a_i,l ⊙ P_l(F_l)`。

4. 训练时加入 LayerDrop/router entropy 的轻量约束，避免永远只选最后一层；同一图像不同 head query 应产生不同路由。

**为什么可能比当前 SASA 强。** 它加入了当前模块缺失的 person appearance/geometry 条件，选择粒度从 4 个标量提升到 layer×channel，同时 residual sum 比 full concat 更省投影计算。

**与最近工作的差异。** Multi-scale Object-Aware 使用人工固定 `αshallow/αmid`；PCLR 是 per-person dynamic。PaGE 使用完整 head DINOv3 与跨分支 attention；PCLR 从已经缓存的 scene pyramid 取 head RoI，不需要第二个重 backbone。

#### 是否恢复 head 分支：建议恢复，但把它当成 query encoder，而不是单独的创新点

PaGE 的消融给出了很直接的信号：在换成 DINOv3 之后，仅加入 token-concat head branch，GF Avg L2 从 `0.1044` 降到 `0.1003`，VAT L2 从 `0.1118` 降到 `0.1000`；这个增益比单独换 DINOv3 更明显。当前 GazeSpot 只把二值 head map 加到 scene tokens，缺少高分辨率 face/head appearance，确实可能是指标上限低的原因。

因此建议把 PCLR 做成两档，并优先保留可升级接口：

- **PCLR-R（低成本）：** `q_i` 只来自 scene pyramid 的 RoI、多层池化与 bbox geometry，用于一天内验证 dynamic routing 是否成立；
- **PCLR-H（推荐最终版）：** 将所有 head crops 批量送入一个轻量 head encoder（例如 ResNet-18 量级，优先比较 ImageNet 与 gaze-orientation pretraining），得到 `h_i`，再令 `q_i = MLP([q_i^ROI,h_i])`。`q_i` 只负责路由和 bias，不再复制一个完整的 patch-level DINOv3 head branch。

这个设计承认“head branch”本身已有大量先例，但利用它补回必要的人物朝向/姿态线索；论文贡献仍是该 query 如何对共享 scene pyramid 做 per-person hierarchical adaptation。若 PCLR-H 明显强于 PCLR-R，就能形成清楚证据链：**bbox/scene RoI 只能定位人物，head appearance 才能形成有效 query；有效 query 再控制 what/where evidence。** 若 PCLR-H 仍无增益，则说明瓶颈不在 head cue，不应继续堆更重分支。

### 5.3 GGSF 的保守替代：Content-Conditioned Relative Position Bias（CRPB）

如果目标是两天内先得到可靠增量，最稳妥的替代不是预测“物理视锥”，而是把 bbox geometry 作为 decoder attention 的 **additive relative-position bias**。

- 对每个 patch 构造 head-centric relative coordinate：`r_i(x,y)=[dx,dy,log ρ,sin φ,cos φ,w,h]`；
- 用 person query `q_i` 生成一组小型 basis coefficients；
- 在 decoder attention logits 中加入 `B_i,h(x,y)`，允许不同 attention head 学习不同空间偏好；
- bias 初始为 0，模型退化时等价于无先验，不会像 `[0,1]` 乘性 mask 一样直接删除全局证据。

这一路线不声称 gaze direction 或物理 FOV，故事是“query-conditioned spatial inductive bias”。它的 novelty 中等，但实现和风险最低，适合作为 PCLR 的第一版搭档。

### 5.4 GGSF 的进取替代：Uncertainty-Aware Gaze Field（UAGF）

如果 oracle direction 实验显示空间方向确有较大上界，再将 CRPB 升级为 uncertainty-aware field：

- 用同一个 `q_i` 预测 2D gaze direction `μ_i` 与 concentration/confidence `κ_i`；
- 用 head center→GT target 的方向做 auxiliary cosine 或 von Mises NLL；
- 根据 `μ_i,κ_i` 构造 soft directional bias；`κ_i` 低时接近均匀，避免遮挡/背头时错误 hard mask；
- 将 field 加到 attention logits 或作为残差特征，禁止乘法截断；
- 报告方向误差与 calibration，证明模型知道何时不可信。

**与 HCLoRA 的差异。** HCLoRA 的 cone 使用 GT direction 做训练惩罚，测试时没有显式预测的不确定性场。**与 ECCV 2026 方法的差异。** 后者使用预测方向和固定 120° FOV；UAGF 的核心必须是可校准不确定性与 soft residual influence，而不是另一个固定 cone。

### 5.5 比另一个 gaze cone 更值得做：Query-Target Contrastive Association（QTCA）

从 novelty、算力和故事一致性看，GGSF 更好的“重做”未必还是一个空间 mask。当前真正未被标准指标直接测量的问题是：同一 scene 有多个 person queries 时，模型是否把 query `i` 绑定到 target `i`，而不是被显著目标或另一个人的 gaze target 吸走。

训练时可直接利用同帧已标注的其他 gaze targets 作为 hard negatives。令 `s(P_i,y_j)` 表示 query `i` 的预测 heatmap 在 target `j` 邻域内的 log probability mass，加入：

`L_QTCA = Σ_i Σ_{j≠i} max(0, m - s(P_i,y_i) + s(P_i,y_j))`。

实现时有三个必要边界：

- 距离很近的 GT targets 先聚成 shared-target cluster；共同注意是 many-to-one positive，不能互相当负例；
- 未完整标注的人物/目标使用 ignore mask，避免制造 false negatives；
- loss 只在同帧有多个可靠 in-frame annotations 时启用，不改变单人物推理成本。

它比继续雕刻 cone 更符合通用抽象：这是 **multi-query dense prediction 的 query-output association**，crowd 只是最容易暴露错误的场景。它也与 PCLR 形成闭环：PCLR-H 提供有辨识力的 person query，QTCA 强迫该 query 真正改变输出目标，CRPB 只提供可退化的空间 inductive bias。

这一路线仍需检查 Sharingan、OmniGF 和多人 gaze association 的重合，不能把“使用其他人的 target 作负例”单独吹成全新范式；更稳妥的贡献组合是 **person-conditioned hierarchical adaptation + association-aware learning/evaluation + amortized inference**。工程上它几乎不增加参数，比 UAGF 更值得优先做。

### 5.6 建议最终命名与贡献结构

先不固定花哨模块名。内部可暂称 **Amortized Person-Conditioned Hierarchical Adapter (APHA)**：

1. 在 Gazelle 已有 scene-once 路径上缓存 DINOv3 pyramid，并让不同 person queries 对同一缓存执行不同的 layer/channel routing；
2. QTCA 用同帧 hard negatives 约束 query-target binding，CRPB 提供低风险的相对空间 bias；
3. 只有 oracle 通过后才把 CRPB 升级为 UAGF；
4. crowd/query-fidelity protocol 验证同图多 query 的 binding 与边际推理成本。

这比“一个 SASA + 一个 GGSF”更像一个方法：同一个 person query 控制 representation selection 与 spatial selection。

## 6. Idea 评价与路线选择

### 6.1 原始 GGSF

**Verdict: Reject and Pivot。** 内部 VAT Crowd `>=4` 对照中，GGSF+SASA 的 AUC 与 CoordConv+SASA 几乎相同，L2/AP 还更差；按照 data-refuted core mechanism 的判定，不能继续把 GGSF 当核心贡献。改名为 bbox gate 只能修正 overclaim，不能恢复贡献有效性。

### 6.2 PCLR-H + QTCA + CRPB

**Verdict: Accept with Revisions，pending pilot。** 这是当前资源下最合理的主线。

| 维度 | 1-10 | 依据 |
|---|---:|---|
| Higher | 7 | 同时补足当前弱 person-conditioning、缺失的 head appearance 与 query-target confusion；尚无新数据 |
| Faster | 7 | residual sum 替代 full concat；scene pyramid 可跨多人复用；需要实测边际 latency |
| Stronger | 7 | person-conditioned routing 与 additive bias 对 query swap/遮挡更稳健；尚待 crowd/OOD 验证 |
| Cheaper | 8 | 不增加第二个 DINO branch，不需要 YOLO/SAM 或 4B VLM |
| Broader | 6 | 可抽象到 query-conditioned dense prediction，但本轮只应在 gaze 上声称 |

Fatal risk 是与 Sharingan 的多人 token、GazeMoE 的自适应专家、PaGE 的 head-scene interaction、Multi-scale 的多层融合分别局部重合，而且 shared scene encoding 已存在于 Gazelle。防御必须是完整组合：**对缓存 pyramid 的 per-person layer/channel routing + query-target association + marginal-cost evaluation**，而不是其中任一单点。

### 6.3 PCLR + UAGF

**Verdict: Conditional Accept，先做 oracle。** 它可能带来更大准确率增益，但 novelty 和工程风险更高。如果 oracle field 都不能显著改善 hard/crowd 样本，就立即停，不做完整方向头。

### 6.4 完整 PaGE-style 双分支

**Verdict: 不作为主方法。** 它可以是强 baseline 或上界，但与 PaGE 高度重合，且 3090 下训练/迭代成本太高。即使提升指标，也很难解释 novelty。

## 7. 单卡 3090 的实验路线与 Go/No-Go

### 7.1 先固定 Crowd Query-Fidelity Protocol

不要再把 `>=4` 当成唯一 crowd 定义。对 VAT 中同一 frame 的全部 annotated heads 一次评估，并固定发布 manifest、样本 ID、构造脚本版本与哈希。核心指标至少包括：

- **Target Confusion Rate：** 对 query `i` 的预测点，在同帧 targets 中最近的若不是自己的 target，则记为关联错误；
- **Association Margin：** `min_{j≠i} d(p_i,y_j) - d(p_i,y_i)`，越大表示人物—目标绑定越稳；
- **Query Swap Sensitivity：** scene 不变而更换 head query 后，heatmap 是否随人物身份合理变化；
- **Heatmap Collapse：** 不同人物目标明显分离时，预测热图却高度相似的比例。

结果按人数、target-target separation、head size/head-head distance、in/out、shared/distinct targets 与遮挡 proxy 分层；把很近的 GT targets 聚为 shared-target cluster，避免把共同注意误判为混淆。置信区间按 video sequence 做 clustered bootstrap，而不是把连续帧当独立样本。最后做 matched non-crowd control，或用回归控制 head size、target distance、in/out 等混杂，证明测到的是 query interference，而不只是“小人头更难”。

### 7.2 Phase A：24 小时，先判定机制有没有上界

1. 统一为 DINOv3、512 输入、相同 decoder/optimizer，公平重跑 `last layer / static residual / current SASA`；所有 variant 从相同初始 backbone 独立训练。
2. 用 GT head→target direction 构造 oracle soft field，只用于判断空间方向的上界。
3. 统计同图不同 head 的 current SASA weights 与 heatmap similarity，确认 query invariance/collapse 的实际频率。
4. 下载并直接评测 PaGE 最轻 checkpoint 与 GazeMoE official checkpoint；先跑标准 VAT/GF，再跑本项目 crowd/query-fidelity split。

**Go UAGF：** oracle field 在 hard/crowd L2 上有清晰且重复的改善，并且不是只改变 AUC 热图形状。  
**No-Go UAGF：** oracle 也无增益或明显损害 out-of-frame/背头样本；此时只保留 CRPB。

### 7.3 Phase B：2-3 天，最小模块筛选

只跑一组相同 seed 的低成本筛选：

- static residual multi-layer；
- image-global SASA；
- bbox-conditioned PCLR；
- ROI appearance-conditioned PCLR；
- lightweight head-branch PCLR-H；
- PCLR-H + QTCA；
- PCLR + CRPB；
- PCLR-H + QTCA + CRPB；
- 若 Phase A 通过，再测 UAGF。

筛选阶段可以减少训练集比例/epoch，但最终主表必须完整训练。不要同时尝试 object detector、完整双分支、关系图和新 loss；那会使两周内无法归因。

### 7.4 Phase C：最终实验

- 标准 GazeFollow、VAT、GOO-Real；ChildPlay 若仍不可得，明确限制；
- 与 Gaze-LLE、Sharingan、PaGE checkpoint 的同协议评测；
- 对不可运行的新作报告作者数值并标状态；
- 关键行 3 seeds；报告 mean±std；
- 报 total/trainable params、FLOPs、N=1/3/5/10 人时的 latency 和 marginal latency；
- crowd 结果同时按人数、target separation、head size/occlusion proxy 分层，避免把所有困难归因于 crowd。

### 建议的停止阈值

- 若 PCLR 对公平 static residual 的 GF/VAT L2 改善均小于约 `0.002`，且无 query-fidelity 改善，不把它升为主贡献；
- 若 CRPB/UAGF 只改善自建 crowd subset、不改善标准或 OOD 指标，则把 spatial module 降为 analysis/ablation；
- 若最终方法不能在现有 GazeSpot 上稳定提升，就不要再把“更高标准精度”放在第一贡献。仍可转为 **同预算精度 + 多人边际成本 + query binding/鲁棒性** 的论文，但前提是这些轴上出现清楚、可复现的优势；若这些轴也没有优势，单靠故事重写不足以支撑转投。

这些数值是项目管理阈值，不是论文 claim，也不是事后挑选显著性的替代品。

## 8. 推荐的新故事

1. Frozen VFMs 提供强场景表示，但其输出是 image-generic；gaze target estimation 本质上是 person-query-conditioned dense prediction。
2. 在同一场景有多个查询人物时，query-invariant representation 会暴露为层选择相同、热图趋同或跟错目标；crowd 只是最自然的压力测试。
3. 重型 head branch 或 VLM 可以改善 query grounding，但为每个 query 重复编码，或需要 H100/大模型；静态多层融合和固定 FOV 又不能适配个体差异和 head-cue uncertainty。
4. APHA 对场景做一次多层编码，用轻量 head branch 构造 person query，路由层/通道证据，并用关联损失抑制“跟到其他人的目标”；空间 bias 保持轻量、可退化。
5. 实验分别回答：query conditioning 是否真的改变了证据选择；是否降低 target confusion；标准精度是否提升；多人推理成本是否随人数更慢增长。

这个故事不要求“从未有人做过多层融合或 head branch”。它的可辩护新意在于把 **representation routing、spatial uncertainty 和 amortized multi-query inference** 放进同一个问题定义和验证框架。

## 9. 对 RQ 的最终回答

**RQ1。** 当前最强 headline 是 PaGE，而不是所有 2026 工作都已全面超过我们；其 GF-only 模型与当前 GazeSpot 仍基本同级。OmniGF 占据大 VLM 多人统一建模，ECCV Multi-scale 占据 object + static multi-scale + fixed FOV，HCLoRA 占据 local adapter + OOC，GazeMoE 占据 input-adaptive expert routing，GazeAnywhere 占据 promptable subject identification。它们的训练协议差异很大，不能只看一列 AUC 排名。

**RQ2。** ECCV/CVPR/ICRA 正式工作必须引用并进入适当表格；PaGE、OmniGF、HCLoRA 虽为预印本，也与核心 novelty 高度重合，应主动引用。只有 PaGE、GazeMoE、Gaze-LLE、Sharingan 这类同协议且有可用代码/ckpt 的方法值得优先本地评测；无代码或不同任务的预印本不需要仓促重实现。

**RQ3。** 原 GGSF 应删除；SASA 应升级为由轻量 head branch 驱动的 person-conditioned residual layer/channel routing。比继续做另一个 cone 更优先的是 QTCA：用同帧其他 target 作为关联 hard negatives，直接抑制 query-target confusion。空间部分先做低风险 additive relative bias，只有 oracle 证明方向上界后再做 uncertainty-aware field。三者共享同一个 person query，并通过 scene-once/multi-query inference 形成统一的准确率、关联忠实度与效率故事。

## References

[1] Fiona Ryan et al., [“Gaze-LLE: Gaze Target Estimation via Large-Scale Learned Encoders,”](https://openaccess.thecvf.com/content/CVPR2025/html/Ryan_Gaze-LLE_Gaze_Target_Estimation_via_Large-Scale_Learned_Encoders_CVPR_2025_paper.html) CVPR, 2025.

[2] Shijing Wang et al., [“Enhancing Gaze Reasoning in Vision Foundation Models for Gaze Following,”](https://arxiv.org/abs/2605.22607) arXiv:2605.22607, 2026.

[3] Qiaomu Miao et al., [“OmniGF: A Dual-Branch Vision-Language Framework for Unified Gaze Following,”](https://arxiv.org/abs/2605.26399) arXiv:2605.26399, 2026.

[4] Jiajie Mi et al., [“Multi-scale Object-Aware Gaze Estimation via Geometric Reasoning,”](https://arxiv.org/abs/2606.29334) ECCV, 2026.

[5] Zhoutong Ye et al., [“PaGE: Towards Practical Human-level Gaze Target Estimation,”](https://arxiv.org/abs/2607.04860) arXiv:2607.04860, 2026.

[6] Xu Cao et al., [“Gaze Target Estimation Anywhere with Concepts,”](https://openaccess.thecvf.com/content/CVPR2026/papers/Cao_Gaze_Target_Estimation_Anywhere_with_Concepts_CVPR_2026_paper.pdf) CVPR, 2026.

[7] Samy Tafasca et al., “Sharingan: A Transformer Architecture for Multi-Person Gaze Following,” CVPR, 2024.

[8] Qiaomu Miao et al., “A Novel Framework for Multi-Person Temporal Gaze Following and Social Gaze Prediction,” NeurIPS, 2024.

[9] Danyang Tu et al., “End-to-End Human-Gaze-Target Detection with Transformers,” CVPR, 2022.

[10] Zhuangzhuang Dai et al., [“GazeMoE: Perception of Gaze Target with Mixture-of-Experts,”](https://arxiv.org/abs/2603.06256) ICRA, 2026.
