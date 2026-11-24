# SOTA-1：近期方法、公开权重与比较目标

检索日期：2026-11-24。只读论文、官方代码/发布 API 和本地已选定结果；没有新训练、下载大权重或 GPU 推理。机器可读来源、版本、PDF SHA、官方权重发布元数据与本地报告 SHA 在 [sota_landscape_20261124.json](sota_landscape_20261124.json)。

SOTA-2 补充：第7–9节逐项列出架构、GazeLLE关系和多层特征类型；新增PaGE、GazeCoT中的GazeLLE-v3及PINTO工程实现。PaGE是上一轮检索遗漏的直接相关强方法，其公开结果高于我们，不能将原8行文献视为最新SOTA上限。架构判据、29份固定commit源码SHA及公开模型元数据见 [architecture_lineage_20261124.json](architecture_lineage_20261124.json)。本轮仍未运行任何模型。

## 1. 当前目标调整

按用户决定，将主目标调整为：在纯 RGB、真实头框、单帧设置下，与近期强方法比较定位精度和完整推理成本。以人物条件化稀疏深层计算为贡献；DINOv3、多层表示沿用。近期 object-aware / vision-language 方法是重点外部对照，不再把全面超过官方 GazeLLE 作为推进前提。

保留已完成 GazeLLE 比较，解释方法与底座的关系；当前 B1 更慢、B8 接近的结果不删除。跨模型比较说明完整系统权衡，Router 增量仍由同表示控制、预算响应和人物条件化消融说明。训练仍为 seed0、train/test、逐轮 test 选单一 checkpoint，不新增 dense 训练。

## 2. 已核实的公开指标

文献数值来自各自原文/官方结果表，尚未在本机或服务器统一复现。GF 的 AUC/Avg/Min，VAT 的 AUC/L2/AP；方向分别为 ↑/↓/↓、↑/↓/↑。不把不同协议结果称为统一复测。我们的行使用已有选中 checkpoint，不取各 epoch 分项最高值。

| 方法 | 发表 | GF AUC | GF Avg L2 | GF Min L2 | VAT AUC | VAT L2 | VAT AP |
|---|---|---:|---:|---:|---:|---:|---:|
| Mi et al., Object-Aware | ECCV 2026 | .961 | .094 | .038 | .948 | .095 | .923 |
| TextGaze-B | IJCAI-ECAI 2026 | .953 | .105 | .047 | .932 | .113 | .879 |
| TextGaze-L | IJCAI-ECAI 2026 | .956 | .099 | .043 | .941 | .102 | .911 |
| OmniGF | NeurIPS 2026 | .941 | .091 | .040 | .917 | .096 | .923 |
| GazeHRNet | Sensors 2026 | .952 | .102 | .055 | .929 | .103 | .909 |
| GazeVLM（会议版） | CVPR Workshops 2026 | .936 | .123 | .061 | .948 | .102 | .898 |
| ViTGaze | Visual Intelligence 2024 | .949 | .105 | .047 | .938 | .102 | .905 |
| Sharingan | CVPR 2024 | .944 | .113 | .057 | — | .107 | .891 |
| Ours-B K50 | 本轮已审查 | .9562 | .1016 | .0434 | .9400 | .1001 | .8832 |
| Ours-L K50 | GF/VAT 已审查 | .9600 | .0941 | .0377 | .9483 | .0897 | .8916 |

来源分别为 [Mi Table 1](https://arxiv.org/pdf/2606.29334)、[TextGaze v2 Table 1](https://arxiv.org/html/2607.10130v2)、[OmniGF v2 Table 9](https://arxiv.org/html/2605.26399v2)、[GazeHRNet 原文](https://www.mdpi.com/1424-8220/26/15/4799)、[GazeVLM 会议版 Table 1](https://openaccess.thecvf.com/content/CVPR2026W/AI4RWC/papers/Mathew_GazeVLM_A_Vision-Language_Model_for_Multi-Task_Gaze_Understanding_CVPRW_2026_paper.pdf)、[ViTGaze 官方表](https://github.com/hustvl/ViTGaze)、[Sharingan 官方论文](https://openaccess.thecvf.com/content/CVPR2024/html/Tafasca_Sharingan_A_Transformer_Architecture_for_Multi-Person_Gaze_Following_CVPR_2024_paper.html)。Mi、TextGaze、OmniGF、GazeVLM 的相关整页已渲染目视核对。GazeVLM 不沿用 Mi 表中较旧版本的数值，也不与其他同名项目混淆。

Ours-L 的 GF 为原 Min L2 规则选中的第 2 轮，完整行 `.960007/.094061/.037721`，实际 support3+decoder4 后由用户收口。Ours-B 的 VAT 为完整 GF→VAT 后 `.940033/.100131/.883183`，不是旧低精度 VAT 初始化结果。VAT-L 现已完成完整GF-L迁移8轮，按test L2选第1轮，全量逐帧 `.948264/.089676/.891592`，工件/导出重聚合审查通过，见 [E2-TL审查](E2TL_GFL_TO_VATL_REVIEW.md)。其效率已在GPU0完成，FP16 B1/B8相对官方GazeLLE-L速度比 `.8630/1.0652×`，相对同权重K100为 `1.0168/1.3123×`，见 [E3-L审查](E3_VATL_EFFICIENCY_REVIEW.md)。尚未与表中外部强方法同机测速。

## 3. ECCV 2026 那篇是否适合作为重点比较

已确认是 Jiajie Mi 等的 *Multi-scale Object-Aware Gaze Estimation via Geometric Reasoning*。原文使用冻结 DINOv3-L/16、多层表示、物体语义和方向/FOV 模块；YOLO11x/SAM2-hiera-large 离线生成 masks，实验用 RTX5090。称其“含多模块和额外物体处理”更准确，不据此假定多个 DINOv3 分支或必然更慢。原文未见可直接采用的 latency/FLOPs 表。[原文方法与设置](https://arxiv.org/html/2606.29334v1)

GF-L 的 Avg L2 与其基本相同，AUC 稍低；Min L2 按公开数值略低，但对方只报三位小数，双方 Min 均可四舍五入为 .038，不称为已显著超过。VAT-L 的 AUC `.948264` 与其公开 `.948` 同量级，L2 `.089676` 比 `.095` 低5.60%，AP `.891592` 比 `.923` 低0.031408。因此可以作为“接近定位精度、比较成本”的重点参照，当前不能写成已全面胜出。

截至检索日，论文、共同作者[主页](https://liuxy1005.github.io/)没有找到官方实现/权重链接，GitHub 标题检索无对应仓库。记录为“未找到”，不证明作者没有公开代码。现阶段可作公开精度参照；不能声称已复现其效率。

## 4. 更适合实际测速的对照

| 方法 | 代码与权重核验 | 当前适合怎样比较 |
|---|---|---|
| TextGaze B/L | [官方代码](https://github.com/idremo/TextGaze-IJCAI2026)，[checkpoints-v1.0](https://github.com/idremo/TextGaze-IJCAI2026/releases/tag/checkpoints-v1.0) API 实际列出 GF/VAT × B/L 四个 PT 和摘要 | 第一优先；同规模比较，训练不用重跑 |
| OmniGF | [官方代码](https://github.com/cvlab-stonybrook/omnigf)，[官方 HF](https://huggingface.co/StonyBrook-CVLab/omnigf) API 列出 GF/VAT safetensors；还需 Qwen3-VL-4B 底座 | 第二优先；近期强 VLM、多人方法，先确认 3090 上官方配置可运行 |
| Sharingan / ViTGaze | 官方仓库均提供任务权重链接，尚未执行大权重下载 | 成熟补充对照；多人结构与纯 RGB ViT 各一个 |
| GazeHRNet | [官方代码](https://github.com/Zhikanyoumai/gazehrnet)存在；论文声称发布权重，但当前 README/源码树未找到任务权重下载 | 先获取 checkpoint，再排测速；注意其 head-aware routing 与本方法的实际 token 稀疏执行区别 |
| GazeVLM | 会议论文可读，尚未核实对应官方代码/任务权重 | 先作公开精度参照；使用深度/HHA，输入标明 |

TextGaze 使用 DINOv2+BERT 和预生成文本线索。当前 GF-L 三项均优于其 L 的报告值；当前 B 在 GF/VAT 两个数据集的六项均优于 TextGaze-B 的公开值。VAT-L 对 TextGaze-L 的AUC/L2更好，AP较低；VAT-B对其L是L2更低、AUC/AP较低。计时应区分给定文本后的 core forward 与从原始 RGB 生成文本的完整流程，不能将缓存输入与在线生成成本混为一个口径。[TextGaze 原文与发布说明](https://arxiv.org/html/2607.10130v2)

OmniGF 的 GF Avg L2 .091 优于我们，但 AUC/Min L2 弱于 GF-L；VAT-L的AUC/L2优于其公开 `.917/.096`，AP低于其 `.923`。它基于 4B VLM，官方仓库说明评测约需20–30GB。Table15 的单人53.62ms是在 RTX PRO6000 Blackwell、BF16、batch64、合成头框及每人生成25tokens下的每图时间，不能除以我们3090的B1/B8时间得到正式加速比。需要复现其产生预测的官方路径。[原文附录H](https://arxiv.org/html/2605.26399v2)

## 5. 另外发现的近期工作

- **GazeAnywhere，CVPR2026**：有[官方代码/权重入口](https://github.com/IrohXu/GazeAnywhere)，但研究概念/语言提示的 PGE，使用筛选后的 Gaze-Co 测试集合。不能把其 GF-Concept/VAT-Concept 数字直接放进标准 GT-headbox GF/VAT 表。[原论文](https://openaccess.thecvf.com/content/CVPR2026/papers/Cao_Gaze_Target_Estimation_Anywhere_with_Concepts_CVPR_2026_paper.pdf)
- **FGI-Gaze，PRCV2025，论文集2026出版**：[出版方摘要](https://link.springer.com/chapter/10.1007/978-981-95-5676-2_32)已核实；完整原表受订阅限制。Mi Table1 转引的 VAT `.958/.091/.911` 是重要强对照线索，尚未独立核原表，不并入“已核实指标”主表。
- **GeoDistillGaze，Pattern Recognition Letters**：2026在线，12月卷期；[出版方摘要](https://www.sciencedirect.com/science/article/pii/S0167865526003533)描述 DINOv2 教师向 DeiT 学生蒸馏以减少计算。它直接涉及效率，应纳入相关工作；当前未核到完整指标与官方任务权重，不预判我们更快。
- **GazeLoom，EAAI2026**：[出版方](https://www.sciencedirect.com/science/article/pii/S0952197626014399)报告 GF AUC .967、距离 .079、VAT AP .942，比“所有最新方法均不如我们”更强；但[官方 README](https://github.com/Pjj0417/GAZELOOM)示例为 GF Avg .112/Min .079，结果表却只标 L2 .079。先核完整原表，不能把 .079 直接当 Avg L2。代码已找到，未确认任务 checkpoint；使用深度信息，输入单独标注。

## 6. 下一项工作及行文

下一项只准备 TextGaze 官方 B/L checkpoint 的推理比较：核对权重/输入文本、原始数据对应关系与预测，再给服务器同硬件评测/测速入口；不训练新 baseline。OmniGF 作为接续项，先检查官方评测的显存与数据要求。Mi、GazeVLM 暂用原文指标；GazeLoom/FGI/GeoDistill 补齐原表或权重后更新，不从未经核对的二手表填主表。

论文主叙事改为“人物条件化深层计算分配，在近期高精度方法之间研究精度—成本权衡”；对象、文本、深度和生成路径的差异在主表标清。是否更快由统一硬件完整测量给出，参数更少、分支更多不能代替时间证据。保留 GazeLLE 参照、当前 K100 控制和既有全部效率结果；原有 train/test 选模与 ICASSP 贡献边界均不改变。

## 7. 哪些采用 GazeLLE 这套框架

这里将GazeLLE骨架定义为：**按图共享场景编码，按人物注入头部位置条件，再用轻量空间decoder预测热图/in-out。** “作者明确从GazeLLE修改”“代码结构沿用这种骨架”“只是同样使用冻结ViT”分别标注；结构相近不证明复用了官方checkpoint。

“多层特征”专指从backbone不同深度取出特征/attention再融合。三层Transformer、五层SIM、多个输入分支和逐层文本监督，都不自动等于backbone多层特征。

| 方法 | 与GazeLLE的关系 | 主干 | backbone多深度融合 | 额外分支/条件 |
|---|---|---|---|---|
| 官方GazeLLE | 原始骨架 | DINOv2 B/L | 否，最后一层 | GT头框位置prompt |
| 我们的Router | 现有本地扩展 | DINOv3 B/L | **是，四层** | 头部外观/相对几何support，多人union，稀疏suffix |
| Mi et al., ECCV2026 | 同属冻结编码器范式；未确认复用GazeLLE decoder | **DINOv3-L** | **是，浅/中/深** | 物体tokens、方向/FOV、YOLO/SAM masks |
| TextGaze | 论文及代码呈现GazeLLE式扩展 | DINOv2 B/L | 否，最后一层 | Qwen生成文本→BERT；视觉文本联合decoder |
| GazeHRNet | 代码沿用GazeLLE式骨架；未证明官方权重复用 | DINOv2 | 否，最后一层 | 相对极坐标、GRSP、head-aware attention、GAZE token |
| GazeLoom | 代码沿用GazeLLE式骨架；论文checkpoint配置未闭合 | DINOv2/SimDINOv2；**代码也有DINOv3入口** | 否，读取单一最终map | 深度编码、几何融合、深度位置编码 |
| **PaGE** | **论文明确从GazeLLE出发，代码说明改编来源** | **DINOv3，场景/头部两套编码** | 否，两分支各取最后一层 | 五层SIM双向交互、RoPE、head crops |
| **GazeLLE-v3（GazeCoT）** | **论文明确替换GazeLLE底座** | **DINOv3-L/H+** | 未报告多层融合 | 单独估计器仍为场景+头框；其后可接MLLM agent |
| PINTO gazelle-dinov3 | 明确基于GazeLLE的工程fork | DINOv3及蒸馏CNN/ViT | 否，所核forward为最终map | ONNX部署、逐步解冻/蒸馏；非已核实SOTA论文 |
| GazeAnywhere | 冻结编码器+轻量decoder结构相近，任务/prompt已改变 | DINOv3-L+文本编码器等 | 否，所核DINOv3配置取最后一层 | 文本指定人物，预测头框而非输入GT头框 |
| OmniGF | 独立VLM架构 | Qwen3-VL-4B+ResNet18 | 最终模型否；四层DPT仅消融 | 语言生成与空间预测双输出 |
| GazeVLM | 独立VLM架构 | Qwen2.5-VL-3B | 不是DINOv3多层方案 | RGB+HHA深度+任务文本 |
| ViTGaze | 早于GazeLLE的独立架构 | 所核配置DINOv2-S/14 | **多层attention maps**，非DINOv3特征拼接 | 头部引导的head-to-scene interaction |
| Sharingan | 早于GazeLLE的独立架构 | MultiMAE初始化ViT-B+ResNet18 | **是，四层进入DPT**；不是DINOv3 | 头部外观/框tokens，多人联合编码 |
| GeoDistillGaze | 轻量化范式相关，decoder继承关系未核 | DINOv2教师→DeiT学生 | 未核实 | 训练时蒸馏，推理时相对坐标/门控 |
| FGI-Gaze | 可读摘要不足以确认 | 未确认 | 未确认 | human-environment交互与过滤，独立全局in/out信息 |

**直接回答配置匹配：**现已核实的外部论文中，Mi最符合“DINOv3＋backbone多层表示”；PaGE、GazeLLE-v3更明确属于GazeLLE改进线，但采用单层backbone输出。当前没有找到同时满足“明确GazeLLE派生＋DINOv3＋多层输出融合＋已验证公开任务权重”的外部对照，不能仅凭仓库名称将其归入这一组合。

## 8. 各自的模型架构

### 8.1 官方GazeLLE与我们的差异

```text
Official: RGB -> DINOv2 final map -> projection(256) -> per-person head prompt
          -> 3 Transformer blocks -> heatmap / in-out
Ours:     RGB -> DINOv3 dense prefix -> head-conditioned support / max-union / TopK
          -> sparse suffix -> early-exit fill -> four-map concat / projection(256)
          -> per-person head prompt -> 3 Transformer blocks -> heatmap / in-out
```

我们的B取`[2,5,8,11]`、L取`[5,11,17,23]`（0-based），后两层回填完整网格。实际改动的是编码器后段的token执行数量，四层表示并非新贡献。[官方实现](https://github.com/fkryan/gazelle)；本地源码为`gazelle/model_v0_dinov2.py`、`gazelle/routing/{model,backbone,router}.py`、`gazelle/model.py`。

### 8.2 Mi et al.：DINOv3＋多层，但未核实为原版GazeLLE decoder

```text
RGB + object masks -> image/object token interaction -> enhanced representation
 -> frozen DINOv3-L -> shallow/middle/deep maps -> deep-dominant residual fusion
Head appearance + eye position -> gaze direction -> smoothed FOV embedding
Fused multi-depth maps + FOV -> spatial decoder -> heatmap / in-out
```

融合为`Fdeep + alpha_mid * Fmid + alpha_shallow * Fshallow`，与我们四层concat不同；具体block编号未核实。YOLO11x/SAM2负责离线mask生成，不能把多个功能模块直接等同多个DINOv3。原文没有给出足以确认其使用官方GazeLLE三层decoder的实现信息。[原文§3.2–3.4](https://arxiv.org/html/2606.29334v1)

### 8.3 TextGaze：GazeLLE式单层视觉分支＋文本

```text
RGB -> frozen DINOv2 final patches -> 256 projection + head-map embedding
RGB -> Qwen2.5-VL scene cues -> frozen BERT -> 64 text tokens -> 256 projection
 -> [in-out?, text, visual] -> 3-layer vision-text Transformer
 -> visual residual -> heatmap; in-out token -> classifier
```

`backbone.py`只读`x_norm_patchtokens`，没有多深度map融合。逐层文本监督发生在融合Transformer，不是DINOv2多层特征。代码保留GazeLLE式load/save别名与头图prompt，但不据此认定其官方checkpoint可直接替换我们的权重。[model](https://github.com/idremo/TextGaze-IJCAI2026/blob/2cede002c4a865bbbdaadf29dfc6f960b693db2c/gazefuze/model.py#L178)、[backbone](https://github.com/idremo/TextGaze-IJCAI2026/blob/2cede002c4a865bbbdaadf29dfc6f960b693db2c/gazefuze/backbone.py#L38)。

### 8.4 GazeHRNet：GazeLLE式骨架＋头部几何/attention改造

```text
DINOv2 final map -> projection(256) -> GRSP -> per-person repeat
 -> head-centric polar encoding + head-map prompt + GAZE token
 -> 3 HAAR blocks -> heatmap / in-out + 8x8 patch-distribution head
```

GRSP计算完整token相似度矩阵；HAAR在完整attention上增加头部偏置和局部抑制。它们没有让DINOv2后段只执行K个tokens，不能把其“routing/sparse propagation”与我们的稀疏suffix混同。对比学习和各向异性监督属于训练侧设计。[model](https://github.com/Zhikanyoumai/gazehrnet/blob/144273307cd7b97150447d6fc876f1360f9b2cd8/models/gazehrnet.py#L118)、[GRSP](https://github.com/Zhikanyoumai/gazehrnet/blob/144273307cd7b97150447d6fc876f1360f9b2cd8/models/grsp.py#L14)、[HAAR](https://github.com/Zhikanyoumai/gazehrnet/blob/144273307cd7b97150447d6fc876f1360f9b2cd8/models/haar.py#L19)。

### 8.5 GazeLoom：单层场景分支＋深度融合

```text
RGB -> selected backbone final map -> semantic projection(256)
Depth map -> resize / depth projection(256)
 -> channel/spatial/pixel gating fusion -> depth-guided positional encoding
 -> head-map prompt -> 6 default Transformer blocks -> heatmap / in-out
```

源码提供DINOv2、DINOv3、SimDINOv2工厂，README训练命令主要用SimDINOv2。各backbone接口返回一张最终map；六层decoder不是六层backbone融合。代码选项与论文报告所对应的checkpoint配置须分别登记。[model](https://github.com/Pjj0417/GAZELOOM/blob/d12d5dbd0c69be29315c30f49eb778d5f5c18ae6/gazeloom/model.py#L153)、[backbone](https://github.com/Pjj0417/GAZELOOM/blob/d12d5dbd0c69be29315c30f49eb778d5f5c18ae6/gazeloom/backbone.py#L323)。

### 8.6 PaGE：明确从GazeLLE发展出的DINOv3双分支

```text
Scene RGB(512) -> scene DINOv3 final map -> projection(256) -> self-attention
Head crops(256) -> head DINOv3 final maps -> projection(256) -> self-attention
 -> 5 bidirectional SIM layers (cross-attention + per-branch self-attention/FFN)
 -> scene map heatmap; paired scene/head in-out tokens -> classifier
```

使用统一坐标2D RoPE、GEGLU和register tokens，最终RoPE方案移除原head-map prompt。两套backbone各取`last_hidden_state`，不是多深度fusion。论文明确以GazeLLE为起点，但最终采用decoder预训练后全模型微调，并有联合数据/额外无标注蒸馏；不能按我们的冻结主干训练配置描述。[原文§3.2](https://arxiv.org/html/2607.04860v1)、[model](https://github.com/OctopusWen/PaGE/blob/4352beaddbd8d2942a6b0152f4d83ac7090f5f5c/page/page_model.py#L148)、[backbone](https://github.com/OctopusWen/PaGE/blob/4352beaddbd8d2942a6b0152f4d83ac7090f5f5c/page/backbone.py#L156)。

### 8.7 GazeCoT中的GazeLLE-v3与PINTO工程实现

```text
GazeLLE-v3: RGB(512) -> DINOv3-L/H+ -> GazeLLE-style decoder -> gaze heatmap
GazeCoT:   gaze estimator -> gaze overlays / ROI text prompts -> MLLM social reasoning
PINTO:     final DINOv3 map -> projection/head prompt -> 3 blocks -> heatmap/in-out
```

GazeCoT论文§3.2明确替换底座、输入尺寸和训练时长，没有报告多层特征融合；不将其后MLLM流程计作单独gaze estimator的必需架构。PINTO是明确基于GazeLLE的可部署工程fork，所核forward也只返回最终map；同时包含蒸馏和逐步解冻方案，不能混称新的同协议论文结果。[GazeCoT原文](https://pi.cs.tsinghua.edu.cn/wp-content/uploads/2026/04/3772318.3790922.pdf)、[PINTO model](https://github.com/PINTO0309/gazelle-dinov3/blob/0c67fb720feb6d2b8178bfba31f00ecca2a2e35d/gazelle/model.py#L13)、[PINTO backbone](https://github.com/PINTO0309/gazelle-dinov3/blob/0c67fb720feb6d2b8178bfba31f00ecca2a2e35d/gazelle/backbone.py#L569)。

### 8.8 GazeAnywhere、OmniGF与GazeVLM

- **GazeAnywhere**：场景与人物描述→冻结视觉/文本编码器→投影256→文本条件head query、图像条件presence token与图文tokens→三层Transformer→头框、热图、in/out。所核DINOv3配置`vision_model_patch_token_layer=1`取最后一层；它预测头框，任务不同于我们的输入GT头框。[model](https://github.com/IrohXu/GazeAnywhere/blob/9616d17405f39ae1f4e03db2e30e1919ae1e8485/modeling/models/anygaze_modeling_mapper.py#L155)、[配置](https://github.com/IrohXu/GazeAnywhere/blob/9616d17405f39ae1f4e03db2e30e1919ae1e8485/modeling/backbone/dinotxt.py#L155)。
- **OmniGF**：RGB场景＋ResNet18头部crop特征→gaze-token注入Qwen3-VL-4B→生成的人物tokens和最后一层视觉hidden states→人物条件卷积上采样热图/in-out，同时生成结构化语言。最终decoder标记`NEEDS_MULTI_LAYER=False`；四层DPT属于消融，不是最终模型。[decoder](https://github.com/cvlab-stonybrook/omnigf/blob/b5554f13233b7ff6690499935e887fbd08c88c17/omnigf/vlm/decoders.py#L25)、[原文附录E](https://arxiv.org/html/2605.26399v2)。
- **GazeVLM**：RGB与HHA深度分别经Qwen2.5-VL视觉编码→结合任务文本→语言decoder输出人物框、gaze坐标、对象信息。它不是GazeLLE的头图prompt＋256维decoder，也不是DINOv3多层方案。[会议版§4](https://openaccess.thecvf.com/content/CVPR2026W/AI4RWC/papers/Mathew_GazeVLM_A_Vision-Language_Model_for_Multi-Task_Gaze_Understanding_CVPRW_2026_paper.pdf)

### 8.9 ViTGaze、Sharingan、GeoDistillGaze与FGI-Gaze

- **ViTGaze**：DINOv2-S/14→`[2,5,8,11]`层attention maps→头部patch引导的head-to-scene交互聚合→热图；最终patch特征结合头mask池化→in/out。它确实使用多层交互信息，但不是DINOv3的多层patch map拼接。[GF配置](https://github.com/hustvl/ViTGaze/blob/4bf2f07e18599baa6e385f8133cdfd7ae7e88dea/configs/gazefollow_518.py#L48)、[model](https://github.com/hustvl/ViTGaze/blob/4bf2f07e18599baa6e385f8133cdfd7ae7e88dea/modeling/meta_arch/gaze_attention_mapper.py#L40)。
- **Sharingan**：ResNet18头部外观＋bbox→每人gaze token；与场景patch tokens联合进入MultiMAE初始化ViT-B→取`[2,5,8,11]`进入人物条件DPT→热图，最终gaze tokens接in/out。它有多层特征，但采用独立头编码器和联合encoder，是GazeLLE之前的路线。[源码](https://github.com/idiap/sharingan/blob/006b69dfb70e83aa8b073d0a8ead10325022f687/src/modeling/sharingan.py#L537)。
- **GeoDistillGaze**：训练时DINOv2教师通过feature-map softmax/KL监督DeiT学生；推理时学生表示与头部相对坐标/自适应门控用于预测。教师学生不等于推理双分支。当前可读内容不足以确认decoder及具体取层，更不能改写成DINOv3。[出版方方法概述](https://www.sciencedirect.com/science/article/pii/S0167865526003533)
- **FGI-Gaze**：摘要可确认图像中的human-environment gaze interaction、特征过滤和独立全局语义in/out预测；backbone、取层和decoder细节未核到，保留未知。[出版方](https://link.springer.com/chapter/10.1007/978-981-95-5676-2_32)

## 9. 新发现的近亲模型怎样进入比较

优先区分两条线：**配置最接近的DINOv3多层参照是Mi；明确从GazeLLE发展的公开DINOv3模型是PaGE。** TextGaze仍是容易复用数据与模型接口的近期对照，但它不满足DINOv3多层这一偏好。

PaGE[官方仓库](https://github.com/OctopusWen/PaGE)和HF模型已公开，本轮API确认[ViT-B](https://huggingface.co/Octopus1/page-vitb)的`model.safetensors`与[H+](https://huggingface.co/Octopus1/page-vithplus)分片权重存在，未下载大权重。[官方项目表](https://page-26.github.io/)公开的B-distill为GF `.966/.081/.029`、VAT `.969/.068/.945`，高于我们当前结果；其联合GF/VAT/ChildPlay标注、全模型微调和MPII/OpenImages蒸馏数据需在比较中注明，不改动我们的冻结主干约束。

GazeCoT的GazeLLE-v3-L/H+在GF报告`.959/.097/.040`与`.960/.093/.038`，更适合检验“DINOv3替换底座”已能带来多少表现；本轮没有核实其具体checkpoint下载入口，不把PINTO或其他fork自动当作对应权重。此补充不启动训练或测速、不替换当前选中模型；下一项准备推理对照时将PaGE列为直接相关候选，仍逐项实施。
