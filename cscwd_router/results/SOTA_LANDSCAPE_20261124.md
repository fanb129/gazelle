# SOTA-1：近期方法、公开权重与比较目标

检索日期：2026-11-24。只读论文、官方代码/发布 API 和本地已选定结果；没有新训练、下载大权重或 GPU 推理。机器可读来源、版本、PDF SHA、官方权重发布元数据与本地报告 SHA 在 [sota_landscape_20261124.json](sota_landscape_20261124.json)。

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
| Ours-L K50 | GF 已审查 | .9600 | .0941 | .0377 | — | — | — |

来源分别为 [Mi Table 1](https://arxiv.org/pdf/2606.29334)、[TextGaze v2 Table 1](https://arxiv.org/html/2607.10130v2)、[OmniGF v2 Table 9](https://arxiv.org/html/2605.26399v2)、[GazeHRNet 原文](https://www.mdpi.com/1424-8220/26/15/4799)、[GazeVLM 会议版 Table 1](https://openaccess.thecvf.com/content/CVPR2026W/AI4RWC/papers/Mathew_GazeVLM_A_Vision-Language_Model_for_Multi-Task_Gaze_Understanding_CVPRW_2026_paper.pdf)、[ViTGaze 官方表](https://github.com/hustvl/ViTGaze)、[Sharingan 官方论文](https://openaccess.thecvf.com/content/CVPR2024/html/Tafasca_Sharingan_A_Transformer_Architecture_for_Multi-Person_Gaze_Following_CVPR_2024_paper.html)。Mi、TextGaze、OmniGF、GazeVLM 的相关整页已渲染目视核对。GazeVLM 不沿用 Mi 表中较旧版本的数值，也不与其他同名项目混淆。

Ours-L 为原 Min L2 规则选中的第 2 轮，完整行 `.960007/.094061/.037721`，实际 support3+decoder4 后由用户收口。Ours-B 的 VAT 为完整 GF→VAT 后 `.940033/.100131/.883183`，不是旧低精度 VAT 初始化结果。本次只读服务器未发现完整 GF-L→VAT-L 的最终 summary，VAT-L 保留空值；不混用旧 VAT-L。

## 3. ECCV 2026 那篇是否适合作为重点比较

已确认是 Jiajie Mi 等的 *Multi-scale Object-Aware Gaze Estimation via Geometric Reasoning*。原文使用冻结 DINOv3-L/16、多层表示、物体语义和方向/FOV 模块；YOLO11x/SAM2-hiera-large 离线生成 masks，实验用 RTX5090。称其“含多模块和额外物体处理”更准确，不据此假定多个 DINOv3 分支或必然更慢。原文未见可直接采用的 latency/FLOPs 表。[原文方法与设置](https://arxiv.org/html/2606.29334v1)

GF-L 的 Avg L2 与其基本相同，AUC 稍低；Min L2 按公开数值略低，但对方只报三位小数，双方 Min 均可四舍五入为 .038，不称为已显著超过。当前 VAT-B 的 AUC/L2/AP 三项均落后，待 VAT-L 完成后再作同规模判断。因此可以作为“接近精度、比较成本”的重点参照，当前不能写成已全面胜出。

截至检索日，论文、共同作者[主页](https://liuxy1005.github.io/)没有找到官方实现/权重链接，GitHub 标题检索无对应仓库。记录为“未找到”，不证明作者没有公开代码。现阶段可作公开精度参照；不能声称已复现其效率。

## 4. 更适合实际测速的对照

| 方法 | 代码与权重核验 | 当前适合怎样比较 |
|---|---|---|
| TextGaze B/L | [官方代码](https://github.com/idremo/TextGaze-IJCAI2026)，[checkpoints-v1.0](https://github.com/idremo/TextGaze-IJCAI2026/releases/tag/checkpoints-v1.0) API 实际列出 GF/VAT × B/L 四个 PT 和摘要 | 第一优先；同规模比较，训练不用重跑 |
| OmniGF | [官方代码](https://github.com/cvlab-stonybrook/omnigf)，[官方 HF](https://huggingface.co/StonyBrook-CVLab/omnigf) API 列出 GF/VAT safetensors；还需 Qwen3-VL-4B 底座 | 第二优先；近期强 VLM、多人方法，先确认 3090 上官方配置可运行 |
| Sharingan / ViTGaze | 官方仓库均提供任务权重链接，尚未执行大权重下载 | 成熟补充对照；多人结构与纯 RGB ViT 各一个 |
| GazeHRNet | [官方代码](https://github.com/Zhikanyoumai/gazehrnet)存在；论文声称发布权重，但当前 README/源码树未找到任务权重下载 | 先获取 checkpoint，再排测速；注意其 head-aware routing 与本方法的实际 token 稀疏执行区别 |
| GazeVLM | 会议论文可读，尚未核实对应官方代码/任务权重 | 先作公开精度参照；使用深度/HHA，输入标明 |

TextGaze 使用 DINOv2+BERT 和预生成文本线索。当前 GF-L 三项均优于其 L 的报告值；当前 B 在 GF/VAT 两个数据集的六项均优于 TextGaze-B 的公开值。VAT-B 对 TextGaze-L 是 L2 更低、AUC/AP 较低。计时应区分给定文本后的 core forward 与从原始 RGB 生成文本的完整流程，不能将缓存输入与在线生成成本混为一个口径。[TextGaze 原文与发布说明](https://arxiv.org/html/2607.10130v2)

OmniGF 的 GF Avg L2 .091 优于我们，但 AUC/Min L2 弱于 GF-L；VAT 的距离与 AP 仍优于当前 B。它基于 4B VLM，官方仓库说明评测约需20–30GB。Table15 的单人53.62ms是在 RTX PRO6000 Blackwell、BF16、batch64、合成头框及每人生成25tokens下的每图时间，不能除以我们3090的B1/B8时间得到正式加速比。需要复现其产生预测的官方路径。[原文附录H](https://arxiv.org/html/2605.26399v2)

## 5. 另外发现的近期工作

- **GazeAnywhere，CVPR2026**：有[官方代码/权重入口](https://github.com/IrohXu/GazeAnywhere)，但研究概念/语言提示的 PGE，使用筛选后的 Gaze-Co 测试集合。不能把其 GF-Concept/VAT-Concept 数字直接放进标准 GT-headbox GF/VAT 表。[原论文](https://openaccess.thecvf.com/content/CVPR2026/papers/Cao_Gaze_Target_Estimation_Anywhere_with_Concepts_CVPR_2026_paper.pdf)
- **FGI-Gaze，PRCV2025，论文集2026出版**：[出版方摘要](https://link.springer.com/chapter/10.1007/978-981-95-5676-2_32)已核实；完整原表受订阅限制。Mi Table1 转引的 VAT `.958/.091/.911` 是重要强对照线索，尚未独立核原表，不并入“已核实指标”主表。
- **GeoDistillGaze，Pattern Recognition Letters**：2026在线，12月卷期；[出版方摘要](https://www.sciencedirect.com/science/article/pii/S0167865526003533)描述 DINOv2 教师向 DeiT 学生蒸馏以减少计算。它直接涉及效率，应纳入相关工作；当前未核到完整指标与官方任务权重，不预判我们更快。
- **GazeLoom，EAAI2026**：[出版方](https://www.sciencedirect.com/science/article/pii/S0952197626014399)报告 GF AUC .967、距离 .079、VAT AP .942，比“所有最新方法均不如我们”更强；但[官方 README](https://github.com/Pjj0417/GAZELOOM)示例为 GF Avg .112/Min .079，结果表却只标 L2 .079。先核完整原表，不能把 .079 直接当 Avg L2。代码已找到，未确认任务 checkpoint；使用深度信息，输入单独标注。

## 6. 下一项工作及行文

下一项只准备 TextGaze 官方 B/L checkpoint 的推理比较：核对权重/输入文本、原始数据对应关系与预测，再给服务器同硬件评测/测速入口；不训练新 baseline。OmniGF 作为接续项，先检查官方评测的显存与数据要求。Mi、GazeVLM 暂用原文指标；GazeLoom/FGI/GeoDistill 补齐原表或权重后更新，不从未经核对的二手表填主表。

论文主叙事改为“人物条件化深层计算分配，在近期高精度方法之间研究精度—成本权衡”；对象、文本、深度和生成路径的差异在主表标清。是否更快由统一硬件完整测量给出，参数更少、分支更多不能代替时间证据。保留 GazeLLE 参照、当前 K100 控制和既有全部效率结果；原有 train/test 选模与 ICASSP 贡献边界均不改变。
