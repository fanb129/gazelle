# CSCWD Router · 人物条件化的深层计算分配

更新：2026-11-23。论文方法覆盖 **DINOv3 ViT-B/16、ViT-L/16 + 多层 raw-concat + gaze-conditioned Router**，四组新训练均 base seed 0。既有 DINOv2 单层适配保留为工程与对照路线。

## 当前 story

冻结视觉编码器降低任务训练的参数量，但推理仍需对完整图像执行编码。我们在编码器中层利用场景、头部外观和相对几何预测 support，让选中位置继续执行深层 blocks；同图多人的需求聚合为共同预算中的一条 sparse suffix。中层回填维持完整解码网格。

DINOv3 是第三方编码器，多层表示是沿用的基础配置。新增贡献集中在人物条件化计算选择、多人共享稀疏执行与实际精度–效率权衡；不再次主张 ICASSP 已覆盖的多层融合和软空间门控。

GazeLLE 本身已支持多人共享场景编码。比较对象应包括其原生共享 dense 前向，不能将它描述为每个人重复执行一次 backbone。

## 写作与执行入口

- [E1_GFB_E2C_REVIEW.md](results/E1_GFB_E2C_REVIEW.md)：GF-B 精度、VAT 续训失败原因分析与后续实验；结果已回传并核验。
- [RUN_VAT_CONTINUE.md](RUN_VAT_CONTINUE.md)：E2-C，VAT 初始化 GF→VAT 来源与固定 K50 低 LR 续训；续训已完成，各追加 8 epochs 未改善 L2，保留旧 best；命令保留供复现。
- [E2_VAT_SEED0_REVIEW.md](results/E2_VAT_SEED0_REVIEW.md)：VAT B/L 已完成并回传；逐帧精度、选模与 14 个大工件核验。
- [RUN_DINOV3_TRAIN_TEST.md](RUN_DINOV3_TRAIN_TEST.md)：已实现入口的完整 GPU 0/1 命令、输出与恢复；GF-B 与 VAT-B 先并行，各自接 L。
- [EXPERIMENT_PLAN.md](EXPERIMENT_PLAN.md)：GF/VAT × B/L 当前清单；四组均 support 3 + sparse decoder 5 epochs，不重训 dense。
- [PAPER_STORY.md](latex/PAPER_STORY.md)：已确定的问题链、DINOv3/多层的引入方式、方法、五段中英文 Introduction 工作稿、图表与实验论证。
- [PAPER_REVIEW.md](latex/PAPER_REVIEW.md)：被拒 MM、rebuttal、在审 ICASSP 的 15 页复核、重叠边界与 MM 图 2 的复用决定。
- [WRITING_EVIDENCE_20261122.json](results/WRITING_EVIDENCE_20261122.json)：3 份 PDF、候选图素材及 19 份旧实验 JSON 的来源 SHA 和真实指标。
- [implementation_plan.md](implementation_plan.md)：当前决定及原 DINOv2 T1–T6 的工程/验收记录。
- [codex_prompt.md](codex_prompt.md)：后续会话说明。

## 已有结果的用途

旧 DINOv3 GF K50 相对 matched dense 的 AUC 差为 `-0.000802`。旧 B1/B4/B8 的 latency speedup 为 `1.001525 / 1.060676 / 1.148545x`，比较的是自身 dense；效率 checkpoint 与正式 GF 准确率 checkpoint 不同，未证明相对官方 GazeLLE 的速度优势。VAT K50/K75 的旧 source-video holdout 定位损失保留。

论文目标是有竞争力的准确率与更低实际成本，不预写尚未测得的“双优”、显存节省、整网计算减半或多人保证无损。

新入口采用全量 train、逐 epoch test、单指标选 checkpoint，标记 `test_selected`。GF 按 Min L2，VAT 按 L2，VAT 最后自动逐帧 test。CPU 32/32 通过；旧 gazelle/ 与历史脚本未改。VAT B/L 已完成回传，逐帧 AUC/L2/AP 为 B `0.913548/0.110971/0.901738`、L `0.923034/0.097970/0.916599`；VAT 各再训 8 轮后仍保留旧 best。GF-B 已验收，AUC/Avg/Min L2 `0.956236/0.101550/0.043420`，GF-L 仍训练；统一效率尚未测量。

四组均复用兼容旧 raw-concat decoder，再以 seed0 重学 Router，不直接继承 seed3106 support。VAT B/L 初始化已存在，不需要等本轮 GF。官方 GazeLLE 后续仅推理比较；旧 matched dense 保留原协议。

## 工作约定

新代码只放 `code/{data_prep,stag,scripts}/`，旧 `gazelle/` 只 import 不改。论文材料放 `latex/`，关键小 JSON/MD 放 `results/`，大型 PT/NPY 不进 Git。只在本机写代码；服务器训练和评测由用户手动运行，代码和正式小报告通过 Git 传递。

用户已授权本机工作完成后由助手自动创建本地 Git commit，仅提交本任务明确文件；不自动 push 或启动服务器实验。
