# CSCWD Router：当前实验清单（GF/VAT × ViT-B/ViT-L，seed 0）

更新：2026-11-24，E3 / E2-TL。按用户最新要求：两个数据集、两个模型规模，**全部新训练 base seed=0**；只新增 Router 模型训练，不重训 dense。官方 GazeLLE checkpoint 仅用于推理对照。

2026-11-24 E3/E2-TL 入口已实现，68/68 CPU 通过，完整命令见 [RUN_EFFICIENCY_VATL.md](RUN_EFFICIENCY_VATL.md)。GPU0 顺序测 GF-B/GF-L，GPU1 测已接受的 GF→VAT-B，GPU2 完整 GF-L→VAT-L；用户手动运行，尚无正式速度或新 VAT-L 指标。显式支持 GF-L 的用户收口来源，保持真实 3+4，不改 completed 标记。

E0 已实现，32 项 CPU 工程检查通过。完整历史命令见 [RUN_DINOV3_TRAIN_TEST.md](RUN_DINOV3_TRAIN_TEST.md)。VAT-B/L 初轮及续训、GF-B 均已回传核验。GF-L 已按用户决定在实际 support 3 + decoder 4 轮收口（最后一轮关机中断，不补跑），以 `459651a` 回传；按 Min L2 选第 2 轮，AUC/Avg/Min L2 `0.960007/0.094061/0.037721`。详见 [E1-L 审查](results/E1L_GF_INTERRUPTED_REVIEW.md)。尚无新的 GPU 效率成绩。

## 1. 四组主实验与并行顺序

E2-T 已完成并通过 `170ace8` 回传：完整 GF-B Router/heatmap decoder→VAT-B，最终逐帧 AUC/L2/AP `0.940033/0.100131/0.883183`，用户接受，E2-H 分类头适配后移。GF-L 收口精度可用，当前接续同硬件效率与完整 GF-L Router→VAT-L，不先新增 GF-L dense。详见 [E2-T 审查](results/E2T_GF_TO_VAT_REVIEW.md)，[原运行命令](RUN_GF_ROUTER_TO_VAT.md)保留供复现。

等待期 [旧快照](results/gfl_interim_20261123.json) 保留；当前以 [interrupted_summary.json](results/testselected_k050_seed0/gazefollow/vitl/interrupted_summary.json) 为准。原 3+5 manifest 不变，报告明确 completed=false、user-closed、实际 3+4；指标来自选中完整 epoch 的 test，无独立复评或逐查询导出。已从 last 内嵌 best 导出标准 `best_test.pt`；新入口使用 --allow-closed-gf 显式接受该收口来源，核对 history/last/best 后继续，不修改 completed 标记。

2026-11-23 E2-C 历史结果：原始 VAT dense 与 routed K100 精度差为 0；两组低 LR 固定 K50 各加 8 epochs 未超过旧 best，不继续原样加轮数。完整 GF-B Router→VAT 此后已恢复定位，GF-L 收口后接效率与完整 VAT-L 迁移。详见 [历史审查](results/E1_GFB_E2C_REVIEW.md)；[续训命令](RUN_VAT_CONTINUE.md)仅保留供复现。

| GPU | 顺序 | 数据集 / 模型 | 新增阶段 | 物理 batch / decoder 累积 | 路由点（0-based） |
|---|---|---|---|---|---|
| 0 | 1 | GazeFollow / ViT-B | support 3 + sparse decoder 5 | 16 / 2 | block 5 |
| 0 | 2 | GazeFollow / ViT-L | 计划 3+5，实际 support 3 + sparse decoder 4 后收口 | 4 / 8 | block 11 |
| 1 | 1 | VAT / ViT-B | support 3 + sparse decoder/inout 5 | 16 / 2 | block 5 |
| 1 | 2 | VAT / ViT-L | support 3 + sparse decoder/inout 5 | 4 / 8 | block 11 |

VAT 的 B/L 都已有可用 decoder，因此不依赖本轮 GF 产物。先并行 GF-B 与 VAT-B，再由两个独立队列各自接 L；某个数据集 B 失败就暂停它自己的队列，另一队列不受影响。

四组原计划共 32 个新增阶段 epoch，实际已完整提交 31 个（GF-L 少最后一轮）；不把未保存的第 5 轮局部状态计为完整 epoch。support 与 decoder 成本不同，不将这个计数换算成未经测量的 GPU 时长。support 梯度累积为 1，decoder 有效 batch 均为 32。训练和精度评测用 FP32。

不新增 dense base、matched dense continuation、多 seed；两种规模只用既有同规模 decoder 初始化。K50 固定主点，K100→K75→K50 是一次训练中的 curriculum。

## 2. 初始化、结构与数据

四份初始化均已在服务器只读核对存在、SHA、任务 tensor key/shape，不含 SASA/GGSF/Router。GF 的 B/L 各 40 个 decoder tensors，VAT 的 B/L 各 45 个，缺失/多余/形状不符为 0。具体路径和 SHA 固定在 `code/stag/dinov3_initializers.py`，兼容性检查见 `results/e0_dinov3_train_test_checks.json`。

最新 L 语义补充见 [L_INITIALIZER_DECISION.md](results/L_INITIALIZER_DECISION.md)：首末轮投影均为 4096 通道，与四路 1024 拼接兼容，不能据此再推断精确历史层号；GF 记录提交与实物矛盾，VAT 记录源码 `[4,11,17,23]` 与当前 `[5,11,17,23]` 不同，runtime 源码未封存。L 暂作 warm-start 补充，不称为历史同配置已认证；不新开两个 L dense，B 完整 GF Router→VAT 已恢复定位，继续优先。

本轮不直接复用 seed3106 的已训练 Router；B 和 L 都重新初始化并训练 seed0 Router。历史 decoder 是预训练初始化，不把其原始训练 seed 改写为 0。

- 编码器：冻结 DINOv3 ViT-B/16 或 ViT-L/16，512×512，32×32 patch 网格。
- B 保留 blocks 2/5/8/11 的四层表示；L 保留 blocks 5/11/17/23。
- Router 在 B block 5 / L block 11 后预测 support，之后只执行选中 patch；未选位置用路由点激活回填。
- `raw_concat`，关闭 SASA/GGSF；仅 RGB、真实头框、单帧，无检测框、深度或时序模块。
- GF：`/newhome/fb/dataset/gazefollow_extended`；VAT：`/newhome/fb/dataset/videoattentiontarget`。直接读取完整 `train_preprocessed.json` 与 `test_preprocessed.json`，没有 holdout 或 validation loader。

## 3. 训练、逐轮 test 与 checkpoint

- base seed 为 0；为使 epoch 恢复可重现，epoch 子 seed 为 `0 + epoch`。所有新模型与阶段均使用这一规则。
- support 3 epochs，只更新 Router、只计算 dense prefix；取末轮，不按另外的 test coverage 选择器挑选 support。
- sparse decoder 5 epochs，Router/backbone 冻结，仅更新 decoder 与 VAT in/out head。训练 keep ratio `[1.0, 0.75, 0.5, 0.5, 0.5]`；逐轮 test 固定 K50。
- support AdamW `lr=1e-3`，decoder `lr=1e-4`，VAT in/out `lr=1e-3`，`weight_decay=0`，clip norm 1，budget weight 0.05。
- GF 按 test Min L2 最小、VAT 按 test L2 最小选一个模型；并列按 AUC 更高，再保留更早 epoch。主表所有指标取这个 checkpoint，标记 `test_selected`。
- VAT 训练与逐轮 test 每 6 帧采样；选中后自动 rate-1 完整 test，不再根据复评反向换 epoch。

`last.pt` 是原子 epoch 提交点，内含当前模型、优化器、scheduler、历史与最佳模型 state；中断后不重复已提交的更新。`best_test.pt` 在阶段完成时从同一份选择记录导出。完成结果再次运行时核对配置与工件 SHA 后跳过，不重复训练。

## 4. E3：完整系统精度与实际效率（只推理，主模型完成后接续）

官方 GF checkpoint：`checkpoints/gazelle_dinov2_vitb14.pt`、`gazelle_dinov2_vitl14.pt`；VAT 对应 `_inout.pt`。均已在服务器目录中确认存在，不重训。

主表按规模明确比较 B 与 B、L 与 L；同时列出文献 reported results。官方 checkpoint 的选择来源与本文 test-selected 模型分别报告。旧 matched dense 保留原协议，不与新的 test-selected 行拼成严格同预算因果消融。

效率测同一张空闲 RTX 3090、FP16、inference mode，关闭 autocast cache；固定同一原始 RGB 与真实头框，保留各模型输入 448/512。必做 B1/B8 正反序，B4 可省。GF/VAT × 两个模型规模 × 两个 batch × 正反序，共 16 个测量单元。

每单元 warmup 50，latency 200、throughput 500，各重复 5 轮；记录每轮 median/p95 latency、images/s、peak memory。计时包含完整 forward（backbone、Router、gather/scatter、融合、decoder），读取/预处理/H2D 在计时外。最终主表准确率用哪个 checkpoint，效率就测哪个。

只测自己的 FPS，或引用不同 GPU 的论文毫秒数，无法得到相对 GazeLLE 的实测加速比。E3 入口在本轮四组训练后接续；E0 没有伪造效率结果。

## 5. E4 与余量实验

### 当前等待期的实验优先级（2026-11-23）

| 顺序 | 工作 | 新训练 | GPU 条件 | 回答的问题 |
|---|---|---|---|---|
| 1 | 已完成 GF-B / VAT-B 的 K25/K50/K75/K100 扫描 | 无 | 有可用显存即可做精度；计时另跑 | 压缩预算改变后，精度与 coverage 如何变化 |
| 2 | GF-B scene-only Router 对照 | support 3 + decoder 5 | 可在允许共用的卡训练，不要求独占 | 人物外观/相对几何条件是否有价值 |
| 3 | VAT 真实同帧多人 union 与逐人 repeat 的精度、选区和案例 | 无 | 推理需显存；速度需空闲卡 | 固定共同预算如何服务多人的不同需求 |
| 4 | 既有逐查询结果按场景、画内/画外、定位误差分组 | 无 | CPU | 改善与失败集中在哪些样本 |

预算扫描只改变同一 checkpoint 的 keep ratio，不将 K100 冒充重新训练的 dense，不用每个预算再选不同 epoch。scene-only 对照从原 E0 的 GF-B dense 初始化开始，保持冻结 backbone、输入、seed 0、数据增强、学习率、3+5 更新预算及 test-selected 规则一致，去掉 head appearance 与 bbox-relative geometry；与现有 full-conditioned GF-B 比较整行结果。输入临时置零或换错头框只能作敏感性诊断，不能代替重训对照。

GF official test 的 4,782 张图均为单头查询，多人实验应使用 VAT test 中同一真实帧的多头标注，不从 train 挑样本充当 test 或伪造头框。union 与 repeat 都报告质量，固定 K 下不保证等价。现有官方 GF 精度报告没有逐查询预测，做配对统计前需补齐同查询 baseline 输出。

DINOv3 正式 benchmark 已实现，使用 `benchmark_router_dinov3.py` 与 `run_router_efficiency.py`；原 DINOv2 入口仅复用其计时器，不直接加载 DINOv3 权重。独立预算评测和 scene-only 入口接续，服务器仍由用户手动运行。

使用已训练模型，在真实多人帧上按同时查询 1/2/3 人分层，比较 GazeLLE 的共享 dense 与 Router 的共享 sparse union；per-person repeat 仅作额外控制。导出真实 support、选中 patch、GT 与热图，保留定位退化样例。它们均不新增训练。

同一 checkpoint 的 K75/K100 推理可作预算响应与同路径控制，不冒充各预算重训的最优 Pareto。若还有一个训练对照的余量，优先 GF 的 scene-only Router（3+5），不扩展新的 dense 训练。未经该控制，不将完整系统相对 GazeLLE 的全部差异归因于人物条件。

## 6. 公开文献参照（不是本轮结果）

| 方法 | GF AUC ↑ | GF Avg L2 ↓ | GF Min L2 ↓ | VAT AUC ↑ | VAT L2 ↓ | VAT AP ↑ |
|---|---:|---:|---:|---:|---:|---:|
| GazeLLE ViT-B | 0.956 | 0.104 | 0.045 | 0.933 | 0.107 | 0.897 |
| GazeLLE ViT-L | 0.958 | 0.099 | 0.041 | 0.937 | 0.103 | 0.903 |
| Multi-scale Object-Aware Gaze Estimation via Geometric Reasoning | 0.961 | 0.094 | 0.038 | 0.948 | 0.095 | 0.923 |

来源：[GazeLLE 原论文](https://arxiv.org/html/2412.09586v2)、[Mi et al. Table 1，PDF 第 10 页](https://arxiv.org/pdf/2606.29334)。Mi et al. 使用 DINOv3 ViT-L/16、RTX 5090，并离线使用 YOLO11x/SAM2 产生 object masks，比较时标明这些差异。

达到或超过 GazeLLE 的精度与效率是待验证目标，超过 2026 方法作为进取目标；不预写双优，不用其可训练参数量推断同硬件推理速度。旧 VAT holdout 负结果和 GF fixed-final 数字保留原标签。

## 7. 实现与输出

- 单模型入口：`code/scripts/train_router_dinov3.py`。
- 单 GPU 的 B→L 队列：`code/scripts/run_router_dinov3_queue.py`。
- 新代码仅在 `cscwd_router/code/`；旧 `gazelle/` 与历史训练脚本只 import 不改。
- 报告根：`cscwd_router/results/testselected_k050_seed0/<dataset>/vit<b|l>/`。
- 大工件根：`experiments/cscwd_router/testselected_k050_seed0/<dataset>/vit<b|l>/`，PT 与压缩逐查询明细只登记路径/SHA。
- 本机授权改动由助手自动本地 commit；push 与服务器启动仍分别执行。具体两条完整 nohup 命令、日志和恢复方式见运行说明。
