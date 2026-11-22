# CSCWD Router：当前实验清单（GF/VAT × ViT-B/ViT-L，seed 0）

更新：2026-11-23，E2。按用户最新要求：两个数据集、两个模型规模，**全部新训练 base seed=0**；只新增 Router 模型训练，不重训 dense。官方 GazeLLE checkpoint 仅用于推理对照。

E0 已实现，32 项 CPU 工程检查通过。完整服务器命令见 [RUN_DINOV3_TRAIN_TEST.md](RUN_DINOV3_TRAIN_TEST.md)。VAT-B/L 已完成并经 Git 回传（`0fc2844`），两组各 3+5 epochs，seed 0；逐帧 AUC/L2/AP 为 B `0.913548/0.110971/0.901738`、L `0.923034/0.097970/0.916599`。工件/指标核验通过，见 [E2 审查](results/E2_VAT_SEED0_REVIEW.md)。GF 尚待审查，尚无新的 GPU 效率成绩。

## 1. 四组主实验与并行顺序

| GPU | 顺序 | 数据集 / 模型 | 新增阶段 | 物理 batch / decoder 累积 | 路由点（0-based） |
|---|---|---|---|---|---|
| 0 | 1 | GazeFollow / ViT-B | support 3 + sparse decoder 5 | 16 / 2 | block 5 |
| 0 | 2 | GazeFollow / ViT-L | support 3 + sparse decoder 5 | 4 / 8 | block 11 |
| 1 | 1 | VAT / ViT-B | support 3 + sparse decoder/inout 5 | 16 / 2 | block 5 |
| 1 | 2 | VAT / ViT-L | support 3 + sparse decoder/inout 5 | 4 / 8 | block 11 |

VAT 的 B/L 都已有可用 decoder，因此不依赖本轮 GF 产物。先并行 GF-B 与 VAT-B，再由两个独立队列各自接 L；某个数据集 B 失败就暂停它自己的队列，另一队列不受影响。

四组共 32 个新增阶段 epoch；support 与 decoder 成本不同，不将这个计数换算成未经测量的 GPU 时长。support 梯度累积为 1，decoder 有效 batch 均为 32。训练和精度评测用 FP32。

不新增 dense base、matched dense continuation、多 seed；两种规模只用既有同规模 decoder 初始化。K50 固定主点，K100→K75→K50 是一次训练中的 curriculum。

## 2. 初始化、结构与数据

四份初始化均已在服务器只读核对存在、SHA、任务 tensor key/shape，均为纯四层 raw-concat，不含 SASA/GGSF/Router。GF 的 B/L 各 40 个 decoder tensors，VAT 的 B/L 各 45 个，缺失/多余/形状不符为 0。具体路径和 SHA 固定在 `code/stag/dinov3_initializers.py`，来源检查见 `results/e0_dinov3_train_test_checks.json`。

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
