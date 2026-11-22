# CSCWD Router · 当前方案、实现规格与进展

> 本文档是给 codex 的实现规格 + 进展记录。codex 开工前读，每完成一个任务写回"进展记录"。
> 2026-11-22 当前论文主方法：DINOv3 多层表示 + gaze-conditioned spatial routing。行文与贡献边界见 `latex/PAPER_STORY.md`；三稿复核见 `latex/PAPER_REVIEW.md`。
> 当前 E0 已实现：GF/VAT × ViT-B/ViT-L，新训练 seed 0，32 项 CPU 检查通过。GPU 0/1 完整命令见 `RUN_DINOV3_TRAIN_TEST.md`。
> 2026-11-23 E2：VAT-B/L 队列完成，结果提交 `0fc2844` 已回传核验。逐帧 AUC/L2/AP 为 B `0.913548/0.110971/0.901738`、L `0.923034/0.097970/0.916599`；详见 `results/E2_VAT_SEED0_REVIEW.md`。GF 尚待审查，效率尚待测量。
> E2-C 已实现低学习率固定 K50 续训，39 项 CPU 检查通过；计划 B/L 各加 8 epochs，命令见 `RUN_VAT_CONTINUE.md`，尚未运行。
> 原 T1–T6 是 DINOv2 单层适配的工程任务，保留原实现、验收和运行记录。以下历史规格不自动变为新的 DINOv3 训练命令。

## 0. 当前决定（2026-11-23，E2）

当前方法覆盖冻结 DINOv3 ViT-B/16 和 ViT-L/16 多层表示上的人物条件化深层计算分配。DINOv3 与多层访问沿用，不再次作为相对在审 ICASSP 的新贡献；不新增 dense/matched dense 训练。旧 matched dense、GF fixed-final 与 VAT holdout 结果保留各自协议，不与新结果混拼。

按用户最新要求，全部新训练 base seed=0，epoch 子 seed 为 `0 + epoch`。四组从各自兼容的旧 raw-concat decoder 初始化，重新学习 support 3 epochs，再训练 sparse decoder/inout 5 epochs；不继承 seed3106 的已训练 Router。GF 按 test Min L2、VAT 按 test L2 选单一 checkpoint，每轮 test 用 K50，VAT 最后自动逐帧评测。VAT 两组已完成并回传核验，GF 尚待审查。

VAT 的 B/L 初始化都已核对存在，因此不用等本轮 GF。GPU 0 顺序 GF-B→GF-L，GPU 1 顺序 VAT-B→VAT-L；详细配置与完整 nohup 命令见 `EXPERIMENT_PLAN.md`、`RUN_DINOV3_TRAIN_TEST.md`。官方 GazeLLE 后续仅做推理比较；旧 DINOv2 T4 按原阈值单独收口。

GazeLLE 原生已共享多人场景编码；新多人主张是共享稀疏 suffix。MM 图 2 是 last-layer PCA，保留素材但当前不直接作为 Router 主动机图。已有 GF K50 相对自身 matched dense AUC 差 `-0.000802`，旧 B8 `1.148545x` 使用不同 split-clean checkpoint，不能据此预写相对官方 GazeLLE 的速度优势。VAT 旧负结果保留。

## 0.1 DINOv2 适配实验的原目标（历史）

在官方 DINOv2 单层 GazeLLE 上适配既有 Router，并验证 K100 等价、真实稀疏执行、训练闭环与精度/效率。原目标和原 gate 保留在下文，用于解释 T1–T4 工件。

## 1. DINOv2 实验底座（历史，复用不动）

- 代码：`gazelle/backbone_v0_dinov2.py`（`DinoV2Backbone`）、`gazelle/model_v0_dinov2.py`（`GazeLLE`）。
- DINOv2 ViT-B/14，输入 448×448，单层特征 32×32（768 维）。
- GazeLLE 官方 GazeFollow 精度：ViT-B AUC 0.956。
- backbone 权重：服务器 `checkpoints/dinov2_vitb14_pretrain.pth`。
- GazeLLE decoder checkpoint（GazeFollow 训练）：用户于 2026-11-21 确认主线使用服务器 `/home/fb/src/paper/gazelleV1/checkpoints/gazelle_dinov2_vitb14.pt`。同目录另有 ViT-B inout、ViT-L 和 ViT-L inout 版本；本轮使用 ViT-B、非 inout。

## 2. DINOv2 实验方法（历史适配规格）

复用 `gazelle/routing/` 的现有 Router（DINOv3 版），把 backbone 从 DINOv3 换成 DINOv2：

```
输入图像（448×448）
  → DINOv2 blocks 0–5（全部 patch，dense）
  → Router：scene feature + head appearance + bbox-relative geometry
        → 每人 free-form support → 同图多人 max-union → Top-K
  → DINOv2 blocks 6–11：只处理 K 个 patch + special tokens
  → scatter 回 32×32，未选位置用 block-5 浅层特征回填（early-exit fill）
  → 原 GazeLLE decoder（linear + pos_embed + head_token + 3 transformer + heads）
```

### DINOv3 → DINOv2 的关键工程差异

1. **patch 14 vs 16**：featmap 都是 32×32，但输入 448（DINOv2）vs 512（DINOv3）。
2. **位置编码**：DINOv2 用**可学习 position embedding**（不是 DINOv3 的 RoPE）。由官方 `prepare_tokens_with_masks` 在 block 0 前完成插值并加入一次；稀疏前向 gather 已带原始位置信息的 block-5 token，suffix 不再重复添加 position embedding。
3. **block 数**：DINOv2 ViT-B = 12 blocks（0–11），route_after_block=5，后段 blocks 6–11 稀疏，与 DINOv3 ViT-B 结构对应。
4. **forward 拆分**：`DinoV2Backbone.forward` 现在是 `forward_features(x)['x_norm_patchtokens']` 一次性前向，要改成 prefix/suffix 拆分（前段 dense 出 block-5 特征做 support，后段只算 K 个 token）。
5. **early-exit fill**：未选 token 用 block-5 浅层特征回填，保持 32×32 完整网格进 decoder。

## 3. DINOv2 工程任务清单（原 T1–T6，按依赖顺序）

### T1 DINOv2 稀疏前向适配（backbone 层）
- 新增 `cscwd_router/code/stag/backbone_sparse_dinov2.py`（不改 `gazelle/backbone_v0_dinov2.py`）。
- 实现：prefix（blocks 0–5，dense）→ 输出 block-5 特征；suffix（blocks 6–11，只算 K 个已带位置信息的 patch + special tokens）；scatter 回 32×32（未选位置 early-exit fill）。
- **验收**：K=全部（不剪）时，输出与 GazeLLE 官方 `forward_features` 逐值一致（K100 等价）；合成测试通过。

### T2 Router 适配（gaze-conditioned support）
- 复用 `gazelle/routing/router.py` 的 support 预测、max-union、Top-K，改输入为 DINOv2 block-5 特征。
- Router 输入：scene feature（block-5）+ head appearance + bbox-relative geometry → 每人 free-form support。
- **验收**：合成测试；support 在合成数据上可训练（coverage loss 下降）。

### T3 训练 recipe + 评测
- staged：GazeLLE 官方 dense checkpoint（冻结）→ support 3ep 只训 Router → K100→K75→K50 5ep 只训 decoder → matched dense 5ep（同预算公平对照）。
- 评测：GazeFollow 标准 AUC / Avg L2 / Min L2（沿用 gazelle 协议）。
- **验收**：合成数据上训练/评测/checkpoint 读写走通；零初始化时输出=官方底座。

### T4 主实验（服务器）
- GazeLLE dense（官方）vs GazeLLE + Router K50，同一 test、同一评测。
- 效率矩阵：RTX 3090，B1/B4/B8 正反序 latency/throughput，K100 control（同路径不删 token）。
- **验收**：pass/fail 阈值——AUC 掉 ≤0.002 且 B8 加速 ≥1.10x，否则判死。

### T5 多人 union（核心贡献，必补）
- 一张图多人：per-image max-union 一次前向 vs per-person 重复，实测 latency/throughput。
- split-clean multi-head 子集分层精度（single/two/three-plus）。
- **验收**：union 相对 per-person repeat 有实测加速；多人精度不崩。

### T6 消融
- learned vs untrained Router（因果）。
- 预算 Pareto（K25/K50/K75）。
- gaze-conditioned vs 通用 saliency 剪枝（证明 gaze 条件化的价值）。

## 4. 硬约束

- 纯 RGB、真实头框，不进深度/未来帧/检测框。
- 严格因果（routing 只用当前帧信息；本任务无跨帧时序，故无时序因果问题）。
- backbone 保持冻结。官方 GazeLLE checkpoint 用于 DINOv2 基线及已有适配实验；DINOv3 多层分支使用自己的 dense decoder 初始化，分别登记工件，不声称直接加载相同官方 decoder。
- 所有数字只写真实跑过的；不编数字、不编对照。
- 新代码全放 `cscwd_router/code/`，旧 `gazelle` 包只 import 不改。

## 5. 数据、路径与工作流

- 数据：`fb@3090.lab:/newhome/fb/dataset/gazefollow_extended`（GazeFollow train/test）。
- 服务器代码仓库：`/home/fb/src/paper/gazelleV1`。
- Python：`/home/fb/anaconda3/envs/py310/bin/python`。
- git 双向：本地写码 → push → 服务器 pull → nohup 手动跑 → 小结果（JSON/MD）git 回传；大文件（.npy/.pt）不进 git，只在 RESULT_PROVENANCE.md 记路径+SHA。
- 2026-11-22 用户更新：本机完成的授权文件由助手自动创建本地 Git commit，明确 stage 范围；push 与服务器执行仍分别处理，不自动启动训练。
- nohup 模板：`CUDA_VISIBLE_DEVICES=<gpu> setsid nohup <python> -u <script> --args > <run>.log 2>&1 &`。

## 6. 进展记录（codex 逐项写回）

> 每完成一个任务，记：任务号 / 完成内容 / 产出文件与数字 / 下一步。不写"风险清单"，只写推进状态。

### 2026-11-21 · T1：实现与本机合成验收完成

- **完成内容**：新增 `code/stag/backbone_sparse_dinov2.py`，继承官方 `DinoV2Backbone`，支持复用已加载的 DINOv2 模型。prefix 默认 blocks 0–5；按图 gather 选中 patch，保留 special tokens；suffix blocks 6–11；scatter 回完整网格，未选位置保留 block-5 原始激活，再按官方顺序对完整 token 序列做末尾 LayerNorm。K100 经过同一条 gather/suffix/scatter 路径，无 dense shortcut。旧 `gazelle/` 未修改。
- **产出文件**：上述 backbone、`code/stag/test_backbone_sparse_dinov2.py`、服务器入口 `code/scripts/check_sparse_dinov2.py`；小结果见 `results/t1_local_checks.json`、`results/LOCAL_CHECKS.md`；来源见 `results/RESULT_PROVENANCE.md`。新增目录级 `.gitignore`，允许小 JSON 入 Git，排除 PT/PTH/NPY 等大型工件。
- **产出数字**：本机 CPU 合成检查 **9/9 通过**；使用官方 DINOv2 源码的随机小模型（24 维、12 blocks、patch 14）。448×448 输入下，K100 的 `[1, 1024, 24]` patch 输出与官方 `forward_features` 逐值一致：`max_abs_diff=0.0`、`mismatched_values=0`。另外覆盖矩形输入、位置插值、不同 route block、K50 suffix 长度、early-exit fill、梯度、state-dict 读写及原版 decoder 的多人查询。该记录是合成工程检查；本机未读取真实数据或官方 checkpoint。
- **下一步**：该阶段之后已完成服务器验收并进入 T2，见下面两条记录。

### 2026-11-21 · T1：服务器验收通过

- **完成内容**：按用户要求 pull 服务器结果提交 `32224be`；核对报告与 provenance 的 SHA256、五个相关源码文件的 SHA256、权重路径与 SHA、六组逐值比较及稀疏 suffix 长度。
- **产出文件与数字**：`results/t1_dinov2_checkpoint_checks.json`、审查记录 `results/T1_REVIEW.md`。实际 GPU 1、RTX 3090；单测 9/9 通过；B1/B2/B8 × FP32/FP16 共 6 组，K100 backbone 与原版 decoder 的 `max_abs_diff=0.0`、不一致元素 0；K50 suffix 全部为 513 tokens，未选位置回填逐值一致。
- **下一步**：T1 完整验收通过，进入 T2。

### 2026-11-21 · T2：Router 接入与本机检查完成

- **完成内容**：新增 `code/stag/router_dinov2.py`，将现有 `CoverageAwareSpatialRouter` 原样接到 DINOv2 block-5 规范化特征，保留 scene/head appearance/bbox-relative geometry、每人 support、每图 max-union、固定预算 Top-K。backbone 全冻结并保持 eval；support-only 只跑 prefix，完整路由每图共享一次 suffix。复用原 `coverage_router_loss`，旧 `gazelle/` 未修改。
- **产出文件**：上述 Router 适配器、`code/data_prep/synthetic_support.py`、`code/stag/test_router_dinov2.py`、服务器入口 `code/scripts/check_router_dinov2.py`；结果见 `results/t2_local_checks.json` 和 `results/LOCAL_CHECKS.md`。
- **产出数字**：本机 T1 回归 9 项 + T2 新增 11 项，**20/20 通过**；随机小模型的 K25/K50/K100 检查全部通过，4×4 网格分别保留 4/8/16 patches；两图三人 K50 前六层输入 `[2, 17]`、后六层 `[2, 9]`，验证了按图共享前向。coverage loss 的有限梯度和三条输入分支的梯度已检查；本机未执行 support 拟合，没有记录 loss 下降数值。
- **下一步**：该阶段之后服务器合成拟合已通过，见下一条 T2 验收记录。

### 2026-11-21 · T2：服务器验收通过

- **完成内容**：pull 服务器结果提交 `a08e944`；核对报告 SHA、八个相关源码 SHA、与 T1 相同的 backbone 权重 SHA、完整 81 条 history、冻结状态和三种预算的执行路径。
- **产出文件与数字**：`results/t2_dinov2_support_checks.json`、`results/T2_REVIEW.md`。RTX 3090、GPU 1、FP32，80 步 Router-only 拟合的 coverage loss 为 `3.465735674 → 0.000011941`，soft/hard coverage 为 `0.999988079 / 0.961666822`；backbone 可训练参数与带梯度参数数均为 0。K25/K50/K100 分别保留 256/512/1024 patches，所有结构检查通过；单测 20/20 通过。
- **下一步**：T2 完整验收通过，进入 T3。

### 2026-11-21 · T3：训练/评测闭环实现与本机检查完成

- **完成内容**：新增 `code/stag/model_dinov2.py`，以 feature-grid adapter 直接调用原版 `GazeLLE.forward`，严格加载官方 decoder。新增 `engine.py`、`checkpoint.py`、`pipeline.py` 与 `code/data_prep/gazefollow.py`，实现固定 support 3ep、K100→K75→K50→K50→K50 decoder 5ep、matched dense 5ep；两条 decoder 分支重置到同一官方权重，使用相同 epoch seed、样本与更新预算。支持按 epoch 恢复、图像分组 holdout、full-train/no-eval、原版 AUC/Avg L2/Min L2 和独立 fixed-final 评测；backbone 始终冻结。compact checkpoint 只保存 decoder/Router，外部 DINOv2 路径和 SHA 随文件登记。
- **产出文件**：上述模型/engine/checkpoint/pipeline/data、`code/scripts/train_router_dinov2.py`、`code/scripts/eval_router_dinov2.py`、`code/stag/test_training_dinov2.py`；小结果 `results/t3_local_checks.json`，说明见 `results/LOCAL_CHECKS.md`。
- **产出数字**：本机 T1/T2/T3 共 **34/34 通过**，新增 T3 14 项；随机小模型 dense/零初始化 K100 与原版 GazeLLE 的最大差值 0.0、不一致元素 0。阶段梯度、checkpoint 往返、多标注指标、图像分组和恢复调度已检查；本机优化器更新数为 0，调度单测的参数扰动仅发生于临时测试目录。
- **下一步**：该阶段之后服务器 synthetic 训练与独立评测已通过，见下一条 T3 验收记录。

### 2026-11-21 · T3：服务器验收通过

- **完成内容**：pull 结果提交 `de1488d`，核对 summary/独立评测 SHA、18 个源码 SHA、逐阶段 history、权重变化范围、七个 checkpoint 的路径/SHA 登记与独立评测一致性。
- **产出文件与数字**：`results/t3_synthetic_seed3106/`、`results/T3_REVIEW.md`。RTX 3090、FP32，实际 support/sparse/matched 更新 3/5/5 次、逐人样本 9/15/15；两条 decoder 分支的起点、epoch seed 和预算相同。初始 dense/K100 与原版 GazeLLE、两个 checkpoint 往返的最大差值均为 0.0；独立 sparse metrics 与总报告逐值一致。34/34 单测和十项 engineering checks 全部通过。
- **下一步**：T3 完整工程验收通过，进入 T4。

### 2026-11-21 · T4：主实验入口、效率矩阵与判定实现完成

- **完成内容**：新增 `code/scripts/run_main_experiment.py`，顺序执行固定 full-train/no-eval 3+5+5、三个 final checkpoint 的一次 official test、B1/B4/B8 正反序共六个新进程 benchmark。新增 `benchmark_router_dinov2.py`，比较原版 dense、同路径不剪 token 的 K100 与 K50；输入为 train 中真实 RGB 和一个 in-frame GT 头框，数据加载/预处理/传输在计时外，统一 inference mode、FP16、关闭 autocast cache。新增 `code/stag/main_results.py`，按预声明公式汇总并输出 pass/stop；恢复核对并复用已完成 official test。
- **产出文件与数字**：上述三个文件、`code/stag/test_main_experiment.py`、`results/t4_local_checks.json`、本机检查说明；此前 34 项 + 新增 10 项，**44/44 通过**。随机小模型 448 输入的 dense/K100/K50 执行图实际保留 1024/1024/512 patches，dense/K100 输出逐值一致。未产生真实 T4 精度、时延或加速数字。
- **下一步**：用户按第 13 节在服务器手动启动完整 T4，回传小报告和 provenance。按 official dense AUC 掉 ≤0.002、B8 latency speedup ≥1.10x 的既定门槛决定继续 T5 或停止本方案。

### 2026-11-22 · W1：三稿复核、idea 与行文逻辑落盘

- **完成内容**：完整读取被拒 MM、rebuttal 与在审 ICASSP，共 15 页；目视核对 MM 图 2 及候选源素材。将论文主方法定位为 DINOv3 多层表示上的人物条件化深层计算分配，明确多层/软门控/共享场景编码的复用边界。写入五段中英文 Introduction 工作稿、方法执行图、贡献、图表和对照逻辑；记录新 full-train/test-selected 口径与现有脚本的区别。
- **产出文件与数字**：`latex/PAPER_STORY.md`、`latex/PAPER_REVIEW.md`、`results/WRITING_EVIDENCE_20261122.json`，同步 README、后续会话提示词、provenance 和 `PROGRESS.md`。核对并摘录 19 份已有实验 JSON 的真实指标与 SHA；旧 GF K50 AUC 为 `0.956462`，相对自身 matched dense 差 `-0.000802`；旧 B8 latency speedup 为 `1.148545x`，其 checkpoint 与 GF 正式准确率 checkpoint 不同。VAT K50/K75 的 holdout 退化保留。未执行新的训练、推理或测速。
- **下一步**：先完成统一跨模型测速入口与同 checkpoint 的测量，再实现本轮选定的全量 train/test-selected 协议。既有 DINOv2 T4 报告另行经 Git 回传收口；多人和消融逐项推进。

### 2026-11-22 · W2：限时实验清单与自动本地提交

- **完成内容**：按用户要求自动创建 W1 本地提交 `2a3bcaf`，仅包含 8 个写作/证据文件，保留已有目录迁移。将下一轮改为不新增 dense/matched dense 训练的 E0–E4 清单；固定 full train/test、逐 epoch test 与单 checkpoint 选模。核对可复用 GF support/VAT 初始化和 Mi et al. 原论文的指标、模型规模及辅助处理。
- **产出文件与数字**：新增 `EXPERIMENT_PLAN.md`，同步当前行文、README、会话提示词、provenance 和 `PROGRESS.md`。计划训练预算是 GF 5、VAT 3+5 epochs（预算，不是已跑结果）；其他核心比较仅推理。Mi et al. 的公开 GF/VAT AUC `0.961 / 0.948` 作为文献目标，不登记为本文成绩。本轮没有新训练或 GPU 评测。
- **下一步**：完成 E0 的最小 DINOv3 train/test-selected 入口与本机工程检查，给出用户手动运行 E1/E2 的完整 nohup 命令；随后对实际选中模型执行 E3/E4。

### 2026-11-22 · E0：双尺寸 train/test 入口与并行队列完成

- **完成内容**：新增 full train/test 数据入口、prefix-only support、稀疏 decoder/inout 更新、单 checkpoint test 选模、原子 epoch 恢复、VAT rate-1 最终评测和逐查询压缩导出。复用现有 DINOv3 Router 与指标，旧 `gazelle/`、历史脚本未改。新增单 GPU B→L 队列；用户指定新 seed 0 后，四组均重学 Router，不新增 dense。
- **产出文件与数字**：7 个新 Python 文件（data_prep 1、stag 4、scripts 2），`RUN_DINOV3_TRAIN_TEST.md`、`results/e0_dinov3_train_test_checks.json`。CPU 32/32 通过，覆盖 B/L factory、梯度冻结、指标一致性、4 种合成闭环与中断恢复；服务器只读确认 GF-B/GF-L/VAT-B/VAT-L 匹配 40/40/45/45 个 tensors，缺失、多余、shape mismatch 均为 0，并核对两份骨干及四份标注 SHA。未进行新真实数据训练或 GPU 测速。
- **下一步**：用户 push/pull 后手动启动 GPU0 GF-B→L 与 GPU1 VAT-B→L；回传小报告后对实际选中模型推进 E3 效率和 E4 多人分析。

### 2026-11-23 · E2：VAT B/L 完成、Git 回传与结果审查

- **完成内容**：检查 GPU 1 的 B→L 队列已完成；按用户指示在服务器仅提交 8 份 VAT JSON 与对应 provenance，push `0fc2844` 后本机 pull。只读核对 checkpoint、源码、数据标注、逐查询身份/指标与单 checkpoint 选模，未重新运行模型或启动训练。
- **产出文件与数字**：`results/testselected_k050_seed0/vat/{vitb,vitl}/`、`results/e2_vat_seed0_server_checks.json`、`results/E2_VAT_SEED0_REVIEW.md`。实际各 3+5 epochs，seed 0；B/L 按 rate-6 test L2 选第 4/2 轮，rate-1 test 各 31,978 人次、20,655 画内，最终 AUC/L2/AP 为 B `0.913548/0.110971/0.901738`、L `0.923034/0.097970/0.916599`。14 个大工件 SHA 全匹配，8 份报告本地 SHA 一致；两组逐查询重聚合指标差值均 0。两组 AUC 均低于同规模 GazeLLE 文献参照，尚无正式效率结果。
- **下一步**：收齐并审查 GF，然后对实际选中模型推进 E3 官方 GazeLLE 对照推理/测速与 E4 多人分析；可用现有 VAT checkpoint 补 K75/K100 推理，不新增 dense 训练。

### 2026-11-23 · E2-C：VAT 迁移来源追溯与低学习率续训

- **完成内容**：从精确 W&B config/output 核实 B/L 都是 GF dense→VAT dense 8 epochs；本轮在此基础重学 VAT Router，没有继承 GF Router。核对 MM Table 4 的 B raw-concat `0.937/0.103`；旧 VAT decoder LR `1e-5`，当前首轮 Router 适配 `1e-4`。新增独立固定 K50 续训入口，不改旧 E0 源码、运行目录或 GF 任务；从原第 5 轮末状态使用新低 LR 优化器，保留旧 best，先做原始初始化 native dense/routed K100 推理核对。
- **产出文件与数字**：`code/stag/dinov3_continue.py`、`code/scripts/continue_router_dinov3.py`、`code/stag/test_dinov3_continue.py`，`RUN_VAT_CONTINUE.md`、`results/e2_vat_initialization_trace.json`、`results/e2_vat_continue_local_checks.json`。39/39 CPU 通过（7 项新增、32 项回归），含 B/L 合成闭环、原工件不变、固定预算、旧 best 保留、恢复逐值一致与 probe 不符时停止；两组完整 CLI 和 shell 语法通过。旧末轮日志 rate-6 AUC/L2 为 B `0.939097/0.103012`、L `0.945401/0.093716`，不是当前逐帧复测。计划各加 8 epochs 是预算，尚无续训结果。
- **下一步**：用户 push/pull 后在 GPU 1 手动执行 B→L 续训，回传 probe/history/summary。GF 继续等待；完成后再比较完整 GF Router→VAT 迁移。以真实结果决定后续预算与 E3，不重训 dense。

## 7. T1 服务器运行与结果回传

本节 T1 已执行并通过，实际使用 GPU 1；以下保留复现命令。脚本只读取官方权重并生成合成 RGB 输入/头框，不读取 GazeFollow 数据，不训练，也不产生 AUC/L2 或效率结论。

先在本机提交第 8 节的文件并 push；服务器在本分支 pull 后运行：

```bash
cd /home/fb/src/paper/gazelleV1
git switch codex/coverage-aware-router
git pull --ff-only origin codex/coverage-aware-router
mkdir -p /home/fb/src/paper/gazelleV1/logs/cscwd_router
mkdir -p /home/fb/src/paper/gazelleV1/cscwd_router/results

CUDA_VISIBLE_DEVICES=1 setsid nohup /home/fb/anaconda3/envs/py310/bin/python -u \
  cscwd_router/code/scripts/check_sparse_dinov2.py \
  --decoder-checkpoint /home/fb/src/paper/gazelleV1/checkpoints/gazelle_dinov2_vitb14.pt \
  --device cuda \
  --batch-sizes 1 2 8 \
  --precisions fp32 fp16 \
  --image-size 448 \
  --route-after-block 5 \
  --seed 3106 \
  --output /home/fb/src/paper/gazelleV1/cscwd_router/results/t1_dinov2_checkpoint_checks.json \
  --provenance /home/fb/src/paper/gazelleV1/cscwd_router/results/RESULT_PROVENANCE.md \
  > /home/fb/src/paper/gazelleV1/logs/cscwd_router/t1_dinov2_checkpoint_checks.log 2>&1 &
```

Backbone 仍通过官方 loader 从仓库的 `dinov2/` 和 `checkpoints/dinov2_vitb14_pretrain.pth` 加载。脚本先跑合成单测，再对 B1/B2/B8 × FP32/FP16 分别检查 K100 backbone 与原版 decoder 的逐值一致性，以及 K50 的 suffix token 数和未选位置回填。通过标准为每项 `equal=true`、数值有限、suffix token 数匹配，且顶层 `passed=true`；不使用放宽误差容限。退出码为 0 表示通过，1 表示结果检查未通过；加载错误直接保留原始错误信息。

输出 JSON 包含实际 Python/PyTorch/CUDA/GPU、代码提交、相关源码 SHA256、两份 checkpoint 的路径与 SHA256。脚本将输入 checkpoint 与结果 JSON 的 SHA256 自动追加到 `RESULT_PROVENANCE.md`；不会保存新的大文件。

运行结束后，只将以下两个小文件经 Git 回传：

```bash
git add cscwd_router/results/t1_dinov2_checkpoint_checks.json cscwd_router/results/RESULT_PROVENANCE.md
git commit -m "Record T1 DINOv2 checkpoint equivalence checks"
git push origin codex/coverage-aware-router
```

## 8. T1 本机提交范围

本轮只提交实现、检查与进展文件。已有 `cscwd/` → `cscwd_v1/` 目录迁移保持原样。

```bash
git add \
  cscwd_router/.gitignore \
  cscwd_router/code/stag/backbone_sparse_dinov2.py \
  cscwd_router/code/stag/test_backbone_sparse_dinov2.py \
  cscwd_router/code/scripts/check_sparse_dinov2.py \
  cscwd_router/results/t1_local_checks.json \
  cscwd_router/results/LOCAL_CHECKS.md \
  cscwd_router/results/RESULT_PROVENANCE.md \
  cscwd_router/implementation_plan.md \
  PROGRESS.md
git commit -m "Add DINOv2 sparse backbone and T1 equivalence checks"
git push origin codex/coverage-aware-router
```

## 9. T2 服务器合成拟合与结果回传

本节已执行并通过；以下保留复现命令。脚本仅使用合成 RGB、合成头框与合成目标，缓存一次冻结的官方 DINOv2 block-5 特征，只更新 Router；这是可训练性检查，不是 GazeFollow 实验。

```bash
cd /home/fb/src/paper/gazelleV1
git pull --ff-only origin codex/coverage-aware-router
mkdir -p /home/fb/src/paper/gazelleV1/logs/cscwd_router

CUDA_VISIBLE_DEVICES=1 setsid nohup /home/fb/anaconda3/envs/py310/bin/python -u \
  /home/fb/src/paper/gazelleV1/cscwd_router/code/scripts/check_router_dinov2.py \
  --device cuda \
  --batch-size 2 \
  --image-size 448 \
  --route-after-block 5 \
  --keep-ratio 0.25 \
  --hidden-dim 256 \
  --temperature 1.0 \
  --escape-tokens 8 \
  --steps 80 \
  --learning-rate 0.001 \
  --budget-weight 0.05 \
  --seed 3106 \
  --output /home/fb/src/paper/gazelleV1/cscwd_router/results/t2_dinov2_support_checks.json \
  --provenance /home/fb/src/paper/gazelleV1/cscwd_router/results/RESULT_PROVENANCE.md \
  > /home/fb/src/paper/gazelleV1/logs/cscwd_router/t2_dinov2_support_checks.log 2>&1 &
```

运行内容：T1/T2 合成单测；两图三人的 80-step Router-only support 拟合（FP32）；拟合后 K25/K50/K100 的固定预算、按图 max-union、逐人映射、逐 block 实际 batch/token 数、early-exit fill 与 K100 官方等价检查。记录每一步实际 coverage/total/soft/hard 指标和第 2 步的三条输入分支梯度，同时记录 backbone 可训练参数数与非空梯度数。

T2 通过标准：`fit.coverage_decreased=true`、三条分支梯度连通、backbone 可训练参数与非空梯度数均为 0、全部路由检查通过、无跳过或失败单测，以及顶层 `passed=true`。只要求合成 coverage loss 相比初始值下降，不将这批合成数据上的 coverage 数字作为论文实验。

权重仍从 `/home/fb/src/paper/gazelleV1/checkpoints/dinov2_vitb14_pretrain.pth` 加载，SHA256 在实际运行时计算；本任务不需要 decoder 权重，不保存新的大型工件。JSON 保存实际命令、环境、配置、输入与源码 SHA、完整 history；provenance 自动追加输入和结果的路径与 SHA。

运行结束后回传：

```bash
git add cscwd_router/results/t2_dinov2_support_checks.json cscwd_router/results/RESULT_PROVENANCE.md
git commit -m "Record T2 DINOv2 support trainability checks"
git push origin codex/coverage-aware-router
```

## 10. T2 本机提交范围

```bash
git add \
  cscwd_router/code/stag/router_dinov2.py \
  cscwd_router/code/stag/test_router_dinov2.py \
  cscwd_router/code/data_prep/synthetic_support.py \
  cscwd_router/code/scripts/check_router_dinov2.py \
  cscwd_router/results/T1_REVIEW.md \
  cscwd_router/results/t2_local_checks.json \
  cscwd_router/results/LOCAL_CHECKS.md \
  cscwd_router/results/RESULT_PROVENANCE.md \
  cscwd_router/implementation_plan.md \
  PROGRESS.md
git commit -m "Connect gaze-conditioned Router to frozen DINOv2 features"
git push origin codex/coverage-aware-router
```

## 11. T3 服务器 synthetic 训练、评测与回传

### 11.1 当前运行：固定 3+5+5 smoke

本节已执行并验收通过，以下保留复现命令。该命令使用官方 ViT-B backbone 和官方 decoder，但只生成合成 RGB/头框/目标；每个 epoch 使用同一批两图三人，共执行规格规定的 3+5+5 个单 batch epoch。`--protocol synthetic` 不读取 `--data-path`。

```bash
cd /home/fb/src/paper/gazelleV1
git pull --ff-only origin codex/coverage-aware-router
mkdir -p /home/fb/src/paper/gazelleV1/logs/cscwd_router

CUDA_VISIBLE_DEVICES=1 setsid nohup /home/fb/anaconda3/envs/py310/bin/python -u \
  /home/fb/src/paper/gazelleV1/cscwd_router/code/scripts/train_router_dinov2.py \
  --protocol synthetic \
  --data-path /newhome/fb/dataset/gazefollow_extended \
  --decoder-checkpoint /home/fb/src/paper/gazelleV1/checkpoints/gazelle_dinov2_vitb14.pt \
  --run-dir /home/fb/src/paper/gazelleV1/experiments/cscwd_router/t3_synthetic_seed3106 \
  --report-dir /home/fb/src/paper/gazelleV1/cscwd_router/results/t3_synthetic_seed3106 \
  --provenance /home/fb/src/paper/gazelleV1/cscwd_router/results/RESULT_PROVENANCE.md \
  --device cuda \
  --batch-size 2 \
  --eval-batch-size 2 \
  --workers 0 \
  --router-lr 0.001 \
  --decoder-lr 0.0001 \
  --weight-decay 0 \
  --budget-weight 0.05 \
  --seed 3106 \
  --split-seed 3106 \
  --val-fraction 0 \
  --router-hidden-dim 256 \
  --router-temperature 1.0 \
  --escape-tokens 8 \
  --log-every 1 \
  --no-augment \
  > /home/fb/src/paper/gazelleV1/logs/cscwd_router/t3_synthetic_seed3106.log 2>&1 &
```

FP32、448 输入、route block 5、decoder 256 维/3 层固定。synthetic 模式先执行本机同套结构单测，再进行实际优化。summary 的 `passed` 表示 T3 工程验收：初始 dense/K100 逐值等于官方模型；support 只改变 Router；另外两阶段只改变 decoder；decoder 起点相同，五轮的更新次数、人数预算与 epoch seed 相同；backbone 冻结；稀疏与 matched checkpoint 重新加载后输出逐值一致。实际 loss/history、synthetic AUC/L2 由运行生成，均作为工程结果保存。

输出位置：

- 小报告：`cscwd_router/results/t3_synthetic_seed3106/summary.json`、`manifest.json`、`initial_equivalence.json` 和逐阶段 history/summary。
- 大文件：`experiments/cscwd_router/t3_synthetic_seed3106/{dense_official,support,sparse_decoder,matched_dense}/final.pt`；后三阶段另有 `last.pt`。实际运行后全部登记路径和 SHA。

中断后在同一命令中加 `--resume`，可从已完成 epoch 的 `last.pt` 继续，已完成阶段不重训；日志改为 `t3_synthetic_seed3106_resume.log`。恢复会核对同一源码、权重、分组与训练配置。

### 11.2 训练结束后：独立加载 sparse checkpoint 评测

完成 11.1 后再执行，验证独立脚本从磁盘读取 checkpoint 与外部 backbone 的路径。该命令仍为 synthetic，不读取 official test。

```bash
cd /home/fb/src/paper/gazelleV1
CUDA_VISIBLE_DEVICES=1 setsid nohup /home/fb/anaconda3/envs/py310/bin/python -u \
  /home/fb/src/paper/gazelleV1/cscwd_router/code/scripts/eval_router_dinov2.py \
  --checkpoint /home/fb/src/paper/gazelleV1/experiments/cscwd_router/t3_synthetic_seed3106/sparse_decoder/final.pt \
  --split synthetic \
  --data-path /newhome/fb/dataset/gazefollow_extended \
  --device cuda \
  --batch-size 2 \
  --workers 0 \
  --output /home/fb/src/paper/gazelleV1/cscwd_router/results/t3_synthetic_seed3106/standalone_sparse_eval.json \
  --provenance /home/fb/src/paper/gazelleV1/cscwd_router/results/RESULT_PROVENANCE.md \
  > /home/fb/src/paper/gazelleV1/logs/cscwd_router/t3_standalone_sparse_eval.log 2>&1 &
```

评测直接沿用 `gazelle.utils.gazefollow_auc/gazefollow_l2`：AUC 在原始图像分辨率计算，Avg L2 为预测点到平均标注坐标的距离，Min L2 为到最近标注的距离，最终按人平均。eval loader 按图聚合全部 in-frame 头框。输出文件存在时拒绝覆盖。

真实数据接口已经实现：`--protocol holdout` 对 train 做稳定图像分组，并在各 decoder 阶段末尾读取 holdout；`--protocol full_train --val-fraction 0` 不构造 eval loader，只保存固定 final。`--split official_test` 只接受 full-train 的 fixed-final baseline/K50/matched checkpoint。T4 真实数据命令见第 13 节。

### 11.3 回传

```bash
git add cscwd_router/results/t3_synthetic_seed3106 cscwd_router/results/RESULT_PROVENANCE.md
git commit -m "Record T3 staged training and checkpoint smoke"
git push origin codex/coverage-aware-router
```

只回传小报告与 provenance；PT 文件保留服务器。

## 12. T3 本机提交范围

```bash
git add \
  cscwd_router/code/stag/model_dinov2.py \
  cscwd_router/code/stag/engine.py \
  cscwd_router/code/stag/checkpoint.py \
  cscwd_router/code/stag/pipeline.py \
  cscwd_router/code/stag/test_training_dinov2.py \
  cscwd_router/code/data_prep/gazefollow.py \
  cscwd_router/code/scripts/train_router_dinov2.py \
  cscwd_router/code/scripts/eval_router_dinov2.py \
  cscwd_router/results/T2_REVIEW.md \
  cscwd_router/results/t3_local_checks.json \
  cscwd_router/results/LOCAL_CHECKS.md \
  cscwd_router/results/RESULT_PROVENANCE.md \
  cscwd_router/implementation_plan.md \
  PROGRESS.md
git commit -m "Add official DINOv2 decoder recipe and evaluation pipeline"
git push origin codex/coverage-aware-router
```

## 13. T4 服务器完整主实验

### 13.1 当前运行命令

本节尚未运行。先按第 14 节提交/push；服务器 pull 后由用户启动这一条入口。它会依次完成训练、独立 official test、效率矩阵和 comparison。

```bash
cd /home/fb/src/paper/gazelleV1
git pull --ff-only origin codex/coverage-aware-router
mkdir -p /home/fb/src/paper/gazelleV1/logs/cscwd_router

CUDA_VISIBLE_DEVICES=1 setsid nohup /home/fb/anaconda3/envs/py310/bin/python -u \
  /home/fb/src/paper/gazelleV1/cscwd_router/code/scripts/run_main_experiment.py \
  --t3-summary /home/fb/src/paper/gazelleV1/cscwd_router/results/t3_synthetic_seed3106/summary.json \
  --data-path /newhome/fb/dataset/gazefollow_extended \
  --decoder-checkpoint /home/fb/src/paper/gazelleV1/checkpoints/gazelle_dinov2_vitb14.pt \
  --run-dir /home/fb/src/paper/gazelleV1/experiments/cscwd_router/t4_gf_fulltrain_seed3106 \
  --report-dir /home/fb/src/paper/gazelleV1/cscwd_router/results/t4_gf_fulltrain_seed3106 \
  --provenance /home/fb/src/paper/gazelleV1/cscwd_router/results/RESULT_PROVENANCE.md \
  --train-batch-size 16 \
  --eval-batch-size 8 \
  --workers 8 \
  --router-lr 0.001 \
  --decoder-lr 0.0001 \
  --weight-decay 0 \
  --budget-weight 0.05 \
  --router-hidden-dim 256 \
  --router-temperature 1.0 \
  --escape-tokens 8 \
  --log-every 50 \
  --seed 3106 \
  --warmup 50 \
  --latency-iters 200 \
  --throughput-iters 500 \
  --repeats 5 \
  > /home/fb/src/paper/gazelleV1/logs/cscwd_router/t4_gf_fulltrain_seed3106.log 2>&1 &
```

执行顺序与固定口径：

1. 核对 T3 通过、官方输入权重和未变的训练源码，执行 T4 控制单测。
2. 沿用 T3 trainer：FP32、448、block 5，support 3ep、K100→K75→K50→K50→K50 decoder 5ep、matched dense 5ep。完整 train，无 validation/test loader；沿用官方训练增强。两条 decoder 分支同起点、同 seed、同更新预算，backbone 冻结。
3. 训练全部完成后才打开 test annotation；对 `dense_official/final.pt`、`sparse_decoder/final.pt`、`matched_dense/final.pt` 各评测一次，统一 FP32、image-unit、batch 8 和原版 AUC/Avg L2/Min L2。已有输出通过 SHA/signature/annotation/batch 核对后复用。
4. 效率为 RTX 3090、每图一个 in-frame GT 查询、B1/B4/B8；正序 `dense→K100→K50` 和反序 `K50→K100→dense` 各一个新进程。每 variant 50 warmup、200×5 synchronized latency、500×5 continuous throughput。输入来自 train 的真实 RGB/头框，稳定选择同一组样本，预处理和图像传输在计时外。原版 dense 不含 Router；K100/K50 使用同一个正式 K50 checkpoint。所有 variant 冻结、inference mode、FP16、autocast cache 关闭；记录逐 block token 数与 peak allocated memory。GPU 状态和其他 compute 进程在运行中检查。
5. 先算每种顺序内的 `dense latency / K50 latency`，再对两种顺序取几何平均。primary gate：K50 相对实际 official dense 的 AUC 掉 ≤0.002，且 B8 synchronized latency speedup ≥1.10x。另报 matched dense 的精度差值、K100 control、throughput 与显存，不将小幅精度/显存收益预先当成结果。阈值和设置在训练前写入 manifest。

退出码 0 表示 T4 门槛通过；2 表示矩阵已完成但门槛未过，按规格停止本方案。脚本出错保留原始日志。

### 13.2 恢复与单独补齐效率单元

中断后使用同一主命令，加 `--resume`，日志改为 `t4_gf_fulltrain_seed3106_resume.log`。它恢复训练的 last epoch，或跳过已完成训练；已存在的 official test 不重跑，已完成的效率单元也复用。源码、数据和参数必须保持与 manifest 一致。

若仅需补齐尚未生成的 B8 forward 单元，可在训练结束后单独执行以下命令，然后对主入口使用 `--resume`。输出已存在时不覆盖。

```bash
cd /home/fb/src/paper/gazelleV1
CUDA_VISIBLE_DEVICES=1 setsid nohup /home/fb/anaconda3/envs/py310/bin/python -u \
  /home/fb/src/paper/gazelleV1/cscwd_router/code/scripts/benchmark_router_dinov2.py \
  --dense-checkpoint /home/fb/src/paper/gazelleV1/experiments/cscwd_router/t4_gf_fulltrain_seed3106/dense_official/final.pt \
  --sparse-checkpoint /home/fb/src/paper/gazelleV1/experiments/cscwd_router/t4_gf_fulltrain_seed3106/sparse_decoder/final.pt \
  --data-path /newhome/fb/dataset/gazefollow_extended \
  --batch-size 8 \
  --order forward \
  --precision fp16 \
  --warmup 50 \
  --latency-iters 200 \
  --throughput-iters 500 \
  --repeats 5 \
  --seed 3106 \
  --output /home/fb/src/paper/gazelleV1/cscwd_router/results/t4_gf_fulltrain_seed3106/efficiency/b8_forward.json \
  > /home/fb/src/paper/gazelleV1/logs/cscwd_router/t4_b8_forward.log 2>&1 &
```

### 13.3 产物与回传

- 小报告：`results/t4_gf_fulltrain_seed3106/manifest.json`、`training/`、三个 `accuracy/*.json`、六个 `efficiency/*.json`、`comparison.json` 和 `comparison.md`。
- 大文件：`experiments/cscwd_router/t4_gf_fulltrain_seed3106/` 下的四个 final 和三个 last PT；训练器实际写出后登记路径/SHA，留服务器。
- comparison 保存全部小证据的路径与 SHA、两个 gate 及总判定。当前没有实际 T4 数字。

```bash
git add cscwd_router/results/t4_gf_fulltrain_seed3106 cscwd_router/results/RESULT_PROVENANCE.md
git commit -m "Record T4 GazeFollow main experiment and efficiency matrix"
git push origin codex/coverage-aware-router
```

## 14. T4 本机提交范围

```bash
git add \
  cscwd_router/code/scripts/run_main_experiment.py \
  cscwd_router/code/scripts/benchmark_router_dinov2.py \
  cscwd_router/code/stag/main_results.py \
  cscwd_router/code/stag/test_main_experiment.py \
  cscwd_router/results/T3_REVIEW.md \
  cscwd_router/results/t4_local_checks.json \
  cscwd_router/results/LOCAL_CHECKS.md \
  cscwd_router/results/RESULT_PROVENANCE.md \
  cscwd_router/implementation_plan.md \
  PROGRESS.md
git commit -m "Add T4 main experiment and DINOv2 efficiency matrix"
git push origin codex/coverage-aware-router
```
