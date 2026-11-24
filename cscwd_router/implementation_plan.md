# CSCWD Router · 当前方案、实现规格与进展

> 本文档是给 codex 的实现规格 + 进展记录。codex 开工前读，每完成一个任务写回"进展记录"。
> 2026-11-24 W5：维护前固定K50，删除完整预算扫描；原表5/6为新表4/5。新增训练集中三个B打分消融，各自GF→VAT；真实多人N=1/2/3、K100单次精度和可视化导出接续。见 `PRE_SHUTDOWN_PLAN.md`，本轮尚未启动GPU任务。
> 2026-11-24 R1：按用户新决定对VAT-L做三指标综合复选。8轮rate-6记录首选第3轮（AP较第1轮高1.1004个百分点、L2增加0.7152%），第5轮AP最高。已准备Pareto第1/2/3/5/8轮的逐帧复评入口，4个本机CPU规则检查通过；完整GPU0命令 `RUN_VATL_EPOCH_COMPARISON.md`。尚未启动GPU复评，正式表暂保留原第1轮。
> 2026-11-24 W4：`latex/CORE_DRAFT_ZH.md` 已加入MM式六张具体表：21行主性能、完整效率、速度/显存变化、预算、条件化勾选消融、多人共享。补回并审查GF-L原有效率后，填入四组精度和24个variant×batch效率汇总项；B/L任务模块参数为GF/VAT `3.88/3.91M`、`4.27/4.31M`，未完成的消融留“—”。
> 2026-11-24 W3核心中文稿已整理：`latex/CORE_DRAFT_ZH.md`。主线为“训练参数效率→稠密推理缺口→人物条件化计算选择”；核对Mi的head/eye→direction→FOV作用，明确当前Router没有直接复用该模块。实验按主性能、效率、预算曲线、条件化消融、真实多人组织，已有数值与待补实验分列。
> 2026-11-24 E3-L VAT效率完成：用户实际使用GPU0，四单元/六份报告经 `afcc39e` 回传审查。VAT-L B1/B8相对官方speedup `0.8630/1.0652×`，相对同权重K100为 `1.0168/1.3123×`；B8延迟低6.12%、峰值显存高28.11%。详见 `results/E3_VATL_EFFICIENCY_REVIEW.md`。本轮未重跑GPU任务，后续接近期强方法推理比较。
> 2026-11-24 E2-TL 验收完成：完整 GF-L Router→VAT-L 8 轮，逐帧 AUC/L2/AP `0.948264/0.089676/0.891592`，按 test L2 选第1轮；20个工件SHA及31,978个查询重聚合核对通过。GPU1效率四请求CPU预检通过，新完整命令 `RUN_VATL_EFFICIENCY.md`；末次GPU1已有其他作业，待空闲后用户手动启动。
> 2026-11-24 SOTA-2：已将16个方法/参照的架构与GazeLLE关系补入 `results/SOTA_LANDSCAPE_20261124.md` 第7–9节，核对9个仓库29份固定commit源码。Mi明确DINOv3多层，PaGE明确从GazeLLE出发但取单层；新增公开PaGE权重及训练范围说明，没有启动实验。
> 2026-11-24 SOTA-1：按用户决定，重点转为与近期 object-aware / vision-language 强方法比较精度—成本，保留 GazeLLE 对照。8行公开指标和4个单独标注线索已核对，TextGaze与OmniGF的官方任务权重入口已验证；下一项优先TextGaze推理比较，不新增dense训练。详见 `results/SOTA_LANDSCAPE_20261124.md`。
> 2026-11-24 E3-B 已回传审查：GF-B B1/B8 相对官方 speedup `0.7822/0.9918×`，VAT-B `0.8255/1.0251×`，B8相对K100约1.25×，峰值显存高约59.5%。详见 `results/E3_B_EFFICIENCY_REVIEW.md`；当前优先阶段profile与等价推理优化，L 任务保持原版本。
> 2026-11-24 E3-F1：VAT 效率的嵌套标注修复 `29c6607` 已部署到独立服务器 worktree，70 项 CPU 与真实 475 序列/58,507 帧预检通过。GPU0/2 保持原运行代码；GPU1 新完整命令见 `RUN_VAT_EFFICIENCY_FIX.md`，尚未由助手启动测速。
> 2026-11-24 E3 / E2-TL 已实现：68/68 CPU 通过。三卡完整命令见 `RUN_EFFICIENCY_VATL.md`：GPU0 GF-B/GF-L 效率，GPU1 当前 GF→VAT-B 效率，GPU2 完整 GF-L→VAT-L。保留 GF-L 收口状态；用户手动启动，尚无真实效率或新 VAT-L 指标。
> 2026-11-22 当前论文主方法：DINOv3 多层表示 + gaze-conditioned spatial routing。行文与贡献边界见 `latex/PAPER_STORY.md`；三稿复核见 `latex/PAPER_REVIEW.md`。
> 当前 E0 已实现：GF/VAT × ViT-B/ViT-L，新训练 seed 0，32 项 CPU 检查通过。GPU 0/1 完整命令见 `RUN_DINOV3_TRAIN_TEST.md`。
> 2026-11-23 E2：VAT-B/L 队列完成，结果提交 `0fc2844` 已回传核验。逐帧 AUC/L2/AP 为 B `0.913548/0.110971/0.901738`、L `0.923034/0.097970/0.916599`；详见 `results/E2_VAT_SEED0_REVIEW.md`。最新 GF 与续训状态见下一条，效率尚待测量。
> E1-B / E2-C：GF-B 与 VAT 续训结果已通过 `8f2ae9c` 回传并核验。GF-B 为 `0.956236/0.101550/0.043420`；VAT 各新增 8 epochs 未超过旧 best L2，最终指标不变。详见 `results/E1_GFB_E2C_REVIEW.md`。
> E1-L 最新：GF-L 关机中断，用户决定不补第 5 轮，以实际 support 3 + decoder 4 收口。`459651a` 已回传，按 Min L2 选第 2 轮 K50，`0.960007/0.094061/0.037721`；52 个 best tensors 已导出核验。见 `results/E1L_GF_INTERRUPTED_REVIEW.md`，后续效率与完整 GF-L→VAT-L，不先新增 GF-L dense。
> E2-T 已完成并经 `170ace8` 回传：完整 GF-B Router→VAT-B 的逐帧 AUC/L2/AP 为 `0.940033/0.100131/0.883183`。用户接受当前 B 结果，E2-H 分类头适配后移。详见 `EXPERIMENT_PLAN.md`。
> L 来源补充：现存 GF-L/VAT-L 投影从 epoch 0 起均为 4096 通道，W&B 指向 DINOv3，但历史精确选层未闭合；不能当作已证实单层，也不能继续称同配置初始化已完全认证。详见 `results/L_INITIALIZER_DECISION.md`。GF-L 已收口，后续用当前 best 接效率和迁移，暂不新增 L dense。
> 原 T1–T6 是 DINOv2 单层适配的工程任务，保留原实现、验收和运行记录。以下历史规格不自动变为新的 DINOv3 训练命令。

## 0. 当前决定（2026-11-24，E3 / E2-TL）

SOTA-1 最新目标：与近期强方法比较完整系统的精度—成本，不以全面超过 GazeLLE 作为推进前提。Mi et al. ECCV2026 为接近底座的精度参照，暂未找到官方代码/权重；先准备可获取四组任务权重的 TextGaze B/L 推理对照，再接 OmniGF。当前训练seed0、train/test与单一checkpoint选模不变，不新增dense训练；已测GazeLLE结果保留。

当前方法覆盖冻结 DINOv3 ViT-B/16 和 ViT-L/16 多层表示上的人物条件化深层计算分配。DINOv3 与多层访问沿用，不再次作为相对在审 ICASSP 的新贡献；不新增 dense/matched dense 训练。旧 matched dense、GF fixed-final 与 VAT holdout 结果保留各自协议，不与新结果混拼。

L 初始化的 keys/shape 兼容不等于历史特征配置一致。当前 GF-L 保留 warm-start 标签，已在实际完成的 4 个 decoder epochs 中按 Min L2 选第 2 轮，指标可用。用户决定关机中断后不补最后一轮；接续同硬件效率和完整 GF-L Router→VAT-L，不先新训 GF-L dense，GazeSpot 暂不替换主线。

按用户要求，全部新训练 base seed=0，epoch 子 seed 为 `0 + epoch`。四组原计划 support 3 + sparse decoder/inout 5，不继承 seed3106 Router；GF-L 实际完成 3+4 后由用户收口，其余三组为 3+5。GF 按 test Min L2、VAT 按 test L2 选单一 checkpoint，每轮 test 用 K50，VAT 最后自动逐帧评测。GF-B 已完成并核验，GF-L 已导出 best 与中断报告。VAT 两组及各新增 8 轮低 LR 续训已核验；新 L2 未改善，保留原 best，不再原样加轮数。

VAT 的 B/L 初始化都已核对存在，因此不用等本轮 GF。GPU 0 顺序 GF-B→GF-L，GPU 1 顺序 VAT-B→VAT-L；详细配置与完整 nohup 命令见 `EXPERIMENT_PLAN.md`、`RUN_DINOV3_TRAIN_TEST.md`。官方 GazeLLE 后续仅做推理比较；旧 DINOv2 T4 按原阈值单独收口。

E2-T 已实际完成：完整继承 GF-B K50 的 52 个 Router/heatmap decoder tensors，仅新增 5 个 in/out tensors，backbone 冻结；8 epochs 中首轮固定 Router，后 7 轮 coverage/BCE 联合适配，初始 LR `1e-4/1e-5/1e-3`，seed 0。按 rate-6 test L2 选第 1 轮，最终 rate-1 AUC/L2/AP `0.940033/0.100131/0.883183`；定位显著改善、AP 低于旧 VAT-B，联合阶段未改善 L2。用户接受当前 B 结果，GF-L 现已收口；E2-H 仅分类头适配后移。等待期优先预算扫描、条件化对照与多人 union 证据，正式效率等空闲卡。

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

### 2026-11-23 · E1-B / E2-C：GF-B 完成与 VAT 续训结果审查

- **完成内容**：服务器提交并 push `8f2ae9c`，本地 pull GF-B、VAT 续训 B/L 及既有官方 GF-B 精度报告，共 13 份原始 JSON。按既定规则核对单 checkpoint 选模、真实参数变化、native dense/K100 probe 与逐查询记录；保留 GF-L 运行任务和旧 T4 其他产物。
- **产出文件与数字**：`results/E1_GFB_E2C_REVIEW.md`、`results/e1_gfb_e2c_server_checks.json`。GF-B 第 5 轮 AUC/Avg/Min L2 `0.956236/0.101550/0.043420`，已有官方 B 报告为 `0.956141/0.104136/0.045332`；相对 Avg/Min L2 降低 2.48%/4.22%，AUC 基本持平。VAT B/L 各新增 8 轮、5,664 次更新，44 个 decoder tensors 改变；最低新 rate-6 L2 `0.111874/0.102485` 未超过旧 best，最终逐帧指标保持不变。原始 dense/routed K100 三项精度差均 0。29 个工件/标注路径（其中 21 个 PT/PTH/gzip）SHA 全匹配，三组逐查询重聚合差值均 0；本轮未新增 GPU forward 或测速。
- **下一步**：GF-L 继续等待；GF-B 可先做 E3 同硬件测速。VAT 先测预算响应、再做完整 GF-B Router→VAT 迁移，不原样延长固定 Router 的 decoder，不新增 dense 训练。

### 2026-11-23 · E2-T：完整 GF Router→VAT 入口完成

- **完成内容**：按用户最新优先级暂停安排正式效率测试，新增完整 GF Router/decoder 的 VAT 迁移入口；校验共享配置/权重逐值继承，仅新增 in/out。全程 K50、冻结 backbone，首轮固定 Router、后续 coverage/BCE 联合适配，保存每轮 compact checkpoint、原子恢复与标准 test 选模；旧 Gazelle/GF/VAT trainer 源码均未改。
- **产出文件与数字**：`code/stag/dinov3_transfer.py`、`code/scripts/transfer_gf_router_to_vat.py`、`code/stag/test_dinov3_transfer.py`、`RUN_GF_ROUTER_TO_VAT.md`、`results/e2t_gf_to_vat_local_checks.json`。51/51 CPU 通过（12 项新增、39 项回归），涵盖真实 B/L factory 的全部共享 tensors 继承与仅新增 5 个 in/out tensors、分阶段梯度、hard Top-K 的梯度来源、全画外 batch、合成闭环、父 GF 工件不变及恢复逐值一致；完整 nohup 命令解析/语法通过。GF-B 来源 SHA `8297c6f621ceeeef6222e31a4e1782ee92d2fb50b697e9da31144cfe5b2a5b14`。8 epochs 是新实验预算，尚无新 VAT 精度。
- **下一步**：用户 push/pull 后在可安排 GPU 1 时手动运行 E2-T；回传初始化诊断、history、summary 与 provenance，比较 GF Router 迁移前后 VAT 精度/coverage。GF-L 继续原任务，效率等空闲卡再测。

### 2026-11-23 · L 初始化语义核对与预算决定

- **完成内容**：根据用户指出的历史来源，补读两组 W&B 精确配置/日志、epoch 0 与末轮 checkpoint、记录 Git 版本的 backbone/model 源码。修正此前把 keys/shape 匹配写成同配置初始化认证的结论；未改旧 trainer/注册表，未停止运行任务。
- **产出文件与数字**：`results/l_initializer_semantics_20261123.json`、`results/L_INITIALIZER_DECISION.md`。两组 L 首末轮投影均 `[256,4096,1,1]`、无 SASA/GGSF，日志参数数为 3,678,720/3,712,001。GF 记录提交是 last-layer/448，与权重结构不符且无 runtime 快照；VAT 记录源码是 `[4,11,17,23]`，当前为 `[5,11,17,23]`，实际历史层号未封存。没有新训练/推理精度或进程操作。
- **下一步**：GF-B→VAT-B 已启动，继续作为首要实验；L 保留当前 warm-start 与 checkpoint，看当前 K75 test 再决定后续预算，不新开两套 dense。暂不切 GazeSpot 底座；如果后续切换，单独适配并做同底座 dense/sparse 比较，不混写贡献。

### 2026-11-23 · E2-T：GF-B→VAT-B 完成、回传与审查

- **完成内容**：服务器仅提交当前 5 份 JSON 与对应 provenance block，push `170ace8`，本地 pull/rebase 保留 L 来源文档和原有目录迁移。核对完整 GF 权重继承、8 轮参数变化、实际 optimizer steps、单模型选模与全部逐帧导出；未新增 GPU forward 或测速。
- **产出文件与数字**：`results/vat_from_gfb_router_k050_joint_seed0/`、`results/E2T_GF_TO_VAT_REVIEW.md`、`results/e2t_gfb_to_vatb_server_checks.json`。最终 31,978 人次、20,655 画内，AUC/L2/AP `0.940033/0.100131/0.883183`，相对旧 VAT-B 差 `+0.026485/-0.010840/-0.018555`，L2 相对降低 9.77%。第 1 轮 Router 未改变，后 7 轮实际更新 4,956 次；decoder/inout 更新 5,664 次，但 best 仍为第 1 轮。20 个登记工件（14 个大型 PT/PTH/gzip）SHA、5 份回传报告与 20 个本地源码 SHA 一致；31,978 条预测身份/标注匹配，三项指标重聚合差值均 0。
- **下一步**：E2-H 优先从当前 best 仅适配 `inout_head`，冻结 Router、heatmap decoder 与 `inout_token`，先改善 AP；需独立实现参数控制与热图一致性检查，再由用户手动运行。暂不原样加联合 epochs；效率等空闲卡，L 来源独立保留。

### 2026-11-23 · E2-W：接受 B 结果，确定等待期与 L 后续分支

- **完成内容**：按用户决定将 E2-H 后移；只读服务器 GF-L history、日志与 GPU 状态，记录等待期实验优先级，不启动/停止任务。GF-L 完成后可用则正式效率与完整 Router→VAT-L；不理想则先训练来源明确的 DINOv3-L 多层 GF 底座。
- **产出文件与数字**：`results/gfl_interim_20261123.json`、`EXPERIMENT_PLAN.md`。GF-L 已提交 2 个 decoder epochs；第 2 轮 train K75、test K50，AUC/Avg/Min L2 `0.960007/0.094061/0.037721`，第 3 轮 K50 训练中，无最终 summary。快照保存三个原始报告内容/SHA、服务器 Git 与日志尾部；不是最终成绩或效率结果。
- **下一步**：先准备 B 的预算响应（无需重训）与 scene-only 条件化对照（同起点、同 seed/预算重训），再做真实多人 union 精度/可视化。正式效率入口需适配 DINOv3，空闲卡测量；GF official test 全部单头，不作为多人证据。

### 2026-11-23 · E1-L：关机中断后收口、回传与审查

- **完成内容**：按用户决定不补最后一轮。只读核验 GF-L 完整 history/原子 last，导出内嵌 best；服务器仅提交当前 4 份小 JSON 和对应 provenance，push `459651a`，本地 pull/rebase 保留 3 条本地提交与目录迁移。未新增训练、GPU forward 或测速。
- **产出文件与数字**：`results/testselected_k050_seed0/gazefollow/vitl/interrupted_summary.json`、`results/E1L_GF_INTERRUPTED_REVIEW.md`、`results/e1l_interrupted_server_checks.json`。实际 support 3 + decoder 4，按 Min L2 选第 2 轮 K50，AUC/Avg/Min L2 `0.960007/0.094061/0.037721`；第 4 轮为 `0.960678/0.093370/0.037724`，不拼指标。52 个 best tensors 导出逐值一致，原 last SHA 不变；6 个 PT/PTH、2 份标注、18 个源码与原生树 SHA 匹配，已提交 optimizer steps 85,095/14,184。保留原 3+5 manifest，报告明确 completed=false / user-closed / 实际 3+4，无独立复评与逐查询重聚合。
- **下一步**：按当前精度进入 E3 同硬件效率与完整 GF-L→VAT-L，暂不新训 GF-L dense。先实现 DINOv3 评测/效率和收口来源接入，再给用户完整命令手动运行；不修改布尔值绕过原 transfer 对 completed 来源的检查。

### 2026-11-24 · E3：DINOv3 与官方 GazeLLE 效率矩阵实现

- **完成内容**：新增 source reader、完整系统 benchmark、单 GPU 顺序队列与矩阵汇总；复用旧 GPU guard/稳定计时器，official/K100/K50 每单元新进程，真实 train RGB/GT 框、B1/B8 正反序、FP16、50/200/500/5，计入完整 forward。复用精度对应 checkpoint，逐 block token trace，源码/权重/输入/GPU 记录，request 一致才恢复复用。
- **产出文件与数字**：`code/stag/dinov3_sources.py`、`dinov3_efficiency.py`、`test_dinov3_efficiency.py`，`code/scripts/benchmark_router_dinov3.py`、`run_router_efficiency.py`，`RUN_EFFICIENCY_VATL.md`、`results/e3_efficiency_vatl_local_checks.json`。新增效率 16 项 CPU 通过，覆盖 B/L 真实 Router prefix/suffix/多层 decoder、收口来源、矩阵/恢复。服务器只读核对三张 RTX3090 均 15 MiB/0%（观察时），官方 B/L/B-inout 权重分别 40/40/45 tensors，两份 DINOv2 backbone 存在。没有 GPU forward 或速度数字。
- **下一步**：用户更新代码后，在 GPU0 测 GF-B→GF-L，在 GPU1 测当前 GF→VAT-B；四单元完整后汇总同卡正反序几何平均，不预写加速。

### 2026-11-24 · E2-TL：GF-L 收口来源迁移接入

- **完成内容**：迁移入口新增显式 `--allow-closed-gf`，读取 committed history/last/best 核对选模与逐值权重，保留 GF-L completed=false、实际3+4。源状态写进新 VAT manifest/summary，不改变旧 GF 工件。L 沿用 B 的 8 轮 K50，首轮固定 Router、后7轮联合适配，batch4×累积8、seed0。
- **产出文件与数字**：更新 `code/stag/dinov3_transfer.py`、`code/scripts/transfer_gf_router_to_vat.py`、`code/stag/test_dinov3_transfer.py`；新增闭环 1 项，与 E3/旧回归合计 **68/68** CPU 通过。旧 `gazelle/` 未改，无真实 VAT-L 训练结果。GPU2 全参数命令、日志与输出在 `RUN_EFFICIENCY_VATL.md`，大型工件将留服务器自动登记 SHA。
- **下一步**：用户手动启动 GPU2 完整 GF-L Router/heatmap decoder→VAT-L；只 train/test、每轮 test L2 单模型选模，最后逐帧 test，回传小报告审查。当前不新增 dense。

### 2026-11-24 · E3-F1：VAT 嵌套标注采样修复与隔离部署

- **完成内容**：修复效率采样把 VAT 序列当作 GF 图像记录的错误，按 source dataset 调用官方 loader，VAT 展开 frames 后选择单帧真实头框。补齐嵌套采样与队列/恢复回归。`29c6607` 已经 Git push，服务器独立 worktree 就绪，原目录 HEAD/三个核心源码 SHA 未变，GPU0 GF 效率和 GPU2 VAT-L 继续运行。
- **产出文件与数字**：更新 `code/stag/dinov3_efficiency.py`、`test_dinov3_efficiency.py`，新增 `RUN_VAT_EFFICIENCY_FIX.md`、`results/e3_vat_schema_fix_checks.json`。70/70 CPU 通过；真实 VAT 475 个序列、58,507 帧，8 个 RGB/GT 框匹配官方 loader，B1 是 B8 前缀，四个正式参数请求 --check-only 通过；GF B8 metadata 与旧版一致。没有新 GPU forward/计时或模型指标。
- **下一步**：用户从 `/home/fb/src/paper/gazelleV1_efficiency_vat_fix` 手动重启 GPU1，输出原仓库 `results/efficiency_dinov3_seed0_20261124_vatfix/vat/vitb/`。原目录保持运行版本直到 GF 矩阵结束，保留失败日志与 GF 已有单元。

### 2026-11-24 · E3-B：GF-B / VAT-B 效率完成与审查

- **完成内容**：通过已空闲的 VAT 修复 worktree 选择性提交两组完整矩阵及两个 provenance blocks，`85210b5` 已 push/pull；随后恢复该 worktree 29c6607，原目录 HEAD4162da7/源码未变。CPU 核对权重/源码版本/输入/GPU/逐 block token 数，重算全部 summary JSON/MD，不重跑计时或训练。
- **产出文件与数字**：两组 `results/efficiency_dinov3_seed0_20261124/gazefollow/vitb/`、`results/efficiency_dinov3_seed0_20261124_vatfix/vat/vitb/`，审查 `results/E3_B_EFFICIENCY_REVIEW.md`，检查 `results/e3_b_efficiency_server_checks.json`。GF官方/K50批次延迟 B1 `17.9309/22.9234ms`、B8 `53.3757/53.8184ms`；VAT B1 `17.9408/21.7323ms`、B8 `51.6012/50.3369ms`。对应speedup `0.7822/0.9918×` 和 `0.8255/1.0251×`；相对K100 B8为 `1.2519/1.2559×`。B8峰值显存约857 vs538MiB，高59.5%。12报告SHA、50个版本化源码、8个source/weight路径（6个PT/PTH）、2份标注、16个RGB与24组token trace核对通过。
- **下一步**：B 当前不写成相对 GazeLLE 明显更快或更省显存。优先GPU1分阶段profile及同权重native full-token控制；检查 none 全一乘法、四层 repeat/cat 等具体开销，在新代码目录验证原版/快路径的输出、精度和实测效率，不新增 dense 训练。GF-L/VAT-L按原任务接续，旧DINOv2 gate不套用本轮。

### 2026-11-24 · E3-P0：推理流程与效率解释落盘

- **完成内容**：逐段核对当前 RGB→dense prefix→人物 support/max-union→TopK→sparse suffix→early-exit fill→四层融合→完整 decoder 的执行路径，只读取得两份旧 B1 阶段报告，区分当前整网实测、历史分段诊断与理论 MACs 推导。明确原生 DINOv3 full-token 推理控制不需要新增 dense 训练。
- **产出文件与数字**：`results/E3_INFERENCE_AND_OPTIMIZATION.md`、`results/e3_efficiency_explanation_evidence.json`。既有 GF-B B8 的额外全 token 路径差额 14.00ms，被 K100→K50 节省的13.56ms抵消；VAT-B对应11.62/12.88ms。旧B1 Router约占插桩时间13–14%，suffix减半patch后的时间并未减半。约27%的blocks主要MACs缩减仅为公式估计，不作为整网FLOPs实测。没有新GPU运行或优化收益。
- **下一步**：只开当前checkpoint的阶段profile/native full-token控制；其后先做none全一门、单人repeat、多层先融合后按人展开和固定网格缓存的等价快路径，核对FP32/FP16与多人输出后再测速度/显存。暂不改预算或重训backbone，原服务器任务不变。

### 2026-11-24 · SOTA-1：近期方法核对与比较目标调整

- **完成内容**：读取近期原文、渲染核对Mi/TextGaze/OmniGF/GazeVLM原表，核实官方代码/权重发布；主目标转为近期强方法的精度—成本比较，保留既有GazeLLE对照和机制控制，不新增dense训练。
- **产出文件与数字**：`results/SOTA_LANDSCAPE_20261124.md`、`results/sota_landscape_20261124.json`；8行公开指标、4项分别标注的线索、3份本地选中结果SHA。GF-L `.960007/.094061/.037721` 接近Mi `.961/.094/.038`，三项优于TextGaze-L `.956/.099/.043`；当前B的GF/VAT六项优于TextGaze-B公开值。Mi未找到公开实现/权重，TextGaze release API确认GF/VAT×B/L四个PT，OmniGF官方HF API确认GF/VAT权重。服务器只读未见完整GF-L→VAT-L最终summary，没有填VAT-L数字或新效率倍数。
- **下一步**：只开TextGaze官方权重推理比较准备，覆盖同数据精度与同硬件效率，区分给定文本后的core与RGB生成文本的完整流程；之后检查OmniGF的3090运行条件。原服务器任务不变，阶段profile/快路径保留为后续优化项。

### 2026-11-24 · SOTA-2：GazeLLE关系、DINOv3与各模型架构补充

- **完成内容**：区分明确派生、GazeLLE式结构扩展、相同冻结编码范式与独立架构；逐项列出输入、backbone取层、融合/decoder、额外分支、训练与推理组件。补查PaGE、GazeCoT中的GazeLLE-v3、PINTO工程实现，保留未知信息，不从模型名推断结构。
- **产出文件与数字**：更新 `results/SOTA_LANDSCAPE_20261124.md` 第7–9节，新增 `results/architecture_lineage_20261124.json`。16项架构（含官方和本方法）、9仓库29份源码commit/SHA；PaGE B/H+官方HF权重元数据已核实。Mi为DINOv3浅/中/深融合，PaGE/TextGaze/GazeHRNet/GazeLoom所核forward为单层输出；ViTGaze是多层attention、Sharingan是多层DPT，两者不是DINOv3。PaGE公开B-distill GF `.966/.081/.029`、VAT `.969/.068/.945` 高于我们，注明联合数据/全模型微调/额外蒸馏，非本轮复测。
- **下一步**：外部官方权重推理对照增加PaGE这一直接相关DINOv3候选；Mi继续作多层配置精度参照，TextGaze保留。先核输入与权重语义后逐项准备服务器脚本；当前配方、checkpoint及服务器任务不变，没有新训练/GPU forward。

### 2026-11-24 · E2-TL 验收 / E3-L：VAT-L 回传审查与 GPU1 效率准备

- **任务号**：E2-TL / E3-L。
- **完成内容**：从隔离 worktree 将完整 GF-L→VAT-L 五份报告和 provenance 以 `2eb4275` push/pull，CPU 检查以 `33e459a` 回传。完整继承52个GF张量、8轮K50训练、冻结backbone、逐轮test L2单模型选模与best/last/epoch工件均核对通过。现有效率入口直接支持L，采用包含VAT loader修复的隔离目录准备GPU1命令；未改旧gazelle包或计时器。
- **产出文件与数字**：`results/vat_from_gfl_router_k050_joint_seed0/`、`results/E2TL_GFL_TO_VATL_REVIEW.md`、`results/e2tl_gfl_to_vatl_server_checks.json`、`code/scripts/review_vat_transfer.py`、`RUN_VATL_EFFICIENCY.md`。最终AUC/L2/AP `0.948264/0.089676/0.891592`，31,978查询（20,655画内），选第1轮；20个工件SHA、22项源码hash及导出指标重聚合通过，三指标差值0；B1/B8正反序4个效率请求CPU预检通过。同期更新文献表，未产生新GPU效率数字。
- **下一步**：GPU1空闲后用户手动运行official/K100/K50效率矩阵，输出 `efficiency_dinov3_vatl_seed0_20261124/vat/vitl/`，完成后Git回传审查。末次GPU1为23,821MiB/89%，当前不与其他作业共卡测速；后续再接近期强方法推理对照。

### 2026-11-24 · E3-L：VAT-L 效率完成、回传与审查

- **任务号**：E3-L（VAT-L）。
- **完成内容**：确认用户改用GPU0的任务有最终COMPLETE、四单元/六报告齐全且已退出。隔离worktree提交 `afcc39e` push/pull；核对执行commit `0a37505`、GPU0 UUID、同一VAT-L best、输入/源码/权重SHA、12组token trace和计时协议，CPU重算汇总通过。实际K50前12层1029、后12层517 tokens，旧gazelle包和计时器均未改。
- **产出文件与数字**：`results/efficiency_dinov3_vatl_seed0_20261124/vat/vitl/`、`results/e3_vatl_efficiency_server_checks.json`、`results/E3_VATL_EFFICIENCY_REVIEW.md`。B1官方/K100/K50为 `33.4730/39.4368/38.7860ms`，B8为 `147.5857/181.8345/138.5573ms`；相对官方speedup `0.8630/1.0652×`，相对K100 `1.0168/1.3123×`。B8峰值allocated `1824.21` vs官方 `1423.95MiB`。六报告SHA、25版本化源码文件、两份native源码树、五个来源/权重引用及八张RGB已核对，汇总/独立速度比重算一致。FP32精度仍为 `0.948264/0.089676/0.891592`，没有新增模型forward或GPU测量。
- **下一步**：保留VAT-L完整系统基线及B1/显存代价，接近期强方法的同机推理对照，优先TextGaze官方任务权重，不新增dense训练；外部速度比待真实测量后填写。

### 2026-11-24 · W3：中文核心稿与实验数据组织

- **任务号**：W3。
- **完成内容**：重新核对GazeLLE原文摘要、Mi方法§3.2–3.4/§4.1及当前Router源码，形成精简中文稿。以人物条件化深层计算选择承接稠密推理问题，明确头部特征来自共享prefix，当前无显式方向/FOV模块；DINOv3/多层为沿用配置，保持ICASSP贡献边界。
- **产出文件与数字**：`latex/CORE_DRAFT_ZH.md`，含动机、方法、贡献、结论及五类实验矩阵；填入四组已审精度和三组正式效率，引用Mi原文与既有审查，不新增实验。明确VAT-L B8相对官方1.0652×、相对routed K100 1.3123×，B1和显存代价保留；GF-L效率尚待回传审查。
- **下一步**：按核心稿补预算下的定位/覆盖率与条件化打分对照，验证“保留多少精度”的主张；真实多人实验及近期强方法推理比较按同一表格口径接续。本轮只完成写作与本机提交，没有训练或GPU运行。

### 2026-11-24 · W4：MM式实验表与真实数据填充

- **任务号**：W4。
- **完成内容**：读取并渲染MM第6/8页，核对主表字段和勾选消融布局；对照官方GazeLLE第6页Table3确认B/L及传统方法的文献数值。将核心稿改为六张具体实验表，按来源填入近期方法、本文四组精度/效率与条件化full行覆盖率。补表时发现GF-L原矩阵已完整完成，经 `5e64c45` 将六报告及CPU检查push/pull，核对25份版本化源码、两份native源码树、五来源/权重、八张RGB、12组token trace和汇总通过。未将旧MM的GazeSpot消融数字填作Router结果。
- **产出文件与数字**：`latex/CORE_DRAFT_ZH.md`、`results/E3_GFL_EFFICIENCY_REVIEW.md`、`results/e3_gfl_efficiency_server_checks.json`、`results/efficiency_dinov3_seed0_20261124/gazefollow/vitl/`。主表21行，GF/VAT各三指标；效率含24个已测variant×batch汇总项；预算8行、条件化勾选4行、多人策略3行。补回GF-L B1/B8相对官方 `0.9177/1.0722×`，B8相对routed K100 `1.3046×`。本文任务参数GF/VAT：B `3,879,681/3,912,962`，L `4,272,897/4,306,178`，来自Router、decoder、in/out实际训练参数数。PaGE更强公开值及其额外训练范围保留。没有新增训练、模型forward或GPU计时。
- **下一步**：按表4/5填充预算精度与同预算条件化消融；表6使用真实同帧多人子集评测。四组原效率矩阵均已审查；近期强方法同机测速完成后再补对应表项，当前不预填。

### 2026-11-24 · R1：VAT-L三指标综合比较与全量复评入口

- **任务号**：R1。
- **完成内容**：提取8轮同次训练的rate-6 AUC/L2/AP，先做Pareto筛选，再用三指标相对最优值的损失等权比较作为辅助排序。新建独立逐帧复评入口，加载已保存epoch，复用原第1轮全量结果，其他候选分别评测；保存每轮完整三指标和导出SHA，不覆盖旧best，不将rate-6与rate-1指标混排。
- **产出文件与数字**：`results/VATL_EPOCH_SELECTION_REVIEW.md`、`results/vatl_epoch_selection_history.json`、`code/stag/vat_epoch_selection.py`、`code/scripts/compare_vat_epochs.py`、`RUN_VATL_EPOCH_COMPARISON.md`。rate-6 Pareto为1/2/3/5/8；第3轮 `0.947861/0.092521/0.903199`，第5轮 `0.947453/0.093317/0.904636`。第4/6/7轮被第3轮三指标全面支配。4个本机CPU选择规则检查通过，脚本语法检查通过；没有新GPU精度。
- **下一步**：部署并做服务器CPU预检；用户手动在GPU0运行候选的统一逐帧test（第1轮复用，新增第2/3/5/8轮），回传后依据完整三项确定正式模型，再导出与核对其效率。

### 2026-11-24 · W5：维护前最小实验闭环

- **任务号**：W5。
- **完成内容**：按约48小时维护窗口删除原表4的完整预算扫描，固定K50；原表5/6改为表4/5，保留三个ViT-B条件化打分消融、各自GF→VAT迁移、真实多人共享与可视化导出。补充同权重K100单次精度参考，不扫描其他预算。L沿用已完成完整模型。
- **产出文件与数字**：`PRE_SHUTDOWN_PLAN.md`、`latex/CORE_DRAFT_ZH.md`、`results/w5_pre_shutdown_planning_evidence.json`。CPU确认VAT test为13,127唯一帧、31,978查询，5,182帧至少三个真实头框；采用共同帧集合和seed0嵌套N=1/2/3查询。三卡本次均15MiB/0%；旧B日志每条GF→VAT纯训练合计8.25小时，不含评测/导出。方案分配三卡各一个消融，维护前预留回传时间；没有新模型指标或GPU运行。
- **下一步**：先实现可配置打分输入的B消融和多人/可视化导出入口，做CPU检查，再提供用户手动启动的三卡完整命令。余量独立复测两组L的B8速度；绘图和排版使用导出数据在本机继续。

### 2026-11-24 · W6-A：三个条件化消融训练入口

- **任务号**：W6-A。
- **完成内容**：新增可配置scoring输入，复用原生Router优先规则、max-union、Top-K和suffix。关闭geometry同时移除绝对头框输入与相对几何；关闭appearance移除头部池化特征。full模式与原版state/输出逐值一致。新增三分支GF3+5→各自VAT8队列，仅冻结骨干上的任务模块训练；source扩展纳入原有恢复fingerprint，旧gazelle包未改。
- **产出文件与数字**：`code/stag/conditional.py`、`code/scripts/train_conditioned_router.py`、`code/scripts/run_final_ablation.py`、`code/stag/test_final_ablation.py`、`results/w6a_conditional_local_checks.json`、`RUN_FINAL_ABLATIONS.md`。新增及原GF/VAT回归合计47项CPU通过（8.39秒），含三个合成迁移闭环、错误分支拒绝、冻结与恢复；没有真实训练指标。
- **下一步**：Git部署独立训练worktree并CPU预检，用户手动启动三卡命令。接续多人、K100精度和真实可视化导出，训练目录保持固定版本。

### 2026-11-24 · W6-B：真实多人和K100精度入口

- **任务号**：W6-B。
- **完成内容**：新增VAT固定共同帧/嵌套GT查询视图；共享官方dense、独立批量sparse、max-union三策略的FP32质量评测和FP16正反序fresh-process计时，保存真实token/batch与图像/查询吞吐。新增四组同权重K100单次精度，保留原选择和GF-L用户收口记录，输出K100−K50，不训练/不换原文件。请求、工件SHA和恢复统一核对。
- **产出文件与数字**：`code/data_prep/final_queries.py`、`code/stag/final_io.py`、`code/stag/final_multipeople.py`、`code/scripts/run_final_multi.py`、`code/scripts/run_final_k100.py`、`code/stag/test_final_evaluation.py`、`RUN_FINAL_EVALUATION.md`、`results/w6b_evaluation_local_checks.json`。新增及旧效率回归合计32项CPU通过（5.89秒），含15-cell合成派发/恢复与原指标一致性；没有真实精度或新GPU时间。
- **下一步**：完成真实可视化导出后统一部署评测worktree、CPU预检；用户在各训练队列结束后手动运行GPU0多人、GPU1 K100，固定训练目录不更新。

### 2026-11-24 · W6-C：两卡收口与真实机制图入口

- **任务号**：W6-C。
- **完成内容**：实现真实预测/support/max-union/Top-K/RGB小案例导出与可脱离服务器的CPU绘图。参照MM图4形成六行同样例对比，另画同帧换人条件图、真实union与独立路径对比；保留较高误差样例，记录选样策略与SHA。新增CPU表4汇总，核对配方与各自GF→VAT继承。按用户最新资源改为只用GPU0/1：0卡scene-only→scene-head，1卡scene-geometry；后接多人、四组K100与可视化。
- **产出文件与数字**：`code/stag/final_visuals.py`、`code/scripts/export_final_visualizations.py`、`render_final_visualizations.py`、`summarize_final_ablation.py`、`run_final_ablation_queue.py`、两份新增检查及 `results/w6c_final_local_checks.json`；全部91项CPU/回归通过，新增C检查12项，合成排版已目视检查。默认导出10个真实案例、生成6张PNG/SVG图；这是配置数量，真实导出尚待用户GPU执行。完整两卡命令在 `RUN_FINAL_ABLATIONS.md`、`RUN_FINAL_EVALUATION.md`。
- **下一步**：提交并部署独立评测worktree、完成服务器CPU预检；用户手动启动两卡训练及后续推理，关键小报告和图经Git回传。固定训练worktree不更新。

### 2026-11-24 · W6-D：两卡入口服务器CPU预检与回传

- **任务号**：W6-D。
- **完成内容**：代码 `d29d1ae` 经Git部署到独立评测worktree，固定训练worktree保持 `c19ba93`。从两份runbook提取真实命令，以空CUDA可见设备执行check-only，训练双分支队列、geometry单分支、多人、K100、可视化共5个入口均exit0；预检报告 `1fed049` push/pull回本地，核对23项报告内源码SHA一致。
- **产出文件与数字**：`results/w6_final_server_preflight.json`。实际服务器Python3.10.19、torch2.6.0+cu124、matplotlib3.10.8；多人共同集合5,182帧，GF/VAT各32个图片候选与多人16个候选可读取。未执行模型forward、GPU训练、精度或计时。评测worktree `/home/fb/src/paper/gazelleV1_final_evaluation` 固定 `1fed049`；完整命令 `RUN_FINAL_ABLATIONS.md`、`RUN_FINAL_EVALUATION.md`，仅用GPU0/1。
- **下一步**：用户手动跑两卡条件化训练，随后0卡独占多人质量/效率，1卡依次K100和真实图导出；三分支结束后CPU汇总表4。回传小结果和RGB/图，保留大工件路径/SHA。

### 2026-11-24 · W7-A：真实VAT多人精度完成并回传

- **任务号**：W7-A。
- **完成内容**：用户授权助手在GPU0以accuracy/resume启动后，N1既有结果复用，N2/N3三策略精度全部完成，日志COMPLETE、进程退出。服务器CPU核对9组预测工件SHA、32项源码fingerprint、同N同人物和嵌套查询，逐查询AUC/L2/AP重聚合差值小于1e-12；小报告经 `7c37446` push/pull回本地，12个报告SHA与30个可读源码文件SHA一致。
- **产出文件与数字**：`results/final_multi_seed0/`、`results/w7a_multi_accuracy_server_checks.json`、provenance。5,182共同帧；N1/2/3分别5,182/10,364/15,546查询，9个FP32质量单元完成。N3 official AUC/L2/AP为0.919423/0.122274/0.866136，union为0.929907/0.112543/0.849850，independent为0.929747/0.112434/0.848231；union GT点覆盖0.959405。N1两Router结果一致，表5精度已填入；大预测gzip仅记录服务器路径/SHA。计时单元为0，GPU0仍有其他用户计算任务。
- **下一步**：可独占GPU0后补6个多人计时单元；GPU1继续原geometry训练。K100和真实图导出按既定顺序运行，不重训主模型。

### 2026-11-25 · W7-B：四组K100与geometry分支完成，核对早先队列恢复点

- **任务号**：W7-B。
- **完成内容**：GPU0四组K100已于11月24日23:59完成；GPU1 geometry分支GF于11月25日01:39、VAT于03:17完成，相关进程均退出。服务器CPU核对40个工件、固定权重/原选模、源码、GF→VAT继承、冻结状态及逐查询指标重聚合；`7ad1de7` 经Git回传，6个主报告SHA在本机一致。补查早先GPU0队列：scene-only已完成support3轮和decoder1轮，last可从decoder第2轮恢复，scene-head尚未开始。
- **产出文件与数字**：`results/final_k100_seed0/`、`results/final_ablation_seed0/scene_geometry/`、`results/w7b_k100_geometry_server_checks.json`，大PT/gzip仅登记服务器路径/SHA。四组K100均实际1024patch、每block1029tokens；GF B/L AUC/Avg L2为0.957218/0.100484、0.961650/0.093095；VAT B/L AUC/L2/AP为0.940170/0.100636/0.878522、0.947314/0.090269/0.898700。geometry GF AUC/Avg L2/Min L2为0.956018/0.101514/0.043611，VAT为0.940438/0.100879/0.886571，已填入表4。CPU核对不执行模型forward；本轮没有启动新GPU任务。
- **下一步**：等待用户确认后在GPU0以原resume队列继续scene-only→scene-head，补齐表4另外两行；后续真实图导出。GPU0仍有其他用户进程，效率待独占后补。

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
