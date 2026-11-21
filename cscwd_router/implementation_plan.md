# CSCWD Router · 实现规格与进展（GazeLLE 底座 + gaze-conditioned routing）

> 本文档是给 codex 的实现规格 + 进展记录。codex 开工前读，每完成一个任务写回"进展记录"。
> 底座：GazeLLE 官方（DINOv2 单层）。方法：gaze-conditioned spatial routing（多人共享一次稀疏前向）。

## 0. 目标（一句话）

在 GazeLLE（官方 DINOv2 单层，`gazelle/backbone_v0_dinov2.py` + `model_v0_dinov2.py`）之上，加 gaze-conditioned routing：在 DINOv2 中层预测一个高召回 gaze support，只让后段 blocks 处理 support 内的 token，多人 max-union 共享一次稀疏前向。精度几乎不掉，省 backbone 深层计算。

## 1. 底座（复用，不动）

- 代码：`gazelle/backbone_v0_dinov2.py`（`DinoV2Backbone`）、`gazelle/model_v0_dinov2.py`（`GazeLLE`）。
- DINOv2 ViT-B/14，输入 448×448，单层特征 32×32（768 维）。
- GazeLLE 官方 GazeFollow 精度：ViT-B AUC 0.956。
- backbone 权重：服务器 `checkpoints/dinov2_vitb14_pretrain.pth`。
- GazeLLE decoder checkpoint（GazeFollow 训练）：用户于 2026-11-21 确认主线使用服务器 `/home/fb/src/paper/gazelleV1/checkpoints/gazelle_dinov2_vitb14.pt`。同目录另有 ViT-B inout、ViT-L 和 ViT-L inout 版本；本轮使用 ViT-B、非 inout。

## 2. 方法（routing 适配 DINOv2）

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

## 3. 任务清单（按依赖顺序）

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
- 复用 GazeLLE 官方 checkpoint，不重训 backbone（dense 阶段）。
- 所有数字只写真实跑过的；不编数字、不编对照。
- 新代码全放 `cscwd_router/code/`，旧 `gazelle` 包只 import 不改。

## 5. 数据、路径与工作流

- 数据：`fb@3090.lab:/newhome/fb/dataset/gazefollow_extended`（GazeFollow train/test）。
- 服务器代码仓库：`/home/fb/src/paper/gazelleV1`。
- Python：`/home/fb/anaconda3/envs/py310/bin/python`。
- git 双向：本地写码 → push → 服务器 pull → nohup 手动跑 → 小结果（JSON/MD）git 回传；大文件（.npy/.pt）不进 git，只在 RESULT_PROVENANCE.md 记路径+SHA。
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
- **下一步**：用户按第 11 节手动执行官方权重上的 synthetic 3+5+5 smoke，再运行独立 synthetic checkpoint 评测并回传小报告。通过后登记 T3 验收，再进入 T4 真实 GazeFollow 主实验与效率矩阵。

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

先按第 12 节提交/push，服务器 pull 后由用户运行。该命令使用官方 ViT-B backbone 和官方 decoder，但只生成合成 RGB/头框/目标；每个 epoch 使用同一批两图三人，共执行规格规定的 3+5+5 个单 batch epoch。`--protocol synthetic` 不读取 `--data-path`。

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

真实数据接口已经实现：`--protocol holdout` 对 train 做稳定图像分组，并在各 decoder 阶段末尾读取 holdout；`--protocol full_train --val-fraction 0` 不构造 eval loader，只保存固定 final。`--split official_test` 只接受 full-train 的 fixed-final baseline/K50/matched checkpoint。真实数据运行命令在 T3 验收后随 T4 发布。

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
