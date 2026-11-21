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
- **下一步**：用户按第 7 节在服务器手动执行真实 ViT-B checkpoint 检查，回传 JSON 和 provenance。检查通过后登记 T1 的服务器验收，再进入 T2 Router 适配；当前未开始 T2。

## 7. T1 服务器运行与结果回传

本节为待用户手动执行的命令。脚本只读取官方权重并生成合成 RGB 输入/头框，不读取 GazeFollow 数据，不训练，也不产生 AUC/L2 或效率结论。

先在本机提交第 8 节的文件并 push；服务器在本分支 pull 后运行：

```bash
cd /home/fb/src/paper/gazelleV1
git switch codex/coverage-aware-router
git pull --ff-only origin codex/coverage-aware-router
mkdir -p /home/fb/src/paper/gazelleV1/logs/cscwd_router
mkdir -p /home/fb/src/paper/gazelleV1/cscwd_router/results

CUDA_VISIBLE_DEVICES=0 setsid nohup /home/fb/anaconda3/envs/py310/bin/python -u \
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
