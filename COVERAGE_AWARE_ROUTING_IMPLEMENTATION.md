# Coverage-Aware Spatial Routing：实施计划与训练命令

## 0. 当前状态

- 基线分支：`v1`，基线提交 `03974fa`。
- 本 idea 分支：`codex/coverage-aware-router`。
- 首版目标：先完成 **固定 25% token 预算** 的闭环，证明 gaze-aware support 能在保留精度的同时减少 DINOv3 后半段的真实计算。
- 首版默认模型：DINOv3 ViT-B/16，输入 `512×512`，patch 网格 `32×32=1024`。
- 首版默认路由位置：`route_after_block=5`。该参数是从 0 开始的 block 下标，即 blocks 0–5 稠密执行，blocks 6–11 稀疏执行。
- 首版有意使用 `spatial_prior=none` 和 `fusion=raw_concat`，不同时叠加 GGSF/SASA，避免无法判断收益来自哪里。

本地机器只用于修改和静态检查。真实 DINOv3、CUDA、训练和速度测试均由服务器 `fb@3090.lab` 在拉取本分支后执行。

### 0.1 进度审计（截至 clean benchmark）

- **P0 工程闭环已完成**：dense/sparse 路径、Top-K、RoPE gather、
  early-fill scatter、训练/评测/checkpoint 和 benchmark 均已实现。
- **P1 support pilot 已完成一个 seed**：只训练 `463,105` 参数的 router；
  K25 exact-point coverage 达到 `82.57%`，K50 为 `94.32%`。
- **P2 已完成 decoder-only proof-of-concept**：成功配置固定 router 和
  整个 DINO，只训练 `3,416,576` 参数的 decoder。K50 最佳完整 tuple
  为 `0.95231 / 0.11003 / 0.05007`；K25 按 Avg L2 选模的 tuple 为
  `0.94654 / 0.11131 / 0.05106`。
- router、decoder 和 DINO suffix 同时训练的 K25 15-epoch run 已失败；
  成功配置中的 DINO suffix **从未微调**，不能表述成成功的端到端训练。
- 当前真实 latency 收益约为 `0–2%` 且无显存收益，效率证据尚未闭环。
- validation 选模、多 seed、关键消融、VAT/多人共享和跨数据集尚未完成。

按完成内容估计：工程原型约 `70–75%`，论文证据闭环约 `35–40%`，
综合进度约 **50%**。这里的百分比是工作量/证据完整度估计，不是方法
成功概率。

## 1. 要解决的问题与论文故事

### 1.1 不是只针对某一篇 object-aware 论文

当前很多 gaze target estimation 方法虽然 decoder 形式不同，但有一个共同假设：**整张图的所有 patch 都值得经过同样深度、同样昂贵的语义推理**。以当前 Gaze-LLE/DINOv3 路径为例，人的 bbox 只在完整 DINO 推理之后进入 decoder；因此 DINO 的计算分配与“谁在看、可能看哪里、当前证据有多不确定”无关。

这会产生一个结构性错配：

- gaze target 往往只落在少量空间区域，但全部 1024 个 patch 都进入深层 blocks；
- 简单样本与困难样本使用相同计算；
- 固定射线或固定角度 FOV 虽能缩小搜索范围，却会把不确定性过早压成一个方向，方向错时容易漏掉真正目标；
- 只在 decoder 上乘 mask 不会减少 DINOv3 的主要计算，不能形成可信的效率贡献。

因此本文问题可以写成：

> Existing gaze-target models perform person-agnostic dense scene reasoning before incorporating the queried person's evidence. Can the model first predict a high-recall, person-conditioned spatial support, and allocate deep visual reasoning only to that support without losing the target under uncertainty?

### 1.2 这里的 mask 到底是什么

这里的 mask 不是：

- 检测器标出的“人 bbox”；
- YOLO/SAM 得到的物体分割；
- 一个固定开角的 FOV cone；
- 最终 gaze heatmap 的第二份复制。

它表示的是：**为了继续进行昂贵深层推理，当前人可能注视到的高召回候选空间 support**。它可以是不规则、多峰的；bbox 只用来说明“谁在看”，而不是指定“看什么”。

### 1.3 方法核心

在 DINOv3 中间层得到稠密特征后，对每个人预测 support：

\[
S_i(p)=\operatorname{softmax}_{p}\left(f(F_r(p),\operatorname{Pool}(F_r,b_i),\Delta(p,b_i)) / T\right),
\]

其中 `F_r(p)` 是位置语义，`Pool(F_r,b_i)` 是 bbox 内的头部外观，`Δ(p,b_i)` 是相对位置和 bbox 尺度。该预测没有 cone 形状约束。

一张图有多个人时，不为每个人重复运行 DINO suffix，而是构造对人数稳定的共享 max-union：

\[
S_{img}(p)=\max_i S_i(p).
\]

然后选择 `Top-K(S_img)`，同时优先保留 head footprint、CLS token、4 个 DINOv3 storage tokens 和少量均匀分布的 escape tokens。被选 patch 及其原始空间位置对应的 RoPE 一起进入后续 blocks。

深层稀疏输出再 scatter 回 `32×32`：选中位置使用深层结果，未选位置使用路由层 early-exit 特征。这样原 v1 四层 decoder 可以不改结构地消费完整特征图。

```text
image
  └─ DINO blocks 0 ... 5：1024 个 patch，稠密
       ├─ head appearance + local scene + geometry
       └─ free-form support → per-image union → Top-K（默认约 256）
            └─ DINO blocks 6 ... 11：K 个 patch + 全部 special tokens，真实稀疏
                 └─ scatter 到 32×32（未选位置由 block 5 回填）
                      └─ 原 v1 multi-layer decoder → gaze heatmap / in-out
```

这回答了“router 是否只针对 decoder”：**不是**。`backbone_sparse` 中 DINOv3 后半段实际只处理保留 token；只有 `support_pilot` 为了先验证假设而运行完整 DINO。

## 2. 已实现代码

| 文件 | 作用 |
|---|---|
| `gazelle/routing/router.py` | person-conditioned 空间分布、多人 max-union、精确固定 Top-K、head/escape token 优先级 |
| `gazelle/routing/backbone.py` | DINOv3 稠密 prefix、patch/RoPE 同步 gather、稀疏 suffix、early-exit scatter |
| `gazelle/routing/model.py` | `support_pilot` 与 `backbone_sparse` 两种路径，并复用 v1 decoder |
| `gazelle/routing/losses.py` | coverage、budget、entropy 损失及 soft/hard coverage 指标 |
| `gazelle/model.py` | 新增 `forward_from_features`，不改变原始 dense forward 行为 |
| `scripts/train_coverage_router.py` | GazeFollow/VAT 统一训练、AMP、梯度累积、分组学习率、初始化/断点恢复 |
| `scripts/eval_coverage_router.py` | 从 checkpoint 的 `model_config` 自动重建并完整评测 |
| `scripts/benchmark_coverage_router.py` | 同 checkpoint 比较 dense/K100/K50/K25 的同步 latency、throughput 与 CUDA 峰值显存 |
| `scripts/smoke_coverage_router.py` | 检查 keep=100% 时稀疏路径与标准 dense DINO 输出等价，并检查 25% 路径形状 |
| `tests/test_coverage_router.py` | support/union/预算/coverage loss 测试 |
| `tests/test_routed_backbone_utils.py` | token gather、RoPE gather、scatter 测试 |

### 2.1 损失

总损失为：

\[
L=\lambda_{hm}L_{hm}+\lambda_{out}L_{out}
+\lambda_{cov}L_{cov}+\lambda_{budget}L_{budget}+\lambda_{ent}L_{ent}.
\]

- `L_cov` 最大化 GT gaze 分布落入 soft support 的概率质量，目标是高召回，不要求 router 逐像素复刻最终 heatmap。
- `L_budget` 用空间分布熵对应的 effective support size 约束 soft union 接近目标预算。
- `L_ent` 是可选的额外锐化项；首版默认权重为 0，避免与 budget 目标重复。
- hard Top-K 不可导，所以主 heatmap loss不能单独训练 router；`L_cov` 是必要的独立梯度路径。
- VAT 的 out-of-frame 样本不计算 heatmap/coverage loss，但仍计算 in/out loss；代码也处理了一个 batch 全部 out-of-frame 的情况。

### 2.2 Checkpoint 规则

新 checkpoint 包含：

- `model_config`：模型、stage、路由 block、预算、router 尺寸与温度；
- `train_config`：训练参数；
- `model_state`：所有非 backbone 参数，以及所有实际解冻的 DINO suffix 参数；
- optimizer、scheduler、AMP scaler、随机数状态；
- epoch、global step、指标和 git commit。

`--init_ckpt` 只初始化模型权重，兼容 v1 的裸 `state_dict` 和本方法的结构化 checkpoint。`--resume` 用于恢复完整训练状态。冻结的 DINO 参数不重复写进 checkpoint，而是在加载时从 `./checkpoints/*_pretrain.pth` 重建；一旦使用 `--train_backbone_after_router`，被训练的 suffix block 一定会保存。

## 3. 分阶段实施计划

### P0：结构正确性，必须先过

1. Python 静态编译和 router 单元测试。
2. `keep_ratio=1.0` 时，标准 dense backbone 与 routed prefix/suffix 的四层输出在数值容差内一致。
3. 检查每个样本的 RoPE 与原 patch index 同步 gather。
4. 检查 ViT-B 默认保留 CLS + 4 storage tokens；这些 token 永不裁剪。
5. 25% 路径完成一次前向、反向和 checkpoint reload，无 NaN/OOM。

只要 100% 等价测试不通过，就不能开始长训练。

### P1：Support pilot，先验证假设

`support_pilot` 完整运行 DINO，只训练 router，冻结现有 gaze decoder。它回答：用中层 head appearance、场景与几何，能否在 25% 空间预算下覆盖真实 gaze target？

记录：

- soft coverage；
- hard Top-K coverage；
- entropy 对应的 effective support ratio；
- actual keep ratio；
- support 可视化，特别是小头、遮挡、背头和多候选目标样本。

Go/No-Go 不使用任意的“必须提升 10%”。Go 的证据应是：

- 学习后的 25% hard coverage 稳定高于随机 25% 和未训练 router；
- 多个 seed 的趋势一致，而非单次波动；
- support 不是只记住 head 周围或固定扇形；
- 困难样本中能出现合理的宽/多峰 support。

如果 support 与随机选择相近，或总是塌缩成固定位置，应先停在这里分析，不进入稀疏微调。

### P2：固定预算真实稀疏模型

先用 P1 的 `best_coverage.pt` 初始化 `backbone_sparse`：

1. smoke：4 个 train batch + 4 个 eval batch；
2. 冻结 DINO suffix 的短 pilot，确认 decoder 能适应 early-exit scatter；
3. 解冻 route 后的 suffix blocks，用 `1e-6` 小学习率完整训练；
4. 对比 dense v1、100% routed control、50% 和 25%。

主要结论必须同时报告两轴：

- 准确率：GazeFollow AUC/Avg L2/Min L2，VAT AUC/L2/In-out AP；
- 效率：真实 batch=1 latency、吞吐、峰值显存、suffix token 数和端到端 FLOPs。

25% 并不意味着端到端必然加速 4 倍：prefix 仍然稠密，MLP 项与 special tokens 也存在。论文只能报告实测值。

### P3：必要消融

固定预算成立后再做：

- 路由位置：ViT-B blocks `2 / 5 / 8`；
- 保留比例：`12.5% / 25% / 50% / 75% / 100%`；
- router 输入：geometry only、head appearance + geometry、完整 scene + head + geometry；
- 回填：zero fill 与 route-layer early-exit fill；
- escape tokens：`0 / 8 / 16`；
- 多人：per-person 重复 suffix 与 per-image union suffix；
- 冻结 suffix 与小学习率微调 suffix。

### P4：自适应预算，暂不混入首版

只有固定预算显示明确的 accuracy-efficiency Pareto 后，才加入离散预算 buckets，例如 `{12.5%, 25%, 50%}`。不要直接预测任意 K，否则每个样本长度不同会降低 GPU batching 效率，也更难复现实验。

### P5：跨数据集与多人场景

- GazeFollow 建立方法有效性。
- VAT 做视频域和 out-of-frame 泛化。
- 当前 `GazeDataset` 仍按“一人一个样本”展开；模型本身已经支持一图多人 union，但要证明 scene-once 的多人效率，还需新增 grouped-frame loader，确保同一帧的多人共享一次 DINO prefix/suffix。
- 最终再做 hard-case、人数分桶、head size/遮挡/边界目标分桶和 paired bootstrap。

## 4. 服务器目录与前置条件

以下命令都假设服务器仓库为：

```bash
/home/fb/src/paper/gazelleV1
```

Python 固定使用：

```bash
/home/fb/anaconda3/envs/py310/bin/python
```

先在服务器拉取本分支，然后进入仓库根目录。DINOv3 使用相对路径，因此不能在其他目录启动脚本。

```bash
cd /home/fb/src/paper/gazelleV1
mkdir -p logs/coverage_router experiments/coverage_router
```

下面统一显式指定 `CUDA_VISIBLE_DEVICES=0`。若 0 号卡正在使用，可整体替换为另一张物理卡；脚本内部仍使用逻辑设备 `cuda`。

## 5. 建议按顺序运行的命令

### 5.1 单元测试

```bash
CUDA_VISIBLE_DEVICES=0 \
nohup /home/fb/anaconda3/envs/py310/bin/python -m pytest -q \
  tests/test_coverage_router.py \
  tests/test_routed_backbone_utils.py \
  > logs/coverage_router/unit_tests.log 2>&1 < /dev/null &
```

查看结果：

```bash
tail -f logs/coverage_router/unit_tests.log
```

### 5.2 DINOv3 100% 等价与 25% 结构 smoke

```bash
CUDA_VISIBLE_DEVICES=0 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
nohup /home/fb/anaconda3/envs/py310/bin/python -u \
  scripts/smoke_coverage_router.py \
  --backbone dinov3_vitb16 \
  --route_after_block 5 \
  --keep_ratio 0.25 \
  --batch_size 2 \
  --device cuda \
  --amp \
  > logs/coverage_router/backbone_smoke.log 2>&1 < /dev/null &
```

日志中必须同时出现 `PASS dense equivalence` 和 `PASS sparse shapes`。

### 5.3 一小时内的小规模 support smoke

```bash
CUDA_VISIBLE_DEVICES=0 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
nohup /home/fb/anaconda3/envs/py310/bin/python -u \
  scripts/train_coverage_router.py \
  --dataset gazefollow \
  --model gazelle_dinov3_vitb16 \
  --data_path /newhome/fb/dataset/gazefollow_extended \
  --init_ckpt /home/fb/src/paper/gazelleV1/experiments/train_gazefollow_baseline/2026-02-10_13-03-16/epoch_14.pt \
  --router_stage support_pilot \
  --route_after_block 5 \
  --keep_ratio 0.25 \
  --escape_tokens 8 \
  --heatmap_loss_weight 0 \
  --router_coverage_weight 1.0 \
  --router_budget_weight 0.05 \
  --router_entropy_weight 0.0 \
  --lr_router 1e-3 \
  --lr_decoder 0 \
  --max_epochs 1 \
  --batch_size 2 \
  --n_workers 0 \
  --max_train_batches 4 \
  --max_eval_batches 4 \
  --amp \
  --wandb_mode disabled \
  --run_dir /home/fb/src/paper/gazelleV1/experiments/coverage_router/support_smoke \
  > logs/coverage_router/support_smoke.log 2>&1 < /dev/null &
```

### 5.4 完整 support pilot

```bash
CUDA_VISIBLE_DEVICES=0 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
nohup /home/fb/anaconda3/envs/py310/bin/python -u \
  scripts/train_coverage_router.py \
  --dataset gazefollow \
  --model gazelle_dinov3_vitb16 \
  --data_path /newhome/fb/dataset/gazefollow_extended \
  --init_ckpt /home/fb/src/paper/gazelleV1/experiments/train_gazefollow_baseline/2026-02-10_13-03-16/epoch_14.pt \
  --router_stage support_pilot \
  --route_after_block 5 \
  --keep_ratio 0.25 \
  --router_temperature 1.0 \
  --escape_tokens 8 \
  --heatmap_loss_weight 0 \
  --router_coverage_weight 1.0 \
  --router_budget_weight 0.05 \
  --router_entropy_weight 0.0 \
  --lr_router 1e-3 \
  --lr_decoder 0 \
  --max_epochs 3 \
  --batch_size 16 \
  --n_workers 8 \
  --amp \
  --wandb_project GazeRoute \
  --wandb_mode online \
  --exp_name gf_support_k025_seed3106 \
  --run_dir /home/fb/src/paper/gazelleV1/experiments/coverage_router/gf_support_k025_seed3106 \
  --seed 3106 \
  > logs/coverage_router/gf_support_k025_seed3106.log 2>&1 < /dev/null &
```

该阶段使用完整 DINO，不能把耗时写成最终加速结果。主要查看 `best_coverage.pt` 和 `history.jsonl`。

### 5.5 真实稀疏路径的 4-batch smoke

```bash
CUDA_VISIBLE_DEVICES=0 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
nohup /home/fb/anaconda3/envs/py310/bin/python -u \
  scripts/train_coverage_router.py \
  --dataset gazefollow \
  --model gazelle_dinov3_vitb16 \
  --data_path /newhome/fb/dataset/gazefollow_extended \
  --init_ckpt /home/fb/src/paper/gazelleV1/experiments/coverage_router/gf_support_k025_seed3106/best_coverage.pt \
  --router_stage backbone_sparse \
  --route_after_block 5 \
  --keep_ratio 0.25 \
  --escape_tokens 8 \
  --train_backbone_after_router \
  --lr_router 3e-4 \
  --lr_decoder 1e-4 \
  --lr_backbone 1e-6 \
  --max_epochs 1 \
  --batch_size 2 \
  --n_workers 0 \
  --max_train_batches 4 \
  --max_eval_batches 4 \
  --amp \
  --wandb_mode disabled \
  --run_dir /home/fb/src/paper/gazelleV1/experiments/coverage_router/sparse_smoke \
  > logs/coverage_router/sparse_smoke.log 2>&1 < /dev/null &
```

### 5.6 GazeFollow 完整稀疏训练

```bash
CUDA_VISIBLE_DEVICES=0 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
nohup /home/fb/anaconda3/envs/py310/bin/python -u \
  scripts/train_coverage_router.py \
  --dataset gazefollow \
  --model gazelle_dinov3_vitb16 \
  --data_path /newhome/fb/dataset/gazefollow_extended \
  --init_ckpt /home/fb/src/paper/gazelleV1/experiments/coverage_router/gf_support_k025_seed3106/best_coverage.pt \
  --router_stage backbone_sparse \
  --route_after_block 5 \
  --keep_ratio 0.25 \
  --router_temperature 1.0 \
  --escape_tokens 8 \
  --train_backbone_after_router \
  --router_coverage_weight 1.0 \
  --router_budget_weight 0.05 \
  --router_entropy_weight 0.0 \
  --lr_router 3e-4 \
  --lr_decoder 1e-4 \
  --lr_backbone 1e-6 \
  --max_epochs 15 \
  --batch_size 8 \
  --grad_accum_steps 4 \
  --n_workers 8 \
  --amp \
  --clip_grad_norm 1.0 \
  --wandb_project GazeRoute \
  --wandb_mode online \
  --exp_name gf_sparse_b5_k025_seed3106 \
  --run_dir /home/fb/src/paper/gazelleV1/experiments/coverage_router/gf_sparse_b5_k025_seed3106 \
  --seed 3106 \
  > logs/coverage_router/gf_sparse_b5_k025_seed3106.log 2>&1 < /dev/null &
```

若 3090 OOM，先把 `--batch_size 8` 改为 `4`，并把 `--grad_accum_steps 4` 改为 `8`，保持有效 batch 不变。不要先降低输入分辨率。

### 5.6.1 5.6 结果判定：当前 joint recipe No-Go

`gf_sparse_b5_k025_seed3106` 的最佳结果仍显著差于 dense v1：

| checkpoint / epoch | AUC ↑ | Avg L2 ↓ | Min L2 ↓ | GT point coverage ↑ | hard coverage ↑ |
|---|---:|---:|---:|---:|---:|
| dense v1 / support pilot | 0.95746 | 0.10268 | 0.04405 | 0.82573 | 0.66334 |
| sparse epoch 0（最佳 L2） | 0.85826 | 0.23706 | 0.16240 | 0.70840 | 0.57764 |
| sparse epoch 2（最佳 AUC） | 0.86679 | 0.25617 | 0.18204 | 0.76353 | 0.64370 |
| sparse epoch 7（最佳 coverage） | 0.76745 | 0.40437 | 0.32595 | 0.80143 | 0.67309 |
| sparse epoch 14 | 0.59460 | 0.48882 | 0.40331 | 0.79059 | 0.66887 |

因此不能继续把 5.7、VAT 或正式消融当作论文实验。这个结果否定的是“从第一个 epoch 起同时更新 router、decoder 和 DINO suffix”的训练 recipe，还没有单独否定 coverage-aware routing：

- coverage 后期恢复，而 gaze 精度继续恶化，问题不只是目标没有进入 Top-K；
- baseline decoder 原来消费 dense blocks `2/5/8/11`，K=25% 后 blocks `8/11` 的 75% 位置突然改成 block 5 early-exit 回填，输入分布发生大幅变化；
- router 同时更新会不断改变离散 Top-K，decoder 和 4253 万参数的 suffix 在追逐一个非平稳输入；
- effective support 多数只有约 4%–10%，远小于实际 K=25%。当前 `router_budget_weight=0.05` 对约 0.03 量级的 budget loss 贡献只有约 0.0015，几乎没有约束剩余低分 token 的排序稳定性。

先完成下面两个诊断阶段。只有分阶段训练恢复出可接受的 accuracy–compute 点，才回到 5.7。

### 5.6.2 不训练的 sparse-init 对照

评测脚本的两个 override 只改变内存中的评测配置，不修改 checkpoint。三个命令必须在同一张 GPU 上**顺序执行**，不要同时启动。

先做 K=100% 完整路径对照：

```bash
CUDA_VISIBLE_DEVICES=0 \
nohup /home/fb/anaconda3/envs/py310/bin/python -u \
  scripts/eval_coverage_router.py \
  --checkpoint /home/fb/src/paper/gazelleV1/experiments/coverage_router/gf_support_k025_seed3106/best_coverage.pt \
  --dataset gazefollow \
  --data_path /newhome/fb/dataset/gazefollow_extended \
  --router_stage_override backbone_sparse \
  --keep_ratio_override 1.0 \
  --batch_size 16 \
  --n_workers 8 \
  --amp \
  --output /home/fb/src/paper/gazelleV1/experiments/coverage_router/gf_sparse_init_controls/k100.json \
  > logs/coverage_router/gf_sparse_init_k100.log 2>&1 < /dev/null &
```

完成后分别做 K=50% 和 K=25%：

```bash
CUDA_VISIBLE_DEVICES=0 \
nohup /home/fb/anaconda3/envs/py310/bin/python -u \
  scripts/eval_coverage_router.py \
  --checkpoint /home/fb/src/paper/gazelleV1/experiments/coverage_router/gf_support_k025_seed3106/best_coverage.pt \
  --dataset gazefollow \
  --data_path /newhome/fb/dataset/gazefollow_extended \
  --router_stage_override backbone_sparse \
  --keep_ratio_override 0.50 \
  --batch_size 16 \
  --n_workers 8 \
  --amp \
  --output /home/fb/src/paper/gazelleV1/experiments/coverage_router/gf_sparse_init_controls/k050.json \
  > logs/coverage_router/gf_sparse_init_k050.log 2>&1 < /dev/null &
```

```bash
CUDA_VISIBLE_DEVICES=0 \
nohup /home/fb/anaconda3/envs/py310/bin/python -u \
  scripts/eval_coverage_router.py \
  --checkpoint /home/fb/src/paper/gazelleV1/experiments/coverage_router/gf_support_k025_seed3106/best_coverage.pt \
  --dataset gazefollow \
  --data_path /newhome/fb/dataset/gazefollow_extended \
  --router_stage_override backbone_sparse \
  --keep_ratio_override 0.25 \
  --batch_size 16 \
  --n_workers 8 \
  --amp \
  --output /home/fb/src/paper/gazelleV1/experiments/coverage_router/gf_sparse_init_controls/k025.json \
  > logs/coverage_router/gf_sparse_init_k025.log 2>&1 < /dev/null &
```

判定：

- K=100% 必须在数值容差内复现 dense baseline；否则先查完整模型加载或 sparse scatter 路径，不能训练。
- K=25% 是“任何更新前”的真实精度。若它已经接近 5.6 epoch 0，主要问题是突变后的 sparse 表示；若它接近 dense、训练后才崩，则主要问题是 joint optimization。
- K=50% 用来区分 K=25% 是否过激，但不要直接复制 5.6 的 joint recipe 长跑。

### 5.6.3 K50：固定 router 和整个 DINO 的 decoder curriculum

5.6.2 已确认：

- K100 完全复现 dense baseline，完整 routed 路径没有通用实现错误；
- K50 虽然 GT point coverage 已有 94.32%，但仍只有 `0.88956 / 0.24057 / 0.17511`；
- K25 为 `0.84675 / 0.26396 / 0.19451`。25%→50% 带来很大的 coverage 增长，却只带来很小的定位改善。

所以不要直接长跑 K25。先用 K50 作为当前 single-mask + early-fill 结构的高覆盖可适配性上界。此时只训练 decoder；router、DINO prefix 和 DINO suffix 全部冻结。前三个 epoch 的训练预算依次为 `100% → 75% → 50%`，后两个 epoch 保持 50%；每个 epoch 的完整评测都固定使用最终 K=50%：

```bash
CUDA_VISIBLE_DEVICES=0 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
nohup /home/fb/anaconda3/envs/py310/bin/python -u \
  scripts/train_coverage_router.py \
  --dataset gazefollow \
  --model gazelle_dinov3_vitb16 \
  --data_path /newhome/fb/dataset/gazefollow_extended \
  --init_ckpt /home/fb/src/paper/gazelleV1/experiments/coverage_router/gf_support_k025_seed3106/best_coverage.pt \
  --router_stage backbone_sparse \
  --route_after_block 5 \
  --keep_ratio 0.50 \
  --router_warmup_epochs 3 \
  --router_temperature 1.0 \
  --escape_tokens 8 \
  --heatmap_loss_weight 1.0 \
  --router_coverage_weight 0 \
  --router_budget_weight 0 \
  --router_entropy_weight 0 \
  --lr_router 0 \
  --lr_decoder 1e-4 \
  --lr_backbone 0 \
  --max_epochs 5 \
  --batch_size 8 \
  --grad_accum_steps 4 \
  --n_workers 8 \
  --amp \
  --clip_grad_norm 1.0 \
  --wandb_project GazeRoute \
  --wandb_mode online \
  --exp_name gf_sparse_fixedrouter_decoder_k050_seed3106 \
  --run_dir /home/fb/src/paper/gazelleV1/experiments/coverage_router/gf_sparse_fixedrouter_decoder_k050_seed3106 \
  --seed 3106 \
  > logs/coverage_router/gf_sparse_fixedrouter_decoder_k050_seed3106.log 2>&1 < /dev/null &
```

不要在这个阶段加入 `--train_backbone_after_router`。判定使用从 K50 zero-shot 到 dense 的 gap recovery，而不是要求任意“提升 10%”：

| 判定 | AUC ↑ | Avg L2 ↓ | Min L2 ↓ | 下一步 |
|---|---:|---:|---:|---|
| 少于 50% gap recovery | < 0.92351 | > 0.17162 | > 0.10958 | 停止当前 early-fill recipe，先改 feature adapter/context preservation |
| 50%–80% | 0.92351–0.94388 | 0.13025–0.17162 | 0.07026–0.10958 | 只做一次短 K25 curriculum probe |
| 至少 80% | ≥ 0.94388 | ≤ 0.13025 | ≤ 0.07026 | K50 明确可行，再测试 K25 和固定-router suffix 微调 |

三个指标应整体落入同一档，同时最后两轮不能相对最佳轮掉 AUC 超过 0.01 或增加 L2 超过 0.015。若指标跨档，按较差的一档决策。

该阶段若能稳定恢复，再从 `best_avg_l2.pt` 初始化，继续保持 router 冻结并以 `1e-6` 解冻 suffix；最后才考虑用更小学习率重新打开 router。若不能恢复，下一步不是继续调学习率，而是用 dense-context masked-fill 对照拆分“混合深度回填”和“稀疏 token 丢失全局语境”这两个原因。

5.6.3 实际结果为：

| epoch | 训练 K | AUC ↑ | Avg L2 ↓ | Min L2 ↓ |
|---:|---:|---:|---:|---:|
| 0 | 100% | 0.88644 | 0.23860 | 0.17296 |
| 1 | 75% | 0.95132 | 0.11113 | 0.05075 |
| 2 | 50% | **0.95231** | **0.11003** | **0.05007** |
| 3 | 50% | 0.95079 | 0.11480 | 0.05307 |
| 4 | 50% | 0.95084 | 0.11497 | 0.05315 |

最佳 epoch 2 相对 K50 zero-shot 到 dense 的 gap recovery 分别为：

- AUC：92.41%；
- Avg L2：94.67%；
- Min L2：95.41%。

最后两轮相对最佳轮只下降约 0.0015 AUC、增加约 0.0049 Avg L2 和 0.0031 Min L2，通过稳定性要求，但已有轻微过拟合。K50 路径因此为强 Go：主要缺口确实来自 decoder 对 mixed-depth feature distribution 不适配，而不是 sparse representation 完全不可恢复。后续使用 epoch 2 的 `best_avg_l2.pt`（它同时也是本次 best AUC/Min L2），不要使用 `last.resume.pt`。

### 5.6.4 K25：固定 router 和整个 DINO 的 decoder curriculum

K50 已通过 80% recovery 门槛，可以独立测试原始 K25 目标。为了让 K25 与 K50 的结论可比较，本阶段仍从 support checkpoint 出发，不从 K50 decoder checkpoint 继续训练；router 和整个 DINO 保持冻结。训练预算为 `100% → 75% → 50% → 25% → 25% → 25%`，每轮评测固定 K25：

```bash
CUDA_VISIBLE_DEVICES=0 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
nohup /home/fb/anaconda3/envs/py310/bin/python -u \
  scripts/train_coverage_router.py \
  --dataset gazefollow \
  --model gazelle_dinov3_vitb16 \
  --data_path /newhome/fb/dataset/gazefollow_extended \
  --init_ckpt /home/fb/src/paper/gazelleV1/experiments/coverage_router/gf_support_k025_seed3106/best_coverage.pt \
  --router_stage backbone_sparse \
  --route_after_block 5 \
  --keep_ratio 0.25 \
  --router_warmup_epochs 4 \
  --router_temperature 1.0 \
  --escape_tokens 8 \
  --heatmap_loss_weight 1.0 \
  --router_coverage_weight 0 \
  --router_budget_weight 0 \
  --router_entropy_weight 0 \
  --lr_router 0 \
  --lr_decoder 1e-4 \
  --lr_backbone 0 \
  --max_epochs 6 \
  --batch_size 8 \
  --grad_accum_steps 4 \
  --n_workers 8 \
  --amp \
  --clip_grad_norm 1.0 \
  --wandb_project GazeRoute \
  --wandb_mode online \
  --exp_name gf_sparse_fixedrouter_decoder_k025_seed3106 \
  --run_dir /home/fb/src/paper/gazelleV1/experiments/coverage_router/gf_sparse_fixedrouter_decoder_k025_seed3106 \
  --seed 3106 \
  > logs/coverage_router/gf_sparse_fixedrouter_decoder_k025_seed3106.log 2>&1 < /dev/null &
```

K25 zero-shot 到 dense 的判定门槛：

| 判定 | AUC ↑ | Avg L2 ↓ | Min L2 ↓ | 下一步 |
|---|---:|---:|---:|---|
| 少于 50% gap recovery | < 0.90211 | > 0.18332 | > 0.11928 | 保留 K50，停止 K25 early-fill 路径 |
| 50%–80% | 0.90211–0.93532 | 0.13493–0.18332 | 0.07414–0.11928 | K50 作为主 Pareto 点；仅在速度收益足够大时微调 K25 suffix |
| 至少 80% | ≥ 0.93532 | ≤ 0.13493 | ≤ 0.07414 | K25 强 Go，进入固定-router suffix 微调与真实速度测试 |

同样要求三个指标整体落入同一档，并检查最后两轮稳定性。

5.6.4 实际结果为：

| epoch | 训练 K | AUC ↑ | Avg L2 ↓ | Min L2 ↓ |
|---:|---:|---:|---:|---:|
| 0 | 100% | 0.84226 | 0.26255 | 0.19293 |
| 1 | 75% | 0.94654 | **0.11131** | **0.05106** |
| 2 | 50% | 0.94862 | 0.11203 | 0.05174 |
| 3 | 25% | **0.94886** | 0.11361 | 0.05208 |
| 4 | 25% | 0.94696 | 0.11290 | 0.05274 |
| 5 | 25% | 0.94848 | 0.11298 | 0.05168 |

必须注意，`summary.json` 中的逐指标最佳值不属于同一个 checkpoint：

- `best_auc.pt` 是 epoch 3，完整 tuple 为 `0.94886 / 0.11361 / 0.05208`；
- `best_avg_l2.pt` 与 `best_min_l2.pt` 是 epoch 1，完整 tuple 为 `0.94654 / 0.11131 / 0.05106`；
- `last.resume.pt` 是 epoch 5，完整 tuple 为 `0.94848 / 0.11298 / 0.05168`。

按照预先采用的 Avg L2 选模规则，应使用 epoch 1 的 `best_avg_l2.pt`。以该单一 checkpoint 计算，K25 zero-shot 到 dense 的 gap recovery 为 AUC 90.14%、Avg L2 94.64%、Min L2 95.34%，三项均为 Strong Go。epochs 3–5 真正使用 K25 训练后的波动范围也只有 0.00190 AUC、0.00071 Avg L2 和 0.00106 Min L2，没有失稳。

K25 相对 K50 的最佳指标只差约 `-0.00345 AUC / +0.00128 Avg L2 / +0.00100 Min L2`。因此暂不解冻 suffix；先用真实速度决定 K25 还是 K50 是主 Pareto 点。

最终多 seed 不能继续从 GazeFollow test 上分别挑每个指标的最佳轮并拼成结果。正式实验前必须固定单一 checkpoint 规则/epoch，或建立 validation split，然后在 test 上一次性报告该 checkpoint 的完整 metric tuple。

### 5.6.5 K100/K50/K25 真实 CUDA benchmark

Benchmark 使用同一个 K25 `best_avg_l2.pt`，以排除权重差异；比较：

- `dense`：真正的原始 GazeLLE 路径，完全绕过 router；
- `support`：完整 dense DINO + router；
- `k100`：带 Top-K、gather/scatter 的 routed 实现开销对照；
- `k50` 和 `k25`：真实稀疏 suffix。

单请求 latency 每次 forward 都执行 CUDA synchronize；连续 throughput 只在一组 forward 的前后 synchronize。输入预先放在 GPU，checkpoint load、数据读取和 image H2D 不计入模型时间；router 在 forward 内部正常处理 bbox，因此其开销会被计入。运行时不要同时启动其他 GPU 任务。

首轮 `benchmark_b1_amp.json` 不能作为最终效率结论。它得到 dense `22.49 ms`、K50 `23.28 ms`、K25 `23.40 ms`，但 dense steady allocated memory 为 `531.6 MiB`，routed 只有 `362.3 MiB`，相差约 `169.3 MiB`，接近 DINO 的 FP16 权重副本大小。原因是 legacy dense 参数默认 `requires_grad=True`，而 routed DINO 在训练阶段被冻结；外层 autocast 因此只缓存 dense 的 FP16 权重，routed 每轮需要重新转换权重。该差异同时污染 latency 与 memory。

修正版在 `inference_mode` 下统一把所有浮点参数设为 autocast-cache eligible。`inference_mode` 仍保证不构建 autograd graph；`requires_grad` 在这里仅用于统一 PyTorch autocast 的权重缓存策略。必须重新运行并写入 `_v2` 文件：

```bash
CUDA_VISIBLE_DEVICES=0 \
nohup /home/fb/anaconda3/envs/py310/bin/python -u \
  scripts/benchmark_coverage_router.py \
  --checkpoint /home/fb/src/paper/gazelleV1/experiments/coverage_router/gf_sparse_fixedrouter_decoder_k025_seed3106/best_avg_l2.pt \
  --variants dense support k100 k50 k25 \
  --device cuda \
  --batch_size 1 \
  --num_people 1 \
  --image_size 512 \
  --warmup_iters 50 \
  --latency_iters 200 \
  --throughput_iters 500 \
  --repeats 5 \
  --amp \
  --output /home/fb/src/paper/gazelleV1/experiments/coverage_router/gf_sparse_fixedrouter_decoder_k025_seed3106/benchmark_b1_amp_v2.json \
  > logs/coverage_router/gf_sparse_fixedrouter_decoder_k025_seed3106_benchmark_b1_amp_v2.log 2>&1 < /dev/null &
```

脚本同时写出 JSON 和 Markdown 表，报告 median/mean/p95 latency、连续推理 FPS、steady/peak/activation memory、实际 patch/special/suffix token 数，以及相对 dense 的 speedup。速度结论不使用任意百分比门槛：要求 K25 相对 dense 和 K50 的每轮 latency median 都同方向改善，且改善幅度明显大于五次 repeat 的波动；否则选择更稳定的 K50，或停止把加速作为主要 contribution。

#### 5.6.5-v2 实际结果与复测要求

`benchmark_b1_amp_v2.json` 已经通过 AMP cache 公平性检查。Dense 的
`autocast_cache_eligible_parameters / total_parameters` 为
`89,086,208 / 89,086,208`，所有 routed variant 为
`89,549,313 / 89,549,313`；两者相差的 `463,105` 个参数正是 router。
v1 中约 `169.3 MiB` 的虚假 steady-memory 差异也已消失，v2 routed
仅比 dense 多 `1.86–5.15 MiB`。

| variant | median / p95 latency | median FPS | peak memory |
|---|---:|---:|---:|
| dense | 42.891 / 44.490 ms | 24.081 | 589.0 MiB |
| support | 62.237 / 64.659 ms | 16.663 | 594.1 MiB |
| K100 | 59.803 / 60.177 ms | 16.672 | 597.7 MiB |
| K50 | 59.880 / 60.059 ms | 17.599 | 600.6 MiB |
| K25 | 59.912 / 61.160 ms | 17.311 | 600.4 MiB |

但是这次 timing **不能作为模型效率结论**。同一个 dense 路径从 v1 的
`22.490 ms` 变成 v2 的 `42.891 ms`，慢了 `90.7%`；K100 的五轮
latency median 又是 `39.376 / 59.884 / 59.903 / 59.769 / 59.880 ms`，
同一次运行内出现约 `52%` 的状态漂移。同步结果后在服务器检查到另一个
`python3` 计算进程同时占用四张 3090，`nvidia-smi pmon` 显示每张卡均为
`99% SM`。因此，v2 只能确认 benchmark 公平性修复成功，不能据此声称
router 固定开销为 `support - dense`，也不能据此给当前方法作最终
效率 No-Go。

不过，v1 和 v2 都没有观察到 K25 优于 K50：v2 中 K25 的 median latency
比 K50 慢 `0.052%`、p95 慢 `1.833%`、FPS 低 `1.634%`，峰值显存只少
`0.254 MiB`。在得到干净复测前，默认保留精度更好的 **K50**，暂停
suffix 微调、VAT 和多 seed。

下一步只做空闲单卡复测。运行前先确认 GPU 0 没有 compute process，
并且利用率持续接近 0：

```bash
nvidia-smi --id=0 \
  --query-compute-apps=pid,process_name,used_memory \
  --format=csv,noheader

nvidia-smi --id=0 \
  --query-gpu=index,utilization.gpu,memory.used,temperature.gpu,power.draw,clocks.sm,pstate \
  --format=csv
```

空闲后先按正序运行，保留 v2 原文件不覆盖：

```bash
CUDA_VISIBLE_DEVICES=0 \
nohup /home/fb/anaconda3/envs/py310/bin/python -u \
  scripts/benchmark_coverage_router.py \
  --checkpoint /home/fb/src/paper/gazelleV1/experiments/coverage_router/gf_sparse_fixedrouter_decoder_k025_seed3106/best_avg_l2.pt \
  --variants dense support k100 k50 k25 \
  --device cuda \
  --batch_size 1 \
  --num_people 1 \
  --image_size 512 \
  --warmup_iters 50 \
  --latency_iters 200 \
  --throughput_iters 500 \
  --repeats 5 \
  --amp \
  --output /home/fb/src/paper/gazelleV1/experiments/coverage_router/gf_sparse_fixedrouter_decoder_k025_seed3106/benchmark_b1_amp_v2_clean_forward.json \
  > logs/coverage_router/gf_sparse_fixedrouter_decoder_k025_seed3106_benchmark_b1_amp_v2_clean_forward.log 2>&1 < /dev/null &
```

只有正序结果满足以下条件，才运行反序确认：

- dense 与 K100 的五个 repeat 不再出现几十个百分点的漂移；
- 整次运行期间没有其他 compute process 进入 GPU 0；
- K50 或 K25 相对 dense/K100 的改善方向一致。

```bash
CUDA_VISIBLE_DEVICES=0 \
nohup /home/fb/anaconda3/envs/py310/bin/python -u \
  scripts/benchmark_coverage_router.py \
  --checkpoint /home/fb/src/paper/gazelleV1/experiments/coverage_router/gf_sparse_fixedrouter_decoder_k025_seed3106/best_avg_l2.pt \
  --variants k25 k50 k100 support dense \
  --device cuda \
  --batch_size 1 \
  --num_people 1 \
  --image_size 512 \
  --warmup_iters 50 \
  --latency_iters 200 \
  --throughput_iters 500 \
  --repeats 5 \
  --amp \
  --output /home/fb/src/paper/gazelleV1/experiments/coverage_router/gf_sparse_fixedrouter_decoder_k025_seed3106/benchmark_b1_amp_v2_clean_reverse.json \
  > logs/coverage_router/gf_sparse_fixedrouter_decoder_k025_seed3106_benchmark_b1_amp_v2_clean_reverse.log 2>&1 < /dev/null &
```

若干净的正反序结果都显示 K50/K25 不快于 dense，则效率路线正式
No-Go，下一步才是用 CUDA event / profiler 分解 dense prefix、router、
Top-K/gather/RoPE、sparse suffix、scatter/norm 和 decoder；在完成该
profile 与实现优化前不继续训练。若 routed 路径快于 dense，但 K25
仍未明显快于 K50，则用 K50 作为主 Pareto 点。

#### 5.6.5-clean 实际结果

正序 `dense → support → K100 → K50 → K25` 全程基本稳定，可用于当前
实现的初步判断：

| variant | median / p95 latency | median FPS | peak / activation memory |
|---|---:|---:|---:|
| dense | 20.532 / 21.339 ms | 48.899 | 588.955 / 57.343 MiB |
| support | 23.655 / 24.694 ms | 42.246 | 594.139 / 57.377 MiB |
| K100 | 20.198 / 21.107 ms | 49.358 | 597.650 / 64.183 MiB |
| K50 | 20.601 / 21.743 ms | 48.465 | 600.643 / 64.271 MiB |
| K25 | 20.167 / 21.120 ms | 49.557 | 600.389 / 64.112 MiB |

正序中 K25 相对 dense 的 median latency 仅降低 `1.78%`，相对同一
routed 路径的 K100 仅降低 `0.15%`；K50 反而比 K100 慢 `2.00%`。
因此，把 suffix patch token 从 1024 减到 256 尚未转化为有意义的
端到端收益。K25 activation peak 还比 dense 多 `6.77 MiB`，没有显存
优势。

反序 `K25 → K50 → K100 → support → dense` 在 K100 的第一轮之后再次
受到外部负载污染：K100 的 repeat median 从 `22.211 ms` 漂移到
`53.688 / 52.450 / 57.356 / 63.377 ms`，后续 support 和 dense 因而
不能与前两个 variant 横向比较。反序只再次观察到 K25 比 K50 快约
`2.05%`；正序对应差异为 `2.10%`。这个 K25/K50 差异方向一致，但幅度
很小，而且两次运行的绝对状态仍不同。

主效率基线必须是 `dense`，因为它代表用户不采用 router 时实际运行的
原始 GazeLLE。`support` 在完整 dense DINO 后额外运行 router，却不利用
路由结果裁剪 token，是为了训练/诊断 support 的刻意控制组；只和它比较
会把“先人为增加 router 开销、再回收一部分开销”误写成最终加速。
`K100` 才是隔离 token pruning 作用的 matched control：它与 K25 使用
同一个 router、prefix/suffix、Top-K、gather/scatter 路径，只是不删
patch token。

当前结论是：机制和精度路线可以继续，但 wall-clock efficiency 尚未
通过。可以暂缓 profiler/内核优化，不能把效率写成已完成贡献；若后续
最终仍没有真实加速，则需要改成更早 routing、直接 sparse decoder 或
其他能够消除 dense reconstruction 的实现。

> **旧版 5.7–5.9 已废弃。** 它们引用了失败的
> `gf_sparse_b5_k025_seed3106` joint checkpoint。下面保留的新 5.7 和
> 5.8 是已经完成的 staged pilot 及 matched control；当前动作见 5.9a。

### 5.7 K50 fixed-router suffix pilot

目的：从已经成功适配 mixed-depth feature 的 K50 decoder checkpoint
继续训练，但仍固定 support router；只以很小学习率微调 DINO blocks
6–11，并同时小幅更新 decoder。它与失败的 joint run 的关键区别是：

- router 固定，`lr_router=0`；
- blocks 0–5 固定，只有 blocks 6–11 可训练；
- 从成功的 K50 decoder checkpoint 初始化，而不是从 support checkpoint
  同时学习所有模块；
- 全程固定 K50，不再做预算 curriculum；
- suffix/decoder 学习率分别只有 `1e-6/1e-5`。

在服务器仓库根目录执行：

```bash
mkdir -p logs/coverage_router

CUDA_VISIBLE_DEVICES=0 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
nohup /home/fb/anaconda3/envs/py310/bin/python -u \
  scripts/train_coverage_router.py \
  --dataset gazefollow \
  --model gazelle_dinov3_vitb16 \
  --data_path /newhome/fb/dataset/gazefollow_extended \
  --init_ckpt /home/fb/src/paper/gazelleV1/experiments/coverage_router/gf_sparse_fixedrouter_decoder_k050_seed3106/best_avg_l2.pt \
  --router_stage backbone_sparse \
  --route_after_block 5 \
  --keep_ratio 0.5 \
  --escape_tokens 8 \
  --train_backbone_after_router \
  --lr_router 0 \
  --lr_decoder 1e-5 \
  --lr_backbone 1e-6 \
  --lr_inout 0 \
  --heatmap_loss_weight 1.0 \
  --router_coverage_weight 0 \
  --router_budget_weight 0 \
  --router_entropy_weight 0 \
  --router_warmup_epochs 0 \
  --weight_decay 0 \
  --max_epochs 3 \
  --batch_size 8 \
  --grad_accum_steps 4 \
  --n_workers 8 \
  --amp \
  --clip_grad_norm 1.0 \
  --wandb_project GazeRoute \
  --wandb_mode online \
  --exp_name gf_sparse_fixedrouter_suffix_k050_seed3106 \
  --run_dir /home/fb/src/paper/gazelleV1/experiments/coverage_router/gf_sparse_fixedrouter_suffix_k050_seed3106 \
  --seed 3106 \
  > logs/coverage_router/gf_sparse_fixedrouter_suffix_k050_seed3106.log 2>&1 < /dev/null &
```

启动后先检查：

```bash
tail -n 80 logs/coverage_router/gf_sparse_fixedrouter_suffix_k050_seed3106.log
```

`run_manifest.json` 中预期 trainable parameter 必须是：

```json
{
  "backbone": 42536448,
  "decoder": 3416576,
  "inout": 0,
  "router": 0
}
```

即总计 `45,953,024` 个参数；如果 router 不为 0，立即停止。三个 epoch
结束后同步以下文件：

```text
experiments/coverage_router/gf_sparse_fixedrouter_suffix_k050_seed3106/
  run_manifest.json
  history.jsonl
  summary.json
```

判断时用同一个 checkpoint 的完整 tuple，与 decoder-only K50 基准
`0.95231 / 0.11003 / 0.05007` 比较，不能拼接逐指标 best：

- 至少两个指标改善，第三个没有明显恶化：继续 validation/多 seed；
- 基本持平：停止 suffix 训练，保留更简单的 decoder-only K50；
- 明显波动或退化：该阶段 No-Go，回退 decoder-only K50。

#### 5.7 实际结果

训练范围和配置符合预期：DINO blocks 6–11 为 `42,536,448` 个可训练参数，
decoder 为 `3,416,576`，router 和 in/out 均为 0。三轮 router eval
指标完全相同，证明 router 确实固定。逐 epoch 的完整
`AUC / Avg L2 / Min L2` 为：

| epoch | AUC ↑ | Avg L2 ↓ | Min L2 ↓ |
|---:|---:|---:|---:|
| 0 | 0.953156 | 0.107457 | 0.047803 |
| 1 | 0.953693 | 0.107747 | 0.047758 |
| 2 | **0.954059** | **0.106171** | **0.046624** |

三项最优都来自 epoch 2 的同一个 checkpoint，不存在 summary envelope。
相对原 decoder-only K50，epoch 2 的 AUC 提升 `0.001754`，Avg L2
降低 `0.003858`（`3.51%`），Min L2 降低 `0.003444`（`6.88%`）。
它分别追回了 decoder-only K50 到 dense 剩余 gap 的
`34.0% / 52.5% / 57.3%`。

这是对“suffix + decoder 小学习率继续训练 recipe”的 Strong Go，但还
不能把提升归因于 suffix：5.7 同时让 decoder 以 `1e-5` 额外训练了三轮。
必须先补一个 matched decoder-only continuation control。

### 5.8 K50 decoder-only continuation control

该 control 使用与 5.7 完全相同的初始化、seed、K、decoder 学习率、
batch 和三个 epoch，唯一差异是 DINO suffix 冻结。不要添加
`--train_backbone_after_router`，并明确设置 `--lr_backbone 0`：

```bash
mkdir -p logs/coverage_router

CUDA_VISIBLE_DEVICES=0 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
nohup /home/fb/anaconda3/envs/py310/bin/python -u \
  scripts/train_coverage_router.py \
  --dataset gazefollow \
  --model gazelle_dinov3_vitb16 \
  --data_path /newhome/fb/dataset/gazefollow_extended \
  --init_ckpt /home/fb/src/paper/gazelleV1/experiments/coverage_router/gf_sparse_fixedrouter_decoder_k050_seed3106/best_avg_l2.pt \
  --router_stage backbone_sparse \
  --route_after_block 5 \
  --keep_ratio 0.5 \
  --escape_tokens 8 \
  --lr_router 0 \
  --lr_decoder 1e-5 \
  --lr_backbone 0 \
  --lr_inout 0 \
  --heatmap_loss_weight 1.0 \
  --router_coverage_weight 0 \
  --router_budget_weight 0 \
  --router_entropy_weight 0 \
  --router_warmup_epochs 0 \
  --weight_decay 0 \
  --max_epochs 3 \
  --batch_size 8 \
  --grad_accum_steps 4 \
  --n_workers 8 \
  --amp \
  --clip_grad_norm 1.0 \
  --wandb_project GazeRoute \
  --wandb_mode online \
  --exp_name gf_sparse_fixedrouter_decoder_continue_k050_seed3106 \
  --run_dir /home/fb/src/paper/gazelleV1/experiments/coverage_router/gf_sparse_fixedrouter_decoder_continue_k050_seed3106 \
  --seed 3106 \
  > logs/coverage_router/gf_sparse_fixedrouter_decoder_continue_k050_seed3106.log 2>&1 < /dev/null &
```

启动后 `run_manifest.json` 必须是：

```json
{
  "backbone": 0,
  "decoder": 3416576,
  "inout": 0,
  "router": 0
}
```

完成后同步：

```text
experiments/coverage_router/gf_sparse_fixedrouter_decoder_continue_k050_seed3106/
  run_manifest.json
  history.jsonl
  summary.json
```

必须比较 5.7 和 5.8 相同 epoch 的完整 tuple，而不是两个 summary 的逐指标
best：

- 5.7 在至少两个指标上一致优于 5.8：suffix 有独立贡献；
- 两者基本相同：提升主要来自 decoder 额外训练，使用更简单的
  decoder-only checkpoint；
- 5.8 更好：suffix 微调 No-Go，回退 decoder-only。

#### 5.8 实际结果与 matched 结论

配置核验通过：decoder 可训练参数为 `3,416,576`，backbone、router 和
in/out 均为 0；三轮 router coverage 和实际 keep ratio 完全不变。

| epoch | 5.8 AUC ↑ | 5.8 Avg L2 ↓ | 5.8 Min L2 ↓ |
|---:|---:|---:|---:|
| 0 | 0.952664 | 0.108858 | 0.049118 |
| 1 | 0.953053 | 0.108793 | 0.049109 |
| 2 | **0.953157** | **0.107365** | **0.048062** |

将 5.7 suffix+decoder 与 5.8 decoder-only 的同 epoch tuple 配对相减：

| epoch | suffix ΔAUC ↑ | suffix ΔAvg L2 ↓ | suffix ΔMin L2 ↓ |
|---:|---:|---:|---:|
| 0 | +0.000492 | -0.001401 | -0.001315 |
| 1 | +0.000640 | -0.001046 | -0.001352 |
| 2 | **+0.000902** | **-0.001194** | **-0.001437** |

5.7 在所有三个 epoch 的所有三个指标上都优于 matched control。epoch 2
的 suffix 独立贡献对应 Avg L2 相对降低 `1.11%`、Min L2 相对降低
`2.99%`。因此 suffix 微调本身是 **Strong Go**，不是 decoder 多训练三轮
造成的假提升；当前 pilot 候选固定为 5.7 epoch 2。

但现有 5.x 仍有两个不能带入论文正式结论的问题：

1. 每个 epoch 都直接在 official test 上评估并保存 best，属于 test
   peeking；
2. `GazeDataset` 把同图不同 head 展平成独立输入，现有结果没有触发
   “多人 support max-union 后共享一个 K50 route”。

因此现在不进入 VAT，也不直接堆多 seed；先做不需训练的真实多人
query-unit 审计。

### 5.9 GazeFollow 多人共享 route 审计

代码新增两种显式评测单位：

- `person`：旧行为，每个 head 重复运行一次图像 backbone；
- `image`：一张图的所有 head 同时输入，DINO 每图运行一次，所有人的
  support 做 max-union，并共享同一个固定 K。

还新增了 image-level 分层结果：`single_head`、`multi_head`、
`two_head`、`three_plus_head`。

#### 5.9a 首次 K100 等价性结果

首次命令使用 person batch 16、image batch 4，并开启 AMP。输出
`passed=false`，但不是 person/GT flatten 错位。失败项只有：

- `avg_l2_equivalent=false`；
- `min_l2_equivalent=false`；
- `image_grouping_reduces_backbone_inputs=false`。

两种模式的 person count、loader item 和 image count 都是 `4782`；
grouped strata 为 `single_head=4782, multi_head=0`。因此当前 official
GazeFollow test 根本没有同图多人 query，无法减少 backbone 输入，也
无法验证 max-union；这个检查在该 split 上应是 N/A，而不是失败。

其余结果支持数据对齐正确：

| metric | image − person |
|---|---:|
| AUC | +0.00000129 |
| Avg L2 | +0.00019267 |
| Min L2 | +0.00008709 |
| soft coverage | -0.00000571 |
| hard coverage | -0.0000000017 |
| point coverage | 0 |

Hard/point coverage 均通过，AUC 也通过。L2 使用 `64×64` heatmap
硬 argmax；batch 16 与 batch 4 在 AMP 下的微小 kernel/舍入差异会让少量
近似并列峰跳格，因此不能据此判定 indexing bug。

比较脚本已经修正：当 `multi_head=0` 时，将 backbone reduction 标记为
不可审计；仍强制检查 image count 不大于 person count、单人数据上二者
相等。

#### 5.9b matched-batch K100 复核（当前下一步）

两条路径都使用 batch 4 和相同 AMP。此时输入顺序、batch shape、精度和
最后一个 batch 都一致，可以隔离首次运行中的 batch-shape 数值漂移：

```bash
mkdir -p logs/coverage_router

CUDA_VISIBLE_DEVICES=0 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
nohup /home/fb/anaconda3/envs/py310/bin/python -u \
  scripts/compare_gazefollow_query_units.py \
  --checkpoint /home/fb/src/paper/gazelleV1/experiments/coverage_router/gf_sparse_fixedrouter_suffix_k050_seed3106/best_avg_l2.pt \
  --data_path /newhome/fb/dataset/gazefollow_extended \
  --keep_ratio_override 1.0 \
  --person_batch_size 4 \
  --image_batch_size 4 \
  --n_workers 8 \
  --amp \
  --expected_person_count 4782 \
  --equivalence_tolerance 1e-5 \
  --output /home/fb/src/paper/gazelleV1/experiments/coverage_router/gf_sparse_fixedrouter_suffix_k050_seed3106/query_unit_k100_b4_amp.json \
  > logs/coverage_router/gf_suffix_k050_query_unit_k100_b4_amp.log 2>&1 < /dev/null &
```

预期顶层为：

```json
{
  "passed": true,
  "dataset_diagnostics": {
    "multi_person_audit_applicable": false,
    "multi_head_person_count": 0
  }
}
```

若 matched B4 AMP 仍不通过，再用完全相同命令去掉 `--amp` 做 FP32
复核；FP32 仍失败才进入逐样本 heatmap/argmax 排查。

#### 5.9c 统计 train holdout 是否能审计多人 union

official test 的 K50 grouped 对比已经取消，因为没有任何 multi-head
query。使用下面的 CPU-only 命令统计完整 train，以及固定
10% image holdout 中的 head-count 分布：

```bash
nohup /home/fb/anaconda3/envs/py310/bin/python -u \
  scripts/audit_gazefollow_query_groups.py \
  --data_path /newhome/fb/dataset/gazefollow_extended \
  --split train \
  --val_fraction 0.10 \
  --split_seed 3106 \
  --output /home/fb/src/paper/gazelleV1/experiments/coverage_router/gazefollow_train_query_groups_seed3106.json \
  > logs/coverage_router/gazefollow_train_query_groups_seed3106.log 2>&1 < /dev/null &
```

只有 holdout 的 `validation.multi_head_record_count > 0`，后续
image-grouped validation 才能为多人 union 提供真实证据。训练集上的旧
5.7 checkpoint 只能做 wiring diagnostic；正式结论仍需按 5.10 从
split-clean baseline 重跑。

#### 5.9b/5.9c 实际结果

Matched B4 AMP 的 official-test K100 comparison 已完全通过：
`passed=true`，person/image 的 AUC、Avg L2、Min L2、soft/hard/point
coverage 和 mean support 全部逐位相同，三项 gaze delta 均为 0。此前
B16/B4 的 L2 小差异确认只是 batch shape 与 AMP 的数值效应，不是
flatten/index bug。

固定 10% holdout 的 query 分布为：

| subset | 可评图像 | 多人图像 | 多人 person | 2-head | 3+ |
|---|---:|---:|---:|---:|---:|
| full train | 107,580 | 4,345 | 10,223 | 3,235 | 1,110 |
| 90% train | 96,834 | 3,917 | 9,215 | 2,907 | 1,010 |
| 10% validation | 10,746 | 428 | 1,008 | 328 | 100 |

Validation 中 multi-head 图占可评图 `3.98%`，multi-head person 占
全部 person `8.90%`；2-head 有 656 人，3+ 有 352 人，最多 7 人。
这个规模足以做 multi overall、2-head、3+ 的 feasibility/guardrail，
但 4+/5+/6+/7 不应分别下统计结论。

Official test 全是 single-head，因此取消在其上运行 K50 union。下一步只
在固定 holdout 的 428 张 multi-head 图上做旧 checkpoint 的短 gate。
旧 checkpoint 已见过这些训练图，结果只能验证 wiring 和排除灾难性容量
竞争，不能写成正式泛化结果，也不能据此扫参。

#### 5.9d Multi-head holdout K100 wiring

该命令只评测固定 holdout 的 `multi` 子集。person 模式有 1,008 次
query/backbone 输入，image 模式有 428 张图；使用 batch 1 和 FP32：

```bash
mkdir -p logs/coverage_router

CUDA_VISIBLE_DEVICES=0 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
nohup /home/fb/anaconda3/envs/py310/bin/python -u \
  scripts/compare_gazefollow_query_units.py \
  --checkpoint /home/fb/src/paper/gazelleV1/experiments/coverage_router/gf_sparse_fixedrouter_suffix_k050_seed3106/best_avg_l2.pt \
  --data_path /newhome/fb/dataset/gazefollow_extended \
  --gazefollow_eval_split train_holdout \
  --gazefollow_val_fraction 0.10 \
  --gazefollow_split_seed 3106 \
  --gazefollow_head_count_subset multi \
  --keep_ratio_override 1.0 \
  --person_batch_size 1 \
  --image_batch_size 1 \
  --n_workers 8 \
  --expected_person_count 1008 \
  --expected_image_count 428 \
  --auc_tolerance 1e-5 \
  --l2_tolerance 5e-4 \
  --coverage_tolerance 1e-6 \
  --output /home/fb/src/paper/gazelleV1/experiments/coverage_router/gf_sparse_fixedrouter_suffix_k050_seed3106/query_unit_multi_holdout_k100_b1_fp32.json \
  > logs/coverage_router/gf_suffix_multi_holdout_k100_b1_fp32.log 2>&1 < /dev/null &
```

必须满足 `passed=true`、`multi_person_audit_applicable=true`、
person/image=`1008/428`，且 grouped hard/point coverage 约为 1。

#### 5.9e Multi-head holdout K50 catastrophic gate

5.9d 通过后运行同一子集的 K50。该 comparison 的 image-person delta
隔离了“每人各有 K50”与“多人共享总 K50”的额外容量竞争：

```bash
CUDA_VISIBLE_DEVICES=0 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
nohup /home/fb/anaconda3/envs/py310/bin/python -u \
  scripts/compare_gazefollow_query_units.py \
  --checkpoint /home/fb/src/paper/gazelleV1/experiments/coverage_router/gf_sparse_fixedrouter_suffix_k050_seed3106/best_avg_l2.pt \
  --data_path /newhome/fb/dataset/gazefollow_extended \
  --gazefollow_eval_split train_holdout \
  --gazefollow_val_fraction 0.10 \
  --gazefollow_split_seed 3106 \
  --gazefollow_head_count_subset multi \
  --keep_ratio_override 0.5 \
  --person_batch_size 1 \
  --image_batch_size 1 \
  --n_workers 8 \
  --expected_person_count 1008 \
  --expected_image_count 428 \
  --output /home/fb/src/paper/gazelleV1/experiments/coverage_router/gf_sparse_fixedrouter_suffix_k050_seed3106/query_unit_multi_holdout_k050_b1_fp32.json \
  > logs/coverage_router/gf_suffix_multi_holdout_k050_b1_fp32.log 2>&1 < /dev/null &
```

该 pilot 不使用任意“必须提升 10%”规则。先看完整 tuple 和 coverage：

- shared-union 额外 AUC 下降不超过约 `0.004`、Avg L2 增加不超过约
  `0.005`：可进入 split-clean chain；
- AUC 下降超过 `0.01` 或 Avg L2 增加超过 `0.02`：灾难性 No-Go，
  正式训练前先做 grouped/mixed-query training 或人数自适应预算；
- 中间区间结合 point coverage 和 2-head/3+ 分层决定。

由于 train validation 每个 head 只有一个 gaze 标注，Avg L2 与 Min L2
等价，只能算一个证据。正式多人置信区间应按 image clustered bootstrap，
不能把同图的 1,008 个人当独立样本。

#### 5.9d/5.9e 实际结果：Pass，进入 split-clean chain

K100 的 `passed=true`，10 项检查全部通过。person 与 image-grouped
都评测同样的 1,008 人；image-grouped 只需 428 次 backbone 输入，
而 person 模式需要 1,008 次。AUC delta 为 `-1.62e-8`，Avg/Min L2
delta 都为 0，point coverage 为 1，hard coverage 约为 1。因此
image grouping、head 顺序、GT 映射和多人预测 flatten 均已验证正确。

K50 的结果为：

| 指标 | 每人独立 K50 | 多人共享总 K50 | grouped - person |
|---|---:|---:|---:|
| AUC ↑ | 0.964709 | 0.963904 | -0.000805 |
| Avg/Min L2 ↓ | 0.111670 | 0.110322 | -0.001348 |
| hard coverage ↑ | 0.874800 | 0.864990 | -0.009809 |
| point coverage ↑ | 0.954365 | 0.945437 | -0.008929 |
| mean support | 0.070416 | 0.090075 | +0.019659 |

其中 mean support 的口径由单人的 support 均值变成多人 max-union 的图像
均值，因此不能把它的增加解释成退化。共享总 K50 只比“每人各有 K50”
少 `0.000805` AUC、少约 `0.9` 个百分点的 point coverage，Avg L2
反而改善 `0.001348`，显著优于前述 catastrophic gate。

从 grouped K100 降到 grouped K50 时，overall AUC 下降 `0.002145`，
Avg L2 增加 `0.001530`。2-head 的 AUC/L2 变化为
`-0.001746/+0.001416`，3+ head 为 `-0.002890/+0.001742`。人数增加
会让固定预算的代价略微扩大，但没有容量崩溃。因此不需要先引入
grouped/mixed-query training 或 adaptive K。

这两个结果仍来自见过全部 GazeFollow train 的旧 checkpoint，只能作为
wiring/capacity gate，不能作为正式泛化结果或论文表格。至此停止在旧
checkpoint 上扫参，下一步只运行 5.10 的 split-clean 正式链路。

### 5.10 正式验证与最终顺序

多人审计通过后，使用代码中的
`--gf_val_fraction 0.10 --gf_split_seed 3106`。split 在
`train_preprocessed.json` 的 image path 层生成，同图所有 heads 不跨
train/validation；validation 默认以 image 为单位运行真实多人 union。
每个 run 会写 `data_split.json`，checkpoint 初始化和 resume 时会核验
源文件与 assignment fingerprint。

正式链路不能从旧 `epoch_14.pt`、旧 support best 或旧 K50 best 开始，
因为它们见过完整 train 或由 test 指标选过。固定顺序是：

1. 从 DINO 预训练和随机 task decoder 重训 split-clean dense baseline；
2. 固定 dense checkpoint，训练 support router，以 validation hard
   coverage 选择唯一 checkpoint；
3. 训练 K50 decoder，以 validation Avg L2 选择唯一 checkpoint；
4. 重做 suffix 与 matched decoder-only control；
5. 三个训练 seed 共用同一个 split seed；
6. recipe 和 epoch 固定后，official test 只评测一次；
7. 再进入 VAT；wall-clock 加速仍作为独立支线。

#### 5.10a split-clean dense protocol smoke（先运行）

这是几分钟级的协议 smoke，只验证 formal split、image-grouped
validation、checkpoint provenance 和训练参数分组。它不是论文实验，
其 checkpoint 也不能初始化后续阶段。命令刻意不传 `--init_ckpt`：
模型会加载本地 DINOv3 预训练权重，而 task decoder 从随机初始化开始。

```bash
mkdir -p logs/coverage_router

CUDA_VISIBLE_DEVICES=0 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
nohup /home/fb/anaconda3/envs/py310/bin/python -u \
  scripts/train_coverage_router.py \
  --dataset gazefollow \
  --model gazelle_dinov3_vitb16 \
  --data_path /newhome/fb/dataset/gazefollow_extended \
  --gf_val_fraction 0.10 \
  --gf_split_seed 3106 \
  --router_stage support_pilot \
  --route_after_block 5 \
  --keep_ratio 1.0 \
  --escape_tokens 8 \
  --lr_router 0 \
  --lr_decoder 1e-3 \
  --lr_backbone 0 \
  --lr_inout 0 \
  --heatmap_loss_weight 1 \
  --router_coverage_weight 0 \
  --router_budget_weight 0 \
  --router_entropy_weight 0 \
  --router_warmup_epochs 0 \
  --weight_decay 0 \
  --max_epochs 1 \
  --batch_size 2 \
  --eval_batch_size 1 \
  --grad_accum_steps 1 \
  --n_workers 0 \
  --max_train_batches 2 \
  --max_eval_batches 2 \
  --wandb_mode disabled \
  --exp_name gf_splitclean_dense_protocol_smoke_seed3106 \
  --run_dir /home/fb/src/paper/gazelleV1/experiments/coverage_router/gf_splitclean_dense_protocol_smoke_seed3106 \
  --seed 3106 \
  > logs/coverage_router/gf_splitclean_dense_protocol_smoke_seed3106.log 2>&1 < /dev/null &
```

进程正常结束后检查：

```bash
tail -n 80 logs/coverage_router/gf_splitclean_dense_protocol_smoke_seed3106.log

jq '{
  train_sample_count,
  eval_sample_count,
  eval_image_count,
  trainable_parameters,
  data_split: {
    selection_is_formal: .data_split.selection_is_formal,
    train_group_count: .data_split.train_group_count,
    validation_group_count: .data_split.validation_group_count,
    assignment_fingerprint: .data_split.assignment_fingerprint
  }
}' \
  /home/fb/src/paper/gazelleV1/experiments/coverage_router/gf_splitclean_dense_protocol_smoke_seed3106/run_manifest.json
```

预期值必须是：

- `train/eval person/eval image = 102132/11326/10746`；
- trainable 参数仅 `decoder=3416576`，router/backbone/inout 均为 0；
- `selection_is_formal=true`；
- train/validation group 为 `105954/11773`；
- assignment fingerprint 为
  `817d9f2200ebc9881f6d2318cdb60d7d8d7af394f98f0e3461a6d8bd2c26b5e3`；
- 生成 `best_val_selection.pt` 和 `summary.json`。

smoke 的 validation 只跑两个 batch，指标没有意义。

#### 5.10b split-clean dense baseline（smoke 通过后运行）

该 run 对齐原 dense baseline 的核心 recipe：FP32、真实 batch 60、
decoder LR `1e-3`、15 epochs、冻结 DINOv3 和 router，并在固定的
image-grouped validation 上按 Avg L2 选 checkpoint（AUC 只作
tie-breaker）。

```bash
CUDA_VISIBLE_DEVICES=0 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
nohup /home/fb/anaconda3/envs/py310/bin/python -u \
  scripts/train_coverage_router.py \
  --dataset gazefollow \
  --model gazelle_dinov3_vitb16 \
  --data_path /newhome/fb/dataset/gazefollow_extended \
  --gf_val_fraction 0.10 \
  --gf_split_seed 3106 \
  --router_stage support_pilot \
  --route_after_block 5 \
  --keep_ratio 1.0 \
  --escape_tokens 8 \
  --lr_router 0 \
  --lr_decoder 1e-3 \
  --lr_backbone 0 \
  --lr_inout 0 \
  --heatmap_loss_weight 1 \
  --router_coverage_weight 0 \
  --router_budget_weight 0 \
  --router_entropy_weight 0 \
  --router_warmup_epochs 0 \
  --weight_decay 0 \
  --max_epochs 15 \
  --batch_size 60 \
  --eval_batch_size 4 \
  --grad_accum_steps 1 \
  --n_workers 8 \
  --wandb_project GazeRoute \
  --wandb_mode online \
  --exp_name gf_splitclean_dense_seed3106 \
  --run_dir /home/fb/src/paper/gazelleV1/experiments/coverage_router/gf_splitclean_dense_seed3106 \
  --seed 3106 \
  > logs/coverage_router/gf_splitclean_dense_seed3106.log 2>&1 < /dev/null &
```

不要添加 `--init_ckpt`、`--resume`、`--train_backbone_after_router` 或
`--amp`。若真实 batch 60 在服务器 OOM，再改成
`--batch_size 30 --grad_accum_steps 2`，不要改输入分辨率、split 或学习率。
完整 run 结束后，后续 support router 必须只从该目录的
`best_val_selection.pt` 初始化。

#### 5.10b 实际结果：正式 dense validation baseline 通过

该 run 已完整运行 15 epochs，`history.jsonl` 连续包含 epochs 0–14，
没有 NaN、样本数变化或 split 漂移。协议核验为：

- FP32，train/eval batch 为 `60/4`；
- 只训练 `3,416,576` 个 decoder 参数；
- DINOv3、router 和 in/out 均冻结；
- `init_ckpt=null`、`resume=null`；
- validation 为固定的 image-grouped GazeFollow train holdout；
- train/eval person 和 eval image 数分别为
  `102132/11326/10746`；
- assignment fingerprint 为
  `817d9f2200ebc9881f6d2318cdb60d7d8d7af394f98f0e3461a6d8bd2c26b5e3`。

预先固定的选模规则是 validation Avg L2 最小、AUC 仅作完全并列时的
tie-breaker。它选择 epoch 14 的单一 checkpoint：

| subset | AUC ↑ | Avg/Min L2 ↓ | persons |
|---|---:|---:|---:|
| overall | 0.964691 | 0.114705 | 11,326 |
| single-head | 0.965114 | 0.114341 | 10,318 |
| multi-head | 0.960356 | 0.118436 | 1,008 |
| two-head | 0.959825 | 0.115968 | 656 |
| three-plus-head | 0.961346 | 0.123036 | 352 |

`summary.best_metrics.auc=0.964715` 来自 epoch 13，不能与 epoch 14 的
L2 拼成一个 tuple。epoch 14 相对 epoch 13 仅少 `0.0000247` AUC，
但 Avg L2 改善 `0.0002705`，因此 epoch 14 的选择符合规则。epoch
0→14 的 overall AUC 增加 `0.01911`，Avg L2 降低 `0.04328`，没有出现
后期崩溃。

服务器正式权重已核验：

```text
best_val_selection.pt
epoch = 14
sha256 = 4752899dbedde10cf211684c07ef3fee9acb2db990f031e484eed07b8f5d911d
```

当前本地目录已同步 JSON/JSONL 结果文件，但没有同步 `.pt` 权重。开始
下一阶段前，应至少归档 `best_val_selection.pt`，并建议同时归档
`last.resume.pt` 和训练日志。该结果是一个 seed 的正式 validation
baseline，不是 official-test 结果，也不是 coverage router 的方法结果。

#### 5.10c FP32 eval batch 4/16 parity：Pass

使用 epoch 14 的同一个 `best_val_selection.pt`，在完全相同的
image-grouped holdout 上以 batch 16 重评。相对训练期间的 batch 4
结果：

| metric | B16 - B4 |
|---|---:|
| overall AUC | `+6.20e-8` |
| overall Avg/Min L2 | `+4.75e-6` |
| single-head AUC | `+6.45e-8` |
| single-head Avg/Min L2 | `+5.21e-6` |
| multi-head AUC | `+3.67e-8` |
| multi-head Avg/Min L2 | `0` |
| two-head AUC | `+1.55e-7` |
| three-plus-head AUC | `-1.84e-7` |

所有 image/person/strata 计数和 split provenance 完全相同；routing
指标最大差异为 mean support 的 `5.95e-7`。结果满足既有 gaze
`1e-5`、L2 `5e-4` 和 routing `1e-6` 容差。因此，后续 formal
validation 固定使用 **batch 16、FP32**。最终横向表也统一使用 B16
重评结果，不混用 B4/B16 数值。

当前评测 JSON 尚未显式记录 batch size、AMP、workers 和 device，这些
runtime 参数只由命令和文件名证明；后续应补入输出 schema。

正式训练 router 前，先用 dense checkpoint 中从未训练过的 router 做
K25 initialization control。它不是纯 random Top-K，因为 head footprint
和 escape-token priority 仍然生效：

```bash
CUDA_VISIBLE_DEVICES=0 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
nohup /home/fb/anaconda3/envs/py310/bin/python -u \
  scripts/eval_coverage_router.py \
  --checkpoint /home/fb/src/paper/gazelleV1/experiments/coverage_router/gf_splitclean_dense_seed3106/best_val_selection.pt \
  --dataset gazefollow \
  --data_path /newhome/fb/dataset/gazefollow_extended \
  --gazefollow_eval_split train_holdout \
  --gf_val_fraction 0.10 \
  --gf_split_seed 3106 \
  --gazefollow_eval_unit image \
  --gazefollow_head_count_subset all \
  --keep_ratio_override 0.25 \
  --batch_size 16 \
  --n_workers 8 \
  --output /home/fb/src/paper/gazelleV1/experiments/coverage_router/gf_splitclean_dense_seed3106/eval_holdout_untrained_router_k025_b16_fp32.json \
  > logs/coverage_router/gf_splitclean_dense_untrained_router_k025_b16_fp32.log 2>&1 < /dev/null &
```

必须满足相同 epoch/split/fingerprint、`10746 images / 11326 persons`、
`query_unit=image` 和 actual keep ratio `0.25`。由于 `support_pilot`
只观察 route 而不裁剪 DINO，gaze tuple 应与 B16 K100 相差不超过
`1e-5 AUC / 5e-4 L2`。hard/point coverage 不预设任意绝对提升门槛；
它们作为正式 support training 的学习前基线，完成后再按相对变化和旧
diagnostic 参考共同判断。

#### 5.10d K25 untrained-router initialization control：Pass

实际输出为
`gf_splitclean_dense_seed3106/eval_holdout_untrained_router_k025_b16_fp32.json`。
它与 K100 B16 FP32 评估使用相同 epoch 14 checkpoint、split、
assignment fingerprint、`10746 images / 11326 persons` 和
image-grouped all-head protocol。除预期的 keep-ratio override 外，
provenance 和模型配置完全一致。

| metric | K100 dense | K25 untrained | K25 - K100 |
|---|---:|---:|---:|
| AUC | 0.964690885 | 0.964690885 | `0` |
| Avg/Min L2 | 0.114710056 | 0.114710056 | `0` |
| actual keep ratio | 1.000000 | 0.250000 | -0.750000 |
| hard coverage | 1.000000 | 0.159233 | -0.840767 |
| exact-point coverage | 1.000000 | 0.153629 | -0.846371 |
| soft coverage | 0.0411611 | 0.0411611 | `0` |
| mean support | 1.0000006 | 1.0000006 | `0` |

overall、single-head、multi-head、two-head 和 three-plus-head 的
所有 gaze 指标也逐值相同。这符合 `support_pilot` 的语义：K25 只改变
Top-K hard route，不裁剪 DINO 或改变 decoder 输入；soft coverage 和
mean support 是 Top-K 前的 router 输出，因此也不会仅因 K 改变。

由此得到正式学习前基线：

```text
H0 (hard coverage)        = 0.1592328867
P0 (exact-point coverage) = 0.1536288186
S0 (soft coverage)        = 0.0411610833
```

这不是失败结果，而是之后衡量 Router 学习增量的零点。正式 support
run 不使用任意的“必须提高 10%/20%”绝对门槛，而是报告相同协议下的
`Δhard = Htrained-H0`、`Δpoint = Ptrained-P0`、`Δsoft = Strained-S0`
及三轮轨迹。论文级显著性与稳定性需在后续 image-clustered bootstrap
和多 seed 中确认。

#### 5.10e split-clean formal support K25

只训练 Router 三轮。训练与评估均固定 FP32；train batch 16 沿用已验证的
support recipe，eval batch 16 来自 5.10c parity 结果。不要添加 `--amp`，
否则训练和选模数值协议会与上述 FP32 baseline/control 不一致。

启动前必须确认目标 `run_dir` 不存在或为空，避免 `history.jsonl`
追加到旧实验：

```bash
CUDA_VISIBLE_DEVICES=0 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
nohup /home/fb/anaconda3/envs/py310/bin/python -u \
  scripts/train_coverage_router.py \
  --dataset gazefollow \
  --model gazelle_dinov3_vitb16 \
  --data_path /newhome/fb/dataset/gazefollow_extended \
  --gf_val_fraction 0.10 \
  --gf_split_seed 3106 \
  --init_ckpt /home/fb/src/paper/gazelleV1/experiments/coverage_router/gf_splitclean_dense_seed3106/best_val_selection.pt \
  --router_stage support_pilot \
  --route_after_block 5 \
  --keep_ratio 0.25 \
  --router_hidden_dim 256 \
  --router_temperature 1.0 \
  --escape_tokens 8 \
  --spatial_prior none \
  --fusion raw_concat \
  --heatmap_loss_weight 0 \
  --inout_loss_lambda 0 \
  --router_coverage_weight 1.0 \
  --router_budget_weight 0.05 \
  --router_entropy_weight 0 \
  --lr_router 1e-3 \
  --lr_decoder 0 \
  --lr_backbone 0 \
  --lr_inout 0 \
  --router_warmup_epochs 0 \
  --weight_decay 0 \
  --max_epochs 3 \
  --batch_size 16 \
  --eval_batch_size 16 \
  --grad_accum_steps 1 \
  --n_workers 8 \
  --log_iter 10 \
  --wandb_project GazeRoute \
  --wandb_mode online \
  --exp_name gf_splitclean_support_k025_seed3106 \
  --run_dir /home/fb/src/paper/gazelleV1/experiments/coverage_router/gf_splitclean_support_k025_seed3106 \
  --seed 3106 \
  > logs/coverage_router/gf_splitclean_support_k025_seed3106.log 2>&1 < /dev/null &
```

正式启动后先检查 `run_manifest.json`：

- trainable 参数必须为 router `463105`，decoder/backbone/inout 均为 0；
- init checkpoint 必须指向 split-clean dense epoch 14；
- `amp=false`，train/eval batch 为 `16/16`；
- assignment fingerprint 必须仍为
  `817d9f2200ebc9881f6d2318cdb60d7d8d7af394f98f0e3461a6d8bd2c26b5e3`；
- train/eval person 和 eval image 必须为
  `102132/11326/10746`。

选模规则由代码固定为 validation hard coverage 最大，其次 exact-point
coverage，再其次 AUC；formal checkpoint 为 `best_val_selection.pt`。
三轮 gaze AUC/L2 应始终与 dense B16 FP32 处于
`1e-5 AUC / 5e-4 L2` 容差内，因为本阶段所有 gaze 模块均冻结。若第一批
意外 OOM，不修改其他超参，换一个新 run directory，改为
`--batch_size 8 --grad_accum_steps 2`，eval 仍保持 16。

#### 5.10e 实际结果：Formal P1 Strong Go

协议和运行完整性全部通过：与 dense baseline 使用同一 formal
split、source SHA、assignment fingerprint 和样本计数；从 dense epoch
14 `best_val_selection.pt` 初始化；FP32 train/eval batch `16/16`；
epochs 0–2 完整且只训练 router `463105` 个参数。decoder、
backbone 和 in/out 均为 0。

相对 5.10d 未训练 K25 零点：

| epoch | hard / delta | exact point / delta | soft / delta | effective support |
|---:|---:|---:|---:|---:|
| untrained | 0.159233 | 0.153629 | 0.041161 | 1.000001 |
| 0 | 0.677605 / `+0.518372` | 0.815645 / `+0.662017` | 0.275250 / `+0.234089` | 0.099416 |
| 1 | 0.692841 / `+0.533608` | 0.828713 / `+0.675084` | 0.282280 / `+0.241119` | 0.105503 |
| **2 selected** | **0.700287 / `+0.541054`** | **0.839926 / `+0.686297`** | **0.333380 / `+0.292219`** | **0.065661** |

hard、point 和 soft coverage 三轮整体上升，所以按预先固定的
hard-coverage 最大化规则正确选中 epoch 2。在 actual keep ratio 始终
精确为 `0.25` 的前提下，selected router 的 hard、point 和 soft
coverage 分别提高 `54.11 / 68.63 / 29.22` 个百分点，不是通过
增加 token 数换取 coverage。

三轮 overall 和所有 head-count strata 的 gaze tuple 都与 dense B16
FP32 逐值相同：overall 始终为
`AUC=0.9646908849 / Avg/Min L2=0.1147100560`。这证明结果没有
混入 decoder/backbone 更新。当前可以成立的结论是：在严格
image-disjoint validation 上，router 能在 25% 固定预算下学到对
gaze target 强得多的空间排序。

该结论仍只是一个 seed 的 formal support 证据，不是 sparse accuracy
或 efficiency 结果。effective support 最终只有 `6.57%`，说明分布高度
聚焦，但也意味着 K25 中低分 tail 的排序可能不稳定；因此下一阶段
仍固定 router，不恢复 joint training。当前 JSON 也没有 per-stratum routing
coverage、per-image 预测或 CI，不能由 overall `83.99%` point coverage
推断每个多人分层都达到同样数值。

#### 5.10f selected checkpoint 重载审计（可选，不阻塞主实验）

可以在独立进程中重载落盘的 epoch 2
`best_val_selection.pt`，确认 checkpoint artifact 能复现 summary。这只是
额外审计，不是新泛化证据，也不再作为启动真实 sparse 主实验的
前置门槛：

```bash
CUDA_VISIBLE_DEVICES=0 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
nohup /home/fb/anaconda3/envs/py310/bin/python -u \
  scripts/eval_coverage_router.py \
  --checkpoint /home/fb/src/paper/gazelleV1/experiments/coverage_router/gf_splitclean_support_k025_seed3106/best_val_selection.pt \
  --dataset gazefollow \
  --data_path /newhome/fb/dataset/gazefollow_extended \
  --gazefollow_eval_split train_holdout \
  --gf_val_fraction 0.10 \
  --gf_split_seed 3106 \
  --gazefollow_eval_unit image \
  --gazefollow_head_count_subset all \
  --keep_ratio_override 0.25 \
  --batch_size 16 \
  --n_workers 8 \
  --output /home/fb/src/paper/gazelleV1/experiments/coverage_router/gf_splitclean_support_k025_seed3106/eval_holdout_selected_k025_b16_fp32.json \
  > logs/coverage_router/gf_splitclean_support_selected_k025_b16_fp32.log 2>&1 < /dev/null &
```

若运行，应复现 checkpoint epoch 2、相同 fingerprint/counts 和 summary 中的
gaze/routing tuple；gaze 容差为 `1e-5 AUC / 5e-4 L2`，routing 容差为
`1e-6`。

#### 5.10g split-clean K50 sparse decoder curriculum（当前主实验）

这是 coverage-router idea 第一个真正的端到端 sparse 正式实验：DINOv3
blocks 0–5 仍稠密运行，固定的 learned router 为每张图像的所有人共享
一个 K50 route，blocks 6–11 只处理保留 token，然后将 sparse 特征
回填给 gaze decoder。

不再等待 5.10f 或重复 K100/K50 zero-shot gate。这些诊断之前已在
同一代码路径上多次通过，如果本 run 异常再回补。本阶段固定 router
和整个 DINO，只训练 decoder；训练 K 为
`100% → 75% → 50% → 50% → 50%`，每轮 validation 始终使用 K50。
选模规则固定为 Avg L2 最小，AUC 只作完全并列时的 tie-breaker。

启动前确认目标 run directory 不存在或为空，然后运行：

```bash
CUDA_VISIBLE_DEVICES=0 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
nohup /home/fb/anaconda3/envs/py310/bin/python -u \
  scripts/train_coverage_router.py \
  --dataset gazefollow \
  --model gazelle_dinov3_vitb16 \
  --data_path /newhome/fb/dataset/gazefollow_extended \
  --gf_val_fraction 0.10 \
  --gf_split_seed 3106 \
  --init_ckpt /home/fb/src/paper/gazelleV1/experiments/coverage_router/gf_splitclean_support_k025_seed3106/best_val_selection.pt \
  --router_stage backbone_sparse \
  --route_after_block 5 \
  --keep_ratio 0.50 \
  --router_hidden_dim 256 \
  --router_temperature 1.0 \
  --escape_tokens 8 \
  --spatial_prior none \
  --fusion raw_concat \
  --router_warmup_epochs 3 \
  --heatmap_loss_weight 1.0 \
  --inout_loss_lambda 0 \
  --router_coverage_weight 0 \
  --router_budget_weight 0 \
  --router_entropy_weight 0 \
  --lr_router 0 \
  --lr_decoder 1e-4 \
  --lr_backbone 0 \
  --lr_inout 0 \
  --weight_decay 0 \
  --max_epochs 5 \
  --batch_size 16 \
  --eval_batch_size 16 \
  --grad_accum_steps 2 \
  --n_workers 8 \
  --clip_grad_norm 1.0 \
  --wandb_project GazeRoute \
  --wandb_mode online \
  --exp_name gf_splitclean_sparse_fixedrouter_decoder_k050_seed3106 \
  --run_dir /home/fb/src/paper/gazelleV1/experiments/coverage_router/gf_splitclean_sparse_fixedrouter_decoder_k050_seed3106 \
  --seed 3106 \
  > logs/coverage_router/gf_splitclean_sparse_fixedrouter_decoder_k050_seed3106.log 2>&1 < /dev/null &
```

预期 `run_manifest.json` 中只有 decoder `3416576` 个参数可训练，
router/backbone/inout 必须都为 0；`router_stage=backbone_sparse`、
`amp=false`、train/eval batch `16/16`、相同 assignment fingerprint。
若 B16 在第一批意外 OOM，停止该进程并换新 run directory，仅改为
`--batch_size 8 --grad_accum_steps 4`，其他参数不变。

暂不同时跑 K25：旧 benchmark 中 K25 相对 K50 仅约 2% 延迟差，
且相对 dense 没有有意义的端到端收益。先用 K50 回答“learned
route 真正作用于 DINO suffix 后，gaze accuracy 能否恢复”这个主问题。

## 6. 运行监控与结果反馈

```bash
tail -f logs/coverage_router/gf_splitclean_sparse_fixedrouter_decoder_k050_seed3106.log
```

```bash
watch -n 1 nvidia-smi -i 0
```

```bash
pgrep -af train_coverage_router.py
```

每个 run directory 会生成：

- `run_manifest.json`：完整配置和 git commit；
- `data_split.json`：正式 validation 的分组策略与 fingerprint；
- `history.jsonl`：逐 epoch train/eval 指标；
- `best_val_selection.pt`：正式 validation 唯一选定 checkpoint；
- 旧模式仍兼容 `best_coverage.pt`、`best_auc.pt`、`best_min_l2.pt`
  或 `best_l2.pt`；
- `last.resume.pt`：断点恢复；
- `summary.json`：最佳结果摘要。

5.10b–5.10e 均已通过。5.10f 改为可选 artifact audit，不再阻塞。
当前直接运行 5.10g split-clean K50 sparse decoder 主实验。暂不运行
official test、VAT、K25 sparse 长训练或多 seed。
