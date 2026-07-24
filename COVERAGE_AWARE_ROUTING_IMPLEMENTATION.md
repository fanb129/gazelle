# Coverage-Aware Spatial Routing：实施计划与训练命令

## 0. 当前状态

- 基线分支：`v1`，基线提交 `03974fa`。
- 本 idea 分支：`codex/coverage-aware-router`。
- 首版目标：先完成 **固定 25% token 预算** 的闭环，证明 gaze-aware support 能在保留精度的同时减少 DINOv3 后半段的真实计算。
- 首版默认模型：DINOv3 ViT-B/16，输入 `512×512`，patch 网格 `32×32=1024`。
- 首版默认路由位置：`route_after_block=5`。该参数是从 0 开始的 block 下标，即 blocks 0–5 稠密执行，blocks 6–11 稀疏执行。
- 首版有意使用 `spatial_prior=none` 和 `fusion=raw_concat`，不同时叠加 GGSF/SASA，避免无法判断收益来自哪里。

本地机器只用于修改和静态检查。真实 DINOv3、CUDA、训练和速度测试均由服务器 `fb@3090.lab` 在拉取本分支后执行。

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

### 5.7 GazeFollow 最终评测

```bash
CUDA_VISIBLE_DEVICES=0 \
nohup /home/fb/anaconda3/envs/py310/bin/python -u \
  scripts/eval_coverage_router.py \
  --checkpoint /home/fb/src/paper/gazelleV1/experiments/coverage_router/gf_sparse_b5_k025_seed3106/best_min_l2.pt \
  --dataset gazefollow \
  --data_path /newhome/fb/dataset/gazefollow_extended \
  --batch_size 16 \
  --n_workers 8 \
  --amp \
  --output /home/fb/src/paper/gazelleV1/experiments/coverage_router/gf_sparse_b5_k025_seed3106/test_metrics.json \
  > logs/coverage_router/gf_sparse_b5_k025_seed3106_eval.log 2>&1 < /dev/null &
```

### 5.8 VAT 微调

```bash
CUDA_VISIBLE_DEVICES=0 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
nohup /home/fb/anaconda3/envs/py310/bin/python -u \
  scripts/train_coverage_router.py \
  --dataset vat \
  --model gazelle_dinov3_vitb16_inout \
  --data_path /newhome/fb/dataset/videoattentiontarget \
  --init_ckpt /home/fb/src/paper/gazelleV1/experiments/coverage_router/gf_sparse_b5_k025_seed3106/best_min_l2.pt \
  --router_stage backbone_sparse \
  --route_after_block 5 \
  --keep_ratio 0.25 \
  --escape_tokens 8 \
  --train_backbone_after_router \
  --lr_router 1e-4 \
  --lr_decoder 1e-5 \
  --lr_inout 1e-3 \
  --lr_backbone 1e-6 \
  --inout_loss_lambda 1.0 \
  --frame_sample_every 6 \
  --eval_frame_sample_every 6 \
  --max_epochs 8 \
  --batch_size 8 \
  --grad_accum_steps 4 \
  --n_workers 8 \
  --amp \
  --clip_grad_norm 1.0 \
  --wandb_project GazeRoute \
  --wandb_mode online \
  --exp_name vat_sparse_b5_k025_seed3106 \
  --run_dir /home/fb/src/paper/gazelleV1/experiments/coverage_router/vat_sparse_b5_k025_seed3106 \
  --seed 3106 \
  > logs/coverage_router/vat_sparse_b5_k025_seed3106.log 2>&1 < /dev/null &
```

训练中的 VAT eval 每 6 帧采样，只用于观察趋势，不是最终结果。

### 5.9 VAT 每帧最终评测

```bash
CUDA_VISIBLE_DEVICES=0 \
nohup /home/fb/anaconda3/envs/py310/bin/python -u \
  scripts/eval_coverage_router.py \
  --checkpoint /home/fb/src/paper/gazelleV1/experiments/coverage_router/vat_sparse_b5_k025_seed3106/best_l2.pt \
  --dataset vat \
  --data_path /newhome/fb/dataset/videoattentiontarget \
  --frame_sample_every 1 \
  --batch_size 16 \
  --n_workers 8 \
  --amp \
  --output /home/fb/src/paper/gazelleV1/experiments/coverage_router/vat_sparse_b5_k025_seed3106/test_every_frame_metrics.json \
  > logs/coverage_router/vat_sparse_b5_k025_seed3106_eval_every_frame.log 2>&1 < /dev/null &
```

### 5.10 断点恢复示例

恢复时继续写入原 run directory：

```bash
CUDA_VISIBLE_DEVICES=0 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
nohup /home/fb/anaconda3/envs/py310/bin/python -u \
  scripts/train_coverage_router.py \
  --dataset gazefollow \
  --resume /home/fb/src/paper/gazelleV1/experiments/coverage_router/gf_sparse_b5_k025_seed3106/last.resume.pt \
  --run_dir /home/fb/src/paper/gazelleV1/experiments/coverage_router/gf_sparse_b5_k025_seed3106 \
  --max_epochs 15 \
  --batch_size 8 \
  --grad_accum_steps 4 \
  --n_workers 8 \
  --amp \
  --wandb_project GazeRoute \
  --wandb_mode online \
  > logs/coverage_router/gf_sparse_b5_k025_seed3106_resume.log 2>&1 < /dev/null &
```

## 6. 运行监控与结果反馈

```bash
tail -f logs/coverage_router/gf_sparse_b5_k025_seed3106.log
```

```bash
watch -n 1 nvidia-smi -i 0
```

```bash
pgrep -af train_coverage_router.py
```

每个 run directory 会生成：

- `run_manifest.json`：完整配置和 git commit；
- `history.jsonl`：逐 epoch train/eval 指标；
- `best_coverage.pt`、`best_auc.pt`、`best_min_l2.pt` 或 `best_l2.pt`；
- `last.resume.pt`：断点恢复；
- `summary.json`：最佳结果摘要。

第一轮请先反馈以下三个日志/文件，不要直接并行启动全部消融：

1. `backbone_smoke.log`；
2. `support_smoke.log`；
3. 完整 support pilot 的 `history.jsonl` 与 `best_coverage.pt` 路径。

确认 P0/P1 后，再决定 25% 是否进入完整稀疏训练，还是先把预算放宽到 50% 排查 coverage 与 downstream accuracy 的关系。
