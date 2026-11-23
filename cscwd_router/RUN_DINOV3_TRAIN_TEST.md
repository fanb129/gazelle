# DINOv3 Router：两个数据集 × ViT-B/ViT-L，seed 0

2026-11-23 最新：GF-L 已关机中断，用户决定不补最后一轮，按实际 support 3 + decoder 4 收口。已回传并核验，详见 [E1-L 审查](results/E1L_GF_INTERRUPTED_REVIEW.md)；不再执行本页 GF 队列的 resume 命令。当前 best 为第 2 轮 K50 `0.960007/0.094061/0.037721`，后续效率与 VAT-L 迁移另用新入口。

当前 VAT 提升优先执行 [完整 GF Router→VAT](RUN_GF_ROUTER_TO_VAT.md)，使用新目录；不要重跑本页已完成的旧 VAT B/L 或续训。正式效率等待空闲卡。

L 来源后来发现精确选层未闭合，见 [L 初始化取舍](results/L_INITIALIZER_DECISION.md)。原 keys/shape 检查仍成立，不再等价于历史配置完全一致；不要据本页旧的双尺寸命令新开 L 训练，当前 L 仅保留已有 warm-start 任务。

更新：2026-11-22。E0 入口已实现，32 项 CPU 工程检查通过。以下命令由用户在服务器手动执行；助手没有启动服务器训练。

2026-11-23 状态：VAT B/L 已正常完成并回传，核验见 [E2 审查](results/E2_VAT_SEED0_REVIEW.md)；不用重复启动 VAT 训练。GF-B 现已回传核验（`0.956236/0.101550/0.043420`），GF-L 收口状态见上；VAT 续训也已完成并保留旧 best，详见 [历史审查](results/E1_GFB_E2C_REVIEW.md)。以下保留原命令供复现，不重启已收口任务。

用户现要求提升 VAT；新增的 E2-C 低 LR 固定 K50 续训见 [RUN_VAT_CONTINUE.md](RUN_VAT_CONTINUE.md)，使用新目录，不修改这里的初始 3+5 epochs。

## 1. 调度与训练规则

VAT 不依赖本轮 GF，两种尺寸都已有兼容的 VAT decoder 初始化。先同时启动两个 B，再由各自队列顺序接 L：

| GPU | 第一项 | 第二项 |
|---|---|---|
| 0 | GazeFollow ViT-B | GazeFollow ViT-L |
| 1 | VAT ViT-B | VAT ViT-L |

四组都使用完整 train/test、不切 val，base seed 为 0。重新初始化 Router，support 3 epochs + sparse decoder/in-out 5 epochs；冻结 backbone，不新增 dense 训练。B 在 block 5 后路由，L 在 block 11 后路由，均使用四层 raw-concat。

B/L 的物理 batch 为 16/4；decoder 梯度累积为 2/8，有效 batch 均为 32。support 梯度累积为 1，分别使用物理 batch 16/4。训练与精度评测固定 FP32。

GF 按 test Min L2、VAT 按 test L2 选一个 checkpoint；并列按 AUC、再保留更早 epoch。训练预算依次为 K100→K75→K50→K50→K50，每轮 test 固定 K50。VAT 逐轮 test 每 6 帧采样，选定后自动逐帧 test；主表全部指标来自该 checkpoint。

## 2. 本机 push，服务器更新

本机：

```bash
git push origin codex/coverage-aware-router
```

服务器：

```bash
cd /home/fb/src/paper/gazelleV1
git switch codex/coverage-aware-router
git pull --ff-only origin codex/coverage-aware-router
mkdir -p /home/fb/src/paper/gazelleV1/logs/cscwd_router
```

## 3. GPU 0：GF-B 后接 GF-L

```bash
CUDA_VISIBLE_DEVICES=0 setsid nohup /home/fb/anaconda3/envs/py310/bin/python -u \
  /home/fb/src/paper/gazelleV1/cscwd_router/code/scripts/run_router_dinov3_queue.py \
  --dataset gazefollow \
  --sizes b l \
  --data-path /newhome/fb/dataset/gazefollow_extended \
  --output-root /home/fb/src/paper/gazelleV1/experiments/cscwd_router/testselected_k050_seed0 \
  --report-root /home/fb/src/paper/gazelleV1/cscwd_router/results/testselected_k050_seed0 \
  --provenance /home/fb/src/paper/gazelleV1/cscwd_router/results/RESULT_PROVENANCE.md \
  --support-epochs 3 \
  --decoder-epochs 5 \
  --keep-ratio 0.5 \
  --batch-b 16 \
  --batch-l 4 \
  --eval-batch-b 16 \
  --eval-batch-l 4 \
  --accumulation-b 2 \
  --accumulation-l 8 \
  --workers 8 \
  --router-lr 0.001 \
  --decoder-lr 0.0001 \
  --inout-lr 0.001 \
  --weight-decay 0.0 \
  --clip-norm 1.0 \
  --budget-weight 0.05 \
  --router-hidden-dim 256 \
  --router-temperature 1.0 \
  --escape-tokens 8 \
  --train-frame-rate 6 \
  --test-frame-rate 6 \
  --seed 0 \
  --log-every 50 \
  --resume \
  > /home/fb/src/paper/gazelleV1/logs/cscwd_router/gazefollow_bl_seed0_gpu0.log 2>&1 < /dev/null &
```

## 4. GPU 1：VAT-B 后接 VAT-L

```bash
CUDA_VISIBLE_DEVICES=1 setsid nohup /home/fb/anaconda3/envs/py310/bin/python -u \
  /home/fb/src/paper/gazelleV1/cscwd_router/code/scripts/run_router_dinov3_queue.py \
  --dataset vat \
  --sizes b l \
  --data-path /newhome/fb/dataset/videoattentiontarget \
  --output-root /home/fb/src/paper/gazelleV1/experiments/cscwd_router/testselected_k050_seed0 \
  --report-root /home/fb/src/paper/gazelleV1/cscwd_router/results/testselected_k050_seed0 \
  --provenance /home/fb/src/paper/gazelleV1/cscwd_router/results/RESULT_PROVENANCE.md \
  --support-epochs 3 \
  --decoder-epochs 5 \
  --keep-ratio 0.5 \
  --batch-b 16 \
  --batch-l 4 \
  --eval-batch-b 16 \
  --eval-batch-l 4 \
  --accumulation-b 2 \
  --accumulation-l 8 \
  --workers 8 \
  --router-lr 0.001 \
  --decoder-lr 0.0001 \
  --inout-lr 0.001 \
  --weight-decay 0.0 \
  --clip-norm 1.0 \
  --budget-weight 0.05 \
  --router-hidden-dim 256 \
  --router-temperature 1.0 \
  --escape-tokens 8 \
  --train-frame-rate 6 \
  --test-frame-rate 6 \
  --seed 0 \
  --log-every 50 \
  --resume \
  > /home/fb/src/paper/gazelleV1/logs/cscwd_router/vat_bl_seed0_gpu1.log 2>&1 < /dev/null &
```

## 5. 查看与恢复

```bash
tail -f /home/fb/src/paper/gazelleV1/logs/cscwd_router/gazefollow_bl_seed0_gpu0.log
tail -f /home/fb/src/paper/gazelleV1/logs/cscwd_router/vat_bl_seed0_gpu1.log
```

`EPOCH` 表示阶段与轮次，`TEST` 表示评测进度，`QUEUE_COMPLETE` 表示该数据集 B/L 都已完成。第一项失败后队列停止，不会继续 L；修正后重新执行对应的同一条命令即可。已完成且哈希一致的模型会跳过，其余从最后已提交 epoch 恢复。

只需单独跑一个尺寸时，把同一命令的 `--sizes b l` 改为 `--sizes b` 或 `--sizes l`。两个尺寸没有产物依赖。

## 6. 输出与回传

四份小报告：

```text
cscwd_router/results/testselected_k050_seed0/gazefollow/vitb/summary.json
cscwd_router/results/testselected_k050_seed0/gazefollow/vitl/summary.json
cscwd_router/results/testselected_k050_seed0/vat/vitb/summary.json
cscwd_router/results/testselected_k050_seed0/vat/vitl/summary.json
```

各目录还有 manifest、support_history.json、sparse_decoder_history.json。大文件在以下根目录的 `<dataset>/vit<b|l>/` 下：

```text
/home/fb/src/paper/gazelleV1/experiments/cscwd_router/testselected_k050_seed0/
```

每组生成 support/final.pt、support/last.pt、sparse_decoder/best_test.pt、sparse_decoder/last.pt 和 predictions.jsonl.gz。压缩明细含逐查询指标、坐标和样本身份，供统计与案例选样；不保存伪造热图。所有大工件自动登记路径与 SHA，不进 Git。

全部完成后在服务器回传：

```bash
cd /home/fb/src/paper/gazelleV1
git add cscwd_router/results/testselected_k050_seed0 cscwd_router/results/RESULT_PROVENANCE.md
git commit -m "Record seed0 B and L Router train-test results"
git push origin codex/coverage-aware-router
```

## 7. 初始化与本轮检查

`code/stag/dinov3_initializers.py` 固定了四个既有纯 raw-concat decoder，无 SASA/GGSF/Router 模块。它们均已在服务器 CPU 只读检查：GF-B/GF-L 各匹配 40 个 decoder tensors，VAT-B/VAT-L 各匹配 45 个，缺失/多余/形状不符均为 0。两份 DINOv3 backbone 也已核对 SHA；新入口启动时再次核对权重并记录数据标注 SHA。

历史 decoder 是初始化，不将其训练 seed 反写为 0；本轮不继承 seed3106 的已训练 Router，四组 Router 都重新初始化。各新阶段以 base seed 0 复现，epoch 子 seed 为 `0 + epoch`。

工程记录见 `results/e0_dinov3_train_test_checks.json`。E0 当时仅执行 CPU 合成闭环与只读工件检查；E2 已回传真实 VAT 逐帧精度，见上述审查。尚无新的 GPU 效率数字，E3 将对实际选中 checkpoint 测量。
