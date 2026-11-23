# 完整 GF-B Router → VAT（E2-T）

更新：2026-11-23。当前优先提升 VAT 精度，正式效率测试等待显卡空闲。此入口已实现，本机 51/51 CPU 检查通过，尚未运行新的真实数据训练。

## 1. 这轮迁移什么

固定来源为已验收 GF-B 的第 5 轮 K50 模型，继承全部 GF Router、四层 raw-concat heatmap decoder，以及同一份冻结 DINOv3 backbone。没有加载旧 VAT decoder/Router。

- GF summary：`/home/fb/src/paper/gazelleV1/cscwd_router/results/testselected_k050_seed0/gazefollow/vitb/summary.json`。
- GF checkpoint：`/home/fb/src/paper/gazelleV1/experiments/cscwd_router/testselected_k050_seed0/gazefollow/vitb/sparse_decoder/best_test.pt`。
- checkpoint SHA256：`8297c6f621ceeeef6222e31a4e1782ee92d2fb50b697e9da31144cfe5b2a5b14`。
- GF 已跑指标：AUC/Avg/Min L2 `0.956236/0.101550/0.043420`；这不是 VAT 的预测成绩。

启动时核对 source SHA、模型配置和全部共享 tensor keys；共享权重读入后逐值核对。VAT 仅新增 `inout_token` 与 `inout_head` 的 5 个 tensors，用 seed 0 初始化。新增 token 会参与 decoder attention，不要求 VAT 初始 heatmap 与 GF 模型逐值一致。

## 2. VAT 训练规则

总计 **8 epochs**，全程 K50、FP32、纯 RGB、真实头框、单帧、冻结 backbone，不新增 dense 训练。

| 阶段 | Epoch | Router | Decoder / in-out |
|---|---|---|---|
| 迁移适配 | 1 | 保持 GF Router 固定 | 更新 decoder 与新增 in/out 分支 |
| 联合适配 | 2–8 | 低 LR 更新 | 继续更新 |

AdamW：Router/decoder/in-out 初始 LR 分别为 `1e-4/1e-5/1e-3`，weight decay 0，8 轮 cosine。每个参数组分别 clip norm 1。物理 batch 16，累积 2，有效 batch 32；base seed 0，epoch 子 seed 为 `0 + epoch`。

联合阶段的 loss 是 heatmap BCE + in/out BCE + 原有 Router coverage/budget loss，coverage weight 1、budget weight 0.05、budget target 0.5。Router 通过 coverage loss 更新；hard Top-K 没有被写成可从 heatmap BCE 获得梯度的可微选择器。没有引入新网络或时序模块。

使用完整官方 train/test 序列，不划 val；train 与逐轮 test 都每 6 帧采样。按 test L2 最小选一个 checkpoint，并列按 AUC、更早 epoch；全部最终指标来自该模型，再自动 rate-1 全 test。初始化阶段另记一次 K50 test，in/out 尚为随机分支，仅诊断，不参与 checkpoint 选择。

本次允许 Router 适配 VAT，与先前“冻结 VAT Router 继续训 decoder”的实验分开登记。当前不重复原续训，也不预写这次一定提升。

## 3. 更新代码

本机：

```bash
cd /Users/fanb/src/gazelle
git push origin codex/coverage-aware-router
```

服务器（保留尚未提交的结果登记）：

```bash
cd /home/fb/src/paper/gazelleV1
git pull --rebase --autostash origin codex/coverage-aware-router
mkdir -p /home/fb/src/paper/gazelleV1/logs/cscwd_router
```

本次只新增 transfer 入口和检查，不修改旧 `gazelle/` 或原 GF trainer 源码。GF-L 继续其原任务。

## 4. GPU 1 完整命令

训练需要可用显存；显卡有其他任务时训练会变慢，这种状态下不测正式效率。由用户在可安排 GPU 1 后手动启动：

```bash
CUDA_VISIBLE_DEVICES=1 setsid nohup /home/fb/anaconda3/envs/py310/bin/python -u \
  /home/fb/src/paper/gazelleV1/cscwd_router/code/scripts/transfer_gf_router_to_vat.py \
  --gf-summary /home/fb/src/paper/gazelleV1/cscwd_router/results/testselected_k050_seed0/gazefollow/vitb/summary.json \
  --data-path /newhome/fb/dataset/videoattentiontarget \
  --run-dir /home/fb/src/paper/gazelleV1/experiments/cscwd_router/vat_from_gfb_router_k050_joint_seed0 \
  --report-dir /home/fb/src/paper/gazelleV1/cscwd_router/results/vat_from_gfb_router_k050_joint_seed0 \
  --provenance /home/fb/src/paper/gazelleV1/cscwd_router/results/RESULT_PROVENANCE.md \
  --epochs 8 --router-warmup-epochs 1 \
  --router-lr 0.0001 --decoder-lr 0.00001 --inout-lr 0.001 \
  --router-loss-weight 1.0 --budget-weight 0.05 \
  --batch-size 16 --eval-batch-size 16 --grad-accumulation 2 \
  --workers 8 --clip-norm 1.0 \
  --train-frame-rate 6 --test-frame-rate 6 \
  --seed 0 --log-every 50 --device cuda --resume \
  > /home/fb/src/paper/gazelleV1/logs/cscwd_router/vat_from_gfb_router_k050_joint_seed0_gpu1.log 2>&1 < /dev/null &
```

同一命令支持中断恢复，原子 `last.pt` 内含 optimizer/scheduler、best state 和已提交 history；已完成结果核对 SHA 后跳过，不重复训练。

## 5. 输出与回传

小报告位于 `cscwd_router/results/vat_from_gfb_router_k050_joint_seed0/`：`manifest.json`、`initialization.json`、`initial_test.json`、`history.json`、`summary.json`。history 分别记录 heatmap/in-out/Router loss、Router 是否启用、真实更新数、可训练参数数、学习率、test 精度和 coverage。

大工件位于 `experiments/cscwd_router/vat_from_gfb_router_k050_joint_seed0/`：`initial.pt`、`epoch_0.pt`–`epoch_7.pt`、`last.pt`、`best_test.pt`、`predictions.jsonl.gz`。逐轮 compact checkpoint 便于后续整行重评候选，主报告仍按预定 L2 选择。所有 PT/gzip 留服务器，实际生成后自动登记路径/SHA，不进 Git。

```bash
cd /home/fb/src/paper/gazelleV1
git add cscwd_router/results/vat_from_gfb_router_k050_joint_seed0 cscwd_router/results/RESULT_PROVENANCE.md
git commit -m "Record complete GF Router to VAT transfer results"
git push origin codex/coverage-aware-router
```

本机工程报告为 `results/e2t_gf_to_vat_local_checks.json`，51 项包括 12 项新增与 39 项回归；覆盖 B/L 真实 factory 的共享 keys、新 in/out、分阶段梯度、hard Top-K 梯度来源、全画外 batch、完整合成闭环、原 GF 工件不变与恢复逐值一致。它们是工程检查，不是新 VAT 实验成绩。
