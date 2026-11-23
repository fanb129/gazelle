# 三卡：GF-B/GF-L/VAT-B 效率与 GF-L→VAT-L

入口已实现，68 项 CPU 工程检查通过；没有在本机或服务器执行真实测速/训练。三张卡由用户手动启动下列命令。当前方案只测完整 forward，不重训 dense。

## 1. 更新代码

本机（助手已自动创建本地 commit）：

```bash
cd /Users/fanb/src/gazelle
git push origin codex/coverage-aware-router
```

服务器：

```bash
cd /home/fb/src/paper/gazelleV1
git pull --rebase --autostash origin codex/coverage-aware-router
mkdir -p /home/fb/src/paper/gazelleV1/logs/cscwd_router
```

## 2. GPU 0：GF-B 后接 GF-L 效率

```bash
CUDA_VISIBLE_DEVICES=0 setsid nohup /home/fb/anaconda3/envs/py310/bin/python -u \
  /home/fb/src/paper/gazelleV1/cscwd_router/code/scripts/run_router_efficiency.py \
  --summaries \
    /home/fb/src/paper/gazelleV1/cscwd_router/results/testselected_k050_seed0/gazefollow/vitb/summary.json \
    /home/fb/src/paper/gazelleV1/cscwd_router/results/testselected_k050_seed0/gazefollow/vitl/interrupted_summary.json \
  --data-path /newhome/fb/dataset/gazefollow_extended \
  --official-checkpoint-dir /home/fb/src/paper/gazelleV1/checkpoints \
  --report-root /home/fb/src/paper/gazelleV1/cscwd_router/results/efficiency_dinov3_seed0_20261124 \
  --provenance /home/fb/src/paper/gazelleV1/cscwd_router/results/RESULT_PROVENANCE.md \
  --batches 1 8 --precision fp16 --warmup 50 \
  --latency-iters 200 --throughput-iters 500 --repeats 5 --seed 0 \
  --allow-closed-gf --resume \
  > /home/fb/src/paper/gazelleV1/logs/cscwd_router/efficiency_gf_bl_gpu0.log 2>&1 < /dev/null &
```

## 3. GPU 1：当前 GF→VAT-B 模型的效率

```bash
CUDA_VISIBLE_DEVICES=1 setsid nohup /home/fb/anaconda3/envs/py310/bin/python -u \
  /home/fb/src/paper/gazelleV1/cscwd_router/code/scripts/run_router_efficiency.py \
  --summaries /home/fb/src/paper/gazelleV1/cscwd_router/results/vat_from_gfb_router_k050_joint_seed0/summary.json \
  --data-path /newhome/fb/dataset/videoattentiontarget \
  --official-checkpoint-dir /home/fb/src/paper/gazelleV1/checkpoints \
  --report-root /home/fb/src/paper/gazelleV1/cscwd_router/results/efficiency_dinov3_seed0_20261124 \
  --provenance /home/fb/src/paper/gazelleV1/cscwd_router/results/RESULT_PROVENANCE.md \
  --batches 1 8 --precision fp16 --warmup 50 \
  --latency-iters 200 --throughput-iters 500 --repeats 5 --seed 0 \
  --resume \
  > /home/fb/src/paper/gazelleV1/logs/cscwd_router/efficiency_vat_b_gpu1.log 2>&1 < /dev/null &
```

这里测的是最终逐帧 AUC/L2/AP `0.940033/0.100131/0.883183` 的第 1 轮模型，不使用旧 `testselected_k050_seed0/vat/vitb/` 的较低精度模型。

## 4. GPU 2：完整 GF-L Router/decoder→VAT-L

```bash
CUDA_VISIBLE_DEVICES=2 setsid nohup /home/fb/anaconda3/envs/py310/bin/python -u \
  /home/fb/src/paper/gazelleV1/cscwd_router/code/scripts/transfer_gf_router_to_vat.py \
  --gf-summary /home/fb/src/paper/gazelleV1/cscwd_router/results/testselected_k050_seed0/gazefollow/vitl/interrupted_summary.json \
  --data-path /newhome/fb/dataset/videoattentiontarget \
  --run-dir /home/fb/src/paper/gazelleV1/experiments/cscwd_router/vat_from_gfl_router_k050_joint_seed0 \
  --report-dir /home/fb/src/paper/gazelleV1/cscwd_router/results/vat_from_gfl_router_k050_joint_seed0 \
  --provenance /home/fb/src/paper/gazelleV1/cscwd_router/results/RESULT_PROVENANCE.md \
  --epochs 8 --router-warmup-epochs 1 \
  --router-lr 0.0001 --decoder-lr 0.00001 --inout-lr 0.001 \
  --router-loss-weight 1.0 --budget-weight 0.05 \
  --batch-size 4 --eval-batch-size 4 --grad-accumulation 8 \
  --workers 8 --clip-norm 1.0 --train-frame-rate 6 --test-frame-rate 6 \
  --seed 0 --log-every 50 --device cuda --allow-closed-gf --resume \
  > /home/fb/src/paper/gazelleV1/logs/cscwd_router/vat_from_gfl_router_k050_joint_seed0_gpu2.log 2>&1 < /dev/null &
```

继承 GF-L 第 2 轮的完整 Router + heatmap decoder（52 个共享 tensors），仅新增 5 个 in/out tensors，冻结 DINOv3-L。原 GF 的 completed=false 和实际 3+4 保留；`--allow-closed-gf` 会读取完整 epoch history、原 last 与导出 best，核对选择和逐值权重后才接受。

与 B 迁移保持同配方：固定 K50、8 epochs，首轮固定 Router、后 7 轮 coverage/BCE 联合适配；batch 4 × 累积 8，有效 batch 32，seed 0。仅官方 train/test，无 val；每轮 rate-6 test 按 L2 选单一 checkpoint，最后自动 rate-1 全 test。输出进入全新 L 目录，不覆盖 B 或旧 VAT-L。

## 5. 效率口径与恢复

每个模型测 4 个单元：B1/B8 × 正反序，各单元为一个新子进程。正序 official→K100→K50，反序 K50→K100→official。官方使用对应规模、对应数据集的 GazeLLE DINOv2 原始 checkpoint；K100/K50 使用当前精度结果对应的同一 Router checkpoint，K100 是同路径不剪 token 的控制，不是重新训练的 dense。

- 同一单元内官方与 Router 使用同一张 RTX 3090、同一原始 train RGB 与真实头框，每图一个查询；不使用 gaze 坐标选择输入。
- 官方输入 448，Router 输入 512，保留完整系统原配置。
- FP16、inference mode、autocast cache 关闭、TF32 关闭；包含 backbone、Router、gather/scatter、多层拼接和 decoder。数据读取、预处理、H2D、checkpoint 读取在计时外。
- 每 variant 50 warmup，200×5 synchronized latency，500×5 continuous throughput。保存 median/mean/p95、repeat medians、FPS、peak allocated memory、逐 block token 数、GPU UUID/温度/P-state、源码与权重 SHA。
- 每轮检查目标卡的其他 compute PID；该卡被其他任务占用时停止，不把共用卡结果写成正式效率。GPU 2 的 VAT-L 训练不使用 GPU 0/1。
- 每种顺序先算 official/K50 和 K100/K50 latency 比，再取几何平均；小于 1 的实测值照实保存，不预设加速 gate。

中断后重复原命令即可：已生成单元核对输入/权重/源码/全部计时参数后复用。源码或参数变了使用新 report-root；不混拼旧单元。

## 6. 输出与回传

效率小报告：`cscwd_router/results/efficiency_dinov3_seed0_20261124/<gazefollow|vat>/vit<b|l>/`，每组 `b1_forward.json`、`b1_reverse.json`、`b8_forward.json`、`b8_reverse.json`、`summary.json`、`summary.md`。没有效率大文件。

VAT-L 小报告：`cscwd_router/results/vat_from_gfl_router_k050_joint_seed0/`，含 manifest、initialization、initial_test、history、summary。大型 initial/epoch/last/best PT 和 predictions.jsonl.gz 留服务器 `experiments/cscwd_router/vat_from_gfl_router_k050_joint_seed0/`，自动登记路径/SHA。

完成后分别回传对应已完成目录；provenance 中其他未提交工作应单独保留，不整包提交无关结果。用户可直接通知助手按此前方式选择性 commit/push/pull 小报告。

```bash
tail -f /home/fb/src/paper/gazelleV1/logs/cscwd_router/efficiency_gf_bl_gpu0.log
tail -f /home/fb/src/paper/gazelleV1/logs/cscwd_router/efficiency_vat_b_gpu1.log
tail -f /home/fb/src/paper/gazelleV1/logs/cscwd_router/vat_from_gfl_router_k050_joint_seed0_gpu2.log
```

本机检查见 `results/e3_efficiency_vatl_local_checks.json`。它记录的是 68 项 CPU 工程检查，不是 GPU 精度/效率结果。
