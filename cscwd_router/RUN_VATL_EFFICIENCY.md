# VAT-L：GPU 0 效率测试已完成

2026-11-24：原计划GPU1，用户实际改用GPU0，已完成全部四单元并回传审查。GPU UUID为 `GPU-9abd86a3-e5fb-bca1-4843-b29aefbfbd46`；原日志文件名仍保留 `gpu1`。B1/B8相对官方速度比为 `0.8630/1.0652×`，详见 [E3_VATL_EFFICIENCY_REVIEW.md](results/E3_VATL_EFFICIENCY_REVIEW.md)。

本次使用完整 GF-L Router→VAT-L 的 `best_test.pt`，不使用旧 VAT-L 初始化实验。来源报告为 `results/vat_from_gfl_router_k050_joint_seed0/summary.json`，最终逐帧 AUC/L2/AP 为 `0.9482644119 / 0.0896762148 / 0.8915918880`。训练结果已回传，审查见 [E2TL_GFL_TO_VATL_REVIEW.md](results/E2TL_GFL_TO_VATL_REVIEW.md)。

代码在独立 worktree `/home/fb/src/paper/gazelleV1_results_vatl_20261124`。该目录已由助手建立，链接原仓库的 `checkpoints`、`dinov2`、`dinov3`，包含 VAT 标注展开修复。原仓库训练目录和其他未提交结果保持原样。CPU 预检只检查输入和来源，不执行 GPU forward。

本次沿用 B 的正式协议：官方 DINOv2 ViT-L/14 GazeLLE、当前 DINOv3 多层 K100、同一选中权重 K50；batch 1/8，各做正反顺序。FP16，seed 0，每变体 50 warmup、200×5 latency、500×5 throughput。真实 VAT train RGB 和 GT 头框，每图一个查询；官方输入 448、Router 输入 512。K100 是同一 K50 权重的计算控制，不是另训 dense。完整 forward 含 backbone、Router 和 decoder，磁盘读图与预处理在计时外，记录显存、block tokens、GPU UUID 和输入/源码/权重 SHA。

以下保留本次实际GPU0的完整命令，输出目录与VAT-B分开；四个已完成单元可由 `--resume` 核对复用。

```bash
cd /home/fb/src/paper/gazelleV1_results_vatl_20261124
mkdir -p /home/fb/src/paper/gazelleV1/logs/cscwd_router
CUDA_VISIBLE_DEVICES=0 setsid nohup /home/fb/anaconda3/envs/py310/bin/python -u \
  /home/fb/src/paper/gazelleV1_results_vatl_20261124/cscwd_router/code/scripts/run_router_efficiency.py \
  --summaries /home/fb/src/paper/gazelleV1/cscwd_router/results/vat_from_gfl_router_k050_joint_seed0/summary.json \
  --data-path /newhome/fb/dataset/videoattentiontarget \
  --official-checkpoint-dir /home/fb/src/paper/gazelleV1/checkpoints \
  --report-root /home/fb/src/paper/gazelleV1/cscwd_router/results/efficiency_dinov3_vatl_seed0_20261124 \
  --provenance /home/fb/src/paper/gazelleV1/cscwd_router/results/RESULT_PROVENANCE.md \
  --batches 1 8 --precision fp16 --warmup 50 \
  --latency-iters 200 --throughput-iters 500 --repeats 5 --seed 0 --resume \
  > /home/fb/src/paper/gazelleV1/logs/cscwd_router/efficiency_vat_l_gpu1.log 2>&1 < /dev/null &
```

查看进度：

```bash
tail -f /home/fb/src/paper/gazelleV1/logs/cscwd_router/efficiency_vat_l_gpu1.log
```

成功标志为 `COMPLETE .../efficiency_dinov3_vatl_seed0_20261124/vat/vitl/summary.json`。目录内应有 `b1_forward.json`、`b1_reverse.json`、`b8_forward.json`、`b8_reverse.json`、`summary.json`、`summary.md`。恢复仍使用同一条命令；`--resume` 只复用输入、权重、源码及参数完全一致的单元。已开始同一矩阵后保持该 worktree 代码版本，避免请求 SHA 改变。

若单独重做 CPU 预检，在完整命令的 `--resume` 后增加 `--check-only`，去掉 `setsid nohup`、后台符号与日志重定向即可。审查脚本的完整复现命令为：

```bash
cd /home/fb/src/paper/gazelleV1_results_vatl_20261124
CUDA_VISIBLE_DEVICES="" /home/fb/anaconda3/envs/py310/bin/python -u \
  cscwd_router/code/scripts/review_vat_transfer.py \
  --summary /home/fb/src/paper/gazelleV1/cscwd_router/results/vat_from_gfl_router_k050_joint_seed0/summary.json \
  --official-checkpoint-dir /home/fb/src/paper/gazelleV1/checkpoints \
  --efficiency-report-root /home/fb/src/paper/gazelleV1/cscwd_router/results/efficiency_dinov3_vatl_seed0_20261124 \
  --provenance /home/fb/src/paper/gazelleV1/cscwd_router/results/RESULT_PROVENANCE.md \
  --output /home/fb/src/paper/gazelleV1_results_vatl_20261124/cscwd_router/results/e2tl_gfl_to_vatl_server_checks.json
```

六份关键 JSON/MD、CPU检查和 provenance 已经 `afcc39e` 经 Git 回传。本次 FP16 完整forward结果为 B1官方/K100/K50 `33.4730/39.4368/38.7860ms`、B8 `147.5857/181.8345/138.5573ms`；B8为整批时间。所有 `.pt/.pth` 与逐查询压缩导出仍留服务器，路径和 SHA 已登记在 `results/RESULT_PROVENANCE.md`。
