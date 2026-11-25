# 维护前：两卡条件化消融

2026-11-25 14:31收口：按用户同意停止scene-head及外层队列，进程组640838已无活动进程。执行停止前，scene-only GF已于11:23完成support3＋decoder5，VAT已于13:03完成8轮；这两组完成结果保留。scene-geometry GF/VAT也已完成。表4保留scene-only、scene＋geometry、full三行；不再启动scene-head或其他训练。

服务器CPU预检已通过并回传：训练目录固定 `c19ba93`，评测目录固定 `1fed049`（代码 `d29d1ae`），见 [预检报告](results/w6_final_server_preflight.json)。两目录已就绪，使用本文对应目录的命令，无需在训练期间pull源码。

W6-A入口与47项CPU/回归已完成；full模式与原版输出逐值一致。scene-only的小结果已回传，并核对25个工件、24份源码及逐查询指标；汇总见 [W7-D审查](results/W7D_ABLATION_REVIEW.md)。多人精度、四组K100与两组真实图均已完成，入口见 `RUN_FINAL_EVALUATION.md`。

固定训练worktree：`/home/fb/src/paper/gazelleV1_final_training`，由助手通过Git部署并CPU预检。训练期间保持此目录版本不变。GPU0的外层队列入口位于独立评测目录，但实际训练始终调用此固定目录的代码。三个分支使用原GF-B decoder初始化和冻结DINOv3-B：support3＋sparse decoder5，再从**各自**GF Router/decoder迁移VAT8轮（首轮固定Router）。seed0，batch16×累积2；仅train/test，GF按Min L2、VAT按L2选同一个checkpoint，VAT最终逐帧test。

关闭geometry时，同时屏蔽head descriptor中的4维绝对头框坐标和6通道相对几何。关闭head appearance时，屏蔽池化头部外观；scene-only两类均关闭。头部/escape优先规则保持一致，主表full模型直接复用已有结果。

<details>
<summary>历史启动命令（队列已停止，不再执行）</summary>

## 历史GPU0：scene only → scene + head appearance（自动串行）

```bash
cd /home/fb/src/paper/gazelleV1_final_evaluation
mkdir -p /home/fb/src/paper/gazelleV1/logs/cscwd_router
CUDA_VISIBLE_DEVICES=0 setsid nohup /home/fb/anaconda3/envs/py310/bin/python -u \
  /home/fb/src/paper/gazelleV1_final_evaluation/cscwd_router/code/scripts/run_final_ablation_queue.py \
  --training-code-root /home/fb/src/paper/gazelleV1_final_training \
  --conditions scene_only scene_head \
  --gf-data-path /newhome/fb/dataset/gazefollow_extended \
  --vat-data-path /newhome/fb/dataset/videoattentiontarget \
  --initialization-root /home/fb/src/paper/gazelleV1 \
  --artifact-root /home/fb/src/paper/gazelleV1/experiments/cscwd_router/final_ablation_seed0 \
  --report-root /home/fb/src/paper/gazelleV1/cscwd_router/results/final_ablation_seed0 \
  --provenance /home/fb/src/paper/gazelleV1/cscwd_router/results/RESULT_PROVENANCE.md \
  --workers 8 --batch-size 16 --eval-batch-size 16 --grad-accumulation 2 \
  --support-epochs 3 --decoder-epochs 5 --vat-epochs 8 --seed 0 --resume \
  > /home/fb/src/paper/gazelleV1/logs/cscwd_router/final_ablation_queue_resume_gpu0.log 2>&1 < /dev/null &
```

## 历史GPU1：scene + geometry

```bash
cd /home/fb/src/paper/gazelleV1_final_training
mkdir -p /home/fb/src/paper/gazelleV1/logs/cscwd_router
CUDA_VISIBLE_DEVICES=1 setsid nohup /home/fb/anaconda3/envs/py310/bin/python -u \
  /home/fb/src/paper/gazelleV1_final_training/cscwd_router/code/scripts/run_final_ablation.py \
  --condition scene_geometry \
  --gf-data-path /newhome/fb/dataset/gazefollow_extended \
  --vat-data-path /newhome/fb/dataset/videoattentiontarget \
  --initialization-root /home/fb/src/paper/gazelleV1 \
  --artifact-root /home/fb/src/paper/gazelleV1/experiments/cscwd_router/final_ablation_seed0 \
  --report-root /home/fb/src/paper/gazelleV1/cscwd_router/results/final_ablation_seed0 \
  --provenance /home/fb/src/paper/gazelleV1/cscwd_router/results/RESULT_PROVENANCE.md \
  --workers 8 --batch-size 16 --eval-batch-size 16 --grad-accumulation 2 \
  --support-epochs 3 --decoder-epochs 5 --vat-epochs 8 --seed 0 --resume \
  > /home/fb/src/paper/gazelleV1/logs/cscwd_router/final_scene_geometry_gpu1.log 2>&1 < /dev/null &
```

</details>

## 产物与恢复

每分支的报告为 `final_ablation_seed0/<condition>/gazefollow/vitb/summary.json` 与 `.../vat/vitb/summary.json`；history、manifest、initialization等小报告同目录。PT和逐查询压缩导出保存在 `experiments/cscwd_router/final_ablation_seed0/<condition>/...`，自动登记路径/SHA。队列最后打印 `COMPLETE conditional branch <condition>`。

恢复使用同一命令和同一worktree；已完成GF会核对来源后跳过，VAT从原子last恢复。CPU预检用相同参数增加 `--check-only`，去掉nohup/后台/重定向，不执行模型forward。子阶段的全部低层超参数会作为 `START [...]` 写入日志，并进入manifest recipe。

训练完成后回传 `cscwd_router/results/final_ablation_seed0/` 的JSON与provenance；大PT/压缩预测不进Git。当前不使用原VAT-L epoch比较入口。

本轮按用户决定主动停止，因此不等待GPU0整队列COMPLETE。scene-head只保留服务器上的部分checkpoint，不作正式消融；路径/SHA已登记provenance。停止记录 `results/w7d_ablation_scope_stop.json`，完成分支的CPU核对 `results/w7d_scene_only_server_checks.json`。本文历史resume命令会重启已取消分支，不作为后续执行项。
