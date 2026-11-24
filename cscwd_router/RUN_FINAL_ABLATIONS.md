# 维护前：两卡条件化消融

W6-A。代码只新增在 `cscwd_router/code/`，旧 `gazelle/` 未改。47项本机CPU/回归通过，包含full模式原版输出逐值一致和三个条件分支的GF→VAT闭环。这里新增真实训练由用户手动启动，尚无新准确率结果。多人/K100/真实可视化入口见 `RUN_FINAL_EVALUATION.md`。当前仅使用GPU0/1，三个条件均保留；GPU0串行两分支，GPU1一分支。

固定训练worktree：`/home/fb/src/paper/gazelleV1_final_training`，由助手通过Git部署并CPU预检。训练期间保持此目录版本不变。GPU0的外层队列入口位于独立评测目录，但实际训练始终调用此固定目录的代码。三个分支使用原GF-B decoder初始化和冻结DINOv3-B：support3＋sparse decoder5，再从**各自**GF Router/decoder迁移VAT8轮（首轮固定Router）。seed0，batch16×累积2；仅train/test，GF按Min L2、VAT按L2选同一个checkpoint，VAT最终逐帧test。

关闭geometry时，同时屏蔽head descriptor中的4维绝对头框坐标和6通道相对几何。关闭head appearance时，屏蔽池化头部外观；scene-only两类均关闭。头部/escape优先规则保持一致，主表full模型直接复用已有结果。

## GPU0：scene only → scene + head appearance（自动串行）

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
  > /home/fb/src/paper/gazelleV1/logs/cscwd_router/final_ablation_queue_gpu0.log 2>&1 < /dev/null &
```

## GPU1：scene + geometry

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

## 产物与恢复

每分支的报告为 `final_ablation_seed0/<condition>/gazefollow/vitb/summary.json` 与 `.../vat/vitb/summary.json`；history、manifest、initialization等小报告同目录。PT和逐查询压缩导出保存在 `experiments/cscwd_router/final_ablation_seed0/<condition>/...`，自动登记路径/SHA。队列最后打印 `COMPLETE conditional branch <condition>`。

恢复使用同一命令和同一worktree；已完成GF会核对来源后跳过，VAT从原子last恢复。CPU预检用相同参数增加 `--check-only`，去掉nohup/后台/重定向，不执行模型forward。子阶段的全部低层超参数会作为 `START [...]` 写入日志，并进入manifest recipe。

训练完成后回传 `cscwd_router/results/final_ablation_seed0/` 的JSON与provenance；大PT/压缩预测不进Git。当前不使用原VAT-L epoch比较入口。

GPU0日志最后应打印 `COMPLETE conditional queue scene_only,scene_head`；GPU1为 `COMPLETE conditional branch scene_geometry`。已有任何分支在跑时，不重复启动该分支。队列使用同一命令恢复，先核对已完成分支再继续下一条；第一条失败时不会开始第二条。GPU2不使用。
