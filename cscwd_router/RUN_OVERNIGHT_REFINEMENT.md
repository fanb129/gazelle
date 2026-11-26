# 今晚：VAT AP 与完整移除 Router 人物条件

2026-11-26。使用 GPU0／1；只 train/test，seed0。独立代码目录 `/home/fb/src/paper/gazelleV1_overnight_20261126`，代码与小报告经 `codex/overnight-refinement-20261126` 传递，原仓库和旧实验保持原样。

## 两条队列

- **GPU0：AP-B → AP-L**。从各自当前 VAT `best_test.pt` 出发，只训练 `inout_head`，固定 LR `1e-2`、8 轮、AdamW／weight decay0／clip1，不用 scheduler、不扫描 LR。冻结 backbone、Router、共享 decoder 和 inout token；共享模型 eval，仅分类 MLP train。沿用原增强训练，B batch16、L batch4；每轮 rate6 test 按 AP 选模，原模型也参与候选；最后 rate1 test 输出同一完整 checkpoint 的 AUC／L2／AP。
- **GPU1：unconditional GF-B → VAT-B**。Router 完全不读取头部外观、绝对框坐标和相对几何，不执行头部池化／人物投影，也不加入 head priority。场景打分和固定 escape 保留；decoder 仍用头框指定人物。复用旧 scene-only 的 3 轮 support 权重，从该阶段保存的原 decoder 开始重新适配5轮，再以相同配方迁移VAT8轮。原 support 阶段的可微损失只依赖软场景分布，head priority只影响离散选择，因此不重复其训练。该继承单独写入 `unconditional/inherited_support.json`。

完整移除的范围是 **Router 人物条件**。去掉 decoder 的人物提示会同时改变任务输入，不作为本轮路由消融。新条件是否产生更大指标差距，以实际结果为准。

## 完整运行命令

先确保代码已部署、工作目录内 `checkpoints`／`dinov2`／`dinov3` 指向原仓库；日志父目录存在。已启动后不重复执行。

GPU0：

```bash
cd /home/fb/src/paper/gazelleV1_overnight_20261126
nohup env CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
  /home/fb/anaconda3/envs/py310/bin/python -u \
  /home/fb/src/paper/gazelleV1_overnight_20261126/cscwd_router/code/scripts/run_overnight_refinement.py \
  --mode ap \
  --original-root /home/fb/src/paper/gazelleV1 \
  --gf-data-path /newhome/fb/dataset/gazefollow_extended \
  --vat-data-path /newhome/fb/dataset/videoattentiontarget \
  --artifact-root /home/fb/src/paper/gazelleV1/experiments/cscwd_router/overnight_20261126 \
  --report-root /home/fb/src/paper/gazelleV1_overnight_20261126/cscwd_router/results/overnight_20261126 \
  --provenance /home/fb/src/paper/gazelleV1_overnight_20261126/cscwd_router/results/overnight_20261126/RESULT_PROVENANCE.md \
  --publish-branch codex/overnight-refinement-20261126 \
  --epochs 8 --head-lr 0.01 --workers 4 --seed 0 --resume \
  > /home/fb/src/paper/gazelleV1/logs/cscwd_router/overnight_ap_gpu0_20261126.log 2>&1 < /dev/null &
```

GPU1：

```bash
cd /home/fb/src/paper/gazelleV1_overnight_20261126
nohup env CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
  /home/fb/anaconda3/envs/py310/bin/python -u \
  /home/fb/src/paper/gazelleV1_overnight_20261126/cscwd_router/code/scripts/run_overnight_refinement.py \
  --mode unconditional \
  --original-root /home/fb/src/paper/gazelleV1 \
  --gf-data-path /newhome/fb/dataset/gazefollow_extended \
  --vat-data-path /newhome/fb/dataset/videoattentiontarget \
  --artifact-root /home/fb/src/paper/gazelleV1/experiments/cscwd_router/overnight_20261126 \
  --report-root /home/fb/src/paper/gazelleV1_overnight_20261126/cscwd_router/results/overnight_20261126 \
  --provenance /home/fb/src/paper/gazelleV1_overnight_20261126/cscwd_router/results/overnight_20261126/RESULT_PROVENANCE.md \
  --publish-branch codex/overnight-refinement-20261126 \
  --epochs 8 --head-lr 0.01 --workers 4 --seed 0 --resume \
  > /home/fb/src/paper/gazelleV1/logs/cscwd_router/overnight_unconditional_gpu1_20261126.log 2>&1 < /dev/null &
```

子任务全部参数由队列生成并在日志 `START [...]` 中完整输出：GF decoder LR1e-4、K100→K75→K50、5轮；VAT Router／decoder／inout LR1e-4／1e-5／1e-3、8轮cosine、首轮冻结Router；batch16×累积2、K50测试。`--resume` 从完整 epoch 保存点恢复。

## 输出与回传

- 大工件：`/home/fb/src/paper/gazelleV1/experiments/cscwd_router/overnight_20261126/{ap/vitb,ap/vitl,unconditional/gazefollow/vitb,unconditional/vat/vitb}/`。
- 小报告：新 worktree 的 `cscwd_router/results/overnight_20261126/`；含 manifest、history、summary 和 provenance，每个子任务完成后自动本地 commit 并 push 指定实验分支。Git发布操作共用锁，不添加大文件。
- 正式论文继续使用已有数值，等新结果回传审查后再更新。新 AP 适配和原消融的训练阶段分别标明，不把新 full AP 与旧消融称为相同配方。
- 非独占卡只做训练／精度；今晚不重测效率。
