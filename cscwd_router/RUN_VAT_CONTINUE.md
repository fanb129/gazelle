# VAT K50 低学习率续训（E2-C，2026-11-23）

最新状态：B/L 各追加 8 轮已完成，经 `8f2ae9c` 回传核验。新 L2 均未超过旧 best，因此最终保留原第 4/2 轮；不要继续原样追加。分析见 [E1_GFB_E2C_REVIEW.md](results/E1_GFB_E2C_REVIEW.md)，下列命令保留供复现/恢复。

当前提升实验已转为完整 GF-B Router/decoder→VAT，见 [RUN_GF_ROUTER_TO_VAT.md](RUN_GF_ROUTER_TO_VAT.md)；本页的固定 Router 续训不再重跑。

## 1. 当前起点与判断

本轮 VAT 的初始化确实有 GF 迁移，不是随机 decoder：

- B：GF `train_gazefollow_baseline/2026-02-10_13-03-16/epoch_14.pt` → VAT `train_vat_baseline/2026-03-11_19-58-20/epoch_7.pt` → 新 VAT support + sparse decoder。
- L：GF `train_gazefollow_vitl/2026-01-30_13-14-15/epoch_14.pt` → VAT `train_vat_vitl/2026-02-02_09-44-48/epoch_7.pt` → 新 VAT support + sparse decoder。

服务器 W&B 的 config 与输出日志已交叉对应到精确目录，路径/SHA 和白名单配置见 `results/e2_vat_initialization_trace.json`。旧 VAT 均训练 8 epochs、decoder LR `1e-5`、batch 60、rate 6；B in/out LR `1e-3`，L `1e-4`。旧末轮日志的 rate-6 AUC/L2/AP 分别为 B `0.939097/0.103012/0.893611`、L `0.945401/0.093716/0.915586`，不当成当前代码逐帧复测值。

MM Table 4 的 DINOv3 ViT-B/16 raw-concat 无 SASA/GGSF 行确实是 VAT `0.937/0.103`。本轮新 Router 用 decoder LR `1e-4`，比旧 VAT 微调高 10 倍，且 5 轮中只有 3 轮 K50。已有轨迹支持进一步适配，但尚未验证是哪一项造成差距。

Decoder 的 GF 迁移已存在；本轮新 Router 则只在 VAT 上学习，未继承正在训练的 GF Router。GF 完成后，可再比较迁移完整 GF Router + decoder 的路线；本入口先完成现有 VAT 模型的续训，不改 GF。

## 2. 续训行为

- 从各自 VAT `sparse_decoder/last.pt`（第 5 轮末）继续，Router/backbone 冻结；不重新做 support，不新增 dense 训练。
- 固定 K50，再训练 8 epochs。新建 AdamW，decoder LR `1e-5`、已适配的 in/out head LR `1e-4`、weight decay 0，8 轮 cosine；这是低学习率再适配，不把已到 0 的旧 scheduler 当成有效续训。
- base seed 仍为 0，新 epoch 子 seed 为 `0 + global_epoch`（从 5 起）；B/L decoder 有效 batch 仍为 32。
- 首先用原始 VAT dense 初始化在 rate-6 test 上测 native dense 与 routed K100，写 `probes.json`。两条路径 AUC/L2/AP 差值超过 `1e-8` 时停止并留下诊断；不带着前向不一致继续训练。这只是推理，没有 dense 训练。
- 每轮 rate-6 test，按 L2 最小、AUC 最大、较早 epoch 选一个模型；**旧 best 仍参与选模**。最终只对选中模型做 rate-1 test，不根据 rate-1 结果重新挑 epoch。
- 原始 E2 结果不覆盖。新输出有原子 `last.pt`、`best_test.pt`、逐查询 gzip；中断用同一命令恢复。

本机 39/39 CPU 检查通过，包含 B/L 合成续训、父工件不变、旧 best 保留、固定 K50、fresh LR、恢复一致和 K100 probe 拒绝路径；这是实现时的 CPU 记录；用户已在服务器完成本轮真实续训和 probes，结果见上述审查。助手本轮仅核验产物，未启动新训练或推理。

## 3. 更新代码

本机由助手已创建本地 commit，用户 push：

```bash
cd /Users/fanb/src/gazelle
git push origin codex/coverage-aware-router
```

服务器保留尚未提交的结果登记并更新。`--autostash` 已核对服务器 Git 2.25.1 支持；本次新增入口，原 GF 训练源码不改。

```bash
cd /home/fb/src/paper/gazelleV1
git pull --rebase --autostash origin codex/coverage-aware-router
mkdir -p /home/fb/src/paper/gazelleV1/logs/cscwd_router
```

## 4. GPU 1：B 续训后接 L

由用户手动执行：

```bash
CUDA_VISIBLE_DEVICES=1 setsid nohup /home/fb/anaconda3/envs/py310/bin/python -u \
  /home/fb/src/paper/gazelleV1/cscwd_router/code/scripts/continue_router_dinov3.py \
  --size all \
  --parent-report-root /home/fb/src/paper/gazelleV1/cscwd_router/results/testselected_k050_seed0/vat \
  --data-path /newhome/fb/dataset/videoattentiontarget \
  --output-root /home/fb/src/paper/gazelleV1/experiments/cscwd_router/vat_continue_k050_lr1e5_seed0 \
  --report-root /home/fb/src/paper/gazelleV1/cscwd_router/results/vat_continue_k050_lr1e5_seed0 \
  --provenance /home/fb/src/paper/gazelleV1/cscwd_router/results/RESULT_PROVENANCE.md \
  --additional-epochs 8 --decoder-lr 0.00001 --inout-lr 0.0001 \
  --batch-b 16 --batch-l 4 --eval-batch-b 16 --eval-batch-l 4 \
  --accumulation-b 2 --accumulation-l 8 --workers 8 \
  --clip-norm 1.0 --seed 0 --log-every 50 --device cuda --resume \
  > /home/fb/src/paper/gazelleV1/logs/cscwd_router/vat_continue_k050_lr1e5_seed0_gpu1.log 2>&1 < /dev/null &
```

只跑一个规模时将 `--size all` 改为 `--size b` 或 `--size l`。日志中的 `CONTINUE epoch=6` 是旧第 5 轮之后的第 1 个新增 epoch；`QUEUE_COMPLETE vat_continuation sizes=b,l` 表示两种规模完成。

## 5. 输出与回传

小报告：`cscwd_router/results/vat_continue_k050_lr1e5_seed0/{vitb,vitl}/{manifest,probes,history,summary}.json`。

大文件：`experiments/cscwd_router/vat_continue_k050_lr1e5_seed0/{vitb,vitl}/{last.pt,best_test.pt,predictions.jsonl.gz}`，实际生成后自动登记路径和 SHA，不进 Git。

```bash
cd /home/fb/src/paper/gazelleV1
git add cscwd_router/results/vat_continue_k050_lr1e5_seed0 cscwd_router/results/RESULT_PROVENANCE.md
git commit -m "Record low-LR VAT K50 continuation results"
git push origin codex/coverage-aware-router
```

原 `--decoder-epochs 5 --resume` 不用于追加 epoch；修改旧 run 的 epoch 数会违反其配置指纹。续训使用本文件的新入口和新目录。
