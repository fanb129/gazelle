# VAT-L：多指标候选的全量逐帧复评

逐轮test比较已完成，见 [VATL_EPOCH_SELECTION_REVIEW.md](results/VATL_EPOCH_SELECTION_REVIEW.md)。旧L2规则选第1轮；当前rate-6三指标综合首选第3轮，AP最高为第5轮。Pareto候选为第1/2/3/5/8轮，本次将这些候选统一到rate-1全量test再比较。

此入口不训练模型。第1轮复用已核对的全量结果，其余四轮各做一次FP32、K50、person-query test，保存各自AUC/L2/AP及逐查询导出。完成后报告Pareto与三指标等权相对损失的推荐；原训练best和论文表中的第1轮来源保留。候选评测完成后再确定正式模型并更新其效率。

代码使用已准备的独立目录 `/home/fb/src/paper/gazelleV1_results_vatl_20261124`，真实checkpoint来自原训练目录的 `epoch_0.pt / epoch_1.pt / epoch_2.pt / epoch_4.pt / epoch_7.pt`；CLI的 `--epochs` 使用 **1-based** 编号。沿用seed0、官方test、RGB和GT头框，不另切val。

GPU0空闲后由用户手动运行：

```bash
cd /home/fb/src/paper/gazelleV1_results_vatl_20261124
mkdir -p /home/fb/src/paper/gazelleV1/logs/cscwd_router
CUDA_VISIBLE_DEVICES=0 setsid nohup /home/fb/anaconda3/envs/py310/bin/python -u \
  /home/fb/src/paper/gazelleV1_results_vatl_20261124/cscwd_router/code/scripts/compare_vat_epochs.py \
  --source-summary /home/fb/src/paper/gazelleV1/cscwd_router/results/vat_from_gfl_router_k050_joint_seed0/summary.json \
  --epochs 1 2 3 5 8 \
  --data-path /newhome/fb/dataset/videoattentiontarget \
  --run-dir /home/fb/src/paper/gazelleV1/experiments/cscwd_router/vatl_joint_seed0_epoch_review \
  --report-dir /home/fb/src/paper/gazelleV1/cscwd_router/results/vatl_joint_seed0_epoch_review \
  --provenance /home/fb/src/paper/gazelleV1/cscwd_router/results/RESULT_PROVENANCE.md \
  --batch-size 4 --workers 8 --seed 0 --device cuda --resume \
  > /home/fb/src/paper/gazelleV1/logs/cscwd_router/vat_l_epoch_comparison_gpu0.log 2>&1 < /dev/null &
```

```bash
tail -f /home/fb/src/paper/gazelleV1/logs/cscwd_router/vat_l_epoch_comparison_gpu0.log
```

成功标志：`COMPLETE .../vatl_joint_seed0_epoch_review/summary.json`。小报告包含 `manifest.json`、五份 `epoch_XXX.json`、`summary.json`、`summary.md`；逐查询压缩导出在上述experiments目录，其路径与SHA会写入provenance。小JSON/MD经Git回传，大文件保留服务器。

恢复使用同一命令，`--resume`核对配置、源码、checkpoint与已有导出的SHA后复用完成项。只做CPU预检可在完整参数后增加 `--check-only`，去掉nohup、后台和日志重定向。
