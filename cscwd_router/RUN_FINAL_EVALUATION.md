# 维护前：多人评测与K100精度

评测代码使用独立目录 `/home/fb/src/paper/gazelleV1_final_evaluation`，与固定训练目录分开。可视化入口完成后助手统一Git部署和CPU预检。以下命令由用户在对应卡的训练结束后手动启动。本轮只完成实现/CPU检查，尚无新GPU指标。

## GPU0：VAT-B真实多人

```bash
cd /home/fb/src/paper/gazelleV1_final_evaluation
mkdir -p /home/fb/src/paper/gazelleV1/logs/cscwd_router
CUDA_VISIBLE_DEVICES=0 setsid nohup /home/fb/anaconda3/envs/py310/bin/python -u \
  /home/fb/src/paper/gazelleV1_final_evaluation/cscwd_router/code/scripts/run_final_multi.py \
  --source-summary /home/fb/src/paper/gazelleV1/cscwd_router/results/vat_from_gfb_router_k050_joint_seed0/summary.json \
  --data-path /newhome/fb/dataset/videoattentiontarget \
  --official-checkpoint /home/fb/src/paper/gazelleV1/checkpoints/gazelle_dinov2_vitb14_inout.pt \
  --artifact-root /home/fb/src/paper/gazelleV1/experiments/cscwd_router/final_multi_seed0 \
  --report-root /home/fb/src/paper/gazelleV1/cscwd_router/results/final_multi_seed0 \
  --provenance /home/fb/src/paper/gazelleV1/cscwd_router/results/RESULT_PROVENANCE.md \
  --task all --batch-size 4 --workers 4 --seed 0 --device cuda --precision fp16 \
  --warmup 50 --latency-iters 200 --throughput-iters 500 --repeats 5 --resume \
  > /home/fb/src/paper/gazelleV1/logs/cscwd_router/final_multi_gpu0.log 2>&1 < /dev/null &
```

质量评测固定VAT test中至少三个真实GT头框的5,182帧，seed0选三个GT人物，N=1/2/3嵌套；画外人物保留。FP32、按person算AUC/L2/AP与GT点覆盖，三个策略分别为official共享dense、同权重independent、共享union。independent把图像按人物展开后批量编码，union每图只编码一次；两者均实际K50，头部与escape规则相同。

计时使用独立选定的真实VAT train帧、同一嵌套GT查询，**batch_images=1**；FP16、50 warmup、200×5 latency和500×5 throughput，正反序，每个单元独立进程。包含复制图像、backbone、Router、suffix/回填与decoder；预处理/H2D在外。记录真实block batch/tokens、GPU UUID、images/s、queries/s与显存；query吞吐不当作image吞吐。

输出：`results/final_multi_seed0/cohort.json`、`accuracy/n{1,2,3}_{official,independent,union}.json`、`timing/n{1,2,3}_{forward,reverse}.json` 和 `summary.json/md`。逐查询压缩预测在artifact-root，不进Git。可以将 `--task all` 改为 `accuracy` 或 `timing` 分开运行；计时需要目标卡无其他compute PID。

## GPU1：四组同权重K100精度

```bash
cd /home/fb/src/paper/gazelleV1_final_evaluation
mkdir -p /home/fb/src/paper/gazelleV1/logs/cscwd_router
CUDA_VISIBLE_DEVICES=1 setsid nohup /home/fb/anaconda3/envs/py310/bin/python -u \
  /home/fb/src/paper/gazelleV1_final_evaluation/cscwd_router/code/scripts/run_final_k100.py \
  --summaries \
    /home/fb/src/paper/gazelleV1/cscwd_router/results/testselected_k050_seed0/gazefollow/vitb/summary.json \
    /home/fb/src/paper/gazelleV1/cscwd_router/results/testselected_k050_seed0/gazefollow/vitl/interrupted_summary.json \
    /home/fb/src/paper/gazelleV1/cscwd_router/results/vat_from_gfb_router_k050_joint_seed0/summary.json \
    /home/fb/src/paper/gazelleV1/cscwd_router/results/vat_from_gfl_router_k050_joint_seed0/summary.json \
  --gf-data-path /newhome/fb/dataset/gazefollow_extended \
  --vat-data-path /newhome/fb/dataset/videoattentiontarget \
  --artifact-root /home/fb/src/paper/gazelleV1/experiments/cscwd_router/final_k100_seed0 \
  --report-root /home/fb/src/paper/gazelleV1/cscwd_router/results/final_k100_seed0 \
  --provenance /home/fb/src/paper/gazelleV1/cscwd_router/results/RESULT_PROVENANCE.md \
  --batch-size 8 --workers 4 --seed 0 --device cuda --allow-closed-gf --resume \
  > /home/fb/src/paper/gazelleV1/logs/cscwd_router/final_k100_gpu1.log 2>&1 < /dev/null &
```

只载入原选中checkpoint，将keep ratio改为1，各运行一次FP32 full test；不训练、不换epoch、不改原summary或checkpoint，不扫描其他预算。保留GF-L的真实3+4/用户收口来源，通过原last/history/best核对后使用。报告包含原K50完整指标、K100完整指标、K100−K50、真实全token trace和相同查询人数检查。

输出：`results/final_k100_seed0/{gazefollow,vat}/vit{b,l}/summary.json`；逐查询大导出在artifact-root。两个入口均支持原命令 `--resume`，只复用相同权重/输入/源码/设置且工件SHA一致的结果。CPU预检在相同命令加 `--check-only` 并去掉nohup/后台/重定向，不执行模型forward。
