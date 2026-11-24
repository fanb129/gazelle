# 维护前：两卡多人评测、K100精度与可视化

服务器CPU预检已通过并回传：训练目录固定 `c19ba93`，评测目录固定 `1fed049`（代码 `d29d1ae`），见 [预检报告](results/w6_final_server_preflight.json)。两目录已就绪，使用本文对应目录的命令，无需在训练期间pull源码。

评测代码使用独立目录 `/home/fb/src/paper/gazelleV1_final_evaluation`，与固定训练目录分开。助手统一Git部署和CPU预检。只用GPU0/1：先完成两卡条件化训练；GPU0队列结束后跑多人（计时独占GPU0），GPU1先跑K100，再跑可视化。后面两个GPU1命令依次手动启动，不能同时启动。尚无本轮新GPU指标。

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

## GPU1：K100结束后，真实机制图与MM式预测对比

```bash
cd /home/fb/src/paper/gazelleV1_final_evaluation
mkdir -p /home/fb/src/paper/gazelleV1/logs/cscwd_router
CUDA_VISIBLE_DEVICES=1 MPLCONFIGDIR=/tmp/gazelle-mpl setsid nohup /home/fb/anaconda3/envs/py310/bin/python -u \
  /home/fb/src/paper/gazelleV1_final_evaluation/cscwd_router/code/scripts/export_final_visualizations.py \
  --gf-summary /home/fb/src/paper/gazelleV1/cscwd_router/results/testselected_k050_seed0/gazefollow/vitl/interrupted_summary.json \
  --vat-summary /home/fb/src/paper/gazelleV1/cscwd_router/results/vat_from_gfl_router_k050_joint_seed0/summary.json \
  --multi-summary /home/fb/src/paper/gazelleV1/cscwd_router/results/vat_from_gfb_router_k050_joint_seed0/summary.json \
  --gf-data-path /newhome/fb/dataset/gazefollow_extended \
  --vat-data-path /newhome/fb/dataset/videoattentiontarget \
  --official-checkpoint-dir /home/fb/src/paper/gazelleV1/checkpoints \
  --report-root /home/fb/src/paper/gazelleV1/cscwd_router/results/final_visualizations_seed0 \
  --figure-root /home/fb/src/paper/gazelleV1/cscwd_router/latex/figures/final_seed0 \
  --provenance /home/fb/src/paper/gazelleV1/cscwd_router/results/RESULT_PROVENANCE.md \
  --candidate-count 32 --examples-per-dataset 4 --multi-candidate-count 16 --multi-examples 2 \
  --seed 0 --device cuda --allow-closed-gf --resume --render \
  > /home/fb/src/paper/gazelleV1/logs/cscwd_router/final_visualizations_gpu1.log 2>&1 < /dev/null &
```

导出10个真实案例（GF4、VAT4、多人2），默认生成6张图，每张PNG/SVG两种格式：

- `gazefollow_comparison`、`vat_comparison`：参照MM图4同列同样例布局，六行依次为RGB头框、GT连线、条件化support、实际保留token、官方热图、Router热图。未保留patch变暗，GT星号、预测叉号；连线只连接真实GT或真实预测点。
- `vat_multi_00/01_mechanism`：同一RGB切换查询人物，各人support与独立K50 mask，最后一列为真实max-union和共享Top-K；下方显示共享路径各人预测及实际suffix patch数量。
- `vat_multi_00/01`：每人support、共享union与逐人独立的预测对照，并列真实误差、p(in)与patch执行总数。patch数量只指suffix空间位置，不当作整网FLOPs。

GF/VAT各从seed0固定的32候选中选2个较低误差和2个较高误差，保留错误案例；多人从seed0固定16个候选中按人物support的Jensen-Shannon差异选前2个，选样规则和分数写入JSON。多人图用于展示机制，不能代替整个5,182帧的量化表。热图比较使用每样例共同色标，不把support称为方向/FOV。原始预测和误差在数组压缩前计算，展示用小数组保留足够精度。

`results/final_visualizations_seed0/manifest.json` 与 `cases/*.json`、`latex/figures/final_seed0/`（含assets小RGB）一并进Git。case与RGB带SHA、相对引用，可在维护后无服务器、无原数据、无GPU重新排图。模型导出请求不绑定绘图脚本，换配色无需重新前向。

## CPU：三分支完成后汇总表4

```bash
cd /home/fb/src/paper/gazelleV1_final_evaluation
CUDA_VISIBLE_DEVICES="" setsid nohup /home/fb/anaconda3/envs/py310/bin/python -u \
  /home/fb/src/paper/gazelleV1_final_evaluation/cscwd_router/code/scripts/summarize_final_ablation.py \
  --report-root /home/fb/src/paper/gazelleV1/cscwd_router/results/final_ablation_seed0 \
  --full-gf-summary /home/fb/src/paper/gazelleV1/cscwd_router/results/testselected_k050_seed0/gazefollow/vitb/summary.json \
  --full-vat-summary /home/fb/src/paper/gazelleV1/cscwd_router/results/vat_from_gfb_router_k050_joint_seed0/summary.json \
  --output /home/fb/src/paper/gazelleV1/cscwd_router/results/final_ablation_seed0/table4_summary.json \
  --provenance /home/fb/src/paper/gazelleV1/cscwd_router/results/RESULT_PROVENANCE.md \
  > /home/fb/src/paper/gazelleV1/logs/cscwd_router/final_table4_cpu.log 2>&1 < /dev/null &
```

核对各条件的权重/数据/源码、同一训练配方与各自GF→VAT继承后输出JSON和MD。缺失分支默认报错；显式 `--allow-incomplete` 才输出空格，绝不补造结果。

## 本机CPU：Git回传后重新渲染

```bash
cd /Users/fanb/src/gazelle
MPLCONFIGDIR=/tmp/gazelle-mpl /private/tmp/gazelle-cscwd-check/bin/python \
  /Users/fanb/src/gazelle/cscwd_router/code/scripts/render_final_visualizations.py \
  --manifest /Users/fanb/src/gazelle/cscwd_router/results/final_visualizations_seed0/manifest.json \
  --output-dir /Users/fanb/src/gazelle/cscwd_router/latex/figures/final_seed0
```

保留 `render_manifest.json` 中的输入/绘图源码/图片SHA。这里不执行模型，不要求数据集或checkpoint。真实图尚需用户执行服务器导出命令后才会产生。
