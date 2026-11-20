# CSCWD：GazeSpot + ConvLSTM

2026-11-20，从新目录开始。目标是在GazeSpot上加入因果时序模块，提高VAT的AUC、L2、AP，保持官方train/test。

ConvLSTM先作为时序基线。基线有效后，再围绕具体不足设计一个新模块，以新增收益支撑创新点。

```text
cscwd/
├── implementation_plan.md  # 当前方案、执行约定与进展
├── code/
│   ├── data_prep/
│   ├── stag/
│   └── scripts/
├── results/                # 新实验的小JSON/MD及provenance
└── latex/                  # 新论文材料
```

已实现冻结任务特征缓存和ConvLSTM基线。真实首轮完成6轮后按用户指令停止，第7轮test中断；尚未观察到有效的AUC/L2/AP共同增益，真实快照见 [stopped summary](results/convlstm_v1_stopped_summary.json)。基线本机检查见 [LOCAL_CHECKS.md](results/LOCAL_CHECKS.md)。

**当前工作是train/test人工失败案例分析**：复用已有原始逐帧预测，导出当前/历史图像，人工标注共同失败机制后再确定新方案。CPU导出命令和离线浏览方式见 [FAILURE_REVIEW.md](FAILURE_REVIEW.md)。下方训练命令保留用于历史复现，当前不重启训练。

旧目录全部归档到 `../cscwd_v0/`，其代码、文献、论文材料和结果均保留；旧根进度见 `../cscwd_v0/PROGRESS_SNAPSHOT.md`。历史脚本对应归档前提交 `a77c8be`，内部原路径/SHA保持原样；新代码独立建立，不从归档代码导入。

本机只实现与检查，服务器由用户手动运行；代码与小结果经Git传递。大型特征/checkpoint留服务器，路径和SHA记入 `results/RESULT_PROVENANCE.md`。

## 实现入口

- `code/data_prep/task_cache.py`：用原GazeSpot的两个head输入钩子提取 `X[256,32,32]` / `z[256]`。原模型eval/冻结、FP32，不做增强；同图多人合并一次backbone。每split输出 `x.npy`、`z.npy`、`heads.pt` 和 `index.json`，保留轨迹/原帧号/图像/数组行映射和来源SHA。
- `code/stag/model.py`：256→64空间/token输入投影、共享3×3单向ConvLSTM、64→256双零初始化残差输出投影。原预测头固定为eval并冻结，但保留对输入的梯度。
- `code/stag/data.py` / `engine.py`：按轨迹保序分chunk，换人/原帧号缺口清空H/C，跨chunk保留并detach。只有轨迹顺序按seed打乱；padding不计loss/指标，GT不进入时序模块。
- `code/scripts/train_convlstm.py`：训练仅访问缓存，先评测冻结GazeSpot-B，再完整train并逐epoch完整test。定位仅GT画内，AP全人物帧；使用原 `gazelle.utils.vat_auc/vat_l2`。
- `code/scripts/run_baseline.py`：手动服务器入口，依次串起两个split缓存与10轮训练。已有缓存/实验目录不覆盖；显式 `--reuse-cache` 可复用已完成且匹配输入的缓存，训练输出仍使用新目录。

首轮固定seed0、epochs10、clip16、batch4、hidden64，AdamW lr0.0001、weight_decay0.01、gradient_clip_norm1.0，损失 `heatmap_MSE + 0.1 * inout_BCE`。

## 服务器手动启动

先由用户登录 `ssh fb@3090.lab`，在服务器执行以下整段命令。日期后缀保证使用新目录。原仓库的本地DINO代码/预训练权重和现有Python环境继续复用，入口从仓库根目录运行。

```bash
cd /home/fb/src/paper/gazelleV1
git switch codex/coverage-aware-router
git pull --ff-only origin codex/coverage-aware-router

RUN_ID="convlstm_v1_seed0_$(date +%Y%m%d_%H%M%S)"
CACHE_ROOT="/newhome/fb/dataset/videoattentiontarget/cscwd_task_features_v1/${RUN_ID}"
OUTPUT_DIR="/home/fb/src/paper/gazelleV1/experiments/cscwd_convlstm_v1/${RUN_ID}"
REPORT_DIR="/home/fb/src/paper/gazelleV1/cscwd/results/${RUN_ID}"
LOG_PATH="/home/fb/src/paper/gazelleV1/experiments/cscwd_convlstm_v1/logs/${RUN_ID}.log"
mkdir -p /home/fb/src/paper/gazelleV1/experiments/cscwd_convlstm_v1/logs

nohup env CUDA_VISIBLE_DEVICES=1 \
  /home/fb/anaconda3/envs/py310/bin/python -u cscwd/code/scripts/run_baseline.py \
  --data-root /newhome/fb/dataset/videoattentiontarget \
  --train-tracks /newhome/fb/dataset/videoattentiontarget/cscwd_tracks/train_preprocessed_tracks.json \
  --test-tracks /newhome/fb/dataset/videoattentiontarget/cscwd_tracks/test_preprocessed_tracks.json \
  --checkpoint /home/fb/src/paper/gazelleV1/experiments/train_vat_sasa_ggsf/2026-03-12_19-24-13/epoch_7.pt \
  --cache-root "$CACHE_ROOT" \
  --output-dir "$OUTPUT_DIR" \
  --report-dir "$REPORT_DIR" \
  --provenance /home/fb/src/paper/gazelleV1/cscwd/results/RESULT_PROVENANCE.md \
  --device cuda:0 \
  --cache-batch-size 4 \
  --num-workers 4 \
  --seed 0 \
  --epochs 10 \
  --clip-frames 16 \
  --batch-size 4 \
  --hidden-channels 64 \
  --learning-rate 0.0001 \
  --weight-decay 0.01 \
  --gradient-clip-norm 1.0 \
  --log-every 100 \
  > "$LOG_PATH" 2>&1 < /dev/null &
echo "PID=$! RUN_ID=$RUN_ID LOG=$LOG_PATH"
```

`CUDA_VISIBLE_DEVICES=1` 指定物理GPU1，程序内固定 `cuda:0`。两份轨迹JSON的全部人物帧都进入缓存和对应train/test；没有validation或稀疏抽帧。

## 输出与选择协议

大型工件只留服务器：缓存位于 `$CACHE_ROOT/{train,test}/`，checkpoint位于 `$OUTPUT_DIR/`，含 `epoch_01.pt` 至 `epoch_10.pt`、`selected.pt`、`last.pt`。checkpoint含模型、优化器、配置、该轮完整指标和来源元数据。

小报告位于 `$REPORT_DIR/`：

- `run_metadata.json`：实际命令、Python/GPU、代码提交、配置、来源与缓存SHA。
- `baseline_metrics.json`：相同test人物帧和当前真实头部框下的冻结GazeSpot-B。
- `epoch_01_metrics.json` 等：每轮train loss和完整test AUC/L2/AP、人物帧数。
- `training_history.json`：完整history和当前selected epoch。
- `artifacts.json`：每轮checkpoint与selected/last的服务器路径和SHA。
- `comparison.json` / `comparison.md`：基准与所选一整轮的AUC/L2/AP及JSON差值。

**test参与checkpoint选择，无validation。** 整轮排序：AUC更高 → L2更低 → AP更高 → 较早epoch；不拼接不同轮的最好列。定位按所有GT画内样本计分，不按预测InOut过滤，AP包含所有有效人物帧。运行时向 [RESULT_PROVENANCE.md](results/RESULT_PROVENANCE.md) 写入真实来源和大型工件路径/SHA。

## 小结果经Git回传

训练结束后在服务器执行（重新登录时先将 `RUN_ID` 设为日志打印的实际值）：

```bash
cd /home/fb/src/paper/gazelleV1
git add "cscwd/results/${RUN_ID}" cscwd/results/RESULT_PROVENANCE.md
git commit -m "Record CSCWD ConvLSTM baseline ${RUN_ID}"
git push origin codex/coverage-aware-router
```

随后本机拉取：

```bash
cd /Users/fanb/src/gazelle
git pull --ff-only origin codex/coverage-aware-router
```

缓存NPY和模型PT由忽略规则排除，新的嵌套小JSON报告允许正常 `git add`。

## 本机合成检查

可用安装了PyTorch、torchvision、timm、NumPy、scikit-learn、Pillow、pytest的Python执行：

```bash
python -m pytest -q cscwd/code/data_prep/test_task_cache.py \
  cscwd/code/stag/test_model.py cscwd/code/scripts/test_baseline.py
```

不要求本机有VAT图片、DINO权重或服务器连接；所有合成缓存和checkpoint写入临时目录。
