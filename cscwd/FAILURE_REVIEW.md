# GazeSpot：train / test 人工失败案例包

当前先做人工看图与机制归纳，不启动新模型训练。原ConvLSTM运行完成6轮后停止，真实记录见 [停止快照](results/convlstm_v1_stopped_summary.json)。

## 数据与抽样

直接读取服务器已完成的GazeSpot-B原始逐人物帧预测，无需重新推理：

```text
/newhome/fb/dataset/videoattentiontarget/cscwd_cache/gazespot_vitb_epoch7/
├── r1_train_predictions/   # 官方train，132,563人物帧
└── test/                   # 官方test，31,978人物帧
```

每目录的 `heatmap.npy`、`inout.npy`、`argmax_xy.npy`、`index.json` 均只读。导出器核验原始cache合同及数组SHA，按自己的旧索引读取原row，不依赖归档代码，也不读取新X/z或旧DINO特征缓存。

train/test独立抽样，各目标48组：

| 类别 | 每split目标数 | 筛选依据 |
|---|---:|---|
| large_l2 | 16 | GT画内L2最高10%中的随机案例 |
| inout_false_negative | 8 | GT画内，原InOut分数低于0.5 |
| inout_false_positive | 8 | GT画外，原InOut分数至少0.5 |
| medium_l2 | 8 | GT画内L2的40%–60%分位 |
| correct_control | 8 | 低L2画内正确与画外正确交错抽样 |

固定seed0；类别轮流抽取，全类别合计每轨迹最多2组、每视频最多6组；同轨迹最大历史窗口不重叠。选定后随机打乱展示顺序，避免从顺序推断错误类别。若不足则报告shortfall，不重复、不自动放宽规则。各类别的抽样数量不代表总体错误占比。

默认看当前帧和过去16帧，包内最多包含过去64帧，允许展开32/64；遇原帧号缺口或人物切换截断，不使用未来。全图仅等比缩小到最长边768像素；另提供从原RGB按真实头框四边扩30%提取、最长边192像素的头部局部图。RGB不叠加GT注视标注，热图预览为原概率乘255后的8bit图，未逐图重归一化，定量L2/分数来自原始数组。

**按用户要求，train和test均用于人工分析与方法开发，不新增validation；后续test不视为独立验证。** 该协议写入页面、manifest、报告和provenance。

## 服务器手动导出

用户登录 `ssh fb@3090.lab` 后执行。CPU导出，不占GPU，不加载模型、不做训练；使用新的输出目录。命令中的时间后缀需记下，后续取回同名ZIP。

```bash
cd /home/fb/src/paper/gazelleV1
git switch codex/coverage-aware-router
git pull --ff-only origin codex/coverage-aware-router

REVIEW_ID="failure_review_v1_$(date +%Y%m%d_%H%M%S)"
REVIEW_OUTPUT="/home/fb/src/paper/gazelleV1/experiments/cscwd_failure_review_v1/${REVIEW_ID}"
REVIEW_REPORT="/home/fb/src/paper/gazelleV1/cscwd/results/${REVIEW_ID}"
REVIEW_LOG="/home/fb/src/paper/gazelleV1/experiments/cscwd_failure_review_v1/logs/${REVIEW_ID}.log"
mkdir -p /home/fb/src/paper/gazelleV1/experiments/cscwd_failure_review_v1/logs

nohup env CUDA_VISIBLE_DEVICES="" \
  /home/fb/anaconda3/envs/py310/bin/python -u cscwd/code/scripts/build_failure_review.py \
  --data-root /newhome/fb/dataset/videoattentiontarget \
  --train-cache /newhome/fb/dataset/videoattentiontarget/cscwd_cache/gazespot_vitb_epoch7/r1_train_predictions \
  --test-cache /newhome/fb/dataset/videoattentiontarget/cscwd_cache/gazespot_vitb_epoch7/test \
  --output-dir "$REVIEW_OUTPUT" \
  --report-dir "$REVIEW_REPORT" \
  --provenance /home/fb/src/paper/gazelleV1/cscwd/results/RESULT_PROVENANCE.md \
  --seed 0 \
  --history-frames 16 \
  --max-history-frames 64 \
  --large-l2-cases 16 \
  --fn-cases 8 \
  --fp-cases 8 \
  --medium-cases 8 \
  --control-cases 8 \
  --score-threshold 0.5 \
  --large-l2-quantile 0.9 \
  --max-per-track 2 \
  --max-per-video 6 \
  --image-max-side 768 \
  --jpeg-quality 85 \
  --archive \
  > "$REVIEW_LOG" 2>&1 < /dev/null &
echo "PID=$! REVIEW_ID=$REVIEW_ID LOG=$REVIEW_LOG ZIP=${REVIEW_OUTPUT}.zip"
```

完成后 `$REVIEW_OUTPUT/` 含 `index.html`、`case_manifest.json`、`annotations_blank.json`、`assets/`；整个目录打包为 `${REVIEW_OUTPUT}.zip`。HTML内嵌案例数据、只读相对图片文件，可以在本机离线打开，无需运行服务或安装前端依赖。

`$REVIEW_REPORT/` 只放 `review_report.json`、`review_report.md` 和 `selected_cases.json`：记录每split候选与实际数量、全体分类混淆、L2分位数、抽样配置、来源和输出SHA。它们是案例选择统计，不是新模型性能。

## 取回本机并标注

将 `REVIEW_ID` 设为服务器打印的实际值，在本机执行：

```bash
REVIEW_ID="替换为服务器打印的实际REVIEW_ID"
LOCAL_CASE_ROOT="/Users/fanb/src/gazelle/experiments/cscwd_failure_review_v1"
mkdir -p "$LOCAL_CASE_ROOT"
scp "fb@3090.lab:/home/fb/src/paper/gazelleV1/experiments/cscwd_failure_review_v1/${REVIEW_ID}.zip" "$LOCAL_CASE_ROOT/"
unzip -q "$LOCAL_CASE_ROOT/${REVIEW_ID}.zip" -d "$LOCAL_CASE_ROOT/${REVIEW_ID}"
open "$LOCAL_CASE_ROOT/${REVIEW_ID}/index.html"
```

每组按顺序观察：

1. 当前帧：先写“当前困难”和“当前猜测”；默认不显示GT、L2、错误类别。
2. 展开历史：记录过去是否提供有用证据（yes/no/uncertain）及具体内容，注意目标是否已经切换。
3. 揭示GT：核对答案并填写失败机制/备注。首次揭示时间会留在标注中；隐藏GT不清除该时间。
4. 完成一批后导出JSON和CSV。浏览器本地保存只是便利，导出文件用于备份和后续共同分析。JSON可以导回同一个package，其他package不能误合并。

页面可过滤train/test和类别，浏览历史帧、原热图、预测点和头框。默认中性案例列表；主动按错误类别过滤会知道该类别标签，不再属于未知类别观察。播放速度仅为浏览速度，不能当作原视频帧率。

## 小报告与人工标注回传

导出完成后在服务器提交小报告：

```bash
cd /home/fb/src/paper/gazelleV1
git add "cscwd/results/${REVIEW_ID}" cscwd/results/RESULT_PROVENANCE.md
git commit -m "Record GazeSpot train/test failure review ${REVIEW_ID}"
git push origin codex/coverage-aware-router
```

本机拉取后，将浏览器导出的实际JSON/CSV复制到 `cscwd/results/${REVIEW_ID}/` 中，建议命名为 `manual_annotations.json` / `manual_annotations.csv`。JSON允许正常加入Git；CSV受仓库全局规则忽略，需明确 `git add -f`。图片、ZIP和原NPY不加入Git。

```bash
cd /Users/fanb/src/gazelle
git pull --ff-only origin codex/coverage-aware-router
git add "cscwd/results/${REVIEW_ID}/manual_annotations.json"
git add -f "cscwd/results/${REVIEW_ID}/manual_annotations.csv"
git commit -m "Annotate GazeSpot failure review ${REVIEW_ID}"
git push origin codex/coverage-aware-router
```

下一步依据重复出现且有具体图像证据的共同机制讨论改进；人工判断“可能被历史补救”只是一项待验证假设，不据此报告模型收益。
