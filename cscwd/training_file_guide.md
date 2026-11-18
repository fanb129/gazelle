# 当前训练会读写哪些文件

适用协议：`r4_official_train_test_v1`，2026-11-18。本文说明当前官方 train/test 作业的实际代码路径、输入和输出。官方首轮已完成并验收，两模型均按AUC规则选epoch 5；真实结果解读见 `cscwd/results/r4_official_local_review.md`。下面说明实际文件组织与生成阶段。

可以把这次训练理解为：**读取已经算好的单帧观察，训练一个小型时序修正模块；每训练一轮，就在官方 test 上完整评测一次，最后挑出 AUC 最好的一轮。**

## 1. 先记住三个根目录

| 用途 | 服务器绝对路径 |
|---|---|
| 仓库、脚本、小报告 | `/home/fb/src/paper/gazelleV1` |
| VAT 原始数据、轨迹和缓存 | `/newhome/fb/dataset/videoattentiontarget` |
| 本轮训练实验输出 | `/home/fb/src/paper/gazelleV1/experiments/cscwd_r4_official_v1` |

Python 为 `/home/fb/anaconda3/envs/py310/bin/python`。外层 `CUDA_VISIBLE_DEVICES=1` 选物理 GPU 1，程序内部 `cuda:0` 指该进程可见的第一张卡。

## 2. 从哪个脚本开始，先后做什么

唯一后台入口：

`/home/fb/src/paper/gazelleV1/cscwd/code/scripts/run_official_round.sh`

执行顺序如下：

```text
生成或复用 test 视觉特征
        ↓
ordinary_gru：完整 train 训练，每轮完整 test，共五轮
        ↓
history_validity：完整 train 训练，每轮完整 test，共五轮
        ↓
读取两份训练报告，比较单帧、普通 GRU、新机制
        ↓
生成比较报告与最佳 checkpoint 清单
```

两种模型依次执行。`ordinary_gru` 是普通 GRU 对照，`history_validity` 是带历史视觉对应、有效性读门和刷新控制的新机制。二者从各自新增模块的随机初始化开始，seed 0，使用同一训练预算。

| 阶段 | 服务器脚本绝对路径 | 做什么 |
|---|---|---|
| 补 test 特征 | `/home/fb/src/paper/gazelleV1/cscwd/code/scripts/cache_official_test_features.py` | 用冻结 GazeSpot 提取当前 RGB 的空间特征 |
| 训练，两种模型共用 | `/home/fb/src/paper/gazelleV1/cscwd/code/scripts/train_official_joint.py` | 训练新增模块，每轮评测，保存权重和指标 |
| 最后比较 | `/home/fb/src/paper/gazelleV1/cscwd/code/scripts/compare_official_joint.py` | 重算选模、核对来源和文件、生成完整比较 |

启动命令保留在实现计划 §11.1。正在运行的这轮无需再次启动。

## 3. 数据集划分到底由哪个 JSON 决定

### 3.1 官方原始分区和最早的轨迹 JSON

官方 train/test 来自两个标注目录：

```text
/newhome/fb/dataset/videoattentiontarget/annotations/train/
/newhome/fb/dataset/videoattentiontarget/annotations/test/
```

RGB 共用以下目录结构，图像相对路径记录在轨迹中：

```text
/newhome/fb/dataset/videoattentiontarget/images/<video>/<clip>/<frame>.jpg
```

T1 将标注恢复为人物轨迹，保存为：

```text
/newhome/fb/dataset/videoattentiontarget/cscwd_tracks/train_preprocessed_tracks.json
/newhome/fb/dataset/videoattentiontarget/cscwd_tracks/test_preprocessed_tracks.json
```

这两份是**原始轨迹清单**：描述哪些人物、哪些帧属于官方 train/test，也保存头框、注视点和画内外标签。它们是生成单帧预测缓存的上游输入。

### 3.2 当前训练实际读取的轨迹清单

当前训练实际读取下面两个预测缓存的 `index.json`：

| 用途 | 当前训练直接读取的 JSON | 使用范围 |
|---|---|---|
| train 轨迹与标签 | `/newhome/fb/dataset/videoattentiontarget/cscwd_cache/gazespot_vitb_epoch7/r1_train_predictions/index.json` | 全部 1,033 tracks / 132,563 人物帧 |
| test 轨迹与标签 | `/newhome/fb/dataset/videoattentiontarget/cscwd_cache/gazespot_vitb_epoch7/test/index.json` | 全部 298 tracks / 31,978 人物帧 |

缓存的 `index.json` 已经包含原轨迹和逐帧标注，因此训练阶段可以直接从这里恢复数据顺序，无需再次读取 CSV 或上游轨迹 JSON。

其中关键字段含义：

| 字段 | 通俗解释 |
|---|---|
| `tracks` / `track_id` | 人物轨迹；`train/` 或 `test/` 前缀标明官方分区 |
| `start` / `stop` | 此人物在预测 NPY 数组中的行区间，左闭右开 |
| `frames` / `frame_index` | 轨迹内的原始帧号，用于保持时序、识别帧号缺口 |
| `frames[].path` | 对应 RGB 的相对路径，也用于找到这张图的视觉特征 |
| `bbox_norm` | 当前帧真实头框，作为模型输入 |
| `gazex_norm` / `gazey_norm` | 真实注视点，用于训练目标和离线计分 |
| `inout` | 真实画内外标签，用于训练损失和离线计分 |
| `files` | 预测数组的文件名、shape、dtype 和 SHA256 |

**本轮没有额外的 train/validation 划分 JSON。** 加载器取上述两个 index 中的全部轨迹。

R1 的旧划分文件仍在：

`/newhome/fb/dataset/videoattentiontarget/cscwd_cache/gazespot_vitb_epoch7/r1_development_v1/split_manifest.json`

它只用于旧开发实验。本轮不读取它来选训练子集；原 validation 的 34,201 人物帧已经包含在完整 train 的 132,563 帧中。旧 train 特征的来源字段仍会记录 R1 lineage，这是来源记录，不会触发旧分组。

## 4. 模型每帧读什么：两类缓存

### 4.1 单帧预测缓存：底座原本预测了什么

两个目录均包含 `index.json`、`heatmap.npy`、`inout.npy`、`argmax_xy.npy`：

```text
train: /newhome/fb/dataset/videoattentiontarget/cscwd_cache/gazespot_vitb_epoch7/r1_train_predictions/
test:  /newhome/fb/dataset/videoattentiontarget/cscwd_cache/gazespot_vitb_epoch7/test/
```

| 文件 | 内容 | shape / dtype |
|---|---|---|
| `heatmap.npy` | 原 GazeSpot 的注视热图 | `[人物帧数,64,64]` / float32 |
| `inout.npy` | 原 GazeSpot 的画内连续分数 | `[人物帧数]` / float32 |
| `argmax_xy.npy` | 原热图峰值坐标，便于离线比较 | `[人物帧数,2]` / float32 |

它们是**原始预测**，不会被本轮修正后的预测覆盖。文件名 `inout.npy` 表示预测分数，JSON 中 `frames[].inout` 表示真实标签，两者用途不同。

### 4.2 视觉特征缓存：图像里有什么

| 用途 | 完整目录 | 特征数组 |
|---|---|---|
| train | `/newhome/fb/dataset/videoattentiontarget/cscwd_cache/gazespot_vitb_epoch7/r2_native_last_fp16/` | `scene_features.npy`，`[58507,768,32,32]` |
| test | `/newhome/fb/dataset/videoattentiontarget/cscwd_cache/gazespot_vitb_epoch7/r4_official_test_native_last_fp16/` | `scene_features.npy`，`[13127,768,32,32]` |

每个目录都有自己的 `index.json`，记录 `image_paths`、特征合同、来源和校验值。特征以 FP16 存储，送入新增模块时转为 FP32。

这里按**唯一图像**保存，每张图只存一份。预测缓存按**人物帧**保存，一张图里多人各有一份预测。因此 train 的图像数 58,507 小于人物帧数 132,563，两个 NPY 的行数不同是正常设计。

加载过程是：

1. 从预测 index 找到某人的当前帧，按 `start + 轨迹内位置` 取原热图和分数。
2. 用该帧的 `path` 在特征 index 的 `image_paths` 中找到图像行号。
3. 读取这张图的空间特征，以及该人的真实头框。
4. 带上此人过去的模型状态，预测当前帧的修正热图和分数。

训练阶段读取缓存，不再逐帧解码 RGB、运行完整 GazeSpot。当前启动流程中的 **test 特征生成阶段** 会读取 test RGB；后续线上从原图推理仍需要冻结底座提取当前帧特征。

## 5. 仓库 results 里的输入 JSON 有什么用

以下文件都位于服务器目录：

`/home/fb/src/paper/gazelleV1/cscwd/results/`

| 输入小报告文件 | 指向哪里、用于什么 |
|---|---|
| `r1_train_single_frame_cache.json` | 指向完整 train 预测缓存，记录模型、权重、index SHA和覆盖 |
| `t2_single_frame_cache.json` | 指向 test 原预测缓存及其来源 |
| `r2_feature_cache.json` | 记录完整 train 特征缓存、原生网格合同和来源 |
| `r4_official_test_features.json` | 本轮 test 特征生成成功后写出，记录新特征路径、首批一致性检查和完整缓存来源 |

这些是**小报告与缓存入口**，大数组保存在上一节的数据集缓存目录。训练读取报告后，再打开其引用的 index 和 NPY，并检查来源是否对齐。启动时打印 `Verifying frozen train/test feature file`，表示在核对大文件 SHA，还未进入梯度训练。

本轮实验协议文件为：

`/home/fb/src/paper/gazelleV1/cscwd/r4_official_train_test_plan.md`

它记录训练预算和选轮规则；其 SHA 写入本轮配置、报告和 checkpoint。

## 6. 原 GazeSpot 权重与本轮训练权重

冻结底座所用 VAT checkpoint：

`/home/fb/src/paper/gazelleV1/experiments/train_vat_sasa_ggsf/2026-03-12_19-24-13/epoch_7.pt`

其骨干预训练权重来源：

`/home/fb/src/paper/gazelleV1/checkpoints/dinov3_vitb16_pretrain.pth`

生成 test 特征时会加载、冻结底座。新增模块训练直接使用缓存，只优化轻量投影、时序模块和输出修正头；本轮的 `epoch_001.pt` 等保存的是这些**新增模块的权重**。

因此，后续从 RGB 完整推理需要：**原 GazeSpot 配置/权重 + 新模块配置/权重**。新 `selected.pt` 不包含整套 GazeSpot。它保存 `model`、`config`、`epoch`、`model_id`、`protocol_version`、`plan_sha256`、`feature_contract`、`sources`；当前 checkpoint 不保存 optimizer 状态，不能直接当作完整断点续训快照。

## 7. 每种模型的权重、每轮预测存在哪里

两个模型分别写入独立目录：

```text
/home/fb/src/paper/gazelleV1/experiments/cscwd_r4_official_v1/ordinary_gru/
/home/fb/src/paper/gazelleV1/experiments/cscwd_r4_official_v1/history_validity/
```

每个目录中的文件结构如下，`002`–`005` 与第一轮结构相同：

```text
<model_dir>/
├── training_config.json         # 本次实际参数、数据来源、共享初始化SHA
├── baseline_details.json        # 原始单帧test的逐轨迹/逐事件明细
├── training_history.json        # 每完成一轮就更新：loss、完整test指标、文件路径
├── epoch_001.pt                 # 第一轮新增模块权重
├── epoch_002.pt
├── epoch_003.pt
├── epoch_004.pt
├── epoch_005.pt
├── test_epoch_001/
│   ├── index.json               # 本轮test预测顺序、轨迹、来源、文件SHA
│   ├── heatmap.npy              # 本轮修正热图，[31978,64,64] float32
│   ├── inout.npy                # 本轮修正状态分数，[31978] float32
│   ├── argmax_xy.npy            # 本轮热图峰值，[31978,2] float32
│   ├── test_details.json        # joint/state的逐轨迹、逐事件计分明细
│   └── history_diagnostics/     # 新机制有此目录；普通GRU无此项
│       ├── index.json
│       ├── controls.npy
│       ├── correspondence_xy.npy
│       └── source_global_indices.npy
├── test_epoch_002/ ... test_epoch_005/
├── selection.json               # 五轮结束后的最佳epoch和权重路径
├── selected.pt                  # 最佳整轮checkpoint的文件副本
└── last.pt                      # 最后一轮epoch_005.pt的文件副本
```

新机制的 `controls.npy` 按人物帧记录八列：`history_validity / refresh / write_strength / match_peak / match_entropy / agreement / person_change / history_available`。`correspondence_xy.npy` 记录历史模板在当前网格中的对应坐标，`source_global_indices.npy` 将诊断行映射回原预测缓存。它们与同帧最终预测来自同一次前向，用于解释模型如何使用历史。

`training_history.json` 的每轮记录包含 `epoch / training / test / checkpoint`。`test` 下有 `standard / temporal / state` 指标；`training` 下有热图损失、状态损失、总损失及训练帧计数。

## 8. 最佳 epoch、论文整行结果怎么找

五轮结束后按完整精度数值比较：test **AUC更高 → L2更低 → AP更高 → 较早epoch**。每种模型各自选一轮，整行指标取自那一轮。

| 要找什么 | 最直接的文件 |
|---|---|
| 新机制最终选中的权重 | `/home/fb/src/paper/gazelleV1/experiments/cscwd_r4_official_v1/history_validity/selected.pt` |
| 普通GRU最终选中的权重 | `/home/fb/src/paper/gazelleV1/experiments/cscwd_r4_official_v1/ordinary_gru/selected.pt` |
| 新机制每轮指标与选中epoch | `/home/fb/src/paper/gazelleV1/cscwd/results/r4_official_history_validity.md` |
| 普通GRU每轮指标与选中epoch | `/home/fb/src/paper/gazelleV1/cscwd/results/r4_official_ordinary_gru.md` |
| 三种方法的完整比较 | `/home/fb/src/paper/gazelleV1/cscwd/results/r4_official_comparison.md` |
| 选中checkpoint的机器清单 | `/home/fb/src/paper/gazelleV1/cscwd/results/r4_official_selection.json` |

上述 Markdown 均有同名 JSON，JSON保存更完整的指标、全精度数值和来源。比较报告的 `selected` 是各模型选中行，`per_model_selection` 含权重路径，`candidate_rows` 含单帧及十个训练候选，`paired_response` 含事件配对比较。manifest 记录 test 参与 checkpoint 选择的本轮协议。

写论文时从各模型 `selected` 对应的同一个 epoch 取 AUC/L2/AP、时序和状态指标。`last.pt` 的含义只是最后一轮，是否同时是最佳轮由实际结果决定；本轮两模型最佳epoch均已确定为5，`selected.pt`与各自 `epoch_005.pt` 一致。

## 9. 运行过程中先看哪些文件

日志都位于：

`/home/fb/src/paper/gazelleV1/cscwd/results/`

| 日志 | 用途 |
|---|---|
| `r4_official_round.log` | 当前处于缓存、GRU、新机制还是比较阶段 |
| `r4_official_test_features.log` | 冻结特征提取、读回与SHA核对；复用已有特征时此阶段跳过 |
| `r4_official_ordinary_gru.log` | 普通GRU各轮训练loss与test AUC/L2/AP |
| `r4_official_history_validity.log` | 新机制各轮训练loss与test AUC/L2/AP |
| `r4_official_comparison.log` | 最终两模型与单帧的比较处理 |

训练过程中，某模型的小汇总报告还没出现时，可先查看该模型实验目录里的 `training_history.json`，它每完成一轮即更新。`selected.pt`、`selection.json` 和该模型仓库results下的小报告在该模型五轮结束后生成；比较报告在两个模型都结束后生成。

外层最终成功标记：`R4 official train/test round completed.`。

## 10. 哪些要经 Git 传回，哪些留在服务器

回传本地的小结果都在 `/home/fb/src/paper/gazelleV1/cscwd/results/`：

```text
r4_official_test_features.json / .md
r4_official_ordinary_gru.json / .md
r4_official_history_validity.json / .md
r4_official_comparison.json / .md
r4_official_selection.json
RESULT_PROVENANCE.md
```

缓存 NPY、模型 PT、每轮完整预测、诊断和逐事件明细留服务器。报告及 `RESULT_PROVENANCE.md` 记录实际路径和 SHA256，具体回传命令见实现计划 §11末尾。

快速查找时：**看进度找日志，看每轮结果找 `training_history.json`，看最终比较找 `r4_official_comparison.md`，找新机制最佳权重用其 `selected.pt`，查数据和标签先看 train/test 预测缓存的 `index.json`。**

## 11. 本说明对应的本地源码

- [后台顺序入口](/Users/fanb/src/gazelle/cscwd/code/scripts/run_official_round.sh)
- [官方train/test缓存加载](/Users/fanb/src/gazelle/cscwd/code/stag/official_data.py)
- [连续轨迹与特征行映射](/Users/fanb/src/gazelle/cscwd/code/stag/temporal_data.py)
- [训练、逐轮保存与选模](/Users/fanb/src/gazelle/cscwd/code/scripts/train_official_joint.py)
- [AUC/L2/AP选轮规则](/Users/fanb/src/gazelle/cscwd/code/stag/official_protocol.py)
- [test评测与原始单帧基线](/Users/fanb/src/gazelle/cscwd/code/stag/official_evaluation.py)
- [完整报告比较](/Users/fanb/src/gazelle/cscwd/code/scripts/compare_official_joint.py)

## 12. R4-D2校准新增文件

本轮规格见 `r4_state_calibration_plan.md`，唯一启动命令见实现计划§13.1。保持上述官方划分、原始预测/视觉特征与epoch-5权重；没有新的epoch训练或权重文件。

train输入直接读 `r1_train_single_frame_cache.json` 指向的完整train prediction index/分数，GRU与新机制另读 `r2_feature_cache.json` 指向的train特征及各自 `selected.pt`，冻结前向只补分数。test输入读 `t2_single_frame_cache.json` 与 `r4_official_comparison.json` 选中的两套epoch-5预测；无需test特征或RGB。

新输出根目录为 `/home/fb/src/paper/gazelleV1/experiments/cscwd_r4_state_calibration_v1/`，其 `single_frame/`、`ordinary_gru/`、`history_validity/` 各存 `calibration.json`、`test_inout_calibrated.npy`、`test_details.json`；后两者另有 `train_scores/inout.npy` 与 `index.json`。train分数与校准test分数分别沿用原官方prediction index的全局行序，校准test以float64保存以保持排序。

日志为 `/home/fb/src/paper/gazelleV1/cscwd/results/r4_state_calibration.log`，成功标记为 `R4 state calibration completed`。回传小报告 `r4_state_calibration.json/.md` 与 `RESULT_PROVENANCE.md`；NPY、完整事件明细及参数JSON留服务器，路径和SHA由入口记录。
