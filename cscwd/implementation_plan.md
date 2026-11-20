# GazeSpot + ConvLSTM：实现计划

更新：2026-11-21。当前主线唯一规格。旧工作整体归档至 `cscwd_v0/`，不再执行旧诊断或候选选择路线。

**当前阶段：第6节的train/test人工失败案例分析。** ConvLSTM首轮在完成6轮后按用户指令停止，第7轮test中断；第2–4节保留其已实现规格，不代表继续训练。暂不实现新的模型模块。

## 1. 目标

以GazeSpot-B VAT checkpoint为底座，增加严格因果的空间时序模块，直接优化VAT AUC、L2、AP。ConvLSTM先作为时序基线；有效后再针对具体不足设计新模块，不把ConvLSTM本身列为创新点。

## 2. 固定架构

使用GazeSpot人物条件Transformer之后、原heatmap/InOut预测头之前的任务特征：空间特征 `X_t [256,32,32]` 和InOut token `z_t [256]`。

```text
u_t = Conv1x1_256_to_64(X_t) + broadcast(Linear_256_to_64(z_t))
(H_t, C_t) = ConvLSTM(u_t, H_previous, C_previous)
X'_t = X_t + Conv1x1_64_to_256(H_t)
z'_t = z_t + Linear_64_to_256(global_average_pool(H_t))
heatmap_t = original_heatmap_head(X'_t)
inout_t = original_inout_head(z'_t)
```

原GazeSpot全部参数冻结，仅训练新增模块。两个残差输出投影零初始化，从原checkpoint的输出开始学习。原heads保持eval，训练时允许梯度经其输入回到新增模块。任务特征缓存使用FP32。

历史只来自同人物的当前/过去帧；单向ConvLSTM，人物切换/帧号缺口重置，跨chunk保留状态并detach。纯RGB和真实头部框；GT只用于loss/计分。

## 3. 训练与选轮

- 官方完整train训练，每epoch完整test，无validation。
- 首轮只跑一个配置：seed0、10epochs、clip16、batch4、hidden64、AdamW lr0.0001/weight_decay0.01、gradient_clip_norm1.0。
- 损失 `heatmap_MSE + 0.1 * inout_BCE`，定位仅GT画内，状态全部有效人物帧。
- 保留每epoch完整指标与checkpoint，按AUC更高→L2更低→AP更高→较早epoch选一整轮；保存selected和last，不拼列。test参与checkpoint选择的约定保持。
- 首轮主比较为原GazeSpot-B与新增ConvLSTM。主表AUC/L2/AP，时序稳定性仅作附加观察。

## 4. 实现与运行

已实现任务特征提取/缓存、ConvLSTM适配器、完整训练与逐epoch test入口。服务器手动入口为 `code/scripts/run_baseline.py`，依次建立train/test缓存，再执行10轮训练。完整命令见 [README.md](README.md)。本机不启动服务器作业。

| 项目 | 约定 |
|---|---|
| 新代码 | `cscwd/code/{data_prep,stag,scripts}/` |
| 原底座 | `gazelle/`只import不改 |
| 新论文 | `cscwd/latex/` |
| 新小结果 | `cscwd/results/` |
| 服务器 | `fb@3090.lab` |
| 服务器仓库 | `/home/fb/src/paper/gazelleV1` |
| Python | `/home/fb/anaconda3/envs/py310/bin/python` |
| GPU | 物理GPU1，外层CUDA_VISIBLE_DEVICES=1，内部cuda:0 |
| VAT数据根 | `/newhome/fb/dataset/videoattentiontarget` |
| 官方train人物轨迹 | `/newhome/fb/dataset/videoattentiontarget/cscwd_tracks/train_preprocessed_tracks.json` |
| 官方test人物轨迹 | `/newhome/fb/dataset/videoattentiontarget/cscwd_tracks/test_preprocessed_tracks.json` |
| GazeSpot-B checkpoint | `/home/fb/src/paper/gazelleV1/experiments/train_vat_sasa_ggsf/2026-03-12_19-24-13/epoch_7.pt` |

既有服务器数据/缓存/checkpoint不随本地归档移动。新代码独立实现，数据只读；新实验输出使用新目录。每个服务器脚本实现后给出Python/GPU/全部参数/日志/输出路径，用户手动执行；不假设已运行。大NPY/PT不进Git，来源写provenance；不填未实测数字。

### 4.1 代码与缓存契约

| 文件 | 职责 |
|---|---|
| `code/data_prep/task_cache.py` | 独立冻结wrapper；原heads的forward pre-hook捕获真实任务特征；FP32缓存、来源SHA与人物行映射 |
| `code/stag/model.py` | 共享单向ConvLSTM、双残差零初始化、冻结且可回传输入梯度的原heads |
| `code/stag/data.py` | 轨迹槽位保序分chunk，跨chunk保留人物，原帧号缺口/人物切换重置；padding排除 |
| `code/stag/engine.py` | 固定MSE/BCE、TBPTT、梯度裁剪、完整VAT AUC/L2/AP |
| `code/scripts/train_convlstm.py` | 只读取缓存训练；逐轮完整test、checkpoint选择、报告和provenance |
| `code/scripts/run_baseline.py` | 新建train/test缓存后串起完整训练；支持显式 `--reuse-cache` |

每个split缓存目录含 `x.npy [N,256,32,32]`、`z.npy [N,256]`（均FP32）、`heads.pt`（原两头的state dict）及最后发布的 `index.json`。索引显式记录track_id、person_id、原frame_index、图像相对/绝对路径、原始头部框、GT监督、array row、轨迹[start,stop)、checkpoint/轨迹/数组SHA。按图像合并多人查询，只将RGB和当前真实头部框送进原GazeSpot；无随机增强，不重建或平滑头部框。GT存于索引用于监督和计分，模型forward只收特征、状态和因果reset。

原GazeSpot（包含GGSF、SASA和人物Transformer）全程eval/冻结；使用原factory和完整匹配的VAT decoder checkpoint。训练仅载入缓存及原heads，不载入backbone。两个split必须来自同一checkpoint、backbone和原heads。两个输出分别经原heatmap/InOut头预测，InOut可由新增token残差改变。

每个batch槽位持续处理同一轨迹，耗尽后下个chunk才换人；每epoch只随机轨迹顺序，不随机轨迹内帧。每个chunk至多16帧，真实帧号不连续时在对应帧之前清空H/C；其余chunk边界保留数值并detach。训练epoch和独立test rollout开始时初始化状态，test不继承train状态。

### 4.2 保存与报告

- 先在相同test任务特征上直接运行冻结heads，得到可比较的GazeSpot-B基准。
- 每轮保存 `epoch_01.pt` 等完整model/optimizer/config/metrics/protocol/metadata；每轮独立JSON记录train loss、test AUC/L2/AP和全部计分人数。
- 持续更新 `training_history.json`、`artifacts.json`、`selected.pt`、`last.pt`。训练完成生成 `comparison.json` / `comparison.md`，包含整轮比较和相对原底座的差值。
- `run_metadata.json`、history、checkpoint和比较报告均声明 **test参与checkpoint选择，无validation**；按AUC降序、L2升序、AP降序、epoch升序选择，不拼列。
- checkpoint/缓存放服务器的新目录；小JSON/MD写入 `cscwd/results/<run_id>/`。实际运行时自动向 `results/RESULT_PROVENANCE.md` 追加缓存、输入与每轮checkpoint的服务器路径/SHA。
- 已有实验目录拒绝覆盖。`--reuse-cache` 仅复用完成且与指定输入匹配的缓存；训练输出和报告仍须使用新目录。

## 5. 进展

- **V1-0 / 2026-11-20 / 目录重建完成**：原 `cscwd/` 全量移动为 `cscwd_v0/`，365个原文件/链接（含隐藏/忽略项）逐项保持内容；旧根进度另存 `cscwd_v0/PROGRESS_SNAPSHOT.md`。新目录仅含本方案、目录骨架与新结果provenance入口。没有新训练/评测数字。
- **V1-1 / 2026-11-20 / 可运行基线实现完成**：上述缓存、ConvLSTM、完整train/逐epoch test/整轮选模/来源记录和服务器手动入口已实现。必要合成检查覆盖真实GazeSpot代码路径的零初始化输出、冻结与梯度、因果状态/重置、FP32缓存与行映射、两轮合成训练/评测/checkpoint读写。检查记录见 `results/LOCAL_CHECKS.md`。`gazelle/`未修改，新代码不依赖 `cscwd_v0/`。
- **V1-2 / 2026-11-21 / 首轮停止**：真实运行 `convlstm_v1_seed0_20261120_130802` 已完成6轮，未观察到有效的共同增益。按用户指令停止PID2458835并确认退出。保留selected epoch4、last epoch6，第7轮评测中断；10轮是原计划预算，不能记录为已完成。完整六轮指标及checkpoint路径/SHA见 `results/convlstm_v1_stopped_summary.json`。
- **V1-3 / 2026-11-21 / 人工分析入口**：用户批准train/test都抽案例并人工看图。实现第6节的原始预测读取、独立分层抽样、连续历史窗口、离线浏览标注和CPU导出入口；真实导出仍由用户手动在服务器执行。

## 6. 当前阶段：train/test人工失败案例分析

目标是明确GazeSpot当前帧缺少什么证据、历史是否提供仍然适用的补充信息，再决定改进机制；不预先假定所有错误都应由时序模型修复。

- 复用服务器 `cscwd_cache/gazespot_vitb_epoch7/r1_train_predictions/` 和 `test/` 的原始 `heatmap.npy/inout.npy/argmax_xy.npy/index.json`。实际文件及索引SHA已与历史报告核对；checkpoint和轨迹来源与新基线一致。仅读取数据，不import `cscwd_v0/` 代码，不重新运行GazeSpot推理。
- 两split独立、同seed抽样，默认各48组：大L2 16、InOut FN 8、FP 8、中等L2 8、正确对照8。大L2从各split画内误差最高10%抽；中等取40%–60%分位；正确对照混合低L2画内正确和画外正确。InOut分类阈值固定0.5，仅用于抽例，不能据此宣称AP改善。
- 各类别轮流抽样，全类别合计每人物轨迹最多2组、每视频最多6组；同轨迹最大历史窗口不重叠。选定后按seed随机打乱展示，避免顺序泄漏类别。约束导致不足时明确记录shortfall，不自动放宽或重复选例。统计全体候选数量和实际所选数量，分层样本比例不是总体错误频率。
- 默认当前帧+过去16帧，包内最多保存过去64帧以便展开32/64。人物切换或原帧号缺口处截断，绝不补未来帧；保留track/frame/path/原数组row映射。旧预测行序与新X/z行序不同，不能直接混用。
- 离线HTML默认只显示当前RGB、头框和原预测，不揭GT、L2和案例类别。先记当前困难/猜测，再展开历史，记录历史是否提供证据及描述，最后揭GT并填失败机制/备注。GT揭示时间记录在标注中；RGB/热图预览不烘焙GT。
- 浏览器保存标注，支持JSON/CSV导出、同包JSON导入；包内含空白标注模板。人工判断只形成机制假设，不能替代后续完整test计分。
- **train和test均参与人工分析/方法开发，无validation；后续test不视为独立验证。** 协议写入页面、manifest、小报告和provenance。
- 新代码：`code/stag/failure_cases.py`、`review_package.py`、`review_viewer.html`；服务器CPU入口 `code/scripts/build_failure_review.py`。完整手动导出/取回/标注回传说明见 [FAILURE_REVIEW.md](FAILURE_REVIEW.md)。大型图片/ZIP留服务器或本机忽略目录，小JSON/MD经Git回传。
