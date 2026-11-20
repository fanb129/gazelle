# 小论文 · 当前进度

分支：`codex/coverage-aware-router`。更新：2026-11-21。

| 投稿线 | 目录 | 状态 |
|---|---|---|
| ACM MM 2026（GazeSpot） | `rebuttal/` | 已投，返修材料与结果保留 |
| ICASSP 2027 | `icassp2027/` | 初稿保留 |
| CSCWD新主线 | `cscwd/` | ConvLSTM完成6轮后按用户要求停止；转为train/test失败案例人工分析 |
| CSCWD旧工作 | `cscwd_v0/` | 全量归档，文献/代码/结果/论文材料保持原样 |

## 当前目标

复用GazeSpot-B原始逐帧预测，分别抽取官方train和test的定位错误、InOut错误及正确对照，查看当前/历史图像并人工标注共同失败机制，再确定下一版改进方案。保持官方train/test，不新增validation。当前不追加ConvLSTM轮次或启动新模型训练。

当前唯一执行规格：`cscwd/implementation_plan.md`。旧候选选择与D7诊断不再执行。

## 工作约定

- 新代码只放 `cscwd/code/{data_prep,stag,scripts}/`，`gazelle/`只import不改。
- 本机实现/必要检查；服务器 `fb@3090.lab` 由用户手动跑。仓库 `/home/fb/src/paper/gazelleV1`，Python `/home/fb/anaconda3/envs/py310/bin/python`，物理GPU1。
- 原模型冻结，仅训练新增时序模块；只用当前/过去RGB和真实头部框。
- 完整train、逐epoch完整test，无validation；AUC优先，随后L2/AP选整轮，保留全部轮次。
- 小JSON/MD经Git回传；大NPY/PT只留服务器，在 `cscwd/results/RESULT_PROVENANCE.md` 记路径和SHA。只记录真实跑过的数字。
- train/test均按用户要求参与人工分析与方法开发；后续test成绩不视为独立验证。精选案例比例不作为总体失败比例。

## 进展

**V1-0 / 2026-11-20 / 归档与初始化完成**：按用户要求把原 `cscwd/` 整体移为 `cscwd_v0/`；365个原文件/链接（含隐藏/忽略项）内容保持。旧根进度保存为 `cscwd_v0/PROGRESS_SNAPSHOT.md`。新建干净 `cscwd/`，仅含新方案、忽略规则、目录骨架与新结果provenance。服务器数据/缓存/checkpoint未移动，没有新训练或评测结果。

**V1-1 / 2026-11-20 / 可运行时序基线实现完成**：独立wrapper捕获原人物Transformer之后两头之前的FP32任务特征，按图像合并多人提取并记录人物轨迹/原帧号/图像/数组行及SHA。新增共享hidden64单向ConvLSTM，双残差输出投影零初始化；原heads冻结且eval，输入梯度保留。轨迹槽位保序，跨chunk保留H/C并detach，人物切换/原帧缺口重置。完整train、逐epoch完整test，固定MSE+0.1BCE、AdamW配置；保存每轮checkpoint/指标、history、selected/last、与冻结GazeSpot-B的整轮比较，明确test参与选模。新代码无旧归档运行依赖，`gazelle/`未修改。本机必要合成检查通过，包含真实GazeSpot代码路径输出一致性与两轮合成训练/评测/读写；记录见 `cscwd/results/LOCAL_CHECKS.md`。服务器任务未启动，无新VAT性能数字。

**V1-2 / 2026-11-21 / 六轮真实结果与主动停止**：用户运行 `convlstm_v1_seed0_20261120_130802`，完整train 132,563人物帧、test 31,978人物帧。底座AUC/L2/AP为0.940181696/0.098994832/0.886800965；按AUC优先选出的epoch4为0.940235693/0.099214758/0.886771586，尚未观察到有效的共同增益。按用户明确指令向PID2458835发送SIGTERM并确认退出；保留1–6轮checkpoint，selected=4、last=6，第7轮test中断。本次不是完成10轮的实验。真实history、路径和SHA的只读快照见 `cscwd/results/convlstm_v1_stopped_summary.json`。

**V1-3 / 2026-11-21 / train和test人工案例包**：用户批准同时分析两split。实现独立原始预测cache reader、可复现分层抽样、人物/视频多样性约束及连续过去帧窗口；离线浏览器支持当前帧→历史→揭GT、标注本地保存与JSON/CSV导入导出。只复用原始预测数据，不import归档代码。CPU导出入口、服务器手动命令与取回方式见 `cscwd/FAILURE_REVIEW.md`。

## 下一步

用户手动运行CPU案例导出入口 `cscwd/code/scripts/build_failure_review.py`，取回离线包后人工分析train/test案例并导出标注。以重复出现、且历史确有可用证据的失败机制为依据再决定优化方向；不预设必须继续使用ConvLSTM。

历史详细进度和全部旧结果见 `cscwd_v0/PROGRESS_SNAPSHOT.md`、`cscwd_v0/results/`。Router旧探索线状态仍见 `COVERAGE_AWARE_ROUTING_IDEA_PROGRESS.md`。
