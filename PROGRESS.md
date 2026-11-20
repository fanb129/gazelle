# 小论文 · 当前进度

分支：`codex/coverage-aware-router`。更新：2026-11-20。

| 投稿线 | 目录 | 状态 |
|---|---|---|
| ACM MM 2026（GazeSpot） | `rebuttal/` | 已投，返修材料与结果保留 |
| ICASSP 2027 | `icassp2027/` | 初稿保留 |
| CSCWD新主线 | `cscwd/` | 干净目录已建立，准备实现GazeSpot+ConvLSTM基线 |
| CSCWD旧工作 | `cscwd_v0/` | 全量归档，文献/代码/结果/论文材料保持原样 |

## 当前目标

GazeSpot-B + 共享单向ConvLSTM，直接提高VAT的AUC、L2、AP，保持官方train/test。ConvLSTM先作为时序基线；有效后再针对具体不足设计一个新模块。

当前唯一执行规格：`cscwd/implementation_plan.md`。旧候选选择与D7诊断不再执行。

## 工作约定

- 新代码只放 `cscwd/code/{data_prep,stag,scripts}/`，`gazelle/`只import不改。
- 本机实现/必要检查；服务器 `fb@3090.lab` 由用户手动跑。仓库 `/home/fb/src/paper/gazelleV1`，Python `/home/fb/anaconda3/envs/py310/bin/python`，物理GPU1。
- 原模型冻结，仅训练新增时序模块；只用当前/过去RGB和真实头部框。
- 完整train、逐epoch完整test，无validation；AUC优先，随后L2/AP选整轮，保留全部轮次。
- 小JSON/MD经Git回传；大NPY/PT只留服务器，在 `cscwd/results/RESULT_PROVENANCE.md` 记路径和SHA。只记录真实跑过的数字。

## 进展

**V1-0 / 2026-11-20 / 归档与初始化完成**：按用户要求把原 `cscwd/` 整体移为 `cscwd_v0/`；365个原文件/链接（含隐藏/忽略项）内容保持。旧根进度保存为 `cscwd_v0/PROGRESS_SNAPSHOT.md`。新建干净 `cscwd/`，仅含新方案、忽略规则、目录骨架与新结果provenance。服务器数据/缓存/checkpoint未移动，没有新训练或评测结果。

## 下一步

实现GazeSpot任务特征缓存、ConvLSTM适配器和一个完整训练/逐轮test入口；首轮seed0、10epochs。完成后给用户完整GPU1 nohup命令。

历史详细进度和全部旧结果见 `cscwd_v0/PROGRESS_SNAPSHOT.md`、`cscwd_v0/results/`。Router旧探索线状态仍见 `COVERAGE_AWARE_ROUTING_IDEA_PROGRESS.md`。
