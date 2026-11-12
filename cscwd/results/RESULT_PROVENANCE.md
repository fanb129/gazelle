# CSCWD 时序方向 · 结果来源（provenance）

> 每个写进论文的数字，在这里记来源。内部写作辅助，不进论文。
> 格式：数字 → 来源脚本/运行 → checkpoint → 结果文件路径 → SHA（如适用）。
> 只记真实跑过的；没跑的不写。

## T1 track 恢复运行约定

- 脚本：`cscwd/code/data_prep/preprocess_vat_tracks.py`；服务器运行命令见 `cscwd/implementation_plan.md` 的 T1 记录。
- 计数核对通过后，脚本在本文件末尾追加真实轨迹/图像帧/人物帧数、命令、代码版本及完整 track JSON 路径/SHA256。
- 小报告：`cscwd/results/t1_track_recovery.json`，随本文件经 git 回传；完整 JSON 留服务器数据集下 `cscwd_tracks/`。

## 单帧 baseline vs STAG 主表（VAT test，真实框，纯 RGB，track 级）

| 配置 | AUC↑ | L2↓ | In/Out AP↑ | 来源 |
|---|---:|---:|---:|---|
| 单帧 GazeSpot | ? | ? | ? | 待填 |
| +STAG 时序门控 | ? | ? | ? | 待填 |

- 单帧 checkpoint：`train_vat_vitl_sasa_ggsf/.../epoch_?.pt`（待填）。
- 评测脚本：`cscwd/code/scripts/eval_temporal.py`，在 track 级 JSON 上跑。
- 结果文件：`cscwd/results/*.json`。

## 时序指标（贡献）

- 帧间抖动 Jitter↓ / 切换响应延迟↓ / 稳定占比↑：脚本 `cscwd/code/stag/metrics.py`。
- 诚实校验：抖动同报真实注视点位移作参照。

## 消融（单帧 / 纯EMA / 规则门控 / 可学习门控）

（待填）

## 待填
