# 小论文 · 进度

> 状态文件。每次开工先读，收工写回。分支：`codex/coverage-aware-router`

## 三条投稿线
| 线 | 目录 | 状态 |
|---|---|---|
| ACM MM 2026（GazeSpot） | `rebuttal/` | 已投，返修材料与实验结果在 `rebuttal/results` |
| ICASSP 2027 | `icassp2027/` | 初稿完成（2026-09 底），源稿在 `icassp2027/2026ACMMM` |
| CSCWD | `cscwd/` | T1 track 恢复本地实现完成，6 项合成测试通过，下一步服务器手动验收 |

## 当前优先级（2026-11-12）
**CSCWD 投稿是第一优先，方向已定：单帧→视频时序，贡献定位"切换感知 + 时序稳定"（STAG）。**
文档齐备：`cscwd/paper_blueprint.md`（论文结构/baseline/指标）、`cscwd/implementation_plan.md`（实现规格+进展）、`cscwd/codex_prompt.md`（codex 提示词）、`cscwd/cscwd_2025_2026_related.md`（CSCWD 2025–2026 同方向论文扫描）。
T1 本地实现完成：`cscwd/code/data_prep/preprocess_vat_tracks.py` 保留原生人物 ID/帧序/缺口，使用当前帧真实框；配套合成回归 6 项通过。完整运行命令已写 `cscwd/implementation_plan.md` 进展记录；服务器仓库路径确认是 `/home/fb/src/paper/gazelleV1`。下一步用户手动跑 T1 并回传真实计数报告，当前停在 T1。CSCWD 2027 截稿 2026-11-30（AoE），6 页双栏 single-blind。

## 已知阻塞
- Router 探索线（coverage-aware-router）：VAT 效果不明显，已暂停；详细状态见 `COVERAGE_AWARE_ROUTING_IDEA_PROGRESS.md`，与当前 CSCWD 时序方向无关。

## 文件约定
- 每个投稿目录各自放 LaTeX 源、图、实验结果；实验结果只写真实跑过的，没跑的对照不写进表格
- `RESULT_PROVENANCE.md` 记录每个数字来自哪次运行，写论文时以它为准
- 每个投稿目录独立放自己的新代码/论文：`<venue>/code/`（新代码）、`<venue>/latex/`（论文）、`<venue>/results/`（关键小结果）；旧 `gazelle` 包只 import 不改
- git 双向传递：本地改 code → push → 服务器 pull → nohup 手动跑 → 小结果（JSON/MD）push 回本地；大文件（.npy/.pt）不进 git，只在 RESULT_PROVENANCE 记路径+SHA

## 下一步
1. 用户手动在服务器运行 T1；回传 `cscwd/results/t1_track_recovery.json` 和 `cscwd/results/RESULT_PROVENANCE.md`。完整 track JSON 留服务器数据集下 `cscwd_tracks/`，路径与 SHA256 由脚本写入 provenance。
2. 用户确认后进入 T2；随后依次推进 T3–T6，走通单帧 vs STAG 对比后动笔 latex。
