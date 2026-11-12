# 小论文 · 进度

> 状态文件。每次开工先读，收工写回。分支：`codex/coverage-aware-router`

## 三条投稿线
| 线 | 目录 | 状态 |
|---|---|---|
| ACM MM 2026（GazeSpot） | `rebuttal/` | 已投，返修材料与实验结果在 `rebuttal/results` |
| ICASSP 2027 | `icassp2027/` | 初稿完成（2026-09 底），源稿在 `icassp2027/2026ACMMM` |
| CSCWD | `cscwd/` | T1–T3 服务器流程验收通过；T3 初始规则定位未改善；T4 本地完成，15 项新增测试与 10 项回归通过 |

## 当前优先级（2026-11-12）
**CSCWD 投稿是第一优先，方向已定：单帧→视频时序，贡献定位"切换感知 + 时序稳定"（STAG）。**
文档齐备：`cscwd/paper_blueprint.md`（论文结构/baseline/指标）、`cscwd/implementation_plan.md`（实现规格+进展）、`cscwd/codex_prompt.md`（codex 提示词）、`cscwd/cscwd_2025_2026_related.md`（CSCWD 2025–2026 同方向论文扫描）。
T1 真实结果已从服务器拉取（`17699cc`）并验收通过：train **1,033 tracks / 58,507 图像帧 / 132,563 人物帧**；test **298 tracks / 13,127 图像帧 / 31,978 人物帧**。原始 CSV 与旧 JSON 的全部计数核对通过，脚本 SHA256 一致；来源见 `cscwd/results/t1_track_recovery.json` 与 `RESULT_PROVENANCE.md`。

T2 真实结果已拉取（`24cf720`）并验收通过：指定 ViT-B GazeSpot VAT `epoch_7.pt`（`train_vat_sasa_ggsf/2026-03-12_19-24-13/epoch_7.pt`），覆盖 **298 tracks / 13,127 图像帧 / 31,978 人物帧**；float32 heatmap、inout、argmax 无损读回与重建全部通过，输入 SHA256 对齐 T1，代码 SHA256 对齐 T2 实现提交。NVIDIA GeForce RTX 3090 上整项缓存作业耗时 **731.430930521 秒**；来源见 `cscwd/results/t2_single_frame_cache.json` 与 provenance。

T3 真实结果已拉取（`eb28855`）并完成运行与口径验收：单帧 **AUC 0.940182 / L2 0.098995 / AP 0.886801**；初始 STAG **AUC 0.936860 / L2 0.099197 / AP 0.886801**。固定权重 `[-2,20,2,10]` 的初始规则定位未改善；来源见 `cscwd/results/t3_stag_comparison.json`。两侧覆盖相同 31,978 人物帧/20,655 画内定位帧，来源与实现 SHA256、轨迹对齐和读回核对通过。

T4 本地时序指标与统一评测入口已实现：`cscwd/code/stag/metrics.py` 同帧对报告预测/GT 位移、稳定及准确稳定占比、三类切换代理事件响应率/延迟/删失计数；`cscwd/code/scripts/eval_temporal.py` 只读 T2/T3 缓存并重现两侧标准三指标。响应半径 0.10、GT 跳变阈值 0.15、稳定位移阈值 0.02、连续 3 帧、最长 30 帧窗口，完整定义与命令见 `cscwd/implementation_plan.md`。当前保持 T3 权重，下一步服务器产出真实时序数字。服务器仓库：`/home/fb/src/paper/gazelleV1`。CSCWD 2027 截稿 2026-11-30（AoE），6 页双栏 single-blind。
T4 本地验证：**15 项新增合成测试、10 项标准/T3 回归通过**；Python 3.10 语法、CLI 和服务器命令/结果忽略规则检查通过。真实时序数字等待服务器运行。

结果回传规则已修正：`cscwd/.gitignore` 放行 `results/` 直属小 JSON，覆盖根目录的 `*.json`；完整 track JSON、NPY 和缓存目录继续忽略。

## 已知阻塞
- Router 探索线（coverage-aware-router）：VAT 效果不明显，已暂停；详细状态见 `COVERAGE_AWARE_ROUTING_IDEA_PROGRESS.md`，与当前 CSCWD 时序方向无关。

## 文件约定
- 每个投稿目录各自放 LaTeX 源、图、实验结果；实验结果只写真实跑过的，没跑的对照不写进表格
- `RESULT_PROVENANCE.md` 记录每个数字来自哪次运行，写论文时以它为准
- 每个投稿目录独立放自己的新代码/论文：`<venue>/code/`（新代码）、`<venue>/latex/`（论文）、`<venue>/results/`（关键小结果）；旧 `gazelle` 包只 import 不改
- git 双向传递：本地改 code → push → 服务器 pull → nohup 手动跑 → 小结果（JSON/MD）push 回本地；大文件（.npy/.pt）不进 git，只在 RESULT_PROVENANCE 记路径+SHA

## 下一步
1. 用户手动运行 T4 统一标准与时序评测，回传 `cscwd/results/t4_temporal_comparison.json` 与 `cscwd/results/RESULT_PROVENANCE.md`。逐轨迹/事件明细留服务器数据集下 `cscwd_cache/gazespot_vitb_epoch7/temporal_rule_v1_test/`，路径与 SHA256 自动记录。
2. 拿到真实时序数字后进入 T5 消融，再推进 T6；当前不填尚未运行的时序比较数字。
