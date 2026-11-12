# 小论文 · 进度

> 状态文件。每次开工先读，收工写回。分支：`codex/coverage-aware-router`

## 三条投稿线
| 线 | 目录 | 状态 |
|---|---|---|
| ACM MM 2026（GazeSpot） | `rebuttal/` | 已投，返修材料与实验结果在 `rebuttal/results` |
| ICASSP 2027 | `icassp2027/` | 初稿完成（2026-09 底），源稿在 `icassp2027/2026ACMMM` |
| CSCWD | `cscwd/` | T1/T2 服务器验收通过；T3 本地实现完成，19 项合成测试及 9 项 T2 回归通过，下一步服务器运行 |

## 当前优先级（2026-11-12）
**CSCWD 投稿是第一优先，方向已定：单帧→视频时序，贡献定位"切换感知 + 时序稳定"（STAG）。**
文档齐备：`cscwd/paper_blueprint.md`（论文结构/baseline/指标）、`cscwd/implementation_plan.md`（实现规格+进展）、`cscwd/codex_prompt.md`（codex 提示词）、`cscwd/cscwd_2025_2026_related.md`（CSCWD 2025–2026 同方向论文扫描）。
T1 真实结果已从服务器拉取（`17699cc`）并验收通过：train **1,033 tracks / 58,507 图像帧 / 132,563 人物帧**；test **298 tracks / 13,127 图像帧 / 31,978 人物帧**。原始 CSV 与旧 JSON 的全部计数核对通过，脚本 SHA256 一致；来源见 `cscwd/results/t1_track_recovery.json` 与 `RESULT_PROVENANCE.md`。

T2 真实结果已拉取（`24cf720`）并验收通过：指定 ViT-B GazeSpot VAT `epoch_7.pt`（`train_vat_sasa_ggsf/2026-03-12_19-24-13/epoch_7.pt`），覆盖 **298 tracks / 13,127 图像帧 / 31,978 人物帧**；float32 heatmap、inout、argmax 无损读回与重建全部通过，输入 SHA256 对齐 T1，代码 SHA256 对齐 T2 实现提交。NVIDIA GeForce RTX 3090 上整项缓存作业耗时 **731.430930521 秒**；来源见 `cscwd/results/t2_single_frame_cache.json` 与 provenance。

T3 本地实现完成：`cscwd/code/stag/gating.py` 纯 NumPy 因果 EMA，固定初始权重 `[-2,20,2,10]`，按 track 初始化并在原始帧号缺口重置；画内概率沿用单帧。`cscwd/code/scripts/run_stag.py` 核验 T2 源缓存、输出平滑预测和门控诊断，用同一个标准指标函数重算单帧/STAG 的 AUC/L2/AP。**19 项 T3 合成测试及 9 项 T2 缓存回归通过**；Python 3.10 语法及命令/结果忽略规则检查通过。完整服务器命令见 `cscwd/implementation_plan.md`。服务器仓库：`/home/fb/src/paper/gazelleV1`。CSCWD 2027 截稿 2026-11-30（AoE），6 页双栏 single-blind。

结果回传规则已修正：`cscwd/.gitignore` 放行 `results/` 直属小 JSON，覆盖根目录的 `*.json`；完整 track JSON、NPY 和缓存目录继续忽略。

## 已知阻塞
- Router 探索线（coverage-aware-router）：VAT 效果不明显，已暂停；详细状态见 `COVERAGE_AWARE_ROUTING_IDEA_PROGRESS.md`，与当前 CSCWD 时序方向无关。

## 文件约定
- 每个投稿目录各自放 LaTeX 源、图、实验结果；实验结果只写真实跑过的，没跑的对照不写进表格
- `RESULT_PROVENANCE.md` 记录每个数字来自哪次运行，写论文时以它为准
- 每个投稿目录独立放自己的新代码/论文：`<venue>/code/`（新代码）、`<venue>/latex/`（论文）、`<venue>/results/`（关键小结果）；旧 `gazelle` 包只 import 不改
- git 双向传递：本地改 code → push → 服务器 pull → nohup 手动跑 → 小结果（JSON/MD）push 回本地；大文件（.npy/.pt）不进 git，只在 RESULT_PROVENANCE 记路径+SHA

## 下一步
1. 用户手动运行 T3 初始规则门控；回传 `cscwd/results/t3_stag_comparison.json` 与 `cscwd/results/RESULT_PROVENANCE.md`。平滑缓存和诊断数组留服务器数据集下 `cscwd_cache/gazespot_vitb_epoch7/stag_rule_v1_test/`，路径与 SHA256 自动记录。
2. 拿到同口径标准指标对比后进入 T4 时序指标，再依次推进 T5–T6；当前不填尚未运行的模型对比数字。
