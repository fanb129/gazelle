# 小论文 · 进度

> 状态文件。每次开工先读，收工写回。分支：`codex/coverage-aware-router`

## 三条投稿线
| 线 | 目录 | 状态 |
|---|---|---|
| ACM MM 2026（GazeSpot） | `rebuttal/` | 已投，返修材料与实验结果在 `rebuttal/results` |
| ICASSP 2027 | `icassp2027/` | 初稿完成（2026-09 底），源稿在 `icassp2027/2026ACMMM` |
| CSCWD | `cscwd/` | T1–T4 服务器验收完成；T5 固定 EMA 对照本地完成，13 项合成测试通过，下一步服务器跑六种方法 |

## 当前优先级（2026-11-12）
**CSCWD 投稿是第一优先，方向已定：单帧→视频时序，贡献定位"切换感知 + 时序稳定"（STAG）。**
文档齐备：`cscwd/paper_blueprint.md`（论文结构/baseline/指标）、`cscwd/implementation_plan.md`（实现规格+进展）、`cscwd/codex_prompt.md`（codex 提示词）、`cscwd/cscwd_2025_2026_related.md`（CSCWD 2025–2026 同方向论文扫描）。
T1 真实结果已从服务器拉取（`17699cc`）并验收通过：train **1,033 tracks / 58,507 图像帧 / 132,563 人物帧**；test **298 tracks / 13,127 图像帧 / 31,978 人物帧**。原始 CSV 与旧 JSON 的全部计数核对通过，脚本 SHA256 一致；来源见 `cscwd/results/t1_track_recovery.json` 与 `RESULT_PROVENANCE.md`。

T2 真实结果已拉取（`24cf720`）并验收通过：指定 ViT-B GazeSpot VAT `epoch_7.pt`（`train_vat_sasa_ggsf/2026-03-12_19-24-13/epoch_7.pt`），覆盖 **298 tracks / 13,127 图像帧 / 31,978 人物帧**；float32 heatmap、inout、argmax 无损读回与重建全部通过，输入 SHA256 对齐 T1，代码 SHA256 对齐 T2 实现提交。NVIDIA GeForce RTX 3090 上整项缓存作业耗时 **731.430930521 秒**；来源见 `cscwd/results/t2_single_frame_cache.json` 与 provenance。

T3 真实结果已拉取（`eb28855`）并完成运行与口径验收：单帧 **AUC 0.940182 / L2 0.098995 / AP 0.886801**；初始 STAG **AUC 0.936860 / L2 0.099197 / AP 0.886801**。固定权重 `[-2,20,2,10]` 的初始规则定位未改善；来源见 `cscwd/results/t3_stag_comparison.json`。两侧覆盖相同 31,978 人物帧/20,655 画内定位帧，来源与实现 SHA256、轨迹对齐和读回核对通过。

T4 真实结果已拉取（`7f15cd7`）并验收通过：同源缓存/代码 SHA256、标准指标重现和样本对齐核对通过。Jitter mean **0.016647→0.012135（−27.10%）**，同帧对 GT 位移 mean **0.005922**；准确且稳定占比 **52.75%→58.15%（+5.40 个百分点）**。但画内落点切换响应率 **58.23%→55.06%**，共同响应的 87 个切换事件平均慢 **0.276 帧**，另损失 5 个单帧可响应事件。当前结论：**稳定性改善，定位略降，切换响应未改善**，门控相对普通 EMA 的价值尚待 T5 对照。来源：`cscwd/results/t4_temporal_comparison.json`。

T4 代码与协议已实现并通过 15 项新增合成测试、10 项标准/T3 回归；固定阈值和完整服务器命令见 `cscwd/implementation_plan.md`。服务器仓库：`/home/fb/src/paper/gazelleV1`。CSCWD 2027 截稿 2026-11-30（AoE），6 页双栏 single-blind。

T5 固定 EMA 对照本地已实现：`cscwd/code/stag/ema.py` 与 `cscwd/code/scripts/run_ema_ablation.py`，覆盖 `α=sigmoid(-2)`、`0.25/0.5/0.75` 四个 EMA，以及单帧/现有 STAG，共六种方法。继承 T4 协议，核验来源链并重现参考指标，全部配置输出标准/时序指标及分别相对单帧、STAG 的配对事件响应。完整缓存与明细留服务器，JSON/Markdown 汇总和 provenance 经 git 回传。可学习门控继续后置。
T5 本地验证：**13 项合成测试通过**（核心 9、完整来源链/CLI 集成 4），包含偏置匹配 EMA 与去门控 STAG 逐值一致、六种方法全报、参考指标重现、缓存不变和配对事件核对；Python 3.10 语法、CLI/命令及结果忽略规则检查通过。真实 EMA 对照数字等待服务器运行。

结果回传规则已修正：`cscwd/.gitignore` 放行 `results/` 直属小 JSON，覆盖根目录的 `*.json`；完整 track JSON、NPY 和缓存目录继续忽略。

## 已知阻塞
- Router 探索线（coverage-aware-router）：VAT 效果不明显，已暂停；详细状态见 `COVERAGE_AWARE_ROUTING_IDEA_PROGRESS.md`，与当前 CSCWD 时序方向无关。

## 文件约定
- 每个投稿目录各自放 LaTeX 源、图、实验结果；实验结果只写真实跑过的，没跑的对照不写进表格
- `RESULT_PROVENANCE.md` 记录每个数字来自哪次运行，写论文时以它为准
- 每个投稿目录独立放自己的新代码/论文：`<venue>/code/`（新代码）、`<venue>/latex/`（论文）、`<venue>/results/`（关键小结果）；旧 `gazelle` 包只 import 不改
- git 双向传递：本地改 code → push → 服务器 pull → nohup 手动跑 → 小结果（JSON/MD）push 回本地；大文件（.npy/.pt）不进 git，只在 RESULT_PROVENANCE 记路径+SHA

## 下一步
1. 用户手动执行实现计划中的 T5 nohup 命令，回传 `cscwd/results/t5_ema_ablation.json`、`cscwd/results/t5_ema_ablation.md` 与 `cscwd/results/RESULT_PROVENANCE.md`。完整 EMA 缓存/明细保存在服务器 `cscwd_cache/gazespot_vitb_epoch7/t5_fixed_ema_test/`。
2. 根据六种方法的真实精度—稳定性—响应折中判断门控额外价值，再开 T6 展示稳定收益、切换滞后与失败片段。当前不填尚未运行的 EMA 数字。
