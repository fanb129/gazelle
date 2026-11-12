# 小论文 · 进度

> 状态文件。每次开工先读，收工写回。分支：`codex/coverage-aware-router`

## 三条投稿线
| 线 | 目录 | 状态 |
|---|---|---|
| ACM MM 2026（GazeSpot） | `rebuttal/` | 已投，返修材料与实验结果在 `rebuttal/results` |
| ICASSP 2027 | `icassp2027/` | 初稿完成（2026-09 底），源稿在 `icassp2027/2026ACMMM` |
| CSCWD | `cscwd/` | T1–T5 服务器验收完成；T6 本地实现完成，15 项合成测试及视觉检查通过，下一步服务器出案例 |

## 当前优先级（2026-11-13）
**CSCWD 投稿是第一优先，方向已定：单帧→视频时序，贡献定位"切换感知 + 时序稳定"（STAG）。**
文档齐备：`cscwd/paper_blueprint.md`（论文结构/baseline/指标）、`cscwd/implementation_plan.md`（实现规格+进展）、`cscwd/codex_prompt.md`（codex 提示词）、`cscwd/cscwd_2025_2026_related.md`（CSCWD 2025–2026 同方向论文扫描）。
T1 真实结果已从服务器拉取（`17699cc`）并验收通过：train **1,033 tracks / 58,507 图像帧 / 132,563 人物帧**；test **298 tracks / 13,127 图像帧 / 31,978 人物帧**。原始 CSV 与旧 JSON 的全部计数核对通过，脚本 SHA256 一致；来源见 `cscwd/results/t1_track_recovery.json` 与 `RESULT_PROVENANCE.md`。

T2 真实结果已拉取（`24cf720`）并验收通过：指定 ViT-B GazeSpot VAT `epoch_7.pt`（`train_vat_sasa_ggsf/2026-03-12_19-24-13/epoch_7.pt`），覆盖 **298 tracks / 13,127 图像帧 / 31,978 人物帧**；float32 heatmap、inout、argmax 无损读回与重建全部通过，输入 SHA256 对齐 T1，代码 SHA256 对齐 T2 实现提交。NVIDIA GeForce RTX 3090 上整项缓存作业耗时 **731.430930521 秒**；来源见 `cscwd/results/t2_single_frame_cache.json` 与 provenance。

T3 真实结果已拉取（`eb28855`）并完成运行与口径验收：单帧 **AUC 0.940182 / L2 0.098995 / AP 0.886801**；初始 STAG **AUC 0.936860 / L2 0.099197 / AP 0.886801**。固定权重 `[-2,20,2,10]` 的初始规则定位未改善；来源见 `cscwd/results/t3_stag_comparison.json`。两侧覆盖相同 31,978 人物帧/20,655 画内定位帧，来源与实现 SHA256、轨迹对齐和读回核对通过。

T4 真实结果已拉取（`7f15cd7`）并验收通过：同源缓存/代码 SHA256、标准指标重现和样本对齐核对通过。Jitter mean **0.016647→0.012135（−27.10%）**，同帧对 GT 位移 mean **0.005922**；准确且稳定占比 **52.75%→58.15%（+5.40 个百分点）**。但画内落点切换响应率 **58.23%→55.06%**，共同响应的 87 个切换事件平均慢 **0.276 帧**，另损失 5 个单帧可响应事件。当前结论：**稳定性改善，定位略降，切换响应未改善**，门控相对普通 EMA 的价值尚待 T5 对照。来源：`cscwd/results/t4_temporal_comparison.json`。

T4 代码与协议已实现并通过 15 项新增合成测试、10 项标准/T3 回归；固定阈值和完整服务器命令见 `cscwd/implementation_plan.md`。服务器仓库：`/home/fb/src/paper/gazelleV1`。CSCWD 2027 截稿 2026-11-30（AoE），6 页双栏 single-blind。

T5 真实结果已拉取（`3de06cd`）并验收通过：六种方法全报，来源/实现 SHA256、T4 协议和参考指标重现、覆盖/事件计数及 Markdown/JSON 一致性通过。STAG 未被固定 EMA 在主要指标上全面支配，存在部分自适应折中收益：相对 Jitter 接近的 EMA `0.5`，准确稳定占比 **58.15% vs 55.72%（+2.43 个百分点）**，L2 更低、AUC 更低、切换响应率同为 **55.06%**；84 个共同切换事件平均快 **0.155 帧**。相对原始稳定占比接近的 EMA `0.25`，切换响应率高 **4.43 个百分点**，80 个共同事件平均快 **0.563 帧**，但 Jitter 和 AUC 更差。

T5 当前判断：门控能补偿强平滑的一部分定位/响应损失，但当前 STAG 未全面优于普通 EMA，且仍未超过单帧的切换响应。EMA `0.75` 实测 AUC **0.940120**、L2 **0.098941**，保留为强简单对照，不作 test 最优参数选择。完整六行结果与配对统计见 `cscwd/results/t5_ema_ablation.json/.md`。T5 代码已通过 13 项合成测试，完整缓存和事件明细留服务器。

T6 本地实现完成：`cscwd/code/scripts/visualize_temporal.py` 核验 T5 来源和六套缓存，确定性选取门控收益、单帧独有响应失败、STAG 滞后、准确稳定收益与真实标注人数 ≥4 的拥挤案例，全部失响应案例完整输出。`case_selection.py/plotting.py` 输出 RGB/真实框/原始热图、原帧号轨迹、误差、alpha/inout/三门控信号和事件计分窗口。**15 项合成测试及合成图视觉检查通过**。完整图/候选与逐帧数据留服务器，JSON/MD 和最多 5 例的小 PNG 预览经 git 回传；真实 VAT 案例等待服务器运行。

结果回传规则已修正：`cscwd/.gitignore` 放行 `results/` 直属小 JSON，覆盖根目录的 `*.json`；完整 track JSON、NPY 和缓存目录继续忽略。

## 已知阻塞
- Router 探索线（coverage-aware-router）：VAT 效果不明显，已暂停；详细状态见 `COVERAGE_AWARE_ROUTING_IDEA_PROGRESS.md`，与当前 CSCWD 时序方向无关。

## 文件约定
- 每个投稿目录各自放 LaTeX 源、图、实验结果；实验结果只写真实跑过的，没跑的对照不写进表格
- `RESULT_PROVENANCE.md` 记录每个数字来自哪次运行，写论文时以它为准
- 每个投稿目录独立放自己的新代码/论文：`<venue>/code/`（新代码）、`<venue>/latex/`（论文）、`<venue>/results/`（关键小结果）；旧 `gazelle` 包只 import 不改
- git 双向传递：本地改 code → push → 服务器 pull → nohup 手动跑 → 小结果（JSON/MD）push 回本地；大文件（.npy/.pt）不进 git，只在 RESULT_PROVENANCE 记路径+SHA

## 下一步
1. 用户手动执行实现计划中的 T6 nohup 命令，回传 `cscwd/results/t6_visualization.json/.md`、`cscwd/results/t6_previews/` 和 provenance。完整输出留服务器 `cscwd_cache/gazespot_vitb_epoch7/t6_diagnostics/`，路径与 SHA256 自动记录。
2. 结合真实成功/失败片段判断旧峰残留、错误峰写入和门控触发，再决定一项机制改动；后续参数选择或拟合转到按源视频划分的 train/validation，当前保持权重。
