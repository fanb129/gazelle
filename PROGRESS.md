# 小论文 · 进度

> 状态文件。每次开工先读，收工写回。分支：`codex/coverage-aware-router`

## 三条投稿线
| 线 | 目录 | 状态 |
|---|---|---|
| ACM MM 2026（GazeSpot） | `rebuttal/` | 已投，返修材料与实验结果在 `rebuttal/results` |
| ICASSP 2027 | `icassp2027/` | 初稿完成（2026-09 底），源稿在 `icassp2027/2026ACMMM` |
| CSCWD | `cscwd/` | T1–T6 完成；R1 本地实现完成，35 项新增合成测试和 17 项原指标回归通过，等待服务器结果 |

## 当前优先级（2026-11-17）
**CSCWD 投稿是第一优先。当前主方案：冻结 GazeSpot ViT-B，训练轻量因果模块，结合人物与场景视觉证据判断历史有效性，联合修正 heatmap 与 In/Out。**
方案已写入 `cscwd/temporal_joint_refinement.md`；R1–R6 的依赖、产出与验收写入 `cscwd/implementation_plan.md`。原规则 STAG 保留为先导基线。R1 数据/协议准备的本地代码已完成，等待用户在服务器手动运行和回传；新时序模块尚未实现/训练，R2–R6 尚未开始。以下 T1–T6 数字保留为真实先导结果。
T1 真实结果已从服务器拉取（`17699cc`）并验收通过：train **1,033 tracks / 58,507 图像帧 / 132,563 人物帧**；test **298 tracks / 13,127 图像帧 / 31,978 人物帧**。原始 CSV 与旧 JSON 的全部计数核对通过，脚本 SHA256 一致；来源见 `cscwd/results/t1_track_recovery.json` 与 `RESULT_PROVENANCE.md`。

T2 真实结果已拉取（`24cf720`）并验收通过：指定 ViT-B GazeSpot VAT `epoch_7.pt`（`train_vat_sasa_ggsf/2026-03-12_19-24-13/epoch_7.pt`），覆盖 **298 tracks / 13,127 图像帧 / 31,978 人物帧**；float32 heatmap、inout、argmax 无损读回与重建全部通过，输入 SHA256 对齐 T1，代码 SHA256 对齐 T2 实现提交。NVIDIA GeForce RTX 3090 上整项缓存作业耗时 **731.430930521 秒**；来源见 `cscwd/results/t2_single_frame_cache.json` 与 provenance。

T3 真实结果已拉取（`eb28855`）并完成运行与口径验收：单帧 **AUC 0.940182 / L2 0.098995 / AP 0.886801**；初始 STAG **AUC 0.936860 / L2 0.099197 / AP 0.886801**。固定权重 `[-2,20,2,10]` 的初始规则定位未改善；来源见 `cscwd/results/t3_stag_comparison.json`。两侧覆盖相同 31,978 人物帧/20,655 画内定位帧，来源与实现 SHA256、轨迹对齐和读回核对通过。

T4 真实结果已拉取（`7f15cd7`）并验收通过：同源缓存/代码 SHA256、标准指标重现和样本对齐核对通过。Jitter mean **0.016647→0.012135（−27.10%）**，同帧对 GT 位移 mean **0.005922**；准确且稳定占比 **52.75%→58.15%（+5.40 个百分点）**。但画内落点切换响应率 **58.23%→55.06%**，共同响应的 87 个切换事件平均慢 **0.276 帧**，另损失 5 个单帧可响应事件。当前结论：**稳定性改善，定位略降，切换响应未改善**，门控相对普通 EMA 的价值尚待 T5 对照。来源：`cscwd/results/t4_temporal_comparison.json`。

T4 代码与协议已实现并通过 15 项新增合成测试、10 项标准/T3 回归；固定阈值和完整服务器命令见 `cscwd/implementation_plan.md`。服务器仓库：`/home/fb/src/paper/gazelleV1`。CSCWD 2027 截稿 2026-11-30（AoE），6 页双栏 single-blind。

T5 真实结果已拉取（`3de06cd`）并验收通过：六种方法全报，来源/实现 SHA256、T4 协议和参考指标重现、覆盖/事件计数及 Markdown/JSON 一致性通过。STAG 未被固定 EMA 在主要指标上全面支配，存在部分自适应折中收益：相对 Jitter 接近的 EMA `0.5`，准确稳定占比 **58.15% vs 55.72%（+2.43 个百分点）**，L2 更低、AUC 更低、切换响应率同为 **55.06%**；84 个共同切换事件平均快 **0.155 帧**。相对原始稳定占比接近的 EMA `0.25`，切换响应率高 **4.43 个百分点**，80 个共同事件平均快 **0.563 帧**，但 Jitter 和 AUC 更差。

T5 当前判断：门控能补偿强平滑的一部分定位/响应损失，但当前 STAG 未全面优于普通 EMA，且仍未超过单帧的切换响应。EMA `0.75` 实测 AUC **0.940120**、L2 **0.098941**，保留为强简单对照，不作 test 最优参数选择。完整六行结果与配对统计见 `cscwd/results/t5_ema_ablation.json/.md`。T5 代码已通过 13 项合成测试，完整缓存和事件明细留服务器。

T6 真实结果已在本地同步（`9e0fdd0`）并验收：T5 来源、六套缓存 index、实现 SHA256、案例计数及 10 张本地 PNG 的 SHA256/大小核对通过。候选数为门控收益 **7**、单帧独有失响应 **5**、STAG 滞后 **11**、稳定收益 episode **726**、拥挤 clip **16**；完整输出 **13 例**，其中全部 5 个失响应案例保留。本地已查看五类各一例的帧图/时间线，完整逐帧 JSON 和其余大图仍在服务器。

T6 可见结论：CBS `2517` 例单帧/STAG 立即响应，EMA `0.25/0.5` 在 3 帧计分窗口内未响应；CBS `12144` 例单帧延迟 **8 帧**、STAG **12 帧**、EMA `0.25` **13 帧**、EMA `0.5` **12 帧**。Downton `13190` 例单帧延迟 **6 帧**后满足连续 3 帧条件，STAG 有短暂低误差段但未满足该条件，并非整段从未接近正确点。所看切换/滞后例中头部位移很小、预测 inout 稳定，alpha 主要随热图位移变化；当前观测符合历史惯性与短暂正确段被缩短，不能把这一成因直接推广到全部 5 个失败例。

T1–T6 的数据、缓存、初版门控、统一评测、固定 EMA 对照和定性输出已走通。当前门控有稳定性/相对强 EMA 的折中收益，但未达到优于单帧切换响应的目标。2026-11-17 已将后续主路线更新为“视觉证据驱动的历史保留/更新/释放 + 定位/状态联合修正”；原四参数校准进入对照，不再作为唯一下一步。旧 train 预测缓存命令尚未执行验收；T2 test 缓存只有预测，不等同于新方案所需的视觉特征缓存。

**R0 / 2026-11-17 / 方案落地**：新增 `cscwd/temporal_joint_refinement.md`；同步实现计划、论文蓝图、README、提示词、文献与指标约束。计划覆盖 source-video 分组、开发集诊断、冻结特征缓存、普通联合时序头、历史有效性机制、训练/验证强对照、冻结配置 test 与论文证据。本次仅修改文档，没有新增实验数字或服务器运行，已有结果文件保持原样。

指标目标：争取 AUC/L2/AP 与准确稳定性、分型响应的更好折中；不预写全部提升。旧规则 AP 与 `leave_frame` 响应不变；新 In/Out 分支可学习分数排序，同时独立评估状态错误翻转与进出响应。当前和过去 RGB/真实框为唯一观察，底座与原 decoder 全冻结；真实注视点/画内外标签只用于当前帧训练监督与离线计分，历史由模型预测产生。

**R1 / 2026-11-17 / 本地实现完成**：新增 `development_split.py`、`state_metrics.py`、`development_diagnostics.py`、`development_protocol.py` 和 `scripts/prepare_development.py`。复用既有 T2 冻结推理脚本，按保守 annotation-video 组划分开发 train/validation；导出原标准/T4 joint 与独立状态基线、预声明选模配置、训练组四类诊断图和未填写人工标签。完整来源/覆盖/SHA256 核对通过后才发布机器报告，`passed` 表示机器准备，`stage_status=awaiting_local_review` 保留人工诊断/分组语义待验收状态。

**R1 / 产出与本地数字**：35 项新增合成检查通过（分组 10、状态指标 11、诊断 8、入口集成 6），另有 17 项旧标准/时序指标回归通过；已查看合成图。旧 `gazelle` 和 T4 指标实现均未修改，未产生真实 VAT 训练/评测数字。本地环境为 `/private/tmp/gazelle-r1-venv`，GPU 前向及真实诊断都由用户在服务器执行；两条完整 nohup 命令与回传文件见 `implementation_plan.md` §8。

结果回传规则已修正：`cscwd/.gitignore` 放行 `results/` 直属小 JSON，覆盖根目录的 `*.json`；完整 track JSON、NPY 和缓存目录继续忽略。

## 已知阻塞
- Router 探索线（coverage-aware-router）：VAT 效果不明显，已暂停；详细状态见 `COVERAGE_AWARE_ROUTING_IDEA_PROGRESS.md`，与当前 CSCWD 时序方向无关。

## 文件约定
- 每个投稿目录各自放 LaTeX 源、图、实验结果；实验结果只写真实跑过的，没跑的对照不写进表格
- `RESULT_PROVENANCE.md` 记录每个数字来自哪次运行，写论文时以它为准
- 每个投稿目录独立放自己的新代码/论文：`<venue>/code/`（新代码）、`<venue>/latex/`（论文）、`<venue>/results/`（关键小结果）；旧 `gazelle` 包只 import 不改
- git 双向传递：本地改 code → push → 服务器 pull → nohup 手动跑 → 小结果（JSON/MD）push 回本地；大文件（.npy/.pt）不进 git，只在 RESULT_PROVENANCE 记路径+SHA

## 下一步
1. **R1（等待服务器结果）**：按实现计划 §8 先生成 `r1_train_predictions`，成功后运行 `prepare_development.py`。回传 `r1_train_single_frame_cache.json`、`r1_split_and_diagnosis.json/.md`、`r1_previews/` 与 provenance；真实覆盖对齐 T1 的 **1,033 tracks / 132,563 人物帧**。本地核对组名/来源并查看全部选中训练案例，另记 `r1_manual_review.md`，随后完成 R1 验收。
2. R1 验收后依次进入 **R2 冻结视觉缓存与普通联合头 → R3 历史有效性机制 → R4 训练/验证与强对照 → R5 冻结配置 test → R6 论文与协作证据**，每次只推进一个任务。
3. 所有新模型选择在开发组完成。冻结 VAT checkpoint 此前已见过 VAT train，validation 留出仅针对新增模块；已查看的 test 为先导分析，不能再用于调参。仅有 VAT 逐人结果时，不声称已验证共同关注或协作收益。
