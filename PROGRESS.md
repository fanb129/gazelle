# 小论文 · 进度

> 状态文件。每次开工先读，收工写回。分支：`codex/coverage-aware-router`

## 三条投稿线
| 线 | 目录 | 状态 |
|---|---|---|
| ACM MM 2026（GazeSpot） | `rebuttal/` | 已投，返修材料与实验结果在 `rebuttal/results` |
| ICASSP 2027 | `icassp2027/` | 初稿完成（2026-09 底），源稿在 `icassp2027/2026ACMMM` |
| CSCWD | `cscwd/` | 精简 R4 首轮入口本地完成，31 项新增检查通过；等待服务器手动训练与比较 |

## 当前优先级（2026-11-17）
**CSCWD 投稿是第一优先。当前主方案：冻结 GazeSpot ViT-B，训练轻量因果模块，结合人物与场景视觉证据判断历史有效性，联合修正 heatmap 与 In/Out。**
方案已写入 `cscwd/temporal_joint_refinement.md`；R1–R6 的依赖、产出与验收写入 `cscwd/implementation_plan.md`。R1/R2 已验收，R3 历史视觉对应/有效性模型与诊断本地完成。用户授权的精简 R4 已写入 `cscwd/r4_compact_plan.md`，版本 `r4_compact_v2`；首轮入口已本地完成，等待服务器手动运行，真实 R3/R4 训练尚未运行。原规则 STAG 保留为先导基线，以下 T1–T6 数字保留为真实先导结果。
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

**R1 / 2026-11-17 / 真实结果验收完成**：已拉取结果提交 `d6adba4`，保留并恢复本地原有 provenance 补充。来源小报告/代码 SHA256、既定模型与输入、计数/响应分母、种子分组和全部 12 张 PNG 的 SHA256/大小核对通过，记录于 `cscwd/results/r1_local_verification.json`。完整 NPY、index、manifest 和逐事件明细留服务器，本地不声称重新计算了全量预测指标。

R1 开发划分为 **32/8 个保守 annotation-video/program 组**：train **827 tracks / 98,362 人物帧 / 44,839 图像帧**；validation **206 tracks / 34,201 人物帧 / 13,668 图像帧**，总计对齐 T1。接受既定 seed 的分组，不将节目/影片名称等同于已核实的独立录制 ID。validation 单帧 **AUC 0.954206 / L2 0.079605 / AP 0.960923**、准确稳定占比 **60.19%**，是开发分区基线，不是新方法增益。

R1 状态结果：validation 画外错误率 **25.73%**、画内错误率 **2.77%**；state-only enter **179/189（94.71%）**、leave **93/169（55.03%）**；joint gaze_shift **135/221（61.09%）**。state-only 与 joint enter 的窗口和分母不同，不能直接把响应率差全归于定位条件。高 AP 与画外阈值错误可以同时存在。

R1 人工诊断已逐例记录于 `cscwd/results/r1_manual_review.json/.md`：查看全部 12 张回传图，每例 5 个快照；标签可重叠，预测误差 7 例、上下文注意区域改变 2 例、真实变化/边界无法判定 8 例。Three Idiots 单帧 in→out→in 不直接当真实释放监督；stable_reference 只保证锚点短窗口。此固定抽样不用于估计全数据类别频率，也未证明视觉对应机制有效。R4 将补充仅在开发 train 拟合的全局单调状态校准对照，区分阈值偏置与时序收益；尚未运行，不回写原 R1 配置或结果。

**R2 / 2026-11-17 / 本地实现完成**：新增 `feature_cache.py`、`temporal_model.py`、`temporal_data.py` 与 `cache_frozen_features.py`、`train_temporal_joint.py`。先 probe 当前帧冻结特征与原始前向，再按唯一图像存原生网格末层 FP16；原 R1 heatmap/In/Out float32 保持原样。普通 GRU 只训练新增投影与双输出残差头，当前场景/分数面积头框/预测区域证据加自身过去预测形成状态；没有提前实现 R3 显式对应或释放机制。

**R2 / 本地产出与数字**：31 项新增合成测试通过（特征 7、模型/损失 10、连续段数据 8、来源与完整训练 CLI 6）。实际跑通临时合成数据的梯度更新、checkpoint 重载和完整验证缓存，不记录其指标为 VAT 实验。训练跨 chunk 仅 detach，不重置同人物状态；跨人/缺口重置；GT 只构造当前 MSE/BCE 目标。原包与已有预测/报告未改。固定 smoke 配置为 seed 0、一个 epoch，不选最优模型。

**R2 / 2026-11-17 / 特征 probe 验收完成**：已拉取 `139a2c7`，32 张开发 train 图像 / 69 人物帧实测末层特征为 **768×32×32**，按原生网格 FP16 存储。FP32 原预测 heatmap/In/Out 最大误差分别 **9.83476639e-7 / 4.17232513e-7**，argmax 变化 **0**；FP16 特征 relative L2 误差 **0.02042%**。58,507 张图像的特征估算 **85.7036 GiB**，probe 时可用空间 **2.4847 TiB**。来源/代码 SHA256、开发 train 抽样、覆盖和存储算式核对通过，记录在 `cscwd/results/r2_probe_review.json/.md`。下一步按 §9.3 缓存，再按 §9.4 训练新增联合头；全量缓存及真实训练尚未运行。

**R2 / 2026-11-17 / 全量缓存与 smoke 验收完成**：拉取 `85d9050`；58,507 张唯一图像的原生 FP16 特征缓存完成，文件 **92,023,554,176 bytes**。固定一个 epoch 实际训练 **98,362 帧 / 821 chunks / 347,522 个新增参数**，验证 **34,201 帧**，两个头均更新。来源/代码、覆盖、loss 与指标分母核对通过，产出 `cscwd/results/r2_local_review.json/.md`，大文件路径/SHA256 已记 provenance。

**R2 / 真实结果与下一步**：validation 单帧→smoke 为 **AUC 0.954206→0.954325 / L2 0.079605→0.079602 / AP 0.960923→0.960799**，基本持平；状态错误 **11.13%→10.53%**，准确稳定占比 **60.19%→59.46%**。state-only enter **179/189→175/189**、leave **93/169→97/169**，joint gaze_shift **135/221→135/221**。当前配置因 enter 响应下降未达到 R4 选模门槛；R2 工程验收完成，下一步 R3 历史目标对应与有效性控制，普通 GRU 保留对照。R3 尚未启动。

**R4 / 2026-11-17 / 精简计划落地**：新增 `cscwd/r4_compact_plan.md`，版本 **r4_compact_v2**。首轮普通 GRU/新机制各单配置、seed 0、五个 epoch，从新模块随机初始化、同源缓存同预算；保留标准精度容忍度，以准确稳定占比优先选 checkpoint 并全报指标/Pareto，取消逐类响应不得下降的硬门槛。有信号后补 validity 读门消融，按主张补状态校准或无历史头；完整搜索、多滤波/单输出矩阵、多 seed/bootstrap 延后。原 R1/R2 记录保持原样。

**R3 / 2026-11-17 / 本地实现完成**：新增 `history_model.py`、`history_diagnostics.py` 与模型/集成测试。以过去预测区域在冻结空间形成模板，在当前网格做视觉对应；validity 控制历史读入，refresh 控制模板写入，双输出残差联合修正。每帧诊断含有效性、刷新/写入强度、对应坐标和匹配摘要，并可按原全局索引落盘。实现说明与结果写入 `cscwd/results/r3_mechanism_design.md`、`r3_local_checks.json/.md`。

**R3 / 实际本地数字与下一步**：R3 **372,677 参数**，普通 GRU **347,522**；同 seed 共享初始化一致，native **768×32×32** 两步前向通过。**24 项新增检查（模型16、集成8）及18项 R2回归通过**，合成完整训练/验证/重载与诊断读回完成。没有新增真实 VAT 性能数字。下一步实现精简 R4 训练/逐 epoch 选择入口并给服务器命令，R3 不追加真实 smoke 作业。

**R4 / 2026-11-17 / 第一轮本地入口完成**：新增固定预算/选模协议、同次前向完整验证、`train_compact_joint.py`、`compare_compact_joint.py` 和 `run_compact_round.sh`。普通 GRU/新机制各从头五个 epoch，保存全部 epoch、selected/last checkpoint；比较报告含完整候选/Pareto、原始帧事件配对和开发选择快照。**31 项新增检查通过**（协议11、验证5、训练6、比较9），合成双模型完整 CLI 已跑通，源文件与检查记录见 `cscwd/results/r4_local_checks.json/.md`。完整服务器命令与回传清单在实现计划 §10，真实作业尚未运行。

结果回传规则已修正：`cscwd/.gitignore` 放行 `results/` 直属小 JSON，覆盖根目录的 `*.json`；完整 track JSON、NPY 和缓存目录继续忽略。

## 已知阻塞
- Router 探索线（coverage-aware-router）：VAT 效果不明显，已暂停；详细状态见 `COVERAGE_AWARE_ROUTING_IDEA_PROGRESS.md`，与当前 CSCWD 时序方向无关。

## 文件约定
- 每个投稿目录各自放 LaTeX 源、图、实验结果；实验结果只写真实跑过的，没跑的对照不写进表格
- `RESULT_PROVENANCE.md` 记录每个数字来自哪次运行，写论文时以它为准
- 每个投稿目录独立放自己的新代码/论文：`<venue>/code/`（新代码）、`<venue>/latex/`（论文）、`<venue>/results/`（关键小结果）；旧 `gazelle` 包只 import 不改
- git 双向传递：本地改 code → push → 服务器 pull → nohup 手动跑 → 小结果（JSON/MD）push 回本地；大文件（.npy/.pt）不进 git，只在 RESULT_PROVENANCE 记路径+SHA

## 下一步
1. **精简 R4（本地就绪，等待服务器）**：服务器 pull 后执行实现计划 §10.1 的单条 nohup 命令，顺序完成普通 GRU、新机制各五个 epoch，再生成比较和开发选择快照；按 §10.5 回传小报告与 provenance。R2 特征缓存直接复用。
2. 有信号后按计划补最少机制证据，再进入 **R5 冻结配置 test → R6 论文与协作证据**。R4 报告版本化修订后的精度条件与完整折中，不按新版规则改写既有 R2 验收。
3. 所有新模型选择在开发组完成。冻结 VAT checkpoint 此前已见过 VAT train，validation 留出仅针对新增模块；已查看的 test 为先导分析，不能再用于调参。仅有 VAT 逐人结果时，不声称已验证共同关注或协作收益。
