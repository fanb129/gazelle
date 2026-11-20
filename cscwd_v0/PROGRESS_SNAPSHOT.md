# 小论文 · 进度

> 状态文件。每次开工先读，收工写回。分支：`codex/coverage-aware-router`

## 三条投稿线
| 线 | 目录 | 状态 |
|---|---|---|
| ACM MM 2026（GazeSpot） | `rebuttal/` | 已投，返修材料与实验结果在 `rebuttal/results` |
| ICASSP 2027 | `icassp2027/` | 初稿完成（2026-09 底），源稿在 `icassp2027/2026ACMMM` |
| CSCWD | `cscwd/` | 用户要求收敛主线：GazeSpot任务特征后接共享ConvLSTM，直接优化VAT AUC/L2/AP；取消D7诊断 |

## 当前优先级（2026-11-20）
**CSCWD 投稿是第一优先。按用户最新指令，主线改为GazeSpot-B + 单向ConvLSTM直接预测heatmap/InOut，以VAT AUC/L2/AP提升为目标，保持官方train/test。取消D7冻结诊断和候选选择路线。推荐方案见 `cscwd/temporal_gazespot_plan.md`，下一任务实现。**
方案已写入 `cscwd/temporal_joint_refinement.md`；R1–R6 的依赖、产出与验收写入 `cscwd/implementation_plan.md`。R1/R2 已验收，R3 历史视觉对应/有效性模型与诊断本地完成。原精简 R4 `r4_compact_v2` validation 首轮已验收，两模型均选 epoch 1、整体仍首选单帧。用户随后明确取消后续 validation，当前执行 `cscwd/r4_official_train_test_plan.md` 的 `r4_official_train_test_v1`：完整官方 train 训练，每 epoch 完整 test，按 AUC 优先、L2 更低、AP 更高选一整轮。官方train/test首轮结果 `aa0ab55` 已通过服务器Git取回并验收，两模型均选epoch 5；AUC/AP小幅改善，但定位误差和时序指标有代价，新机制尚未显示超过普通GRU的额外收益。原规则 STAG 保留为先导基线，以下 T1–T6 数字保留为真实先导结果。
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

**R4 / 2026-11-17 / 中断后完整性复核**：重新核对实现提交 `bda1501` 的训练、验证、选模、比较、后台入口与 §10 命令，源码 SHA256 与原记录一致；31 项 CPU 合成检查重新通过，另以临时替身实际验证顺序运行、第一/第二训练失败停止和含空格参数传递三种场景。未发现缺文件或未接通流程，无需修改源码。记录为 `cscwd/results/r4_completeness_review.json/.md`；下一步仍是用户手动运行服务器首轮并回传。

结果回传规则已修正：`cscwd/.gitignore` 放行 `results/` 直属小 JSON，覆盖根目录的 `*.json`；完整 track JSON、NPY 和缓存目录继续忽略。

**R4 / 2026-11-17 / 启动说明简化**：实现计划 §10.1 为唯一后台启动入口，外层设 `CUDA_VISIBLE_DEVICES=1` 使用物理 GPU 1，自动依次完成两项训练和 CPU 比较；删除重复分步命令，回传清单保留于 §10 末尾。仅更新本文档与实现计划，无新增实验数字；下一步由用户在服务器启动。

**R4 / 2026-11-18 / 第一轮真实结果验收**：已拉取 `f1d41e4`，来源/预算/覆盖/指标算式/选模与 Pareto 共 **235 项本地核对通过**，产出 `cscwd/results/r4_local_review.json/.md`，大文件仍留服务器。普通 GRU/新机制各 seed 0、五个 epoch，均按既定规则选 **epoch 1**。同一 validation 单帧→新机制：AUC **0.954206→0.954290**、L2 **0.079605→0.079562**、AP **0.960923→0.960808**；准确稳定 **60.1874%→59.4964%**，状态错误 **11.1254%→10.5114%**，joint gaze_shift **135/221→135/221**。新机制仅比 GRU 多8个准确稳定窗口、少6帧状态错误；总体仍选单帧。当前是状态纠偏折中，核心机制额外收益尚未建立；这些 validation 数字不能与 ViT-L test 直接比较。下一步复用 epoch-1 缓存分解状态/定位贡献并汇总门控诊断，按结果补状态校准或调整训练，暂不进入 R5。

**R4 / 2026-11-18 / 分区与训练约定确认**：已读取用户指定 PDF 第6页表2，原 ViT-B VAT主表 **0.940 / 0.099 / 0.887** 与已跑官方test单帧结果舍入一致；当前更高数字来自official train内的开发validation，原底座曾见过official train。后续所有训练均每epoch完整验证、记录全部轮次及checkpoint，按训练前固定规则选择一轮整套最佳指标并保存 `selected.pt`。当前R4已实现，本次只将约定补入实现计划与状态文件，无新增实验；下一步仍为已有缓存贡献诊断。

**R4-B / 2026-11-18 / 官方 train/test 协议与入口完成**：按用户指定只使用官方 train/test，132,563 人物帧完整训练、每轮完整评测31,978人物帧；用户确认AUC优先、并列L2更低/AP更高选择一轮整套指标，不使用旧precision门槛。新增官方数据/评测/选模模块、test特征缓存、训练/比较/顺序入口及测试，代码全部位于 `cscwd/code/`；旧源码和validation报告保留。**36项本地合成检查通过**（数据5、特征6、协议5、训练6、比较8、后台6），产出 `cscwd/results/r4_official_local_checks.json/.md`，没有新增真实VAT指标。完整train特征和train/test原预测复用，需补test特征；两模型seed0各五轮，从头初始化。下一步由用户在服务器按实现计划§11.1单条命令（物理GPU1）运行并回传 `r4_official_*`。

**R4-B / 2026-11-18 / 服务器数据与缓存只读确认**：已SSH检查，官方train/test的轨迹与原图均在，train **1,033 tracks / 132,563人物帧 / 58,507图像**、test **298 / 31,978 / 13,127**，全部图像路径存在且分区不交叉。R1只逻辑拆分官方train，完整train特征与train/test原预测均保留；新test视觉特征目录和完成报告尚不存在，需由§11.1入口先生成。服务器仍为 `f1d41e4`、未pull新脚本。产出 `cscwd/results/r4_official_server_preflight.json/.md`，10项既有数据检查通过；未启动服务器作业。下一步先pull，再手动运行GPU1单条入口。

**R4-B / 2026-11-18 / 训练文件说明完成**：用户已启动官方train/test服务器作业。新增 `cscwd/training_file_guide.md`，按实际源码说明官方分区与训练直接读取的index、预测/视觉特征缓存、标签和行映射、原始/新增权重、逐轮输出、选模、日志及回传文件；同步README入口。只在本地更新文档，未修改运行脚本、协议文件或服务器任务，没有新增实验数字；下一步等待本轮结果回传。

**R4-B / 2026-11-18 / 官方train/test首轮结果验收**：结果已在服务器提交为 `aa0ab55`，GitHub当时仍为 `c966239`；已通过只读Git/SSH取回并合并，保留双方历史。来源/预算/覆盖/算式/选模等 **283项本地核对通过**，产出 `cscwd/results/r4_official_local_review.json/.md`。test特征13,127图像FP16缓存完成，20,646,985,856 bytes；两模型完整train各五轮、每轮完整test，均按AUC规则选 **epoch 5**。单帧→新机制：AUC **0.940182→0.940466**、L2 **0.098995→0.099244**、AP **0.886801→0.892778**，准确稳定 **52.7531%→50.3334%**，状态错误 **19.5165%→19.3289%**，gaze_shift **92/158→88/158**。普通GRU epoch 5为AUC首选（0.940541）；新机制AUC/AP与准确稳定低于GRU，L2更低、leave多1个响应。当前保留小幅标准/状态排序改善与出画响应取舍，尚不宣称核心机制有效。下一步复用缓存分解定位/状态贡献并汇总门控诊断，再决定一次针对性调整；本次未启动新作业。

**R4-D1 / 2026-11-18 / 缓存贡献与门控诊断入口完成**：新增 `cache_diagnosis.py`、`history_diagnosis.py`、`diagnose_official_cache.py` 与三套测试，计划 `cscwd/r4_cache_diagnosis_plan.md`（`r4_cache_diagnosis_v1`）。四种坐标/状态组合计分，准确稳定窗口坐标/状态/交互及对称贡献分解，lost/gained互斥失败组合；新机制控制量按历史可用、GT状态/稳定帧和原帧±3的代理事件邻域汇总。源缓存按人物/原帧/source行对齐，并重现两端L2/AP与全部时序/状态摘要。**23项本地合成检查通过**（交叉11、门控8、入口4），包括禁止Torch导入的实际CLI、缓存不变、轨迹重排和手算分解；记录 `cscwd/results/r4_cache_diagnosis_local_checks.json/.md`。无新增真实VAT数字或训练作业；下一步由用户按§12.1单条CPU nohup运行并回传诊断结果，再决定训练调整。

**R4-D1 / 2026-11-18 / 真实缓存诊断验收**：拉取 `287d063`，169项本地来源/算式/端点/分母核对通过，产出 `cscwd/results/r4_cache_diagnosis_local_review.json/.md`。新机制M00/M10/M01/M11准确稳定窗口为 **10,443 / 10,399 / 9,992 / 9,964**，总−479；坐标项−44、分数项−451、交互+16；对称分数项−443，约占净下降92.48%。GT画内引入1,155帧画外误判，仅纠正40帧；画外净纠正1,175帧，因此总错帧只少60。历史可用时validity均值0.671961、范围0.496976–0.896991，未见明确事件低门释放形态，聚合不当机制因果证据。下一步R4-D2：固定epoch 5与坐标，为单帧/GRU/新机制做train拟合的一维单调输出状态校准；新增模块先冻结前向补train分数，校准不反馈历史，保持AP排序。D2尚未实现或运行，本次无新增训练作业。

**R4-D2 / 2026-11-18 / 校准对照本地入口完成**：新增 `state_calibration.py`、`calibration_rollout.py`、`calibrate_official_state.py` 与三套测试，规格为 `cscwd/r4_state_calibration_plan.md`（`r4_state_calibration_v1`）。单帧/GRU/新机制各用完整train拟合一个全局log-odds偏置，固定区间[-16,16]、未加权BCE有限可变部分、80轮导数二分；固定epoch 5与坐标，新增模块只冻结前向，校准不反馈历史。float64输出保留原0/1、不同分数顺序和ties，核对train/test AP不变及train目标不增。

**R4-D2 / 本地产出与下一步**：**18项新增CPU合成检查通过**（数学10、因果前向4、完整入口4），覆盖已知偏置、极端float32保序、GT不进入预测、跨chunk一致性/人物与缺口重置、原行映射、来源不变、原端点重现、排序失败阻止成功报告和完整CLI/provenance。记录 `cscwd/results/r4_state_calibration_local_checks.json/.md`；旧源码、真实结果和原训练协议保持。没有新增真实VAT数字，未启动服务器作业。下一步用户按 `implementation_plan.md` §13.1单条命令用物理GPU 1运行，回传 `r4_state_calibration.json/.md` 与provenance，再决定针对性训练调整。

**R4-D2 / 2026-11-18 / 真实校准结果验收**：已pull `75f305c`，393项本地小报告来源/源码/覆盖/端点/比例/配对算式核对通过，记录 `cscwd/results/r4_state_calibration_local_review.json/.md`。三个train偏置为单帧 **−0.423750**、GRU **+0.041511**、新机制 **+0.090667**；固定epoch 5与历史，AUC/L2/AP保持。新机制准确稳定 **50.3334%→50.8386%（+100窗口）**、状态错误 **6,181→6,055**、错误翻转 **798→758**，只补回相对原始单帧479窗口损失的 **20.88%**；gaze_shift仍 **88/158**，校准后仍比校准单帧/GRU少96/41个准确稳定窗口。单帧仅校准也使leave **91→107**，因此原出画响应数量提升不足以证明历史贡献。来源和执行通过，效果为部分纠偏，机制额外收益尚未建立；完整NPY/PT/事件明细留服务器，本地未重新拟合或逐帧计分。

**R4-D2 / 下一步决策**：不继续在test扫描偏置。优先R4-D3一次训练目标调整：保留逐帧MSE/BCE，增加稳定GT段的相邻状态一致性与真实状态变化后的当前帧响应监督；GRU/新机制同设置、各从头五轮，冻结GazeSpot，沿用官方train/逐epoch test与AUC选整轮。GT只构造loss；掩码、权重、跨chunk处理在下一任务实现前固定。本次仅验收与计划，无新增训练入口或服务器任务；有实际时序信号后再补最少无历史/读门对照。

**R4-D3 / 2026-11-18 / 状态时序监督本地实现完成**：新增 `state_supervision.py`、独立训练/比较入口、顺序wrapper与四套测试，规格固定于 `cscwd/r4_state_supervision_plan.md`（`r4_state_supervision_v1`）。loss为 `L_heatmap + 0.1 * (L_BCE + 0.1 * (L_stable + L_transition))`；稳定同状态相邻概率L1，真实状态翻转当前及随后两帧的当前状态BCE，独立分母。监督carry与模型历史分开，同人物跨chunk保留/detach，跨人/缺口清空；GT仅loss。普通GRU/新机制均从头五轮、seed 0，复用冻结缓存与原完整train/逐epoch test/AUC选整轮，推理架构保持、不应用D2校准。

**R4-D3 / 本地产出与下一步**：**29项CPU合成检查通过**（loss/epoch12、训练及实际比较CLI6、比较7、wrapper4），覆盖手算/梯度、跨chunk完整监督计数、两模型真实合成五轮、重载、同次诊断、test标签不影响五轮权重、前轮同源差值、来源SHA与GPU1/失败停止。记录 `cscwd/results/r4_state_supervision_local_checks.json/.md`；七个新Python文件的3.10语法与shell命令检查通过。旧源码/原结果不改，没有新增真实VAT数字或服务器作业。下一步用户按实现计划§14.1单条命令运行，回传两模型/比较/selection小报告与provenance，再验收稳定/响应取舍及相对GRU效果。

**R4-D3 / 2026-11-19 / 真实结果验收与分析**：已pull `db239cf`，240项本地小报告来源/源码/覆盖/loss/选轮/模板/provenance核对通过，记录 `cscwd/results/r4_state_supervision_local_review.json/.md`。每轮train 132,563帧、stable pairs 129,666、transition frames 5,358；两模型完整五轮，按原规则选 **GRU epoch 2 / 新机制 epoch 3**。新机制整行 **AUC 0.940481 / L2 0.099299 / AP 0.888879**，准确稳定 **51.2225%**、状态错帧 **6,168**、错误翻转 **717**、gaze_shift **89/158**；仍比单帧少303个准确稳定窗口、3个gaze响应，比本轮GRU少63个稳定窗口。原首轮选epoch 5→D3选epoch 3的+176稳定窗口不能直接归因于loss；同epoch 3实际 **10,232→10,140**、翻转 **660→717**。两模型五个同轮AP均下降/L2均增加，稳定只有epoch 4增加。本固定配置无一致收益，尚无历史有效性机制优势；大数组/权重/事件未在本地重算，原报告保持。

**R4-D3 / 下一步决策**：不继续追加D3权重扫描或校准。下一项 **R4-D4当前帧无历史联合修正对照**：保留普通GRU当前视觉输入/原预测/真实头框与双输出结构，每帧清空预测历史，使用原逐帧MSE/BCE、单配置seed 0、从头五轮，沿用官方train/逐epoch test/AUC选整轮。它用于判定原AP增益是否依赖历史，并决定保留或调整当前历史机制；重置约定及有效参数区别在实现前固定。本次仅验收和计划，未实现D4或启动作业。

**R4方向复盘 / 2026-11-19 / 可行性与底座选择讨论**：新增 `cscwd/idea_feasibility_review.md`，核对现有T5/D1–D3、模型路径、官方Gaze-LLE论文/代码及本地GazeSpot源稿。现有B底座EMA已达准确稳定54.2534%、gaze90/158、L2 0.098941，优于D3新机制对应时间/L2指标；当前失败不能归因于B太小。Gazelle指官方DINOv2 Gaze-LLE，关闭本地SASA/GGSF不是同一模型。建议先用现成B缓存做最小归因/可行性判定，机制有增量后再加公开Gazelle和L规模验证；候选重构为显式当前/对齐历史预测、train效用监督与分离状态记忆，可先固定In/Out收缩目标。理想实验用同底座增量与稳定—响应关系表达，未填假想数字。D4仍未实施，本次无新实验或底座切换；下一步仍先按既定D4确认原AP归因，重构前分析train历史候选的可用空间。

**R4架构说明 / 2026-11-19**：新增 `cscwd/method_architecture.md`，分开绘制已实现GRU门控双残差、讨论中的显式候选/效用选择、待实施D4无历史对照。按代码核对HV未显式warp过去heatmap、validity/refresh读写路径、D3仅训练loss、D2输出校准无历史反馈。下一步仍本地实现D4→用户GPU1五轮→回传决定结构，重构前分析train候选效用。本次仅图和说明，无新代码/结果/实验。

**R4-D4 / 2026-11-19 / 当前帧无历史对照本地完成**：固定 `cscwd/r4_current_frame_plan.md`（`r4_current_frame_v1`），新增 `CurrentFrameRefiner`、独立训练/比较入口、单模型顺序wrapper与四套测试。沿用原GRU层和初始化，每帧强制空状态且返回None，原逐帧MSE+0.1×BCE、seed 0、从头五轮；复用官方完整train/test与既有冻结B缓存，不加D3时序loss或D2校准。名义参数 **347,522**，其中 **74,112** 个权重元素对应结构性零历史输入，报告明确有效参数作用不同。

**R4-D4 / 本地产出与下一步**：**23项CPU合成检查通过**（模型7、训练/实际比较CLI6、小报告比较7、wrapper3），覆盖历史/顺序/chunk独立、活跃梯度、共享初始化、五轮完整test、test标签不改变训练权重、选中/末轮重载、全部同epoch参照与GPU1失败传播；记录 `cscwd/results/r4_current_frame_local_checks.json/.md`。唯一服务器命令在 `implementation_plan.md` **§15.1**，新输出位于 `experiments/cscwd_r4_current_frame_v1/current_frame/`。无新增真实VAT性能数字，未启动服务器作业；下一步用户手动运行并回传小报告，判断原AP收益是否依赖历史。

**R4-D4 / 2026-11-19 / 真实结果验收**：已pull `6eeb594`，235项本地小报告来源/代码/plan SHA、覆盖/预算/loss、整轮选择、同epoch差值及manifest/模板核对通过，记录 `cscwd/results/r4_current_frame_local_review.json/.md`。按原规则选epoch 5：**AUC0.940407 / L2 0.099362 / AP0.893067**，准确稳定10,050/19,796、状态错帧6,123、错误翻转742、gaze87/158、leave107/192。AP略高于原GRU/HV，leave与HV相同，说明原AP/出画收益无需历史即可重现；稳定比单帧少393窗、gaze少5个。GRU全部五个同epoch AUC仍略高，不能断言历史完全无用。服务器大文件未本地重算，原结果保持；当前不继续扩展旧GRU/HV。

**R4-D5 / 2026-11-19 / 训练集候选效用分析本地完成**：规格 `cscwd/r4_history_utility_plan.md`（`r4_history_utility_v1`），新增候选生成、独立计分和单条CLI及三套测试。只读完整官方train，固定current/current_grid/previous_raw/EMA0.75/视觉传输五候选；source→target patch质量传输temperature0.1，一跳raw历史，原InOut保持。候选全量落盘后再读GT，按稳定/变化组报四象限、补救/伤害、状态阻断和固定集合min-L2参考；GT/oracle不反馈历史，不读取test、不训练门。

**R4-D5 / 本地产出与下一步**：**28项合成检查通过**（生成12、计分9、实际CLI集成7），覆盖原生特征/搬运方向、EMA一致性、人物/缺口/未来因果、手算四象限/分组/选择参考、原指标/状态保持、无test文件运行、GT隔离与来源/坐标读回。记录 `cscwd/results/r4_history_utility_local_checks.json/.md`；无新增真实VAT D5数字，未启动服务器。下一步用户按实现计划 **§16.1** 用GPU 1手动运行，回传 `r4_history_utility.json/.md` 与provenance，再决定是否训练显式效用选择门。

**R4-D5 / 2026-11-20 / 真实结果验收**：已pull `a6928e6`，879项小报告源码/来源/plan SHA、覆盖/四象限/分组/算式、状态/leave一致、oracle/工件/模板核对通过，记录 `cscwd/results/r4_history_utility_local_review.json/.md`。主分母88,270帧、当前错22,831，固定集合GT参考可救3,126（13.6919%），固定状态阻断154；train oracle L2 0.081660不是部署成绩。transport救2,624/伤3,145，需要选择；current_grid无历史准确稳定47,676→51,364（+4.3015pp）、L2 0.094313→0.094019、gaze499→504/816。大数组未在本地重算；后续必须与grid及grid-EMA比较。

**R4-D6 / 2026-11-20 / 成本监督二选一门本地完成**：规格 `cscwd/r4_utility_selector_plan.md`（`r4_utility_selector_v1`），新增无标签观察/MLP、候选bank、整图评测、单条训练入口与四套测试。C=current_grid/H=一跳raw视觉传输，logit>0且有历史才选H，否则C，原InOut保持；监督仅train GT画内且有历史，用L2优势方向和绝对值成本加权BCE。六冻结token+17标量、hidden64、**296,129参数**，初始全选C，无GRU/选择结果反馈。新模块seed0从头五轮，先一次缓存train/test bank，逐epoch完整test/AUC选整轮；五固定对照包含grid-EMA。

**R4-D6 / 本地产出与下一步**：**38项合成检查通过**（核心15、bank7、评测8、实际训练CLI8），覆盖成本loss手算/梯度、标签/未来/人物/缺口隔离、双EMA、无损bank/整图输出、完整五轮/同次logits选择、重载在线与bank一致、test GT不改变五轮权重和强控制/AP/state保持。记录 `cscwd/results/r4_utility_selector_local_checks.json/.md`；无新增真实VAT D6数字，未启动服务器。下一步用户按实现计划 **§17.1** 用GPU1手动运行，回传selector报告/selection/provenance，再决定机制是否有额外时序收益。

**R4-D6 / 2026-11-20 / 真实结果验收**：已pull `4bbfdad`，1,485项小报告代码/plan/来源SHA、预算/双bank/覆盖、成本算式、选轮/工件/差值/事件配对核对通过，记录 `cscwd/results/r4_utility_selector_local_review.json/.md`。原规则选 **epoch1**，H选择 **0/31,978**；预测三文件/事件明细SHA、全部指标与current_grid相同。整行 **AUC0.942892 / L2 0.097854 / AP0.886801**，准确稳定 **11,226/19,796（56.7084%）**、gaze **91/158**。相对raw的+783稳定窗和AUC/L2提升属于当前重采样，不是历史门收益；grid-EMA稳定11,479窗、gaze90/158，仍是折中。

**R4-D6 / 结果解释与下一步**：在线训练weighted BCE **0.652967→0.596276**，每轮监督88,270帧、H偏好24,583、tie11,430。后四轮实际H选择188/1,156/2,908/2,087帧，但AUC/L2/准确稳定均差于grid，稳定分别少10/89/214/155窗；不能称未训练或五轮都不用历史。AP/状态错帧6,241/错误翻转592/leave91/192均保持。大数组/权重未本地重算，原报告不改。本次仅验收和计划；下一任务D7只冻结已有checkpoint、在train bank前向并分解成本收益/伤害、二候选oracle与常数先验，区分train决策无效或泛化不足，再决定新训练。

**方案重置 / 2026-11-20 / 用户指令**：停止继续D7诊断，回到“GazeSpot上加时序模块，提高VAT主指标”。新增 `cscwd/temporal_gazespot_plan.md`：使用人物条件Transformer输出的256通道空间特征和InOut token，接共享ConvLSTM64与零初始化残差，再用原双heads直接预测；原GazeSpot冻结，只训练新增模块，MSE+0.1BCE、单配置seed0/10轮、逐轮完整test/AUC选整轮。旧结果保留，本次只确定推荐方案，无新实验数字；下一任务实现任务特征缓存和完整训练入口。

## 已知阻塞
- Router 探索线（coverage-aware-router）：VAT 效果不明显，已暂停；详细状态见 `COVERAGE_AWARE_ROUTING_IDEA_PROGRESS.md`，与当前 CSCWD 时序方向无关。

## 文件约定
- 每个投稿目录各自放 LaTeX 源、图、实验结果；实验结果只写真实跑过的，没跑的对照不写进表格
- `RESULT_PROVENANCE.md` 记录每个数字来自哪次运行，写论文时以它为准
- 每个投稿目录独立放自己的新代码/论文：`<venue>/code/`（新代码）、`<venue>/latex/`（论文）、`<venue>/results/`（关键小结果）；旧 `gazelle` 包只 import 不改
- git 双向传递：本地改 code → push → 服务器 pull → nohup 手动跑 → 小结果（JSON/MD）push 回本地；大文件（.npy/.pt）不进 git，只在 RESULT_PROVENANCE 记路径+SHA

## 下一步
1. **实现GazeSpot + ConvLSTM**：按 `cscwd/temporal_gazespot_plan.md` 缓存原模型人物条件任务特征，新增共享空间时序适配器和直接双输出训练入口，完整train/test、10轮、物理GPU1。
2. 首轮直接看相对原GazeSpot的AUC/L2/AP，保持逐轮test及整轮选模；不继续D7诊断或候选选择机制。
3. 结果整行分轮次保留：原首轮GRU/HV均epoch 5，D3为GRU epoch 2/HV epoch 3，D2校准另列；不混用最好列，test选模协议保留。
