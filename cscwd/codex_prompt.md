# Codex 提示词（CSCWD 历史效用选择 · 2026-11-20 当前版）

以下整段是给 codex 的完整提示词，直接复制到 codex 新会话开头即可：

**最新指令优先（2026-11-20）**：用户要求停止继续分析/候选选择，取消D7。下一任务直接按 `temporal_gazespot_plan.md` 实现GazeSpot人物条件任务特征后的共享ConvLSTM64、原双heads输出、MSE+0.1BCE、单配置10轮/逐epoch test/AUC选整轮。原GazeSpot冻结，保持train/test，主目标AUC/L2/AP；下文D阶段为历史，不再据此执行冻结诊断。

---

你在 gazelle 仓库（~/src/gazelle，分支 codex/coverage-aware-router）实现 CSCWD 2027 投稿的时序方向。工作模式：你只在本机写代码；训练/评测在服务器（ssh fb@3090.lab）手动跑，代码经 git 传递、关键结果经 git 传回。用中文回复，代码/标识符/CLI/错误信息用英文。

先读这些文件：
- ~/src/gazelle/PROGRESS.md（状态）
- ~/src/gazelle/cscwd/implementation_plan.md（**实现规格，照它干活**）
- ~/src/gazelle/cscwd/temporal_gazespot_plan.md（最新用户重置：取消D7，直接实现GazeSpot+ConvLSTM双输出优化VAT指标）
- ~/src/gazelle/cscwd/r4_official_train_test_plan.md（当前协议：完整 train、每 epoch test、AUC 选模）
- ~/src/gazelle/cscwd/r4_compact_plan.md（原 validation 实验存档）
- ~/src/gazelle/cscwd/temporal_joint_refinement.md（当前方案、训练目标、创新边界）
- ~/src/gazelle/cscwd/paper_blueprint.md（论文结构，理解贡献定位）
- ~/src/gazelle/cscwd/cscwd_2025_2026_related.md（venue 扫描结论，motivation 锚点）
- ~/src/gazelle/cscwd/literature_review.md、datasets_and_metrics.md（文献与指标约束）
- ~/src/gazelle/cscwd/results/t5_ema_ablation.md、t6_visualization.md（已跑先导结果）
- ~/src/gazelle/cscwd/results/r1_manual_review.md、r1_local_verification.json（已验收开发划分、状态指标与人工诊断）
- ~/src/gazelle/cscwd/results/r2_local_review.md、r2_local_review.json（已验收冻结缓存、普通联合头 smoke 与真实指标折中）
- ~/src/gazelle/cscwd/results/r3_mechanism_design.md、r3_local_checks.json（R3 已实现机制与本地检查）
- ~/src/gazelle/cscwd/results/r4_local_review.md（原validation首轮结果验收）
- ~/src/gazelle/cscwd/results/r4_official_local_checks.md（官方train/test入口检查）
- ~/src/gazelle/cscwd/results/r4_official_local_review.md（真实首轮验收、选epoch 5）
- ~/src/gazelle/cscwd/r4_cache_diagnosis_plan.md、results/r4_cache_diagnosis_local_checks.md（已完成缓存诊断方案与检查）
- ~/src/gazelle/cscwd/results/r4_cache_diagnosis_local_review.md（真实诊断验收与当前R4-D2校准对照决策）
- ~/src/gazelle/cscwd/results/r4_state_calibration_local_review.md（D2真实校准验收与当前D3状态时序训练决策）
- ~/src/gazelle/cscwd/r4_state_supervision_plan.md（D3固定loss、跨chunk监督与运行规格）
- ~/src/gazelle/cscwd/results/r4_state_supervision_local_review.md（D3真实验收、同epoch对比和当前D4无历史对照决策）
- ~/src/gazelle/cscwd/results/r4_current_frame_local_review.md（D4真实验收，AP/出画收益不需要历史即可重现）
- ~/src/gazelle/cscwd/results/r4_history_utility_local_review.md（D5真实候选补救/伤害与grid强控制验收）
- ~/src/gazelle/cscwd/r4_utility_selector_plan.md、results/r4_utility_selector_local_review.md（D6已真实验收，epoch1等同grid；下一任务冻结train诊断见进展记录）
- ~/src/gazelle/cscwd/idea_feasibility_review.md（方向复盘、底座选择与理想实验关系；候选重构尚未实现）
- ~/src/gazelle/cscwd/code/data_prep/preprocess_vat_tracks.py（T1 已完成的 track 恢复）
- ~/src/gazelle/gazelle/model.py、gazelle/dataloader.py（单帧底座与数据加载）

当前状态：T1–T6 和 R1/R2 已验收。R2 全量特征缓存与普通联合头固定一个 epoch smoke 结果 85d9050 已回传，契约为末层 768×32×32、FP16，58,507 张图像。实际训练/验证 98,362/34,201 帧，347,522 个新增参数，两个头更新。标准指标基本持平，状态错误 11.13%→10.53%，准确稳定 60.19%→59.46%，state enter 179/189→175/189，leave 93/169→97/169；该配置未达到最终选模条件。记录见 results/r2_local_review.json/.md。原规则 STAG 和普通 GRU 保留为对照，R3 本地完成情况见下一段。

R3 已本地完成：HistoryValidityRefiner 在原始冻结空间匹配过去预测区域模板，validity 控制历史读入、refresh 控制模板写入；step兼容R2，诊断可按原全局索引落盘。372,677 参数；24项新增检查和18项回归通过。validation和官方train/test均已真实训练验收，当前进度以下段为准。

目标：按 implementation_plan.md §3.2 继续当前官方 train/test R4；不重做 T1–T6、R1–R3。用户已明确取消后续 validation，完整 train 训练、每 epoch 完整 test，按 AUC 更高、L2 更低、AP 更高、较早 epoch 选一整轮；如实记录 test 参与 checkpoint 选择。任务是“推进 idea”，不是“审计问题”。

主方案：复用既定 ViT-B VAT checkpoint，冻结完整 GazeSpot 和原 decoder，仅训练新增因果模块，以当前人物/场景证据控制历史保留、位置更新与释放，联合输出修正 heatmap 和 In/Out 连续分数。四参数校准、自适应滤波和普通因果头保留为对照；不把门控、记忆、模板或因果本身宣称为新贡献。

目录与工作流（硬约束）：
1. 本次所有新代码放 ~/src/gazelle/cscwd/code/（data_prep / stag / scripts 三个子目录），旧 gazelle 包只 import 不改。论文 latex 放 cscwd/latex/，关键结果放 cscwd/results/。
2. 代码传递：你只在本机写代码，写完后明确告诉用户要 git commit/push 哪些文件；服务器 pull 后由用户手动跑。
3. 每个需要服务器运行的脚本，给出完整 nohup 命令（GPU 号、python 路径、全部参数、日志路径、输出路径），格式参考 implementation_plan.md 第 4 节模板。命令由用户手动执行；你不假设本机有数据、不假设已经跑过。
4. 关键实验结果（小 JSON/MD）写 cscwd/results/，由用户经 git 传回本地；大文件（.npy 缓存、.pt checkpoint）不进 git，只在 RESULT_PROVENANCE.md 记路径。

技术硬约束（不可违反）：
1. 时序严格因果，仅当前及过去 RGB/特征/模型预测，绝不用未来帧；训练历史也不用 GT 点或状态填充。
2. 复用 GazeSpot 的 VAT checkpoint，不重训 backbone。
3. 纯 RGB、真实头部框，不进深度/未来帧/检测框设置。
4. 所有数字只写真实跑过的；不编数字、不编对照实验。
5. 每完成一个任务，把"任务号 / 完成内容 / 产出文件与数字 / 下一步"写回 implementation_plan.md 的进展记录区，并同步 PROGRESS.md。
6. 当前协议为 `r4_official_train_test_v1`：官方 train 全部训练、test 每 epoch 全部评测并选 epoch，不再保留 validation。旧 R1/原 R4 validation 报告保持原标签。所有对照共享底座、数据与预算；GT 画内定位 mask 保持固定，AP 使用连续分数。

风格要求（很重要）：
- 避免防御性编程：不为不可能发生的边界写大量 if/assert/try-except；不堆空检查和未决 TODO；先让主路径跑通，只在真正会出错处处理。
- 避免防御性写作：进展记录不写"风险/待解决问题"清单，不反复强调"可能不对/需验证"；直接写做了什么、结果、下一步。
- 一步一个任务，本地实现及相应验收后再进入下一个 R 任务；需要真实作业的任务完成服务器回传验收后再推进，不一次铺开多个半成品。R3 是本地结构/检查任务，不追加服务器 smoke。

官方train/test首轮、D1–D6均已验收。D6结果 `4bbfdad` 经1,485项小报告核对通过，按原规则选epoch1，历史选择0/31,978，预测SHA/全部指标等同current_grid：AUC0.942892/L2 0.097854/AP0.886801、稳定11,226/19,796、gaze91/158。收益属于空间处理，不能称历史门有效。在线BCE0.652967→0.596276；后四轮实际采用188/1,156/2,908/2,087帧历史，但AUC/L2/稳定均比grid差。grid-EMA稳定11,479窗、gaze90/158，原InOut/state保持。下一任务D7复用train bank和五个checkpoint，冻结前向并分解C=grid/H=transport二候选效用、常数cost先验、模型收益/伤害/regret和固定GT组；不新增训练、不扫描test阈值、不改变epoch1。D5 raw多候选oracle不代表该二候选空间，在线训练loss不代表冻结hard决策收益。当前仅完成D6验收/该计划，§11–§17保留复现，原源码与机器报告不改写。

---

（使用注意：若 codex 会话的实际 cwd 就是仓库根，`~/src/gazelle/...` 可简写为相对路径；路径以你实际环境为准。）
