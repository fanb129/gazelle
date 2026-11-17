# Codex 提示词（CSCWD 联合时序修正 · 2026-11-17 当前版）

以下整段是给 codex 的完整提示词，直接复制到 codex 新会话开头即可：

---

你在 gazelle 仓库（~/src/gazelle，分支 codex/coverage-aware-router）实现 CSCWD 2027 投稿的时序方向。工作模式：你只在本机写代码；训练/评测在服务器（ssh fb@3090.lab）手动跑，代码经 git 传递、关键结果经 git 传回。用中文回复，代码/标识符/CLI/错误信息用英文。

先读这些文件：
- ~/src/gazelle/PROGRESS.md（状态）
- ~/src/gazelle/cscwd/implementation_plan.md（**实现规格，照它干活**）
- ~/src/gazelle/cscwd/temporal_joint_refinement.md（当前方案、训练目标、创新边界）
- ~/src/gazelle/cscwd/paper_blueprint.md（论文结构，理解贡献定位）
- ~/src/gazelle/cscwd/cscwd_2025_2026_related.md（venue 扫描结论，motivation 锚点）
- ~/src/gazelle/cscwd/literature_review.md、datasets_and_metrics.md（文献与指标约束）
- ~/src/gazelle/cscwd/results/t5_ema_ablation.md、t6_visualization.md（已跑先导结果）
- ~/src/gazelle/cscwd/code/data_prep/preprocess_vat_tracks.py（T1 已完成的 track 恢复）
- ~/src/gazelle/gazelle/model.py、gazelle/dataloader.py（单帧底座与数据加载）

当前状态：T1–T6 先导阶段已经完成。原规则 STAG 仅修正热图、In/Out 原值不变，已有稳定性收益与切换代价。新方案尚未实现/训练，不把已有缓存说成已具备视觉特征。

目标：按 implementation_plan.md §3.2 从首个未完成的 R 任务继续，当前为 R1；不重新从 T1 开始。R1 分组/诊断 → R2 冻结特征与普通联合头 → R3 历史有效性机制 → R4 训练/验证强对照 → R5 冻结配置 test → R6 论文证据。任务是“推进 idea”，不是“审计问题”。

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
6. 按 source-video 分组选择新增模块 train/validation；冻结 VAT checkpoint 本身此前已见过 VAT train，所有对照共享。test 已做先导分析，后续不再用来选择参数。GT 画内定位 mask 保持固定，AP 使用连续分数，不以改阈值声称 AP 提高。

风格要求（很重要）：
- 避免防御性编程：不为不可能发生的边界写大量 if/assert/try-except；不堆空检查和未决 TODO；先让主路径跑通，只在真正会出错处处理。
- 避免防御性写作：进展记录不写"风险/待解决问题"清单，不反复强调"可能不对/需验证"；直接写做了什么、结果、下一步。
- 一步一个任务，本地实现、服务器真实结果回传与验收完成后，再进入下一个 R 任务；不一次铺开多个半成品。

当前入口为 R1。用户本轮仅要求方案和计划落文件，尚未开始 R1；收到继续实施指令后，先完成 R1 的本地实现与必要检查，给出完整服务器命令，等待用户手动执行和回传结果，再验收并更新状态。不要自动执行旧 T5 train cache 命令或跳到模型训练。

---

（使用注意：若 codex 会话的实际 cwd 就是仓库根，`~/src/gazelle/...` 可简写为相对路径；路径以你实际环境为准。）
