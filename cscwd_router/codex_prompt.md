# 后续会话提示词：CSCWD Router

更新：2026-11-23。复制以下任务上下文即可，不要重新从 T1 开始，也不要重新询问已确认的 checkpoint 路径。

---

你在 `~/src/gazelle`、分支 `codex/coverage-aware-router` 推进 CSCWD 2027 的人物条件化空间路由。用中文回复；代码、标识符、CLI 和错误信息用英文。

先读：

L 来源更新先读 `results/L_INITIALIZER_DECISION.md` 与 `results/l_initializer_semantics_20261123.json`。用户怀疑 L 是旧单层；实际两组首末轮投影均 4096、无 SASA/GGSF，W&B 指向 DINOv3，但 GF 记录提交是 last-layer/448 与实物矛盾，VAT 记录源码的 `[4,11,17,23]` 也不同于当前 `[5,11,17,23]`。没有 runtime 快照，不能将 shape 兼容写成历史同配置认证，也不能直接判成单层。未改旧代码、未停止进程。GF-B→VAT-B 已完成并恢复定位，B 继续优先；L 暂保留 warm-start 补充，不新开两个 L dense，不切 GazeSpot 主线。用户尚未下达停止 L 的明确指令。

最新先读 `results/E2T_GF_TO_VAT_REVIEW.md` 与 `results/e2t_gfb_to_vatb_server_checks.json`。E2-T 完整 GF-B Router/decoder→VAT-B 已完成并经 `170ace8` push/pull：最终逐帧 AUC/L2/AP `0.940033/0.100131/0.883183`，相对旧 VAT-B L2 降低 9.77%，AP 下降 0.018555。实际 8 epochs K50，首轮固定 Router、后 7 轮 coverage/BCE 联合适配；按 test L2 选第 1 轮，选中 Router 与 GF 逐值一致，不能宣称联合 VAT Router 适配提升。52 个共享 tensors、20 个工件 SHA、8 轮真实更新和全部 31,978 条逐查询记录核验通过，重聚合指标差均 0。原命令 `RUN_GF_ROUTER_TO_VAT.md` 保留供复现，不重复运行。

下一任务优先 E2-H：从该 best 仅训练 post-transformer 的 `inout_head` MLP，冻结 backbone、Router、heatmap decoder 和 `inout_token`，保护定位、改善 AP。尚未实现或运行。现有 inout 参数组包含 `inout_token`，不能直接用 router/decoder LR=0 实现此控制；需独立参数范围和逐查询热图一致性检查，再给用户完整命令手动跑。正式效率等待空闲卡，原 GF-L 不擅停，旧 trainer 源码不改。

0. `cscwd_router/EXPERIMENT_PLAN.md` 与 `RUN_DINOV3_TRAIN_TEST.md`：E0 四组配置、GPU 0/1 命令、输出与恢复。最新 GF-B/VAT 续训结果先读 `results/E1_GFB_E2C_REVIEW.md` 与 `results/e1_gfb_e2c_server_checks.json`，优先于 W1/W2 的历史预算。
1. `cscwd_router/latex/PAPER_STORY.md`：当前 idea、行文与贡献边界。
2. `cscwd_router/latex/PAPER_REVIEW.md`：MM/rebuttal/ICASSP 的复核及图 2 处理。
3. `cscwd_router/implementation_plan.md` 与 `PROGRESS.md`：最新执行状态。原 T1–T6 对应 DINOv2 适配，T1–T3 已通过服务器验收；T4 官方 GF baseline accuracy 已回传，其余原始报告与完整验收待收口。
4. `cscwd_router/results/WRITING_EVIDENCE_20261122.json`：已有真实结果与来源。
5. `COVERAGE_AWARE_ROUTING_IDEA_PROGRESS.md`、`gazelle/routing/{backbone,router,model,losses}.py`：旧 DINOv3 Router 的机制与完整实验历史。

当前论文完整方法采用冻结 DINOv3 ViT-B/16、ViT-L/16 + 多层 raw-concat + Router。DINOv3 和多层表示沿用，不作为相对在审 ICASSP 的新贡献。新增主张是 support、共同预算下的多人 sparse suffix、回填与实测权衡。Router 在 B block 5 / L block 11 后决策（0-based），不依赖尚未计算的深层特征；头框是身份/几何，头部外观不是已标定的 gaze direction。GazeLLE 原生已共享多人场景编码，不能描述成每人重复 backbone。

E1-B / E2-C 历史结果见 `results/E1_GFB_E2C_REVIEW.md` 与 `results/e1_gfb_e2c_server_checks.json`。服务器 W&B 已核对 VAT B/L 都是 GF dense→VAT dense 8 epochs→本轮新 Router，不是随机初始化；原 decoder LR `1e-5`，本轮用了 `1e-4`。独立低 LR 续训已完成，各新增 8 epochs、5,664 次更新，但最低新 L2 未超过旧 best；最终指标不变。原始 dense/K100 三项精度差均 0。GF-B 第 5 轮 AUC/Avg/Min L2 为 `0.956236/0.101550/0.043420`，定位好于已回传的官方 B 报告，AUC 基本持平。结果提交 `8f2ae9c` 已 pull，13 份原始 JSON 与 29 个工件/标注路径已核验。原 E0 七个源码 SHA 不变；不要改旧目录的 epoch 数后 resume。完整 GF Router→VAT 迁移此后已完成，以最上方 E2-T 审查和 E2-H 下一步为准。

旧 DINOv3 GF K50/matched dense AUC 为 `0.956462 / 0.957264`；旧 B8 `1.148545x` 相对自身 dense，且测速与正式准确率 checkpoint 不同。VAT K50/K75 的 source-video holdout 定位损失保留。尚未证明相对官方 GazeLLE 的统一速度优势。

用户已选择后续新训练使用全量官方 train、逐 epoch test、不切 val；GF 以 Min L2、VAT 以 L2 选择单一 checkpoint，全部指标来自该 checkpoint，登记 `test-selected`。VAT 逐轮 test rate 6，选中后最终逐帧评测。旧 fixed-final / validation-selected 的结果保留原标签。新独立入口已实现，CPU 32/32 通过。VAT B/L 已完成并以 `0fc2844` 回传，按 test L2 选第 4/2 轮，逐帧 AUC/L2/AP 为 B `0.913548/0.110971/0.901738`、L `0.923034/0.097970/0.916599`；GF-B 已审查，GF-L 仍训练，效率待测。

用户限时决定：不新增 dense 或 matched dense 训练。最新明确四组 GF/VAT × B/L 全部新训练 base seed 0，不用 3106；四组均从已有兼容 raw-concat decoder 初始化，重学 support 3 epochs + sparse decoder/inout 5 epochs，不继承旧已训练 Router。旧 decoder 的训练 seed 不改写为 0。各阶段 epoch 子 seed 为 `0 + epoch`，decoder K100→K75→K50→K50→K50，每轮 test 固定 K50。官方 GazeLLE 只推理评测与测速。旧 matched dense 按旧协议单独保留，新主表定位为完整系统对比，不冒充同预算因果消融。

E0 已完成：单模型入口 `code/scripts/train_router_dinov3.py`、独立 B→L 队列 `code/scripts/run_router_dinov3_queue.py`；四份 decoder key/shape 和两份 backbone SHA 已服务器只读核对。VAT 初始化已有，因此不等本轮 GF。用户手动同时启动 GPU0 GF-B→GF-L、GPU1 VAT-B→VAT-L；完整命令见运行说明，小报告写 `results/testselected_k050_seed0/`。VAT 已回传且 14 个大工件 SHA、两组 31,978 条逐查询数据核验通过，重聚合精度差值均 0；不重复训练。两组 AUC 低于同规模 GazeLLE 文献参照，不能预写双优。先收齐并审查 GF，再执行 E3 统一测速与 E4 多人推理；可对已有 VAT checkpoint 补 K75/K100 推理。一次完成一个任务，不同时铺开所有实验。

工作硬约束：

- 本机只写代码和做必要工程检查，训练/评测由用户在 `ssh fb@3090.lab` 手动跑；不得自动启动任务或终止他人 GPU 进程。
- 新代码全部放 `cscwd_router/code/{data_prep,stag,scripts}/`；旧 `gazelle/` 只 import 不改；论文在 `cscwd_router/latex/`，关键结果在 `cscwd_router/results/`。
- 纯 RGB、真实头框、单帧，无深度、未来帧、检测框或时序模块。backbone 冻结。官方 DINOv2 decoder 是外部基线，不直接冒充 DINOv3 多层 decoder 的初始化。
- 服务器仓库 `/home/fb/src/paper/gazelleV1`，Python `/home/fb/anaconda3/envs/py310/bin/python`；每个新服务器入口给完整 nohup 命令（GPU、全部参数、日志、输出），由用户手动执行。
- 已确认官方 GF decoder：`/home/fb/src/paper/gazelleV1/checkpoints/gazelle_dinov2_vitb14.pt`；DINOv2 backbone：同目录 `dinov2_vitb14_pretrain.pth`。DINOv3 初始化使用旧报告中各自登记的路径和 SHA，不假定本机有权重或数据。
- 小 JSON/MD 经 Git 传递，PT/NPY 留服务器并登记路径/SHA；所有结果只写真实跑过的，旧协议和新协议不混写。
- 每完成任务将任务号、完成内容、文件与真实数字、下一步写入 implementation_plan 进展区，并同步 PROGRESS。
- 保留用户已有 `cscwd/` → `cscwd_v1/` 目录迁移，不全量 stage。
- 本机授权工作完成后由助手自动创建本地 Git commit，不再只让用户手动提交。只 stage 当前任务文件；本条不自动授权 push 或启动服务器训练。

W1 写作与证据文件已由助手提交到本地 Git，commit `2a3bcaf`；W2 计划 commit `8756c1e`。E0 实现及 CPU 检查见 `results/e0_dinov3_train_test_checks.json`，E2 的真实 VAT 精度已回传，尚无新的 GPU 效率数字。

---
