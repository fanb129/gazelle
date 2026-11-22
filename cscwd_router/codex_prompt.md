# 后续会话提示词：CSCWD Router

更新：2026-11-22。复制以下任务上下文即可，不要重新从 T1 开始，也不要重新询问已确认的 checkpoint 路径。

---

你在 `~/src/gazelle`、分支 `codex/coverage-aware-router` 推进 CSCWD 2027 的人物条件化空间路由。用中文回复；代码、标识符、CLI 和错误信息用英文。

先读：

1. `cscwd_router/latex/PAPER_STORY.md`：当前 idea、行文与贡献边界。
2. `cscwd_router/latex/PAPER_REVIEW.md`：MM/rebuttal/ICASSP 的复核及图 2 处理。
3. `cscwd_router/implementation_plan.md` 与 `PROGRESS.md`：最新执行状态。原 T1–T6 对应 DINOv2 适配，T1–T3 已通过服务器验收，T4 原始小报告待 Git 回传收口。
4. `cscwd_router/results/WRITING_EVIDENCE_20261122.json`：已有真实结果与来源。
5. `COVERAGE_AWARE_ROUTING_IDEA_PROGRESS.md`、`gazelle/routing/{backbone,router,model,losses}.py`：旧 DINOv3 Router 的机制与完整实验历史。

当前论文完整方法采用冻结 DINOv3 ViT-B/16 + 多层 raw-concat + Router。DINOv3 和多层表示沿用，不作为相对在审 ICASSP 的新贡献。新增主张是 support、共同预算下的多人 sparse suffix、回填与实测权衡。Router 在 block 5 后决策，不依赖尚未计算的深层特征；头框是身份/几何，头部外观不是已标定的 gaze direction。GazeLLE 原生已共享多人场景编码，不能描述成每人重复 backbone。

旧 DINOv3 GF K50/matched dense AUC 为 `0.956462 / 0.957264`；旧 B8 `1.148545x` 相对自身 dense，且测速与正式准确率 checkpoint 不同。VAT K50/K75 的 source-video holdout 定位损失保留。尚未证明相对官方 GazeLLE 的统一速度优势。

用户已选择后续新训练使用全量官方 train、逐 epoch test、不切 val；GF 以 Min L2、VAT 以 L2 选择单一 checkpoint，全部指标来自该 checkpoint，登记 `test-selected`，VAT 最终逐帧评测。旧 fixed-final / validation-selected 的结果保留原标签。**现有脚本尚未改为新协议，不得宣称已实现或已跑。**

下一项先准备官方 GazeLLE / DINOv3 多层 matched dense / Router 的统一测速入口，闭合同一最终 checkpoint 的精度与效率；之后实现新训练协议。一次完成一个任务，不同时铺开多人和消融。

工作硬约束：

- 本机只写代码和做必要工程检查，训练/评测由用户在 `ssh fb@3090.lab` 手动跑；不得自动启动任务或终止他人 GPU 进程。
- 新代码全部放 `cscwd_router/code/{data_prep,stag,scripts}/`；旧 `gazelle/` 只 import 不改；论文在 `cscwd_router/latex/`，关键结果在 `cscwd_router/results/`。
- 纯 RGB、真实头框、单帧，无深度、未来帧、检测框或时序模块。backbone 冻结。官方 DINOv2 decoder 是外部基线，不直接冒充 DINOv3 多层 decoder 的初始化。
- 服务器仓库 `/home/fb/src/paper/gazelleV1`，Python `/home/fb/anaconda3/envs/py310/bin/python`；每个新服务器入口给完整 nohup 命令（GPU、全部参数、日志、输出），由用户手动执行。
- 已确认官方 GF decoder：`/home/fb/src/paper/gazelleV1/checkpoints/gazelle_dinov2_vitb14.pt`；DINOv2 backbone：同目录 `dinov2_vitb14_pretrain.pth`。DINOv3 初始化使用旧报告中各自登记的路径和 SHA，不假定本机有权重或数据。
- 小 JSON/MD 经 Git 传递，PT/NPY 留服务器并登记路径/SHA；所有结果只写真实跑过的，旧协议和新协议不混写。
- 每完成任务将任务号、完成内容、文件与真实数字、下一步写入 implementation_plan 进展区，并同步 PROGRESS。
- 保留用户已有 `cscwd/` → `cscwd_v1/` 目录迁移，不全量 stage。

本轮 W1 已完成写作文件与证据摘录，未改训练代码、未启动服务器实验。

---
