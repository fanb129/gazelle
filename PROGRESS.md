# 小论文 · 当前进度

分支：`codex/coverage-aware-router`。更新：2026-11-22。

| 投稿线 | 目录 | 状态 |
|---|---|---|
| ACM MM 2026（GazeSpot） | `rebuttal/` | 已拒，返修材料与结果保留（供 ICASSP 复用） |
| ICASSP 2027 | `icassp2027/` | 初稿保留，在审 |
| CSCWD 新主线 | `cscwd_router/` | E0 双尺寸入口已实现，32/32 CPU 通过；GF/VAT × B/L 均 seed0、3+5 epochs，不重训 dense。GPU0 GF-B→L，GPU1 VAT-B→L，待用户手动运行 |
| CSCWD 旧工作 | `cscwd_v0/`、`cscwd_v1/` | 全量归档（v0=STAG 时序探索；v1=ConvLSTM 时序基线 + 失败分析，均已证伪"冻结底座 + 时序"） |

## 当前目标

在冻结的 DINOv3 ViT-B/16、ViT-L/16 多层表示上做 gaze-conditioned spatial routing：中层联合场景、头部外观和相对几何预测 support，后段只处理选中 patch，多人 max-union 共享 sparse suffix，未选位置回填以维持完整解码网格。**新贡献是人物条件化计算分配；DINOv3 和多层表示作为沿用的基础配置。** 本轮不新增 dense 训练，与官方 DINOv2 GazeLLE 比较完整系统，旧 matched dense 证据保留原协议；精度与速度是否同时超过由实测决定。

执行状态见 `cscwd_router/implementation_plan.md`；行文逻辑与三稿边界见 `cscwd_router/latex/PAPER_STORY.md`、`PAPER_REVIEW.md`；后续会话提示词为 `cscwd_router/codex_prompt.md`。旧 DINOv2 T1–T6 记录保留，不自动变成 DINOv3 的新命令。

当前执行清单为 `cscwd_router/EXPERIMENT_PLAN.md`，完整 GPU 0/1 命令见 `cscwd_router/RUN_DINOV3_TRAIN_TEST.md`。VAT 不依赖本轮 GF；两队列分别先跑 B 再接 L。所有新训练 base seed 0，四组各 support 3 + sparse decoder/inout 5 epochs。

## 工作约定

- 新代码只放 `cscwd_router/code/{data_prep,stag,scripts}/`，`gazelle/` 只 import 不改。
- 本机实现/必要检查；服务器 `fb@3090.lab` 由用户手动跑。仓库 `/home/fb/src/paper/gazelleV1`，Python `/home/fb/anaconda3/envs/py310/bin/python`。
- backbone 冻结；GazeLLE 官方 checkpoint 用于 DINOv2 基线和已有适配，DINOv3 分支分别登记自身 dense 初始化。纯 RGB、真实头框；routing 是单帧计算分配，无跨帧时序。
- 小 JSON/MD 经 Git 回传；大 NPY/PT 只留服务器，在 `cscwd_router/results/RESULT_PROVENANCE.md` 记路径和 SHA。只记录真实跑过的数字。
- 本机授权工作完成后由助手自动创建本地 Git commit，只 stage 本任务文件；push 与服务器运行不自动执行。

## 进展

**方向决策链（更新于 2026-11-22）**：时序路线归档 → 回到 Router 效率线 → 先完成官方 DINOv2 单层的 T1–T4 适配与实验入口 → 重新核对旧 DINOv3 正式结果和三份稿件，按用户当前意图将论文方法定为 **DINOv3 多层 + Router**。此前“DINOv3 多层又贵又弱”的概括不成立：旧 GF K50 精度接近官方 GazeLLE，批量推理相对自身 dense 有收益；跨模型速度与 VAT 适用性分别验证。

**关键依据**：
- 官方 GazeLLE 是冻结 DINOv2 + 轻量 decoder 的外部基线；少量可训练参数不等于省去编码器推理。它本身已支持多人共享场景编码。
- Router 机制复用 `gazelle/routing/` 的 DINOv3 实现；DINOv2 适配与原 K100 等价证据保留。
- novelty 定位：人物条件化 support、共同预算下的多人 sparse suffix、完整网格回填与实测权衡；不再将 ICASSP 已覆盖的多层表示和软门控作为新贡献。

**T1（2026-11-21）**：用户确认主线 decoder 为服务器 `/home/fb/src/paper/gazelleV1/checkpoints/gazelle_dinov2_vitb14.pt`。新增 `cscwd_router/code/stag/backbone_sparse_dinov2.py`：dense prefix → gather → sparse suffix → early-exit scatter，位置编码只在官方 token preparation 时加入一次，末尾 LayerNorm 的顺序与官方一致。K100 经过同一条稀疏实现路径；旧 `gazelle/` 未修改。

**T1 本机产出与数字**：本机 CPU 合成 **9/9 通过**（官方 DINOv2 源码、随机 24 维/12 blocks 小模型），448×448 的完整 32×32 网格 K100 与官方 `forward_features` 逐值一致，`max_abs_diff=0.0`、`mismatched_values=0`；另已验证稀疏 token 数、原位置、early-exit fill、梯度、checkpoint 读写和原版 decoder 多人查询。小结果在 `cscwd_router/results/t1_local_checks.json`，说明在 `results/LOCAL_CHECKS.md`；服务器复现入口和命令见执行规格第 7 节。

**T1 服务器验收（2026-11-21）**：已 pull 结果提交 `32224be` 并核对通过。实际 RTX 3090、GPU 1；B1/B2/B8 × FP32/FP16 六组 K100 backbone 与原版 decoder 全部逐值一致，最大差值 0.0、不一致元素 0；K50 suffix 实际为 513 tokens，回填逐值一致。报告 SHA 与 provenance 一致，五个相关源码 SHA 与本地一致。原始结果 `cscwd_router/results/t1_dinov2_checkpoint_checks.json`，审查记录 `results/T1_REVIEW.md`。T1 已完成。

**T2 实现与本机检查（2026-11-21）**：新增 `code/stag/router_dinov2.py`，原样复用现有 Router 和 coverage loss，将 scene/head/geometry 的输入接到 DINOv2 block-5 特征；backbone 冻结并保持 eval，support-only 只跑 prefix，完整路由按图 max-union 后只跑一次 sparse suffix。T1/T2 共 **20/20** 结构与 backward 检查通过（新增 11 项），K25/K50/K100 小模型检查通过；两图三人 K50 的 prefix/suffix batch 均为 2，后段每图实际为 8 patches + CLS。结果在 `results/t2_local_checks.json`；服务器 80-step support 拟合入口与 GPU 1 完整命令见执行规格第 9 节。本机未执行 support 拟合，loss 下降数值由服务器实际产生。

**T2 服务器验收（2026-11-21）**：已 pull 结果提交 `a08e944` 并审查通过。80 步 coverage loss `3.465735674 → 0.000011941`，soft/hard coverage `0.999988079 / 0.961666822`，scene/head/geometry 均有梯度，backbone 可训练参数和带梯度参数数均为 0；K25/K50/K100 结构检查全部通过。报告 SHA、八个源码 SHA 和与 T1 相同的 backbone SHA 均已核对。证据为 `results/t2_dinov2_support_checks.json`、`results/T2_REVIEW.md`。T2 已完成。

**T3 实现与本机检查（2026-11-21）**：新增原版单层 GazeLLE decoder 接口、固定 support 3ep → sparse decoder 5ep / matched dense 5ep pipeline、图像分组数据视图、标准指标和 compact checkpoint。两条 decoder 分支从同一官方权重开始，使用相同 epoch seed 和更新预算，支持从 last epoch 恢复；backbone 始终冻结。T1/T2/T3 共 **34/34 通过**（新增 14 项），随机小模型初始 dense/K100 与原版 GazeLLE 的最大差值 0.0、不一致元素 0。恢复调度用临时状态扰动做控制单测，本机未执行优化器更新；本轮小结果在 `results/t3_local_checks.json`。GPU 1 的训练与独立评测完整命令见执行规格第 11 节。

**T3 服务器验收（2026-11-21）**：已 pull 提交 `de1488d`。实际完成 3/5/5 更新、9/15/15 人次，两条 decoder 分支同起点/seed/预算；backbone 冻结。初始 dense/K100 与原版 GazeLLE、两个 checkpoint 往返最大差值均为 0.0；独立 sparse metrics 与训练报告完全一致。34/34 单测、十项 engineering checks、18 个源码 SHA、两个小报告 SHA 和七个 PT 的登记均核对通过。证据为 `results/t3_synthetic_seed3106/`、`results/T3_REVIEW.md`。T3 已完成。

**T4 实现与本机检查（2026-11-21）**：新增完整主入口 `code/scripts/run_main_experiment.py`、真实 RGB/GT 头框 benchmark 和 gate 汇总模块。固定 full-train/no-eval 后，对 official dense/K50/matched 各 test 一次，再执行 B1/B4/B8 正反序 dense/K100/K50 矩阵；恢复复用已核对的 test 和效率单元。primary gate 按原规格：相对 official dense AUC 掉 ≤0.002，B8 latency speedup ≥1.10x。本机共 **44/44 通过**，没有真实 T4 精度或速度数字；小结果 `results/t4_local_checks.json`，GPU 1 完整启动命令见执行规格第 13 节。

**W1 行文落盘（2026-11-22）**：完整读取 MM 9 页、rebuttal 1 页、ICASSP 5 页，核对 MM 图 2 的 PCA 内容。新增 `latex/PAPER_STORY.md` 与 `PAPER_REVIEW.md`，写入问题链、DINOv3/多层的引入方式、中英文 Introduction 工作稿、贡献边界与统一实验表。新增 `results/WRITING_EVIDENCE_20261122.json`，登记 3 份 PDF、候选图素材及 19 份旧实验 JSON 的 SHA/真实指标。MM 图 2 保留素材，当前不直接作为 Router 主动机图。未新增训练、推理或测速。

**已有证据的当前口径（2026-11-22）**：旧 DINOv3 GF K50/matched dense 的 AUC 为 `0.956462 / 0.957264`；旧 B1/B4/B8 相对自身 dense 的 latency speedup 为 `1.001525 / 1.060676 / 1.148545x`，准确率与效率使用不同 checkpoint，跨模型正式速度仍待统一测量。VAT 旧 holdout 的 K50/K75 未保持 dense 定位精度。此前会话只读核对过 DINOv2 T4 的服务器准确率，原始小报告尚未经 Git 回传，不能把本轮文档汇总视为 T4 完整验收。

**后续新训练协议（2026-11-22）**：按用户选择，全量官方 train、逐 epoch test、不另切 val；GF 按 Min L2、VAT 按 L2 选择单一 checkpoint，再报告其完整指标，登记 `test-selected`。旧 fixed-final 与 holdout 结果保留原标签。协议已写入文档，现有训练脚本尚未修改。

**W2 限时实验设计（2026-11-22）**：按用户要求完成 W1 本地提交 `2a3bcaf`（8 文件，未包含已有目录迁移）。新增 `EXPERIMENT_PLAN.md`，复用 GF 全量 support 和已有 VAT 初始化，不新增 dense/matched dense；计划新增 GF 5、VAT 3+5 个训练 epoch，其余核心对照只推理。核对 Mi et al. 公开指标 GF `0.961/0.094/0.038`、VAT `0.948/0.095/0.923` 与 ViT-L/16、离线 object-mask 设置，作为文献目标。本轮只完成计划与来源核对，没有新模型指标或测速。

**E0 双尺寸实现（2026-11-22）**：GF/VAT × B/L 的新训练统一 base seed=0。新增 7 个 Python 文件与独立 B→L 队列；四组从兼容旧 decoder 重学 support 3 epochs，再训 sparse decoder/inout 5 epochs，不继承 seed3106 Router、不重训 dense。GF 选 test Min L2、VAT 选 test L2，VAT 选中后自动逐帧 test。CPU 32/32 通过，含四种合成闭环、冻结梯度、官方指标一致性、原子恢复与错误 seed 拒绝；只读服务器核对四份初始化、两份骨干和四份数据标注。工程报告 `results/e0_dinov3_train_test_checks.json`，完整命令 `RUN_DINOV3_TRAIN_TEST.md`；未启动真实训练或 GPU 推理。

## 下一步

1. 本机 push、服务器 pull 后，用户按运行说明启动 GPU0 GF-B→L 和 GPU1 VAT-B→L；四组均 seed0、3+5 epochs，仅 train/test。
2. 回传四组小报告，对实际选中 checkpoint 做对应 B/L 官方 GazeLLE 推理、B1/B8 正反序效率与多人分析。
3. 旧 DINOv2 T4 按原协议单独收口；旧 matched dense / VAT holdout 不混入新协议。余量再补 scene-only Router。

历史详细进度见 `cscwd_v0/PROGRESS_SNAPSHOT.md`、`cscwd_v1/implementation_plan.md`。Router 旧探索线（DINOv3 版）状态仍见 `COVERAGE_AWARE_ROUTING_IDEA_PROGRESS.md`。
