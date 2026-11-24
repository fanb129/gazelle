# 小论文 · 当前进度

分支：`codex/coverage-aware-router`。更新：2026-11-24。

| 投稿线 | 目录 | 状态 |
|---|---|---|
| ACM MM 2026（GazeSpot） | `rebuttal/` | 已拒，返修材料与结果保留（供 ICASSP 复用） |
| ICASSP 2027 | `icassp2027/` | 初稿保留，在审 |
| CSCWD 新主线 | `cscwd_router/` | GF/VAT×B/L精度与效率四组齐全，GF-L效率已补回；L的GF/VAT B8相对官方speedup为1.0722/1.0652×。维护前固定K50，集中B消融、真实多人和可视化 |
| CSCWD 旧工作 | `cscwd_v0/`、`cscwd_v1/` | 全量归档（v0=STAG 时序探索；v1=ConvLSTM 时序基线 + 失败分析，均已证伪"冻结底座 + 时序"） |

## 当前目标

在冻结的 DINOv3 ViT-B/16、ViT-L/16 多层表示上做 gaze-conditioned spatial routing：中层联合场景、头部外观和相对几何预测 support，后段只处理选中 patch，多人 max-union 共享 sparse suffix，未选位置回填以维持完整解码网格。**新贡献是人物条件化计算分配；DINOv3 和多层表示作为沿用的基础配置。** 2026-11-24按用户决定，将重点改为近期强方法的精度—成本比较，Mi ECCV2026为精度参照，TextGaze/OmniGF为可获取官方权重的推理对照；保留GazeLLE既有结果，不新增dense训练，旧matched dense保留原协议。

执行状态见 `cscwd_router/implementation_plan.md`；行文逻辑与三稿边界见 `cscwd_router/latex/PAPER_STORY.md`、`PAPER_REVIEW.md`；后续会话提示词为 `cscwd_router/codex_prompt.md`。旧 DINOv2 T1–T6 记录保留，不自动变成 DINOv3 的新命令。

当前执行清单为 `cscwd_router/PRE_SHUTDOWN_PLAN.md`，完整GPU0/1命令见 `cscwd_router/RUN_FINAL_ABLATIONS.md`、`RUN_FINAL_EVALUATION.md`。只增加三个B条件化分支，各自GF3+5→VAT8；0卡scene-only→scene-head，1卡scene-geometry；seed0。已有GF/VAT×B/L主模型保留，后续只补多人、同权重K100精度与可视化。旧 `RUN_DINOV3_TRAIN_TEST.md` 保留历史命令。

## 工作约定

- 新代码只放 `cscwd_router/code/{data_prep,stag,scripts}/`，`gazelle/` 只 import 不改。
- 本机实现/必要检查；服务器 `fb@3090.lab` 由用户手动跑。仓库 `/home/fb/src/paper/gazelleV1`，Python `/home/fb/anaconda3/envs/py310/bin/python`。
- backbone 冻结；GazeLLE 官方 checkpoint 用于 DINOv2 基线和已有适配，DINOv3 分支分别登记自身 dense 初始化。纯 RGB、真实头框；routing 是单帧计算分配，无跨帧时序。
- 小 JSON/MD 经 Git 回传；大 NPY/PT 只留服务器，在 `cscwd_router/results/RESULT_PROVENANCE.md` 记路径和 SHA。只记录真实跑过的数字。
- 本机授权工作完成后由助手自动创建本地 Git commit，只 stage 本任务文件；push 与服务器运行不自动执行。

## 进展

**W7-A多人精度完成（2026-11-24）**：GPU0 accuracy/resume任务正常结束，5,182共同帧N=1/2/3×三策略共9个FP32单元齐全；`7c37446` 已回传，源码/工件/查询身份和逐查询指标重聚合核对通过。N3 union AUC/L2/AP为0.929907/0.112543/0.849850，coverage0.959405；独立路径为0.929747/0.112434/0.848231，官方为0.919423/0.122274/0.866136。表5精度已补齐，详见 `results/final_multi_seed0/summary.md`。GPU0仍有其他用户进程，效率计时未运行；下一项独占后补计时，GPU1保留原训练。

**W6-D部署完成（2026-11-24）**：两卡命令的5个服务器CPU预检入口全部exit0，`1fed049` 报告经Git回本地，23项记录内源码SHA一致。多人5,182帧、可视化GF/VAT各32候选及多人16候选可读取，绘图库已就绪。训练目录固定 `c19ba93`，评测目录 `gazelleV1_final_evaluation` 固定 `1fed049`。没有执行新GPU任务；用户按两份 `RUN_FINAL_*` 文档手动启动，先0卡两分支串行/1卡geometry，再0卡多人、1卡K100→可视化。

**W6-C两卡与可视化入口（2026-11-24）**：全部91项本机CPU/回归通过（含新增12项C检查），真实导出/CPU绘图、表4自动核对汇总及两卡串行队列已实现。MM式六行图展示RGB→GT→support→实际token→官方/本文热图，另有同帧换人及max-union图；小案例相对引用可离线重画。用户仅授权GPU0/1：0卡scene-only→scene-head，1卡scene-geometry；后接多人、四组同权重K100和可视化。入口和完整命令 `cscwd_router/RUN_FINAL_ABLATIONS.md`、`RUN_FINAL_EVALUATION.md`，证据 `results/w6c_final_local_checks.json`。没有新增真实GPU结果，下一项服务器CPU预检与用户手动运行。

**W6-B评测入口（2026-11-24）**：真实VAT共同帧N=1/2/3的三策略质量/效率与四组同权重K100精度已实现。32项CPU/旧效率回归通过，含原指标一致、N1共享/独立逐值一致、真实native suffix的batch/token、15-cell派发恢复与原checkpoint不变。命令 `cscwd_router/RUN_FINAL_EVALUATION.md`，证据 `results/w6b_evaluation_local_checks.json`；没有新GPU评测。可视化完成后统一部署评测目录，不更新已部署训练目录。

**W6-A条件化训练入口（2026-11-24）**：三种B输入消融与各自GF3+5→VAT8队列已实现，47项本机CPU/回归通过；full原版输出逐值一致，关闭几何也移除绝对bbox坐标。source扩展纳入恢复fingerprint，旧gazelle包未改。证据 `cscwd_router/results/w6a_conditional_local_checks.json`；完整三卡命令 `cscwd_router/RUN_FINAL_ABLATIONS.md`。没有新GPU训练结果，接续部署/CPU预检与其余评测入口。

**W5维护前收口（2026-11-24）**：删除完整预算扫描，固定K50；原表5/6改为表4/5，新增训练仅三种B打分消融（scene-only、scene＋geometry、scene＋head appearance），三卡并行，各自GF3+5→VAT8，full与L沿用现有结果。同权重K100只补一次精度参考，其他GPU任务为真实多人、可视化导出与余量B8复测。CPU统计VAT test中5,182帧至少三个真实头框，可用同帧N=1/2/3嵌套查询；三卡本次均空闲。旧B日志每条纯训练约8.25小时（不含test/IO），详细资源预算与回传顺序见 `cscwd_router/PRE_SHUTDOWN_PLAN.md`，元数据证据 `results/w5_pre_shutdown_planning_evidence.json`。本轮只更新方案、未训练或推理；下一项实现入口并提供手动运行命令。

**R1 VAT-L综合复选（2026-11-24）**：用户决定同时比较AUC/L2/AP。8轮rate-6 test记录已整理至 `cscwd_router/results/VATL_EPOCH_SELECTION_REVIEW.md`，第3轮比原第1轮AP高1.1004个百分点，L2仅增加0.7152%，AUC下降0.000166；第5轮AP最高0.904636。Pareto为1/2/3/5/8，4/6/7被第3轮全面支配。新增纯CPU比较辅助及独立全量复评入口，4个选择规则检查通过；原第1轮rate-1与效率不覆盖。GPU0全参数命令 `cscwd_router/RUN_VATL_EPOCH_COMPARISON.md`，尚无新GPU复评结果，下一步先部署CPU预检再由用户手动运行。

**W4实验表落盘（2026-11-24）**：`cscwd_router/latex/CORE_DRAFT_ZH.md` 已按MM主表/勾选消融格式展开六张表，21行方法主表明确GF/VAT六指标及Params，含官方GazeLLE、近期方法、PaGE额外训练范围与Router-B/L。补表发现GF-L原有效率已完成，`5e64c45` 回传六报告与CPU检查，本地六报告SHA/25源码和服务器原生源码/权重/输入/12组trace/汇总核对通过；GF-L B1/B8相对官方 `0.9177/1.0722×`，B8相对K100 `1.3046×`。至此完整效率表填入四组24个variant×batch汇总项；另有速度/显存变化、8行预算、4行条件化消融、3行多人策略。K50行和覆盖率来自当前选中模型，未完成消融“—”。任务参数GF/VAT：B3.88/3.91M、L4.27/4.31M。GF-L审查 `results/E3_GFL_EFFICIENCY_REVIEW.md`；只核对原有结果，没有新GPU实验。后续按固定表格补预算、条件化和多人数据。

**W3中文核心稿（2026-11-24）**：新增 `cscwd_router/latex/CORE_DRAFT_ZH.md`，按用户构思压缩为动机、方法、贡献、实验矩阵与结论。核对Mi的头部外观＋眼位置→方向→FOV→多层融合流程，本文把人物条件用于编码器中途的计算选择，当前代码没有直接搬用其方向/FOV模块。DINOv3/多层仍为基础配置。稿件填入四组已审精度与三组正式效率；列明预算精度、同预算条件化消融、真实多人共享和近期方法比较应补的数据。现有结论保留VAT-L B8小幅快、B1慢及更高显存，未把“保证所有指标”写成已验证；没有新训练或GPU评测。

**E3-L VAT-L效率验收（2026-11-24）**：用户实际在GPU0运行完成，日志COMPLETE、四单元/六报告齐全，结果和CPU检查由隔离worktree提交 `afcc39e` push/pull。本机六报告SHA与25份按执行commit读取的源码、服务器27项源码hash/五来源权重/八张RGB、12组token trace、原始选模与汇总计算均核对通过。B1官方/K100/K50延迟 `33.4730/39.4368/38.7860ms`，B8 `147.5857/181.8345/138.5573ms`，相对官方speedup `0.8630/1.0652×`、相对K100 `1.0168/1.3123×`。B8延迟比官方低6.12%、吞吐57.3449 vs53.7989 FPS，峰值显存1824.21 vs1423.95MiB（高28.11%）；B1延迟高15.87%。精度仍为FP32 `0.948264/0.089676/0.891592`，效率为FP16完整forward。审查 `cscwd_router/results/E3_VATL_EFFICIENCY_REVIEW.md`，证据 `results/e3_vatl_efficiency_server_checks.json`；没有重跑GPU任务。下一项接近期强方法同机推理对照，保留B1和显存代价，不写成全面占优。

**E2-TL验收 / E3-L效率准备（2026-11-24）**：完整GF-L Router→VAT-L 8轮已由隔离worktree提交 `2eb4275` push/pull，CPU检查 `33e459a` 回传通过。seed0、仅train/test、首轮固定Router后7轮联合适配，按rate-6 test L2选第1轮，最终逐帧AUC/L2/AP **0.948264/0.089676/0.891592**，31,978查询（20,655画内）。52个GF共享张量、20个工件SHA、22项源码hash、选模/optimizer/冻结状态一致；导出AUC/L2/AP重聚合差值0。相对旧VAT-L定位改善（L2低8.47%），AP较旧L低2.50个百分点；当前L三项均优于完整GF→VAT-B。审查 `cscwd_router/results/E2TL_GFL_TO_VATL_REVIEW.md`，证据 `results/e2tl_gfl_to_vatl_server_checks.json`，文献表已同步。现有runner直接支持L，B1/B8正反序四请求CPU预检通过，GPU1完整命令 `cscwd_router/RUN_VATL_EFFICIENCY.md`。末次1卡另有作业占23,821MiB/89%，待空闲后用户手动启动；未产生新GPU效率数字。

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

**E2 VAT 服务器结果与审查（2026-11-23）**：GPU 1 的 B/L 队列已正常结束，实际各 support 3 + decoder 5 epochs，seed 0，仅 train/test。服务器结果提交 `0fc2844` 已 push 并 pull 到本机。B/L 分别按 rate-6 test L2 选第 4/2 轮，最终 rate-1 test 均为 31,978 个查询（20,655 画内）；AUC/L2/AP 为 B `0.913548/0.110971/0.901738`，L `0.923034/0.097970/0.916599`。14 个大工件 SHA、源码/标注/选模/冻结 state 均核对通过，两组逐查询指标重聚合差值均 0。审查 `results/E2_VAT_SEED0_REVIEW.md`，CPU 工件检查 `results/e2_vat_seed0_server_checks.json`；两组 AUC 低于同规模 GazeLLE 文献参照，尚未测速。

**E2-C 初始化追溯与低 LR 续训入口（2026-11-23）**：服务器精确 W&B config/output 确认 B/L 初始化均为 GF dense→VAT dense 8 epochs，再接本轮新 Router；没有遗漏 decoder 的 GF 迁移，但 Router 未在 GF 预训练。旧 VAT decoder LR 为 `1e-5`，本轮 `1e-4`；旧末轮 rate-6 AUC/L2 为 B `0.939097/0.103012`、L `0.945401/0.093716`，仅登记历史日志值。新增独立续训入口：原第 5 轮末状态、固定 K50、新低 LR 优化器、旧 best 继续参与选模、原始 dense/K100 推理核对，计划 B/L 各追加 8 epochs。39/39 CPU 通过，旧 E0 源码 SHA 全部不变；`RUN_VAT_CONTINUE.md` 含 GPU 1 B→L 完整命令。新续训尚未运行，无新精度数字。

**E1-B / E2-C 结果回传与审查（2026-11-23）**：结果提交 `8f2ae9c` 已从服务器 push 并 pull 本地，回传 GF-B 4 份、VAT 续训 B/L 8 份及官方 GF baseline 1 份 JSON。GF-B 完成 3+5 epochs，以 Min L2 选第 5 轮；AUC/Avg/Min L2 为 `0.956236/0.101550/0.043420`，相对已有官方 GazeLLE-B 报告差 `+0.000095/-0.002586/-0.001912`，定位误差相对降低 2.48%/4.22%，AUC 基本持平。VAT 各实际新增 8 epochs、5,664 次更新、44 个 decoder tensors 改变，但最低新 rate-6 L2 `0.111874/0.102485` 未超过旧 best `0.110842/0.099612`；最终逐帧指标保持原值。原始 VAT dense 与 routed K100 的 AUC/L2/AP 差均 0。29 个工件/标注路径（21 个大型 PT/PTH/gzip）SHA 核验通过，三组逐查询重聚合差值均 0；未新增 GPU forward。详见 `results/E1_GFB_E2C_REVIEW.md` 与 `results/e1_gfb_e2c_server_checks.json`。GF-L 继续训练，不再原样延长 VAT decoder。

**E2-T 完整 GF Router→VAT 实现（2026-11-23）**：按用户要求先提升 VAT，效率等空闲卡再测。新增 3 个 Python 文件，严格继承完整 GF Router/heatmap decoder，仅新建 5 个 in/out tensors。固定 K50、冻结 backbone，8 epochs（首轮固定 Router，后 7 轮 coverage/BCE 联合适配），LR `1e-4/1e-5/1e-3`、seed 0、完整 train/test；逐轮选 test L2 最低模型，最终 rate-1 test，保存每轮 checkpoint。51/51 CPU 通过，原 E0 与续训源码 SHA 不变；报告 `results/e2t_gf_to_vat_local_checks.json`，完整 GPU 1 命令 `RUN_GF_ROUTER_TO_VAT.md`。尚未启动真实训练，没有新 VAT 指标。

**L 来源与预算决定（2026-11-23）**：两组 L 的 epoch 0/末轮投影都是 `[256,4096,1,1]`，无 SASA/GGSF，W&B 配置指向 DINOv3；不能归为普通 1024 维单层 decoder。但 GF 记录 Git 源码为 last-layer/448，与实际结构矛盾；VAT 记录选层 `[4,11,17,23]` 不同于当前 `[5,11,17,23]`，没有 runtime 源码快照，精确历史语义未闭合。此前仅凭 keys/shape 认证同配置的表述已修正。证据 `results/l_initializer_semantics_20261123.json`，取舍 `results/L_INITIALIZER_DECISION.md`。GF-B→VAT-B 已启动，优先继续；L 暂保留 warm-start 实验，看当前 K75 test，不新增两个 dense，不直接替换 GazeSpot；本轮未停止任何进程。

**E2-T 结果回传与审查（2026-11-23）**：服务器结果提交 `170ace8` 已 push 并 pull 本地，5 份原始 JSON 与新 provenance block 已归档，rebase 保留本地 L 文档及用户目录迁移。完整 GF Router/decoder→VAT-B 实际完成 8 epochs，最终逐帧 AUC/L2/AP `0.940033/0.100131/0.883183`，相对旧 VAT-B 差 `+0.026485/-0.010840/-0.018555`，L2 相对降低 9.77%。按 test L2 选择第 1 轮，此时 Router 仍为 GF 权重；后 7 轮联合适配未超过它。52 个共享 tensors 继承、20 个工件（14 个 PT/PTH/gzip）SHA、8 轮真实更新/选模均通过；31,978 条逐查询身份/标注一致，三项指标重聚合差值均 0。详见 `results/E2T_GF_TO_VAT_REVIEW.md`、`results/e2t_gfb_to_vatb_server_checks.json`。本轮未新增模型 forward 或测速；下一项优先仅适配分类头，保护定位。

**E2-W 等待期安排（2026-11-23）**：用户接受当前 VAT-B，分类头适配后移。只读 GF-L 当前 history：已提交 2 个 decoder epochs，第 2 轮训练 K75、test K50，AUC/Avg/Min L2 `0.960007/0.094061/0.037721`；第 3 轮 K50 训练中，无最终 summary。快照 `results/gfl_interim_20261123.json` 保存服务器路径、原始小报告内容/SHA 与日志尾部。等待期优先准备 B 的 K25/K50/K75/K100 预算扫描、scene-only 条件化对照及真实多人 union 分析；训练/精度不要求独占卡，正式测速等空闲卡。本轮未启动新实验。

**E1-L 中断收口与审查（2026-11-23）**：用户决定不补 GF-L 最后一轮。服务器结果 `459651a` 已 push 并 pull，实际 support 3 + decoder 4 epochs，第 5 轮途中关机，按 test Min L2 仍选第 2 轮；AUC/Avg/Min L2 `0.960007/0.094061/0.037721`。从 last 内嵌 best 导出 52 个 tensors 逐值一致的 `best_test.pt`，原 last SHA 不变；原 manifest 3+5 不改，收口报告明确 `completed=false`、实际 3+4，无独立复评或逐查询导出。6 个 PT/PTH、2 份标注、18 个源码及原生树 SHA 和实际 optimizer steps（85,095/14,184）核对通过。报告 `results/E1L_GF_INTERRUPTED_REVIEW.md`、检查 `results/e1l_interrupted_server_checks.json`。当前精度可继续效率与 VAT-L 迁移，不先新训 GF-L dense；未启动新 GPU 任务。

**E3 效率实现（2026-11-24）**：新增 DINOv3 对官方 GazeLLE 的单元/队列入口，共享旧稳定计时器；每单元新进程，official/K100/K50，B1/B8 正反序，FP16、448/512 原配置、真实 train RGB/GT 单查询，50 warmup、200×5 latency、500×5 throughput。记录逐 block tokens、实际速度/FPS/显存、GPU UUID 和源码/权重 SHA；恢复核对全部 request 后复用。效率新增 16 项 CPU 检查通过，未执行 GPU 测速。

**E2-TL 迁移接入（2026-11-24）**：共享 source reader 显式接入 GF-L 用户收口状态，核对原 last/history 与导出 best 逐值一致，保留 completed=false 和实际 3+4。VAT-L 使用原 B 配方 8 epochs K50，batch4×累积8、seed0，首轮固定 Router、后7轮联合适配，逐轮 test L2 选模、最后逐帧 test。新增合成闭环通过，含旧训练/续训/迁移回归合计 **68/68**；旧 gazelle/ 未改。完整 GPU0/1/2 命令 `cscwd_router/RUN_EFFICIENCY_VATL.md`，工程报告 `results/e3_efficiency_vatl_local_checks.json`。本轮仅本机实现/CPU 检查和服务器只读权重/GPU 核对，未启动真实训练或测速。

**E3-F1 VAT 效率入口修复（2026-11-24）**：用户报告 `KeyError: 'heads'`，原因是 VAT 顶层序列需展开 frames，旧效率采样只覆盖 GF 扁平结构。已按 dataset 复用官方 loader，补嵌套采样与队列/恢复测试，70/70 CPU 通过。修复 `29c6607` 已 push，服务器独立 worktree `/home/fb/src/paper/gazelleV1_efficiency_vat_fix` 就绪，原目录 HEAD4162da7/核心源码未变，GPU0/2 继续原任务。真实 VAT 475 序列/58,507 帧、8 个 RGB/GT 框及 B1/B8 正反序四个请求预检通过；未启动 GPU1 计时。完整重启命令 `RUN_VAT_EFFICIENCY_FIX.md`，证据 `results/e3_vat_schema_fix_checks.json`；新结果写原仓库 `_vatfix` 目录。

**E3-B 两组 B 正式效率审查（2026-11-24）**：已完成 GF-B/VAT-B 各 B1/B8 正反序，12 份 JSON/MD 经隔离 worktree 提交 `85210b5` push/pull，本机原始报告 SHA 与 50 个版本化源码检查通过。8 个单元/24 组 token trace、权重/输入/源码/GPU 和汇总核验一致，原目录 HEAD/代码保持不变。相对官方 GazeLLE 的 latency speedup 为 GF B1/B8 `0.7822/0.9918×`，VAT `0.8255/1.0251×`；B1 延迟增加27.84%/21.13%，B8 GF基本持平、VAT小幅快2.5%。相对同一路由 K100，B8为1.2519/1.2559×；B8峰值 allocated memory约857MiB，官方约538MiB，高59.5%。报告 `results/E3_B_EFFICIENCY_REVIEW.md`、检查 `results/e3_b_efficiency_server_checks.json`；本轮未新增 GPU forward 或训练。当前不支持 B 完整系统明显更快/更省显存的表述。

**E3-P0 推理与效率说明（2026-11-24）**：完整推理流程、算术缩减与硬件耗时的区别、历史B1阶段证据和优化顺序已写入 `cscwd_router/results/E3_INFERENCE_AND_OPTIMIZATION.md`，来源/SHA与推导在 `results/e3_efficiency_explanation_evidence.json`。GF-B B8的K100额外路径差额14.00ms与K50节省13.56ms相抵，解释了相对官方基本持平；旧B1中Router约占插桩时间13–14%，suffix缩短未兑现等比例时间收益。优先当前checkpoint分段profile和同权重native控制，再做保持语义的融合/复制/全一门快路径；本轮仅源码与既有报告分析，没有新GPU测量或优化结果。

**SOTA-1 近期方法与目标调整（2026-11-24）**：核对8行公开指标与4项单独标注线索，Mi/TextGaze/OmniGF/GazeVLM原表已渲染复核。GF-L三项接近Mi，均优于TextGaze-L公开值；当前B的GF/VAT六项优于TextGaze-B。Mi未找到官方代码/权重，TextGaze四个PT发布资产与OmniGF GF/VAT safetensors已由官方API确认；完整GF-L→VAT-L最终summary本轮只读未见，不填结果。表与公开链接 `cscwd_router/results/SOTA_LANDSCAPE_20261124.md`，来源/版本/SHA `sota_landscape_20261124.json`。主目标改为近期强方法的精度—成本，保留GazeLLE参照与ICASSP贡献边界；本轮没有训练、权重下载或GPU评测。

**SOTA-2 架构与框架关系（2026-11-24）**：`SOTA_LANDSCAPE_20261124.md` 第7–9节已补16项架构（含官方/本方法），9仓库29份固定commit源码的SHA与判据在 `architecture_lineage_20261124.json`。明确DINOv3多层配置的外部参照是Mi；PaGE论文明确从GazeLLE发展且有公开B/H+任务权重，但两分支各取最后一层。TextGaze、GazeHRNet、GazeLoom沿用相似解码骨架而非四层DINO融合。补入上一轮遗漏的PaGE强结果及联合训练/SFT/蒸馏范围；不预写我们胜出，未改当前配方或运行实验。

## 下一步

1. 按 `cscwd_router/PRE_SHUTDOWN_PLAN.md` 实现并启动三个B条件化消融，各自GF→VAT；主模型GF/VAT×B/L结果保持已有来源。随后完成真实多人、K100单次精度和可视化导出，维护前回传小结果。
2. 有余量优先独立复测GF-L/VAT-L的B8速度；绘图和排版在本机继续。近期强方法推理对照、阶段profile和等价优化后移，完整预算扫描取消。
3. 旧DINOv2 T4、matched dense与VAT holdout保留各自协议；本轮不扩展dense训练、L消融或其他长训练。

历史详细进度见 `cscwd_v0/PROGRESS_SNAPSHOT.md`、`cscwd_v1/implementation_plan.md`。Router 旧探索线（DINOv3 版）状态仍见 `COVERAGE_AWARE_ROUTING_IDEA_PROGRESS.md`。
