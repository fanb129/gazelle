# CSCWD 2027 · 因果联合时序修正

> 投稿：CSCWD 2027（IEEE SMC，CCF-C），截稿 2026-11-30（AoE），6 页双栏 single-blind。
> 当前方案（2026-11-19）：冻结 GazeSpot ViT-B，以当前视觉证据判断历史有效性，训练轻量因果模块，联合修正 heatmap 与 In/Out。
> 状态：D3真实结果已验收，固定时序loss无一致收益。D4当前帧无历史对照本地完成，23项合成检查通过；用户按实现计划 **§15.1** 用GPU 1手动运行五轮，确定原AP收益是否依赖历史。

## 目录结构

```
cscwd/
├── code/            # 本次时序方向代码（新代码全放这里，旧 gazelle 只 import 不改）
│   ├── data_prep/   # T1 track 恢复
│   ├── stag/        # 既有基线、共用评测与后续联合时序模块
│   └── scripts/     # 缓存/推理/评测/可视化
├── latex/           # 论文（main.tex + refs.bib + fig/）
├── results/         # 关键小结果（JSON/MD），RESULT_PROVENANCE.md 记来源
└── *.md             # 方案/蓝图/规格/文献
```

## 文档入口

| 文档 | 作用 |
|---|---|
| [temporal_joint_refinement.md](temporal_joint_refinement.md) | 当前方案：motivation、方法假设、双输出训练、先导证据与创新边界 |
| [idea_feasibility_review.md](idea_feasibility_review.md) | 方向复盘：当前实例、ViT-B/L与Gazelle/GazeSpot选择、理想结果关系与最小路线 |
| [method_architecture.md](method_architecture.md) | 三张架构图：已实现方法、候选重构和D4对照；下一步任务 |
| [paper_blueprint.md](paper_blueprint.md) | 论文结构：motivation / method / baseline / 指标 |
| [implementation_plan.md](implementation_plan.md) | 权威执行规格：R1–R6 依赖/产出/验收；保留 T1–T6 结果与历史命令 |
| [r4_cache_diagnosis_plan.md](r4_cache_diagnosis_plan.md) | 已完成诊断：四种坐标/状态组合、准确稳定贡献与历史控制量 |
| [r4_state_calibration_plan.md](r4_state_calibration_plan.md) | 已完成R4-D2：train拟合全局状态偏置、固定原预测历史与坐标 |
| [r4_state_supervision_plan.md](r4_state_supervision_plan.md) | 已完成R4-D3：固定状态稳定/变化监督、跨chunk处理与同预算训练 |
| [r4_current_frame_plan.md](r4_current_frame_plan.md) | R4-D4：每帧清空历史、原目标/五轮预算与参数使用区别 |
| [training_file_guide.md](training_file_guide.md) | 当前训练文件说明：数据JSON、特征、权重、每轮结果与日志的具体路径 |
| [r4_official_train_test_plan.md](r4_official_train_test_plan.md) | 当前 R4：完整 train、逐 epoch test、AUC 选模 |
| [r4_compact_plan.md](r4_compact_plan.md) | 原 R4 validation 历史协议与选模规则 |
| [codex_prompt.md](codex_prompt.md) | codex 提示词 + 追加指令 |
| [cscwd_2025_2026_related.md](cscwd_2025_2026_related.md) | CSCWD 2025–2026 同方向论文扫描（venue 空白 + 锚点） |
| [literature_review.md](literature_review.md) | 既有逐篇解读及本轮视频机制/相邻工作补充入口 |
| [datasets_and_metrics.md](datasets_and_metrics.md) | 数据集标注、指标定义、比较约束 |
| [results/r1_manual_review.md](results/r1_manual_review.md) | R1 分组验收、开发基线与全部 12 例人工诊断 |
| [results/r2_probe_review.md](results/r2_probe_review.md) | R2 实测特征契约、精度与存储验收 |
| [results/r2_local_review.md](results/r2_local_review.md) | R2 全量缓存、真实 smoke 结果与验收结论 |
| [results/r3_mechanism_design.md](results/r3_mechanism_design.md) | R3 已实现机制、状态/诊断和参数量 |
| [results/r3_local_checks.md](results/r3_local_checks.md) | R3 本地模型、合成训练/验证与诊断检查 |
| [results/r4_local_checks.md](results/r4_local_checks.md) | 原 R4 validation 入口与合成 CLI 检查 |
| [results/r4_official_local_checks.md](results/r4_official_local_checks.md) | 官方 train/test 入口与36项本地检查 |
| [results/r4_official_local_review.md](results/r4_official_local_review.md) | 官方train/test首轮真实结果、epoch 5选择与下一步诊断 |
| [results/r4_cache_diagnosis_local_checks.md](results/r4_cache_diagnosis_local_checks.md) | 缓存诊断入口与23项本地合成检查 |
| [results/r4_cache_diagnosis_local_review.md](results/r4_cache_diagnosis_local_review.md) | 真实诊断结论：状态输出取舍、门控观测与下一项校准对照 |
| [results/r4_state_calibration_local_checks.md](results/r4_state_calibration_local_checks.md) | 校准入口与18项本地合成检查 |
| [results/r4_state_calibration_local_review.md](results/r4_state_calibration_local_review.md) | 真实校准验收、部分恢复与D3状态时序训练决策 |
| [results/r4_state_supervision_local_checks.md](results/r4_state_supervision_local_checks.md) | D3 loss/训练/比较/后台入口及29项本地合成检查 |
| [results/r4_state_supervision_local_review.md](results/r4_state_supervision_local_review.md) | D3真实验收、同epoch分析与D4当前帧对照决策 |
| [results/r4_current_frame_local_checks.md](results/r4_current_frame_local_checks.md) | D4模型/训练/比较/后台入口及23项本地合成检查 |

## 工作流

本地改 `cscwd/code/` → git push → 服务器 `fb@3090.lab` pull → nohup 手动跑 → 关键小结果（JSON/MD）push 回本地。大文件（.npy/.pt）不进 git，只在 RESULT_PROVENANCE.md 记路径 + SHA。

当前执行顺序：**官方train/test、D1–D3已验收 → 用户运行§15.1的D4当前帧无历史对照 → 决定历史机制取舍 → 论文证据**。D4本地入口已完成；原首轮两模型均epoch 5，D3为GRU epoch 2/HV epoch 3；§9–§14保留复现，已完成作业无需重跑。

## 论文索引（快速查，详见 literature_review.md）

| 编号 | 论文 | 输入→输出 |
|---|---|---|
| P01 | Detecting Attended Visual Targets in Video，CVPR 2020 | 视频帧+头部框→逐帧热图、画内概率 |
| P02 | Patch-Level Gaze Distribution Prediction，WACV 2023 | 图像/视频+头部信息→精细热图+粗分布 |
| P03 | MTGS，NeurIPS 2024 | 多帧+多人轨迹→每人热图+社会注视 |
| P04 | 时空 Transformer 注视检测概率模型，2026 | 当前+历史帧→头部与注视关系 |
| P05 | Gaze-LLE，CVPR 2025 | 单帧 RGB+头部框→热图 |
| P06 | Sharingan，CVPR 2024 | 单帧+多人头部框→每人热图 |
| P07 | ChildPlay，ICCV 2023 | 场景+头部+三维几何→热图 |
| P08 | ARGaze，2026 预印本 | 第一人称帧+历史热图→当前热图 |

"VAT" 指 VideoAttentionTarget。"AP" 须结合任务理解（In/Out AP、共同注意 AP、目标检测 AP 不是同一指标）。

## 选题边界（已消歧，写 motivation 以此为入口）

时间融合、outside token、人物 token 时序、自适应滤波和历史热图/模板均已有工作。当前研究假设是：第三人称人物条件下，**结合当前人物与场景证据，控制历史的保留、位置更新与释放，同时修正定位和状态**。实际贡献由同输入普通时序头、强滤波和机制消融确认。VAT 位移事件是代理标签；逐人稳定性不等同于已证明共同关注收益。
