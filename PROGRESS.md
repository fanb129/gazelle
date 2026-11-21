# 小论文 · 当前进度

分支：`codex/coverage-aware-router`。更新：2026-11-21。

| 投稿线 | 目录 | 状态 |
|---|---|---|
| ACM MM 2026（GazeSpot） | `rebuttal/` | 已拒，返修材料与结果保留（供 ICASSP 复用） |
| ICASSP 2027 | `icassp2027/` | 初稿保留，在审 |
| CSCWD 新主线 | `cscwd_router/` | T1 稀疏前向已实现，本机合成 9/9 通过；下一步服务器官方 checkpoint 验收 |
| CSCWD 旧工作 | `cscwd_v0/`、`cscwd_v1/` | 全量归档（v0=STAG 时序探索；v1=ConvLSTM 时序基线 + 失败分析，均已证伪"冻结底座 + 时序"） |

## 当前目标

在 GazeLLE（官方 DINOv2 单层，`gazelle/backbone_v0_dinov2.py` + `model_v0_dinov2.py`）上做 gaze-conditioned spatial routing：DINOv2 中层预测 gaze support，后段 blocks 只处理 support 内 token，多人 max-union 共享一次稀疏前向。**卖点是效率 + 多人可部署，不是精度刷点。**

唯一执行规格：`cscwd_router/implementation_plan.md`；codex 提示词：`cscwd_router/codex_prompt.md`。

## 工作约定

- 新代码只放 `cscwd_router/code/{data_prep,stag,scripts}/`，`gazelle/` 只 import 不改。
- 本机实现/必要检查；服务器 `fb@3090.lab` 由用户手动跑。仓库 `/home/fb/src/paper/gazelleV1`，Python `/home/fb/anaconda3/envs/py310/bin/python`。
- 复用 GazeLLE 官方 checkpoint（dense 阶段不重训 backbone）；纯 RGB、真实头框；routing 是单帧计算分配，无跨帧时序。
- 小 JSON/MD 经 Git 回传；大 NPY/PT 只留服务器，在 `cscwd_router/results/RESULT_PROVENANCE.md` 记路径和 SHA。只记录真实跑过的数字。

## 进展

**方向决策链（2026-11-21）**：时序路线（STAG/GRU/ConvLSTM）已证伪（oracle 上限 13.7%、selector 退化成不用历史、ConvLSTM loss 降 0.5%）→ 社会注视要新数据集（放弃）→ 回到 Router 效率线 → 底座的"DINOv3+多层 raw_concat"被判定为"又贵又弱"的历史包袱 → 定版：**在 GazeLLE（DINOv2 单层）上做 routing**。

**关键依据**：
- GazeLLE 官方 GazeFollow ViT-B AUC 0.956（公开最强轻量底座），backbone 权重在服务器 `checkpoints/dinov2_vitb14_pretrain.pth`。
- Router 机制复用 `gazelle/routing/`（DINOv3 版），适配 DINOv2 的差异见 `cscwd_router/implementation_plan.md` 第 2 节。
- novelty 定位：机制（token 剪枝）对标 DynamicViT 等，差异化在 gaze 条件化 + 多人 union。

**T1（2026-11-21）**：用户确认主线 decoder 为服务器 `/home/fb/src/paper/gazelleV1/checkpoints/gazelle_dinov2_vitb14.pt`。新增 `cscwd_router/code/stag/backbone_sparse_dinov2.py`：dense prefix → gather → sparse suffix → early-exit scatter，位置编码只在官方 token preparation 时加入一次，末尾 LayerNorm 的顺序与官方一致。K100 经过同一条稀疏实现路径；旧 `gazelle/` 未修改。

**T1 产出与数字**：本机 CPU 合成 **9/9 通过**（官方 DINOv2 源码、随机 24 维/12 blocks 小模型），448×448 的完整 32×32 网格 K100 与官方 `forward_features` 逐值一致，`max_abs_diff=0.0`、`mismatched_values=0`；另已验证稀疏 token 数、原位置、early-exit fill、梯度、checkpoint 读写和原版 decoder 多人查询。小结果在 `cscwd_router/results/t1_local_checks.json`，说明在 `results/LOCAL_CHECKS.md`。服务器入口 `code/scripts/check_sparse_dinov2.py` 与完整 GPU 0 nohup 命令已写入执行规格第 7 节；官方权重前向尚未在本机执行。

## 下一步

1. 用户提交/push T1 文件；服务器 pull 后手动运行 `cscwd_router/code/scripts/check_sparse_dinov2.py`，完成真实 ViT-B 权重的 B1/B2/B8、FP32/FP16 K100 等价验收，并回传结果 JSON 和 provenance。
2. 登记服务器 T1 验收通过后进入 T2，复用现有 support/max-union/Top-K 机制。当前尚未开始 T2。

历史详细进度见 `cscwd_v0/PROGRESS_SNAPSHOT.md`、`cscwd_v1/implementation_plan.md`。Router 旧探索线（DINOv3 版）状态仍见 `COVERAGE_AWARE_ROUTING_IDEA_PROGRESS.md`。
