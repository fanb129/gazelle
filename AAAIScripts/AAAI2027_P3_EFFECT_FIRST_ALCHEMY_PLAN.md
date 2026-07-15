# AAAI 2027 P3：效果优先炼丹计划

> 状态：`PLAN_FROZEN / CODE_READY / P3A_COMPLETE / P3B_HOLD`
> 日期：2026-07-15  
> 服务器 Python：`/home/fb/anaconda3/envs/py310/bin/python`  
> 服务器仓库：`/home/fb/src/paper/gazelleV1`  
> GPU：单卡 RTX 3090  
> 当前决定：停止 P2/P2.1，不做人工标注；P3A 未选出优于 R0 的新候选，暂不运行 P3B，先按第 9 节在 R1 residual refinement 家族内做单因素下一轮。

## 1. 本轮目标

本轮不再为预设 idea 做机制验证，只回答一个工程问题：在保留 DINOv3 多层特征和 Gazelle decoder 的前提下，能否用轻量的新融合分支或损失函数稳定提高 GazeFollow/VAT 指标。

搜索阶段允许：

- 单 seed；
- 从现有 SASA+GGSF checkpoint warm-start；
- 短训练筛选；
- 只在 train-derived validation split 上选择候选；
- 先追求效果，胜出后再讨论命名与故事。

投稿结果不允许直接使用搜索阶段的 test-set 最优 checkpoint。候选确认有效后，必须按第 8 节完成 GazeFollow → VAT 的干净训练链路。

## 2. 已知参考结果

以下只作为效果参考，不把 ViT-B 与 ViT-L 混为同一条比较线：

| 方法 | GazeFollow AUC ↑ | Min L2 ↓ | Avg L2 ↓ | VAT AUC ↑ | VAT L2 ↓ | VAT Inout AP ↑ |
|---|---:|---:|---:|---:|---:|---:|
| Baseline，单层，ViT-B | 0.9550 | 0.0486 | 0.1073 | 0.9362 | 0.1041 | 0.8988 |
| Baseline，多层，ViT-B | 0.9575 | 0.0440 | 0.1026 | 0.9391 | 0.1030 | 0.8936 |
| SASA+GGSF，ViT-B | 0.9578 | 0.0432 | 0.1012 | 0.9401 | 0.0993 | 0.8870 |
| GGSF，ViT-B（VAT AP 参考） | 0.9574 | 0.0435 | 0.1019 | 0.9416 | 0.1034 | 0.9053 |
| SASA+GGSF，ViT-L | 0.9609 | 0.0382 | 0.0934 | 0.9442 | 0.0926 | 0.9121 |

P3 的第一目标是改善 ViT-B Pareto frontier，不把换成 ViT-L 所得的收益算作融合方法收益。

## 3. 本轮候选

所有候选都保留当前 SASA+GGSF 主干作为 warm-start base。新增分支采用零初始化，因此训练开始时应尽量复现原 checkpoint 的输出，避免随机新融合头破坏已有性能。

| ID | 候选 | 结构/损失 | 搜索阶段 epochs | 目的 |
|---|---|---|---:|---|
| R0 | continued baseline | 原 SASA+GGSF 继续训练 | 3 | 排除“只是多训练几轮”的影响 |
| R1 | residual refinement | base fusion + zero-init multi-layer residual refinement | 3 | 低风险提高多层空间/通道交互能力 |
| R2 | cross-layer attention | base fusion + zero-init per-location four-layer attention residual | 3 | 让每个空间位置自适应选择层级信息 |
| R3 | R1 + coordinate loss | R1，附加 soft-argmax coordinate loss，初始权重 0.05 | 3 | 直接优化 L2，同时观察 AUC/AP trade-off |

本轮不运行 `equal_weight`：现有实现只是将每层乘以常数后再接可学习卷积，缩放可被后续卷积吸收，信息量与 `raw_concat` 基本相同。

## 4. 代码隔离与待实现文件

历史 `gazelle/` 目录保持只读，不在其中增加或修改模型。P3 模型放在新的仓库根目录：

```text
AAAIAlchemyModels/
├── __init__.py
├── fusion.py                    # residual refinement / cross-layer attention
├── model.py                     # 包装原 Gazelle，保留 base fusion 并注入零初始化残差
├── factory.py                   # GF/VAT ViT-B factory
└── tests/
    └── test_alchemy_model.py    # shape、zero-init equivalence、gradient smoke

AAAIScripts/
├── train_p3_alchemy.py          # GF/VAT 训练；sequence-level train/validation split
├── evaluate_p3_alchemy.py       # full test inference 与 records/report 输出
├── select_p3_candidate.py       # leaderboard、Pareto 与 winner.json
├── run_p3a_vat_screen.sh        # R0/R1/R2/R3 单卡顺序短训
├── run_p3b_vat_confirm.sh       # winner VAT 8 epochs warm-start 确认
├── run_p3c_gazefollow.sh        # winner GazeFollow 15 epochs
└── run_p3d_vat_clean.sh         # 用 P3C GF best.pt 初始化 VAT 8 epochs
```

以上 P3 代码已完成并通过本地无需 DINO/真实数据的 smoke test，状态为 `CODE_READY`。服务器 pytest、DINO/checkpoint 加载 smoke 和第 7～8 节真实训练仍为 `NOT_RUN`；在服务器 smoke 通过前不得执行长训练命令。

## 5. 固定路径与初始化

```text
VAT data:
/newhome/fb/dataset/videoattentiontarget

GazeFollow data:
/newhome/fb/dataset/gazefollow_extended

VAT SASA+GGSF warm-start:
/home/fb/src/paper/gazelleV1/experiments/train_vat_sasa_ggsf/2026-03-12_19-24-13/epoch_7.pt

GazeFollow SASA+GGSF warm-start:
/home/fb/src/paper/gazelleV1/experiments/train_gazefollow_sasa_ggsf/2026-02-26_15-51-06/epoch_14.pt

P3 results:
/home/fb/src/paper/gazelleV1/AAAIResults/P3

P3 logs:
/home/fb/src/paper/gazelleV1/AAAIResults/logs
```

所有正式命令直接使用绝对路径，不要求手工设置环境变量。训练任务用 `nohup` 后台顺序执行；不得在单张 3090 上并行启动两个训练任务。

## 6. 选择规则

### 6.1 P3A 短训筛选

固定使用 VAT train sequences 的 90%/10% train/validation split，seed=3106。test split 不参与 P3A 选型。

相对 R0，候选满足以下任一主要改善即可进入候选集合：

- validation L2 至少降低 0.002；
- validation AUC 至少提高 0.001；
- validation Inout AP 至少提高 0.005。

同时不得出现明显崩坏：AUC 下降超过 0.002、L2 增加超过 0.003 或 AP 下降超过 0.010。若多个候选通过，以 Pareto dominance 优先；仍并列时依次按 L2、AUC、AP 选择。

### 6.2 P3B full VAT 确认

winner 从原 VAT warm-start 重新训练 8 epochs，不从 P3A 的第 3 epoch 继续。完成后只评估一次 full VAT test。

值得进入 GazeFollow 的目标是至少满足以下三项中的两项：

- VAT AUC ≥ 0.9420；
- VAT L2 ≤ 0.0980；
- VAT Inout AP ≥ 0.9050。

这是炼丹晋级线，不是论文显著性声明。

## 7. P3A：VAT 短训筛选命令

### 7.1 代码完成后的 smoke test

```bash
cd /home/fb/src/paper/gazelleV1
/home/fb/anaconda3/envs/py310/bin/python -m pytest -q AAAIAlchemyModels/tests
/home/fb/anaconda3/envs/py310/bin/python AAAIScripts/train_p3_alchemy.py --self-test
/home/fb/anaconda3/envs/py310/bin/python AAAIScripts/evaluate_p3_alchemy.py --self-test
/home/fb/anaconda3/envs/py310/bin/python AAAIScripts/select_p3_candidate.py --self-test
```

预期结果：

```text
7 passed
Self-test passed: candidate mapping, differentiable coordinate loss, and sequence-disjoint 90/10 split.
Self-test passed: full-test metric aggregation and candidate mapping.
Self-test passed: thresholds, collapse guard, Pareto dominance, and tie-break order.
```

本地 2026-07-15 实测（使用已有 PyTorch/timm 与 pytest 运行时组合，不加载 DINO 权重或真实数据）：

| Smoke item | 真实结果 |
|---|---|
| `AAAIAlchemyModels/tests` | `7 passed in 4.36s` |
| synthetic shape / invalid hierarchy | 通过 |
| R0/R1/R2 zero-init vs historical SASA+GGSF | 逐元素严格相等，`rtol=0, atol=0` |
| R1/R2 gradient smoke | output projection 梯度非零 |
| legacy checkpoint / GF→VAT coverage | shared base 与 cross-dataset shared coverage 均为 `1.0` |
| `train_p3_alchemy.py --self-test` | `Self-test passed: candidate mapping, differentiable coordinate loss, and sequence-disjoint 90/10 split.` |
| `evaluate_p3_alchemy.py --self-test` | `Self-test passed: full-test metric aggregation and candidate mapping.` |
| `select_p3_candidate.py --self-test` | `Self-test passed: thresholds, collapse guard, Pareto dominance, and tie-break order.` |
| Python `py_compile` | 通过 |
| 四个 shell 的 `bash -n` | 通过 |
| `git diff --check` | 通过 |

服务器 smoke 与长训练均为 `NOT_RUN`。本地没有读取服务器 DINO 权重、SASA+GGSF checkpoint 或数据集，因此这里不填写任何真实集指标。

### 7.2 后台顺序运行 R0/R1/R2/R3

```bash
cd /home/fb/src/paper/gazelleV1
mkdir -p /home/fb/src/paper/gazelleV1/AAAIResults/P3/screen /home/fb/src/paper/gazelleV1/AAAIResults/logs
chmod +x AAAIScripts/run_p3a_vat_screen.sh
nohup bash AAAIScripts/run_p3a_vat_screen.sh \
  > /home/fb/src/paper/gazelleV1/AAAIResults/logs/p3a_vat_screen.log 2>&1 &
```

查看日志：

```bash
tail -f /home/fb/src/paper/gazelleV1/AAAIResults/logs/p3a_vat_screen.log
```

确认单卡上只有一个 P3 训练进程：

```bash
ps -ef | grep '[t]rain_p3_alchemy.py'
nvidia-smi
```

P3A 应生成：

```text
AAAIResults/P3/screen/r0_continued_baseline/{run_manifest,history}.json
AAAIResults/P3/screen/r1_residual_refinement/{run_manifest,history}.json
AAAIResults/P3/screen/r2_cross_layer_attention/{run_manifest,history}.json
AAAIResults/P3/screen/r3_residual_coord005/{run_manifest,history}.json
AAAIResults/P3/screen/leaderboard.csv
AAAIResults/P3/screen/winner.json
```

### 7.3 P3A 实测结果

| Candidate | Best epoch | Val AUC ↑ | Val L2 ↓ | Val Inout AP ↑ | ΔAUC vs R0 | ΔL2 vs R0 | ΔAP vs R0 | 结果 |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| R0 continued baseline | 0 | 0.93316 | 0.12682 | 0.95516 | 0 | 0 | 0 | reference；selector winner |
| R1 residual refinement | 0 | 0.93245 | 0.12804 | 0.95206 | -0.00070 | -0.00123 | -0.00310 | No-Go；无主要改善 |
| R2 cross-layer attention | 0 | 0.93125 | 0.12944 | 0.95313 | -0.00191 | -0.00263 | -0.00203 | No-Go；无主要改善 |
| R3 R1 + coord 0.05 | 0 | 0.93223 | 0.12779 | 0.95231 | -0.00093 | -0.00097 | -0.00285 | No-Go；无主要改善 |

```text
P3A selector winner: R0 continued baseline
P3A new-method winner: none；R1/R2/R3 均未达到任一主要改善门槛
P3A wall time: 未记录；同步结果不含 p3a_vat_screen.log，manifest/history 也未写 wall time
GPU peak memory: 未记录；同步结果不含 nvidia-smi 或显存日志
异常/中断记录: 无运行中断；四组 manifest 均为 complete。实验异常信号是四组 best epoch 全为 0，继续训练时 validation 整体变差
```

结果完整性核验：四组均使用 seed=`3106`、VAT train sequence-level 90%/10% split、20,462/2,179 train/validation queries，且 `test_used_for_selection=false`。train annotation SHA256 为 `9831fe491277083dfc82cb843729a37e622c88c510e9c2c9e345158c03389e43`，test annotation SHA256 为 `75c1097ac5ba417326b158f5f177720ae1c834b00d82f2588b3c28abc3b2c4aa`。R0 task checkpoint coverage 为 `1.0`；R1/R2/R3 的 shared-base coverage 均为 `1.0`，missing keys 仅为 12/13/12 个预期新分支参数，`unexpected=[]`、shape mismatch 为空，zero-init audit 均通过。

四组训练 loss 都下降，但下降主要来自 in/out loss；heatmap BCE 基本不变。与此同时 validation AUC/L2/AP 在 epoch 1～2 普遍退化。R1/R2/R3 没有满足 L2 `+0.002`、AUC `+0.001` 或 AP `+0.005` 的任何一项主要改善；它们虽然没有越过“明显崩坏”护栏，但不能进入候选集合。自动 selector 因此只保留 R0。这里的 R0 是 control winner，不构成新的融合方法收益。

### 7.4 P3A 决策与下一步

当前不运行 `run_p3b_vat_confirm.sh`。P3B 的目的应是确认一个通过 P3A 的新候选；直接把 R0 continued baseline 跑 8 epochs 只会确认“继续训练旧模型”，而原 VAT SASA+GGSF 参考结果本身也没有达到 P3B 的 2/3 晋级线。

下一步保留 R1 residual refinement 家族：R1 与 R3 是同一结构，R3 的 coordinate loss 使 epoch-0 L2/AP 相对 R1 略好，但仍未超过 R0，因此按第 9 节先做 coordinate loss weight=`0.02/0.10` 的单因素 P3A2。启动 P3A2 前先补一个不接触 test 的低成本 sanity：在同一 validation split 上评估 warm-start `initial.pt`，确认四组训练前输出/指标一致，并记录首个 epoch 后 base、inout head 与 residual branch 的参数漂移。若 initial 一致而首个 epoch 即统一退化，则把问题定位为优化协议而不是结构表达能力；完成冻结的 coordinate-weight 两点后，不应盲目延长 epoch。代码侧需先把当前 R3 固定的 `0.05` 暴露为受 manifest 记录的 CLI 参数，并新增独立 P3A2 顺序脚本与输出目录；P3A 的 R0 结果保持为冻结 reference，不覆盖现有 screen 目录。

## 8. Winner 后续命令

以下阶段严格顺序执行。只有上一阶段通过晋级线，才运行下一阶段。

### 8.1 P3B：winner full VAT warm-start 确认

`run_p3b_vat_confirm.sh` 从 `screen/winner.json` 读取候选，不需要手工修改 shell 脚本。

```bash
cd /home/fb/src/paper/gazelleV1
chmod +x AAAIScripts/run_p3b_vat_confirm.sh
nohup bash AAAIScripts/run_p3b_vat_confirm.sh \
  > /home/fb/src/paper/gazelleV1/AAAIResults/logs/p3b_vat_confirm.log 2>&1 &
```

```bash
tail -f /home/fb/src/paper/gazelleV1/AAAIResults/logs/p3b_vat_confirm.log
```

| Candidate | VAT AUC ↑ | VAT L2 ↓ | VAT Inout AP ↑ | 是否满足 2/3 晋级线 |
|---|---:|---:|---:|---|
| Current SASA+GGSF | 0.9401 | 0.0993 | 0.8870 | reference |
| P3 winner | `[待补充]` | `[待补充]` | `[待补充]` | `[待补充]` |

### 8.2 P3C：winner GazeFollow 15 epochs

仅当 P3B 通过时运行：

`run_p3c_gazefollow.sh` 会先读取 P3B 的 `test.report.json`，并强制检查冻结的 2/3 晋级线；未通过会立即退出，不会启动 GazeFollow 训练。

```bash
cd /home/fb/src/paper/gazelleV1
chmod +x AAAIScripts/run_p3c_gazefollow.sh
nohup bash AAAIScripts/run_p3c_gazefollow.sh \
  > /home/fb/src/paper/gazelleV1/AAAIResults/logs/p3c_gazefollow.log 2>&1 &
```

```bash
tail -f /home/fb/src/paper/gazelleV1/AAAIResults/logs/p3c_gazefollow.log
```

| Candidate | GF AUC ↑ | GF Min L2 ↓ | GF Avg L2 ↓ | 结论 |
|---|---:|---:|---:|---|
| Current SASA+GGSF | 0.9578 | 0.0432 | 0.1012 | reference |
| P3 winner | `[待补充]` | `[待补充]` | `[待补充]` | `[待补充]` |

### 8.3 P3D：用 P3C best.pt 初始化干净 VAT 训练

只有 P3C 没有明显退化时运行。这一步产生可用于最终论文比较的 VAT checkpoint：

为把“没有明显退化”变成可执行保护，`run_p3d_vat_clean.sh` 要求相对当前 GazeFollow SASA+GGSF 参考值：AUC 下降不超过 `0.002`，Min L2 和 Avg L2 各自增加不超过 `0.003`；任一项失败就退出，不启动 VAT。

```bash
cd /home/fb/src/paper/gazelleV1
chmod +x AAAIScripts/run_p3d_vat_clean.sh
nohup bash AAAIScripts/run_p3d_vat_clean.sh \
  > /home/fb/src/paper/gazelleV1/AAAIResults/logs/p3d_vat_clean.log 2>&1 &
```

```bash
tail -f /home/fb/src/paper/gazelleV1/AAAIResults/logs/p3d_vat_clean.log
```

| Candidate | VAT AUC ↑ | VAT L2 ↓ | VAT Inout AP ↑ | 与 P3B 是否一致 | 最终判断 |
|---|---:|---:|---:|---|---|
| P3 winner，warm-start search | `[待补充]` | `[待补充]` | `[待补充]` | — | search only |
| P3 winner，clean GF→VAT | `[待补充]` | `[待补充]` | `[待补充]` | `[待补充]` | `[待补充]` |

## 9. 失败后的下一轮炼丹顺序

P3A 已确认 R1/R2/R3 全部失败。不回到 P2/P2.1，也不立即编故事。下一轮只在最好的 R1 residual refinement 家族上依次搜索：

1. coordinate loss weight：0.02 / 0.10；
2. residual width：128 / 256；
3. fusion learning rate：当前值的 0.5× / 2×；
4. inout loss weight：0.5 / 1.5；
5. 仍无收益则结束 ViT-B 轻量融合搜索，评估是否只保留 ViT-L 结果或转向更匹配当前贡献强度的 venue。

每轮只改变一个主要因素，仍先用 VAT 3 epochs 筛选；禁止一次并行堆叠多个无法归因的改动。

## 10. 当前禁止事项

- 不继续填写或汇总 P21 人工标注；
- 不在 `gazelle/` 目录修改历史模型；
- 不同时运行多个 3090 训练任务；
- P3A 未产生 `winner.json` 前，不运行 P3B；
- P3B 未达到晋级线前，不运行 GazeFollow；
- 不把 warm-start search checkpoint 直接当作最终论文结果；
- 不在新结果产生前修改本文件中的 `[待补充]` 数据。
