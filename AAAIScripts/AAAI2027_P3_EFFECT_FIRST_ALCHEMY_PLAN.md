# AAAI 2027 P3：效果优先炼丹计划

> 状态：`PLAN_FROZEN / CODE_READY / NOT_RUN`
> 日期：2026-07-15  
> 服务器 Python：`/home/fb/anaconda3/envs/py310/bin/python`  
> 服务器仓库：`/home/fb/src/paper/gazelleV1`  
> GPU：单卡 RTX 3090  
> 当前决定：停止 P2/P2.1，不做人工标注；先搜索真实指标增益，再根据胜出结构包装论文故事。

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

### 7.3 P3A 结果占位

| Candidate | Best epoch | Val AUC ↑ | Val L2 ↓ | Val Inout AP ↑ | ΔAUC vs R0 | ΔL2 vs R0 | ΔAP vs R0 | 结果 |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| R0 continued baseline | `[待补充]` | `[待补充]` | `[待补充]` | `[待补充]` | 0 | 0 | 0 | reference |
| R1 residual refinement | `[待补充]` | `[待补充]` | `[待补充]` | `[待补充]` | `[待补充]` | `[待补充]` | `[待补充]` | `[待补充]` |
| R2 cross-layer attention | `[待补充]` | `[待补充]` | `[待补充]` | `[待补充]` | `[待补充]` | `[待补充]` | `[待补充]` | `[待补充]` |
| R3 R1 + coord 0.05 | `[待补充]` | `[待补充]` | `[待补充]` | `[待补充]` | `[待补充]` | `[待补充]` | `[待补充]` | `[待补充]` |

```text
P3A winner: [待补充]
P3A wall time: [待补充]
GPU peak memory: [待补充]
异常/中断记录: [待补充]
```

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

如果 R1/R2/R3 全部失败，不回到 P2/P2.1，也不立即编故事。下一轮只在最好的结构上依次搜索：

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
