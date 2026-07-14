# AAAI 2027 诊断与最小候选实验方案

> 状态：P0、P0.5、P1a 与 P2 已完成并同步；P1a hierarchy-router 方法线为 No-Go；P2 检出一致的 transition degradation，但因每个切换方向仅有 238/251 个样本，未通过预先冻结的 500/方向门槛，因此 temporal-module 方法线暂为 No-Go。
> 服务器解释器：`/home/fb/anaconda3/envs/py310/bin/python`
> 原则：先验证机制，再训练候选；所有 `[待补充]` 都必须由真实输出填写，不得根据预期补数。

## 1. 这套代码回答什么问题

本轮不先假设 SASA、GGSF 或新 router 一定有效，而是依次回答三个可证伪问题：

1. **Q1：bbox-conditioned prediction 是否具有足够的人物特异性？**  
   对人物 A 单独预测后，检查其峰值是否在事后评价中更接近同帧另一个标注目标，而不是 A 自己的目标。本文将这种错误称为 **cross-target confusion (X-TCR)**。它只是 post-hoc error taxonomy：不能证明模型在内部把 A“绑定”给 B，也不能证明存在 inter-person interaction/interference。
2. **Q2：DINOv3 多层特征是否具有可利用的互补性？**  
   不同层是否只是高度冗余；SASA 是否学到明显偏离 uniform 的、随人物变化的权重？
3. **Q3：为什么 GGSF 的历史增益很小，是否应删除？**  
   ACM MM 实验已经显示 GGSF 提升很小。本轮只审计其 mask 是否接近常数/identity，以及不同人物之间是否真的有差异；它不再作为默认核心贡献候选。

P0/P0.5 已经回答：crowded/query binding 不足以作为主故事，GGSF 应删除，原 SASA 近似任务级固定层级配方。P1a 进一步说明 query-conditioned hierarchy router 虽然学到了人物间权重差异，但没有稳定改善整体指标。P2 则发现静态模型在 VAT in/out 边界帧存在一致退化；该现象目前只作为待质检的问题线索，不能直接升级为 temporal 方法贡献。

## 2. 目录与职责

```text
AAAIScripts/
├── common.py                         # SHA256 与严格 checkpoint 校验
├── failure_taxonomy.py               # 逐人错误、X-TCR、association margin、collapse
├── compare_failure_reports.py        # 两模型 paired delta + sequence bootstrap CI
├── hierarchical_feature_probe.py     # raw DINO 多层冗余/互补 proxy
├── audit_sasa_ggsf.py                # raw GGSF mask 与 SASA weight 审计
├── p05_inference_interventions.py   # P0.5 SASA/GGSF 无训练干预
├── run_p05_vat.sh                   # 3090.lab 顺序运行与 paired comparison
├── run_p1a_vat.sh                   # 3090.lab 顺序训练/评估 prior-residual pilot
├── p2_transition_reliability_audit.py # 无推理的 VAT transition/calibration 审计
├── run_p2_transition_audit.sh        # P2 四组已有 records 的后台执行入口
├── p21_transition_validity_audit.py  # P2.1 contact sheets、人工表与冻结门槛汇总
├── run_p21_transition_validity_audit.sh # P2.1 无 GPU 后台生成入口
├── matched_control_runner.py          # 现有 Gazelle controls 的 dry-run 计划/汇总
├── train_gazelle_control.py           # unchanged Gazelle control，严格 matching init/best.pt
├── train_person_router.py             # 独立候选 GF/VAT 训练，保存 best.pt
└── AAAI2027_DIAGNOSTIC_AND_PILOT_PLAN.md

AAAIModules/
├── person_hierarchical_router.py     # bbox ROI pooling + per-person layer weights
├── person_hierarchical_gazelle.py    # 复用原 Gazelle decoder，scene backbone 只跑一次
├── factory.py                        # GF/VAT、ViT-B/L factory
└── tests/                            # 不需要 DINO 权重的 synthetic tests
```

没有修改 `gazelle/` 下的模型文件。`AAAIModules` 只导入原 Gazelle 的 backbone/decoder 接口。

## 3. 服务器固定路径

以下命令按当前服务器历史路径写死，不再预设 `PY/VAT/GF/CKPT` 环境变量：

```bash
cd /home/fb/src/paper/gazelleV1
mkdir -p /home/fb/src/paper/gazelleV1/AAAIResults/P0 /home/fb/src/paper/gazelleV1/AAAIResults/P05 /home/fb/src/paper/gazelleV1/AAAIResults/P1a /home/fb/src/paper/gazelleV1/AAAIResults/logs
```

必须从仓库根目录运行，因为 DINOv3 本地仓库与预训练权重目前使用相对路径：

```text
dinov3/
checkpoints/dinov3_vitb16_pretrain.pth
```

## 4. Step 0：代码与环境 smoke test

### 4.1 纯函数与诊断 smoke

```bash
/home/fb/anaconda3/envs/py310/bin/python AAAIScripts/failure_taxonomy.py --self-test
/home/fb/anaconda3/envs/py310/bin/python AAAIScripts/compare_failure_reports.py --self-test

/home/fb/anaconda3/envs/py310/bin/python AAAIScripts/hierarchical_feature_probe.py \
  --synthetic-smoke \
  --output-dir /home/fb/src/paper/gazelleV1/AAAIResults/smoke/hierarchy

/home/fb/anaconda3/envs/py310/bin/python AAAIScripts/audit_sasa_ggsf.py \
  --synthetic-smoke \
  --output-dir /home/fb/src/paper/gazelleV1/AAAIResults/smoke/module_audit
```

预期不是论文结果，只是连通性检查：

```text
Self-test passed: bins, shared-target clustering, cross-target confusion, margin, and heatmap similarity.
Self-test passed: paired alignment and metric directions.
Wrote 28 raw probe rows ...
Wrote 32 raw gate/weight audit rows ...
```

### 4.2 独立候选模型 test

```bash
/home/fb/anaconda3/envs/py310/bin/python -m pytest -q AAAIModules/tests/test_person_hierarchical_router.py
```

预期：

```text
5 passed
```

若服务器环境未安装 `pytest`，可先运行语法检查，再用下一节的一批次训练 smoke 验证完整 forward：

```bash
/home/fb/anaconda3/envs/py310/bin/python -m py_compile AAAIScripts/*.py AAAIModules/*.py AAAIModules/tests/*.py
```

## 5. P0-A：VAT failure taxonomy 与 query fidelity

### 5.1 先跑 20 帧快速检查

历史 baseline 为 448 输入、GazeSpot 为 512 输入，因此下面的旧 checkpoint 对比只能用于 **现象探索**，不能进入正式归因表。

```bash
/home/fb/anaconda3/envs/py310/bin/python AAAIScripts/failure_taxonomy.py \
  --dataset vat \
  --data-path /newhome/fb/dataset/videoattentiontarget \
  --json-path /newhome/fb/dataset/videoattentiontarget/test_preprocessed.json \
  --checkpoint /home/fb/src/paper/gazelleV1/experiments/train_vat_vitb_v0/2026-03-20_22-30-50/epoch_7.pt \
  --model-source v0 \
  --model-label baseline_v0_448_exploratory \
  --device cuda:2 \
  --max-frames 20 \
  --output-prefix /home/fb/src/paper/gazelleV1/AAAIResults/P0/vat_base_smoke

/home/fb/anaconda3/envs/py310/bin/python AAAIScripts/failure_taxonomy.py \
  --dataset vat \
  --data-path /newhome/fb/dataset/videoattentiontarget \
  --json-path /newhome/fb/dataset/videoattentiontarget/test_preprocessed.json \
  --checkpoint /home/fb/src/paper/gazelleV1/experiments/train_vat_sasa_ggsf/2026-03-12_19-24-13/epoch_7.pt \
  --model-source current \
  --spatial-prior ggsf \
  --fusion sasa \
  --model-label gazespot_512_exploratory \
  --device cuda:2 \
  --max-frames 20 \
  --output-prefix /home/fb/src/paper/gazelleV1/AAAIResults/P0/vat_spot_smoke
```

检查以下文件都不为空：

```text
*.records.csv
*.frames.csv
*.summary.csv
*.report.json
```

`report.json` 会记录 checkpoint 和 annotation JSON 的绝对路径、SHA256、样本数及 checkpoint 加载覆盖率。架构不匹配时脚本应直接失败，不能继续评估部分随机初始化的模型。

### 5.2 完整 VAT 推理

确认 smoke 无误后使用 `nohup` 跑完整推理。单卡上两条命令应顺序执行，不要同时启动：

```bash
nohup /home/fb/anaconda3/envs/py310/bin/python -u AAAIScripts/failure_taxonomy.py \
  --dataset vat \
  --data-path /newhome/fb/dataset/videoattentiontarget \
  --json-path /newhome/fb/dataset/videoattentiontarget/test_preprocessed.json \
  --checkpoint /home/fb/src/paper/gazelleV1/experiments/train_vat_vitb_v0/2026-03-20_22-30-50/epoch_7.pt \
  --model-source v0 \
  --model-label baseline_v0_448_exploratory \
  --device cuda:1 \
  --output-prefix /home/fb/src/paper/gazelleV1/AAAIResults/P0/vat_base_full \
  > /home/fb/src/paper/gazelleV1/AAAIResults/logs/vat_base_full.log 2>&1 &

nohup /home/fb/anaconda3/envs/py310/bin/python -u AAAIScripts/failure_taxonomy.py \
  --dataset vat \
  --data-path /newhome/fb/dataset/videoattentiontarget \
  --json-path /newhome/fb/dataset/videoattentiontarget/test_preprocessed.json \
  --checkpoint /home/fb/src/paper/gazelleV1/experiments/train_vat_sasa_ggsf/2026-03-12_19-24-13/epoch_7.pt \
  --model-source current \
  --spatial-prior ggsf \
  --fusion sasa \
  --model-label gazespot_512_exploratory \
  --device cuda:2 \
  --output-prefix /home/fb/src/paper/gazelleV1/AAAIResults/P0/vat_spot_full \
  > /home/fb/src/paper/gazelleV1/AAAIResults/logs/vat_spot_full.log 2>&1 &
```

### 5.3 Paired comparison 与 sequence bootstrap

```bash
/home/fb/anaconda3/envs/py310/bin/python AAAIScripts/compare_failure_reports.py \
  --reference-records /home/fb/src/paper/gazelleV1/AAAIResults/P0/vat_base_full.records.csv \
  --candidate-records /home/fb/src/paper/gazelleV1/AAAIResults/P0/vat_spot_full.records.csv \
  --bootstrap-iterations 2000 \
  --seed 3106 \
  --output-prefix /home/fb/src/paper/gazelleV1/AAAIResults/P0/vat_base_vs_spot
```

所有 `improvement_*` 都统一为“正值代表 candidate 更好”。VAT CI 以视频 sequence directory 为 cluster 重采样，避免把连续帧当作完全独立样本。

### 5.4 P0-A 结果

| 模型 | VAT AUC ↑ | VAT L2 ↓ | In/Out AP ↑ | X-TCR ↓ | Association margin ↑ | Distinct-target collapse ↓ |
|---|---:|---:|---:|---:|---:|---:|
| Baseline v0 448（仅探索） | 0.9366 | 0.1033 | 0.8986 | 0.2081 | 0.2048 | 0.0539 |
| GazeSpot 512（仅探索） | 0.9402 | 0.0990 | 0.8868 | 0.1961 | 0.2137 | 0.0513 |

| 分层变量 | 区间 | 有效 query 数 | Baseline X-TCR | GazeSpot X-TCR | GazeSpot 改善与 sequence-bootstrap 95% CI |
|---|---|---:|---:|---:|---|
| 人数 | 2 | 4,608 | 0.0872 | 0.0842 | +0.0030 `[-0.0081, 0.0142]` |
| 人数 | 3 | 3,265 | 0.2156 | 0.2178 | -0.0021 `[-0.0603, 0.0743]` |
| 人数 | 4 | 4,509 | 0.2240 | 0.2007 | +0.0233 `[-0.0069, 0.0575]` |
| 人数 | 5+ | 2,358 | 0.4037 | 0.3762 | +0.0276 `[-0.0085, 0.0893]` |
| Head size | large | 8,483 | 0.1222 | 0.1230 | -0.0007 `[-0.0217, 0.0274]` |
| Head size | medium | 5,029 | 0.3366 | 0.3015 | +0.0352 `[-0.0019, 0.0813]` |
| Head size | small | 1,228 | 0.2752 | 0.2704 | +0.0049 `[-0.0371, 0.0435]` |
| Target separation | close | 2,202 | 0.3901 | 0.3715 | +0.0186 `[-0.0211, 0.0566]` |
| Target separation | medium | 6,994 | 0.2239 | 0.2086 | +0.0153 `[-0.0149, 0.0445]` |
| Target separation | far | 5,544 | 0.1160 | 0.1108 | +0.0052 `[-0.0163, 0.0315]` |

结果解释：

P0 观察：原始分桶中 X-TCR 随人数从 `0.0872 (2人)` 升至 `0.4037 (5+人)`，也随 target separation 从 far 的 `0.1160` 升至 close 的 `0.3901`。这说明“错误落到其他标注目标附近”是可测现象，但人数、目标间距、head size 强烈混杂，不能直接解释为 inter-person binding。

Baseline → GazeSpot 的 paired 结果为：AUC `+0.00361 [0.00027, 0.00706]`；L2 改善 `+0.00425 [-0.00284, 0.01178]`；X-TCR 改善 `+0.01201 [-0.00697, 0.03373]`；association margin `+0.00893 [-0.00175, 0.02138]`。只有 AUC 的 cluster-bootstrap CI 未跨 0；L2、X-TCR 和 margin 均没有稳定差异，同时 In/Out AP 从 `0.8986` 降到 `0.8868`。

进一步的 post-hoc controlled analysis 使用 sequence-cluster robust standard errors，并同时控制标准化 people count、head size、head-to-target distance 与 target separation：

| Outcome | Model | People-count effect | Head-size effect | Target-distance effect | Target-separation effect |
|---|---|---|---|---|---|
| X-TCR（logistic OR） | Baseline | 1.267 `[0.990,1.622]`, p=.060 | 0.489 `[0.320,0.745]` | 1.536 `[1.165,2.024]` | 0.553 `[0.434,0.704]` |
| X-TCR（logistic OR） | GazeSpot | 1.292 `[0.999,1.672]`, p=.051 | 0.607 `[0.390,0.944]` | 1.484 `[1.140,1.932]` | 0.549 `[0.418,0.720]` |
| L2（cluster-robust OLS β） | Baseline | +0.0217 `[0.0064,0.0370]`, p=.006 | -0.0137 `[-0.0289,0.0015]` | +0.0185 `[0.0030,0.0340]` | -0.0053 `[-0.0166,0.0060]` |
| L2（cluster-robust OLS β） | GazeSpot | +0.0208 `[0.0077,0.0339]`, p=.002 | -0.0103 `[-0.0252,0.0045]` | +0.0141 `[-0.0014,0.0297]` | -0.0032 `[-0.0136,0.0072]` |

结论是：人数增加与 L2 退化仍有独立关联，所以 crowded 可以保留为 localization stress test；但 people count 对 X-TCR 的独立效应仅在显著性边缘，而 target separation/head size/target distance 更稳定，因此 P0 不支持把“跨人物绑定错误”作为主机制故事。

重要解释：X-TCR 不是“模型看到了 B 的 bbox 后选错了 B”。当前模型对每个 bbox 独立处理；该指标只是在预测完成后，用同帧其他标注目标作为参照，描述 A 的错误落点更像哪一个已标注目标。它可能来自人物条件不足、显著物体偏置、普通定位误差或偶然接近，必须结合 target separation/head size 控制后才能讨论。

当前没有实现 counterfactual query swap。脚本在 report 中明确写为 `not implemented`，不得在论文中声称已经完成。

## 6. P0-B：冻结 DINOv3 层级特征诊断

该分析直接读取 raw backbone feature，不需要任务训练；checkpoint 参数只用于可选架构核验。

```bash
nohup /home/fb/anaconda3/envs/py310/bin/python -u AAAIScripts/hierarchical_feature_probe.py \
  --dataset vat \
  --data-path /newhome/fb/dataset/videoattentiontarget \
  --json-path /newhome/fb/dataset/videoattentiontarget/test_preprocessed.json \
  --max-samples 1000 \
  --device cuda:2 \
  --output-dir /home/fb/src/paper/gazelleV1/AAAIResults/P0/hierarchy_probe \
  > /home/fb/src/paper/gazelleV1/AAAIResults/logs/hierarchy_probe.log 2>&1 &
```

输出：

- `per_sample.csv`：每帧、每层及层对的指标；
- `aggregate.csv`：overall 以及 crowd/head-area/inout 分层；
- `results.json`：配置、annotation hash 和 caveat。

P0 结果（前 1,000 帧）：

| 层/层对 | Token cosine ↓ | Normalized effective rank ↑ | Spatial variance | Linear CKA ↓ | Directional novelty ↑ |
|---|---:|---:|---:|---:|---:|
| L2 | 0.3988 | 0.0900 | 6.4636 | - | - |
| L5 | 0.2379 | 0.1354 | 6.1136 | - | - |
| L8 | 0.1861 | 0.2381 | 1.3235 | - | - |
| L11 | 0.4316 | 0.2865 | 0.0911 | - | - |
| L2–L5 | - | - | - | 0.9401 | 0.1528 |
| L5–L8 | - | - | - | 0.9678 | 0.2485 |
| L8–L11 | - | - | - | 0.8873 | 0.6690 |
| L2–L11 | - | - | - | 0.7380 | 0.7708 |

相邻中层高度相关（L2–L5 CKA `0.9401`、L5–L8 `0.9678`），而浅层与深层差异更明显（L2–L11 CKA `0.7380`、directional novelty `0.7708`）。因此下一轮不值得把四个单层和所有组合全部训练；最有信息量的 matched comparison 是 **last-only vs shallow+deep（L2+L11）**。

这些只属于 representation proxies；它们支持“浅层与深层并非完全冗余”，但不能单独证明这种互补性对 gaze task 有用。另需注意，脚本取的是数据顺序中的前 1,000 帧，而非随机或分层抽样，样本偏向 4+ crowded；该结果只用于候选筛选，最终结论仍需 matched task controls。

## 7. P0-C：SASA 与 GGSF 行为审计（用于降级/删除决策）

ACM MM 实验已经表明 GGSF 增益很小，因此该步骤不是为了重新证明 GGSF 是核心贡献，而是解释它是否几乎为 identity/常数，并为删除、降级为实现细节或重新设计提供证据。

```bash
nohup /home/fb/anaconda3/envs/py310/bin/python -u AAAIScripts/audit_sasa_ggsf.py \
  --dataset vat \
  --data-path /newhome/fb/dataset/videoattentiontarget \
  --json-path /newhome/fb/dataset/videoattentiontarget/test_preprocessed.json \
  --checkpoint /home/fb/src/paper/gazelleV1/experiments/train_vat_sasa_ggsf/2026-03-12_19-24-13/epoch_7.pt \
  --max-samples 1000 \
  --device cuda:1 \
  --output-dir /home/fb/src/paper/gazelleV1/AAAIResults/P0/sasa_ggsf_audit \
  > /home/fb/src/paper/gazelleV1/AAAIResults/logs/sasa_ggsf_audit.log 2>&1 &
```

该脚本只使用 raw forward 输出；不会调用旧 eval 脚本中的 GT enhance/degrade/beautification 函数。

P0 结果：

| 模块 | 指标 | Overall | Crowd 1 | Crowd 2–3 | Crowd 4+ | 判断 |
|---|---|---:|---:|---:|---:|---|
| GGSF | mask mean | 0.9482 | 0.9579 | 0.9596 | 0.9467 | 非常接近 identity 1 |
| GGSF | dynamic range | 0.0640 | 0.0568 | 0.0578 | 0.0650 | 空间调制幅度很小 |
| GGSF | inter-person L1 | 0.0093 | - | 0.0067 | 0.0096 | 人物间几乎不变 |
| SASA | normalized entropy | 0.6719 | 0.7105 | 0.6330 | 0.6736 | 非 uniform，但不是强动态证据 |
| SASA | inter-person L1 | 0.0009 | - | 0.0007 | 0.0009 | 人物间几乎完全一致 |

SASA 的全局平均层权重为 `L2=0.0340, L5=0.0974, L8=0.2007, L11=0.6678`。它明显偏向最后一层，说明不是简单 equal weighting；但是 inter-person L1 只有 `0.0009`，且当前实现不直接使用 `head_token`，所以现有证据不支持“随人物自适应选层”的表述。更准确的描述是：当前 SASA 近似学习了一个以深层为主的全局静态深度配方。

GGSF 的 mask mean 为 `0.9482`、spatial entropy 为 `0.99997`、identity L1 为 `0.0518`，结合 ACM MM 中很小的增益，已经足以判定它不适合作为核心贡献。后续只需用 identity intervention 做最后确认，不应投入时间复杂化 GGSF。

注意：当前 SASA 不直接使用 `head_token`；实测同图不同人物的 layer weights 几乎一致。这是需要如实报告的实现事实，不能继续称为 person-adaptive fusion。

## 8. P0 Go / No-Go 判据

### 8.1 P0 最终判断

| 原假设/模块 | P0 证据 | 决策 |
|---|---|---|
| crowded 中存在可归因的“跨人物绑定”问题 | X-TCR 随人数上升，但 people count 的 controlled X-TCR effect 仅处于显著性边缘，并受目标间距、head size、target distance 影响 | **不作为主机制故事**；crowded 仅保留为 localization stress test |
| GGSF 提供有效、人物相关的空间选择 | mask 近 identity、空间熵近 1、inter-person L1 仅 0.0093，且历史增益很小 | **删除为核心贡献**；最多作为被否定的旧设计/消融 |
| SASA 是 person-adaptive layer selection | 平均权重非 uniform，但 L11 占 0.6678，inter-person L1 仅 0.0009 | **原 claim 不成立**；先判断它是否只是静态深度加权 |
| DINOv3 层级特征可能互补 | L2–L11 的 CKA 0.7380、directional novelty 0.7708，明显区别于相邻中层 | **有条件保留**；必须用 task-level matched controls 验证 |
| GazeSpot 已稳定改善 query fidelity | 旧实验只有 AUC CI 不跨 0；L2、X-TCR、margin 均不稳定，AP 下降 0.0118；且 448 vs 512 混杂 | **不能据此宣称有效** |

对原来的“**SASA + GGSF 解决 crowded 中跨人物绑定**”故事，结论是 **Reject and Pivot**。这不是要求推翻整篇论文，而是停止维护已被 P0 否定的机制解释：去掉 GGSF 核心地位、去掉 binding/interaction overclaim，把尚可验证的主线收缩为“**冻结视觉基础模型中，浅层空间细节与深层语义是否能以低成本方式服务 bbox-conditioned gaze estimation**”。

对“浅层–深层互补 + 最小 bbox-conditioned routing”这一收缩方向，结论是 **有条件进入 pilot**，但当前还不能直接启动约 1.5 天的 GazeFollow router 训练。

### 8.2 下一步：先做 P0.5 无训练干预

在同一个 GazeSpot checkpoint 上只改变推理时的 gate/weight，比较：

1. 原始 learned SASA + GGSF；
2. 固定全局均值权重 `[0.0340, 0.0974, 0.2007, 0.6678]`；
3. 在帧内/样本间 shuffle SASA 权重，保持边际分布；
4. equal weights `[0.25, 0.25, 0.25, 0.25]`；
5. last-only `[0, 0, 0, 1]`；
6. shallow+deep static `[0.5, 0, 0, 0.5]`；
7. GGSF identity（mask 固定为 1，SASA 保持原样）。

全部在完整 VAT 上评估 AUC、L2、AP、X-TCR、margin，并使用 sequence-level paired bootstrap。该步骤不训练，成本远低于一次 VAT；它直接回答两个关键问题：

- fixed mean 若与 learned SASA 基本相同，则删除“dynamic/adaptive”表述，把 SASA 降级为 static depth weighting；
- identity 若与 GGSF 基本相同，则从候选模型中彻底删除 GGSF。

这里的干预只改变内存中当前模型实例的 forward，不修改 checkpoint，也不修改 `gazelle/` 源码。`shuffle_previous_frame` 使用前一帧的平均 learned SASA weights 替换当前帧权重，从而打断图像–权重对应；除首尾边界外保持跨帧经验分布。由于 P0 已发现同帧人物权重几乎相同，人物内 permutation 本身没有诊断意义。

先跑 20 帧 smoke：

```bash
/home/fb/anaconda3/envs/py310/bin/python -u AAAIScripts/p05_inference_interventions.py \
  --dataset vat \
  --data-path /newhome/fb/dataset/videoattentiontarget \
  --json-path /newhome/fb/dataset/videoattentiontarget/test_preprocessed.json \
  --checkpoint /home/fb/src/paper/gazelleV1/experiments/train_vat_sasa_ggsf/2026-03-12_19-24-13/epoch_7.pt \
  --intervention fixed_mean \
  --model-label p05_fixed_mean_smoke \
  --device cuda:0 \
  --max-frames 20 \
  --output-prefix /home/fb/src/paper/gazelleV1/AAAIResults/P05/vat_fixed_mean_smoke
```

确认生成 `.records.csv`、`.frames.csv`、`.summary.csv` 和 `.report.json` 后，从仓库根目录后台顺序运行完整 P0.5：

```bash
cd /home/fb/src/paper/gazelleV1
mkdir -p /home/fb/src/paper/gazelleV1/AAAIResults/P05 /home/fb/src/paper/gazelleV1/AAAIResults/logs
chmod +x AAAIScripts/run_p05_vat.sh
nohup bash AAAIScripts/run_p05_vat.sh \
  > /home/fb/src/paper/gazelleV1/AAAIResults/logs/p05_vat_all.log 2>&1 &
```

查看进度：

```bash
tail -f /home/fb/src/paper/gazelleV1/AAAIResults/logs/p05_vat_all.log
```

批量脚本复用已有的 `AAAIResults/P0/vat_spot_full.records.csv` 作为 original learned SASA+GGSF，不重复运行它；随后顺序运行其余六个干预，并自动执行 2,000 次 sequence-bootstrap paired comparisons。任何一步失败时脚本会停止，不会带着缺失结果继续比较。

P0.5 结果表：

| Inference intervention（同一 checkpoint） | AUC ↑ | L2 ↓ | AP ↑ | X-TCR ↓ | Margin ↑ | 相对 learned 的 paired 结论 |
|---|---:|---:|---:|---:|---:|---|
| Learned SASA + GGSF | 0.9402 | 0.0990 | 0.8868 | 0.1961 | 0.2137 | reference |
| Fixed global mean | 0.9411 | 0.0994 | 0.8834 | 0.1931 | 0.2135 | AUC `+0.00086 [-0.00011,0.00209]`，L2 改善 `-0.00041 [-0.00196,0.00096]`；均不稳定 |
| Previous-frame shuffled | 0.9402 | 0.0990 | 0.8868 | 0.1959 | 0.2138 | 几乎完全相同；图像与动态权重对应关系没有可测贡献 |
| Equal four-layer | 0.8724 | 0.1729 | 0.8673 | 0.3112 | 0.1172 | 显著退化，但属于未重训 decoder 的分布外干预 |
| Last-only | 0.9042 | 0.1360 | 0.8391 | 0.2611 | 0.1658 | 显著退化，但属于未重训 decoder 的分布外干预 |
| Shallow+deep static（L2+L11） | 0.7427 | 0.2817 | 0.7691 | 0.5802 | -0.0341 | 显著退化，但属于未重训 decoder 的分布外干预 |
| GGSF identity | 0.9402 | 0.0992 | 0.8865 | 0.1965 | 0.2130 | 所有 paired CI 跨 0；GGSF 可删除 |

主要输出：

```text
AAAIResults/P05/vat_{fixed_mean,shuffle,equal,last_only,shallow_deep,ggsf_identity}.*
AAAIResults/P05/learned_vs_{fixed_mean,shuffle,equal,last_only,shallow_deep,ggsf_identity}.report.json
AAAIResults/P05/last_only_vs_shallow_deep.report.json
```

P0.5 的核心结论不是“固定均值优于 learned SASA”，而是两者在统计上不可区分；shuffle 又几乎逐点复现 learned 结果。因此现有 SASA 的有效部分是一个**任务级固定层级配方**，不是样本级或人物级动态选择。GGSF identity 同样与原模型不可区分。equal、last-only、L2+L11 的退化只能说明旧 decoder 依赖训练时的四层缩放，不能替代重训后的架构消融。

### 8.3 P0.5 后的决策

P0.5 已否定两个旧 claim：GGSF 没有可测贡献，SASA 也不是动态的人物自适应融合。与此同时，固定均值明显优于未经重训的单层和任意两层替换，说明旧 MM 提升确实主要依赖四层层级配方。由于 2026 年的 *Multi-scale Object-Aware Gaze Estimation via Geometric Reasoning* 已覆盖“冻结 DINOv3 多层特征 + 静态融合 + 几何/物体引导”，不能再把“使用多层特征”本身写成主要创新。

下一步不重新发明复杂 GGSF，也不立即花约 1.5 天跑 GazeFollow。先做一个约 4 epoch、router-only 的 VAT-first 可证伪 pilot：保留 P0.5 实测的全局层级先验，只学习 bbox-conditioned 的小残差。它回答一个比“多层是否有用”更具体的问题：**同一个任务级层级配方，是否需要随被观察人物及其局部/全局上下文做有限修正？**

## 9. P1a：全局层级先验 + 人物条件残差（VAT-first）

### 9.1 设计边界

候选不再声称“首次使用多层 DINOv3”，而把可检验贡献限定为：从一个跨样本共享的任务级层级先验出发，只允许 bbox query 对它做有界的小修正。

```text
w_p = softmax(log(w_train) + tanh(Delta_p))
w_train = mean legacy-SASA weights measured on VAT train only
```

其中 `Delta_p` 由人物 bbox 的多层 ROI、全局 scene context、bbox geometry 和 layer embedding 产生。P0.5 在 test 上观测到的 `.0340/.0974/.2007/.6678` 只用于提出候选，正式 P1a 会以 seed 3106 从 VAT train 均匀抽取 1,000 帧重新估计 `w_train`，不把 test 统计量写入模型。输出层零初始化，因此 `initial.pt` **严格等于 train-only 固定全局先验**；训练后 `best.pt` 与它构成唯一变量为人物条件残差的 matched comparison。

`AAAIModules.PersonHierarchicalGazeLLE` 的边界如下：

- DINOv3 scene backbone 每个 batch 只运行一次；
- 不使用 GGSF，不增加第二个 DINO 分支；
- 冻结 backbone、Gazelle decoder、heatmap head 和 in/out head，pilot 只训练 `layer_router`；
- 加权后仍使用原 Gazelle `4C → dim` projection、transformer 和 heatmap/inout heads；
- 输出 raw `layer_weights` 与明确 metadata；
- 不含 relational loss，不观察同帧其他 query，不解决真正的 inter-person interaction。

与 2026 年 object-aware geometric reasoning 工作的差异必须写成“**静态任务级层级选择 vs query-conditioned residual routing**”，而不是泛泛的“我们也做多尺度”。这个差异只有在 `best.pt` 显著超过 `initial.pt` 时才成立；否则停止包装 router。

### 9.2 为什么先跑 VAT，而不是立即跑 GazeFollow

GazeFollow 一次约 1.5 天，VAT 一次约 7 小时。P1a 只训练小 router，并使用现有 VAT SASA+GGSF checkpoint 中兼容的 decoder/head 参数；旧 `sasa.*`、`ggsf.*` 参数会被显式记录为 ignored，不会进入新模型。固定先验只从 VAT train 图像估计；随后 VAT train 按 sequence directory 固定划分 90%/10% train/validation，test 只在训练完成后评估，避免用 test 统计量初始化或选 epoch。

这个 pilot 不是最终公平 SOTA 实验，而是回答“人物条件残差是否值得继续投入”。只有通过预先定义的 Go 门槛，才启动耗时的 GazeFollow matched training。

### 9.3 服务器 smoke test（先运行）

先检查新增测试。当前测试数应为 5：

```bash
cd /home/fb/src/paper/gazelleV1
/home/fb/anaconda3/envs/py310/bin/python -m pytest -q AAAIModules/tests/test_person_hierarchical_router.py
```

然后只跑一个 train batch 和一个 validation batch，预计几分钟内完成。smoke 只检查连通性，临时使用 P0.5 默认先验；完整实验会改用 train-only JSON：

```bash
/home/fb/anaconda3/envs/py310/bin/python -u AAAIScripts/train_person_router.py \
  --dataset vat \
  --data-path /newhome/fb/dataset/videoattentiontarget \
  --init-checkpoint /home/fb/src/paper/gazelleV1/experiments/train_vat_sasa_ggsf/2026-03-12_19-24-13/epoch_7.pt \
  --allow-legacy-sasa-ggsf \
  --train-scope router_only \
  --validation-from-train \
  --validation-fraction 0.1 \
  --validation-seed 3106 \
  --router-prior-weights 0.0340173,0.0974448,0.2007198,0.6678180 \
  --router-residual-scale 1.0 \
  --epochs 1 \
  --batch-size 2 \
  --workers 2 \
  --max-train-batches 1 \
  --max-eval-batches 10 \
  --lr 1e-3 \
  --seed 3106 \
  --device cuda:0 \
  --output-dir /home/fb/src/paper/gazelleV1/AAAIResults/smoke/p1a_prior_residual_vat
```

应生成：

```text
AAAIResults/smoke/p1a_prior_residual_vat/run_manifest.json
AAAIResults/smoke/p1a_prior_residual_vat/initial.pt
AAAIResults/smoke/p1a_prior_residual_vat/epoch_0.pt
AAAIResults/smoke/p1a_prior_residual_vat/best.pt
AAAIResults/smoke/p1a_prior_residual_vat/history.json
```

smoke 日志没有单独同步，但完整 P1a 的 `run_manifest.status=complete`，训练、双模型完整测试、paired bootstrap 与 router audit 均已成功完成，因此端到端连通性已由正式运行覆盖。

检查 `run_manifest.json` 时必须满足：`test_used_for_selection=false`、`candidate=P1a_global_prior_plus_person_residual_router`，并且 `initialization_report.unexpected=[]`、`incompatible_shapes=[]`。`ignored_source_keys` 中出现 `sasa.*`/`ggsf.*` 是预期行为。

### 9.4 完整 P1a：一个后台命令顺序完成训练、双模型测试与审计

`run_p1a_vat.sh` 会依次执行：从 train 均匀抽取 1,000 帧估计固定先验 → 4 epoch router-only 训练 → 测试 `initial.pt` 固定先验 → 测试 `best.pt` 残差 router → 2,000 次 sequence-bootstrap paired comparison → 1,000 帧 router weight audit。单卡只会同时运行一个任务。

```bash
cd /home/fb/src/paper/gazelleV1
mkdir -p /home/fb/src/paper/gazelleV1/AAAIResults/P1a /home/fb/src/paper/gazelleV1/AAAIResults/logs
chmod +x AAAIScripts/run_p1a_vat.sh
nohup bash AAAIScripts/run_p1a_vat.sh \
  > /home/fb/src/paper/gazelleV1/AAAIResults/logs/p1a_vat_seed3106.log 2>&1 &
```

查看进度：

```bash
tail -f /home/fb/src/paper/gazelleV1/AAAIResults/logs/p1a_vat_seed3106.log
```

确认进程：

```bash
ps -ef | grep '[r]un_p1a_vat.sh'
```

主要输出：

```text
AAAIResults/P1a/prior_residual_vat_seed3106/run_manifest.json
AAAIResults/P1a/train_prior_1000/{per_sample,aggregate}.csv
AAAIResults/P1a/train_prior_1000/results.json
AAAIResults/P1a/prior_residual_vat_seed3106/{initial,best,epoch_0,epoch_1,epoch_2,epoch_3}.pt
AAAIResults/P1a/prior_residual_vat_seed3106/history.json
AAAIResults/P1a/vat_static_prior_full.{records,frames,summary}.csv
AAAIResults/P1a/vat_prior_residual_full.{records,frames,summary}.csv
AAAIResults/P1a/static_prior_vs_prior_residual.{paired,summary}.csv
AAAIResults/P1a/static_prior_vs_prior_residual.report.json
AAAIResults/P1a/router_audit/{per_sample,aggregate}.csv
AAAIResults/P1a/router_audit/results.json
```

### 9.5 结果填写区

训练使用 20,462 个 train queries 和 2,179 个 sequence-disjoint validation queries，只训练 250,625 个 router 参数；test 未用于选 epoch。验证集历史如下：

| Epoch | Train loss ↓ | Validation AUC ↑ | Validation L2 ↓ | Validation AP ↑ |
|---:|---:|---:|---:|---:|
| 0 | 0.3525 | 0.9343 | 0.1280 | 0.9570 |
| 1 | 0.3382 | 0.9349 | 0.1242 | 0.9569 |
| 2 | 0.3424 | 0.9340 | 0.1253 | 0.9563 |
| 3 | 0.3316 | 0.9343 | **0.1234** | 0.9561 |

| P1a model（同架构、同 checkpoint 起点） | AUC ↑ | L2 ↓ | AP ↑ | X-TCR ↓ | Margin ↑ | 备注 |
|---|---:|---:|---:|---:|---:|---|
| `initial.pt`：train-only 固定层级先验、无 GGSF | 0.94095 | 0.10039 | 0.88180 | 0.19315 | 0.21427 | 严格 control；prior=`.01176/.08664/.21456/.68703` |
| `best.pt`：全局先验 + 人物条件残差 | 0.94143 | 0.09983 | 0.88647 | 0.19410 | 0.21396 | validation best epoch=3 |

| Paired metric（positive means `best.pt` better） | Mean improvement | Sequence-bootstrap 95% CI | 判断 |
|---|---:|---:|---|
| AUC | +0.00048 | `[-0.00037, 0.00129]` | CI 跨 0，不稳定 |
| L2 | +0.00055 | `[-0.00144, 0.00235]` | CI 跨 0，不稳定 |
| In/Out AP | +0.00467 | `[-0.00073, 0.00868]` | point estimate 改善，但 CI 跨 0 |
| X-TCR | -0.00095 | `[-0.00676, 0.00366]` | point estimate 退化，CI 跨 0 |
| Association margin | -0.00031 | `[-0.00338, 0.00214]` | point estimate 退化，CI 跨 0 |

| Router behavior | `initial.pt` | `best.pt` | 判断 |
|---|---:|---:|---|
| Mean weights L2/L5/L8/L11 | `.01176/.08664/.21456/.68703` | `.01956/.09838/.22850/.65355` | 保留深层主导，同时向浅/中层移动 |
| Inter-person L1 | `0` | `0.01407` | 确实产生了人物条件差异 |
| Normalized entropy | `0.61482` | `0.65734` | 未塌缩，分布反而更平坦 |

补充观察：router 在 3 人、near target、medium head 等部分分桶改善，但在单人 L2 和 far-target AUC 上出现显著反向退化；不同分桶方向不一致，不能选择性包装为 crowded 或 query-specific 收益。与旧 learned SASA+GGSF（0.94023/0.09902/0.88680）相比，P1a 的 AUC 略高、L2 略差、AP 基本相同，同样没有形成新的 SOTA 证据。

### 9.6 预先冻结的 Go / No-Go 门槛

P1a 进入下一阶段必须同时满足：

1. AUC 或 L2 至少一个 overall paired 95% CI 完全位于改善方向，另一个不能出现稳定退化；
2. AP 相对 `initial.pt` 的 point drop 不超过 `0.003`；
3. `best.pt` 的权重没有塌缩为几乎固定或单层，并出现可复现的人物/场景条件差异；
4. 改善不能只存在于某一个极小分桶，crowded 只作为 stress-test 分层，不再作为主问题定义。

实际判定：第 1 条失败；第 2、3 条通过；第 4 条失败，因为分桶收益方向相反。因此结论为 **No-Go**。不运行 GazeFollow、不补 3 seeds，也不通过增大 router、延长 epoch 或恢复 GGSF 挽救这一机制。

### 9.7 P1a 最终判断与下一步

**Paper type：Novel Method。** 一句话故事原本是“用任务级先验加人物条件残差，使冻结视觉基础模型的层级表示适配每个 bbox query”。致命问题是：动态性已经被成功学出，但主要定位指标被简单固定 control 匹配，且没有一个 overall paired CI 支持改善。按照预先冻结的判据，这是被数据否定的核心机制，而不是多跑几次可以修复的方差问题。

**Verdict：Reject and Pivot（针对 P1a 版本，不是放弃整个 gaze 方向）。**

下一步回到 problem-first，暂不继续改网络。优先做一个无需 GPU 长训练的 **P2：VAT transition-conditioned reliability audit**：

1. 在连续帧中用 bbox IoU 匹配同一人物，划分 stable-in、in→out、out→in、stable-out 四类状态；
2. 对已有 raw predictions 计算 in/out ECE、Brier、AP，以及扣除 GT 运动后的 localization jitter；
3. 控制人数、head size、target distance 与 scene cut，判断“状态转换时的不可靠性”是否是独立且可重复的真实问题；
4. 只有诊断成立，才调研并设计轻量的 transition-aware calibration/temporal residual head，优先在缓存特征或输出层上训练，不重跑 DINO backbone。

这个方向不能简单表述为“加入时序”：CVPR 2020 的 [Detecting Attended Visual Targets in Video](https://openaccess.thecvf.com/content_CVPR_2020/html/Chong_Detecting_Attended_Visual_Targets_in_Video_CVPR_2020_paper.html) 已建模动态 attention，2024 年的 [multi-person temporal gaze framework](https://arxiv.org/abs/2403.10511) 已联合多人、时序和 social gaze；静态不确定性/in-out 联合建模也已有 [Patch-Level Gaze Distribution Prediction](https://openaccess.thecvf.com/content/WACV2023/html/Miao_Patch-Level_Gaze_Distribution_Prediction_for_Gaze_Following_WACV_2023_paper.html)。潜在差异只能是“**scene-level gaze following 在 in/out 状态转换下的可靠性与校准**”，且仍需系统文献核验后才能称为 gap。

如果 P2 诊断也不成立，则不再强行修改原 MM 方法投 AAAI。届时有两个诚实选择：把现有工作按静态多层融合的真实贡献转向要求较低的 venue，或围绕新的任务/数据设置重启一篇论文。AAAI 近年的 gaze 文章更依赖明确的新问题或外部语义，例如 activity cues 的 [AAAI 2024 工作](https://ojs.aaai.org/index.php/AAAI/article/view/28480)、自闭症儿童新场景与数据的 [AAAI 2026 工作](https://ojs.aaai.org/index.php/AAAI/article/view/41177)，以及 concept-conditioned/OOD setting 的 [CVPR 2026 GazeAnywhere](https://openaccess.thecvf.com/content/CVPR2026/papers/Cao_Gaze_Target_Estimation_Anywhere_with_Concepts_CVPR_2026_paper.pdf)。仅靠重新命名多层融合不足以达到这一问题强度。


## 10. 已完成的执行与结果完整性

以下同时记录开发机代码检查与已同步的服务器 P0 实验：

| 检查 | 结果 |
|---|---|
| `failure_taxonomy.py --self-test` | 通过 |
| `compare_failure_reports.py --self-test` | 通过 |
| hierarchy synthetic smoke | 成功，28 rows |
| SASA/GGSF synthetic smoke | 成功，32 rows |
| router synthetic（PyTorch 环境） | 权重 shape `[3,4]`，逐人和为 1，`models_query_interaction=False` |
| 全部新 Python 文件 `py_compile` | 通过 |
| `git diff --check` | 通过 |
| 完整 VAT baseline/GazeSpot inference | 完成；各 31,978 person records、13,127 frames |
| VAT annotation 一致性 | 两次推理 SHA256 均为 `75c1097a...c4aa` |
| Paired comparison | 完成；31,978 对齐 records，2,000 次 sequence bootstrap |
| hierarchy probe | 完成；前 1,000 帧，结论仅作候选筛选 |
| SASA/GGSF audit | 完成；53,431 raw audit rows，无 GT post-processing |
| P0.5 六种 inference interventions | 完成；fixed mean/shuffle/GGSF identity 支持删除旧 dynamic/GGSF claim |
| P1a prior-residual 语法与纯函数检查 | 通过；epoch-0 prior 与 VAT sequence split 已作本地检查 |
| P1a 完整训练与测试 | 完成；20,462/2,179 train/validation queries，best epoch=3 |
| P1a paired comparison | 完成；31,978 对齐 records，localization 与 AP 均完成 sequence bootstrap |
| P1a router audit | 完成；uniform 1,000 test frames、2,344 queries，inter-person L1=0.01407 |
| P1a Go/No-Go | **No-Go**；overall AUC/L2/AP/X-TCR/margin 的 CI 全部跨 0 |
| P2 transition audit | 完成；131 sequences、31,680 matched transitions/模型、2,000 次 sequence bootstrap |
| P2 Go/No-Go | **严格 No-Go**；退化效应显著，但 in→out/out→in 仅 238/251，未达到冻结的 500/方向门槛 |

## 11. 当前结论、P2 结果与复现入口

P0/P0.5/P1a/P2 的结果已经回传并完成检查：

```text
AAAIResults/P0/vat_base_full.report.json
AAAIResults/P0/vat_spot_full.report.json
AAAIResults/P0/vat_base_vs_spot.report.json
AAAIResults/P0/hierarchy_probe/results.json
AAAIResults/P0/sasa_ggsf_audit/results.json
AAAIResults/P05/vat_*.report.json
AAAIResults/P05/learned_vs_*.report.json
AAAIResults/P1a/prior_residual_vat_seed3106/{run_manifest,history}.json
AAAIResults/P1a/vat_{static_prior_full,prior_residual_full}.report.json
AAAIResults/P1a/static_prior_vs_prior_residual.report.json
AAAIResults/P1a/router_audit/results.json
AAAIResults/P2/transition_reliability/{per_transition,summary,comparisons}.csv
AAAIResults/P2/transition_reliability/report.json
```

当前不要再运行第 9.4 节命令，也不要启动旧 GazeFollow controls。下面的 P2 命令保留作复现记录，不需要重复执行；下一步按第 11.5 节先做无训练的 P2.1 人工有效性审计。

### 11.1 P2 代码 smoke

从服务器仓库根目录运行：

```bash
cd /home/fb/src/paper/gazelleV1
/home/fb/anaconda3/envs/py310/bin/python AAAIScripts/p2_transition_reliability_audit.py --self-test
```

预期输出：

```text
Self-test passed: IoU tracking, transition states, ECE, and AP.
```

### 11.2 P2 完整后台命令

该步骤不加载模型或 PT 文件，只读取 VAT test annotation 和四组已有 `.records.csv`。预计主要消耗 CPU；四组模型中的 448 baseline 只用于检查问题是否跨架构存在，不能与 512 模型做绝对性能归因。

```bash
cd /home/fb/src/paper/gazelleV1
mkdir -p /home/fb/src/paper/gazelleV1/AAAIResults/P2/transition_reliability /home/fb/src/paper/gazelleV1/AAAIResults/logs
chmod +x AAAIScripts/run_p2_transition_audit.sh
nohup bash AAAIScripts/run_p2_transition_audit.sh \
  > /home/fb/src/paper/gazelleV1/AAAIResults/logs/p2_transition_reliability.log 2>&1 &
```

查看进度：

```bash
tail -f /home/fb/src/paper/gazelleV1/AAAIResults/logs/p2_transition_reliability.log
```

确认进程：

```bash
ps -ef | grep '[p]2_transition_reliability_audit.py'
```

输出文件：

```text
AAAIResults/P2/transition_reliability/per_transition.csv
AAAIResults/P2/transition_reliability/summary.csv
AAAIResults/P2/transition_reliability/comparisons.csv
AAAIResults/P2/transition_reliability/report.json
```

其中：

- `per_transition.csv`：相邻帧 bbox-IoU 匹配、stable/switch 状态、in/out score、Brier、L2 和 motion residual；
- `summary.csv`：按 model、state、regime、crowd、head size、target distance 汇总；
- `comparisons.csv`：2,000 次 sequence-bootstrap 的 switch-vs-stable Brier/absolute error，以及 out→in-vs-stable-in L2；
- `report.json`：annotation/records SHA256、tracking coverage、完整汇总和 claim boundary。

### 11.3 P2 实测结果与预先冻结的门槛

| Tracking diagnostic | 结果 |
|---|---:|
| Sequence count | 131 |
| Adjacent frame pairs | 12,996 |
| Matched people | 31,680/模型（四模型合计 126,720 rows） |
| IoU match-rate upper bound | 1.0000（31,680 / 31,680） |
| stable-in / in→out / out→in / stable-out counts | 20,225 / 238 / 251 / 10,966（每模型相同） |

| Model | Switch−Stable Brier（95% CI） | Switch−Stable absolute error（95% CI） | Out→In−StableIn L2（95% CI） | 判断 |
|---|---:|---:|---:|---|
| Baseline v0 448（只作跨架构现象核验） | +0.23862 [0.21102, 0.26624] | +0.26503 [0.23572, 0.29381] | +0.06185 [0.03199, 0.09257] | 三项显著；样本门槛失败 |
| Learned SASA+GGSF 512 | +0.25058 [0.21611, 0.28363] | +0.27672 [0.24110, 0.31071] | +0.06110 [0.03578, 0.08970] | 三项显著；样本门槛失败 |
| P1a static prior 512 | +0.25156 [0.21622, 0.28603] | +0.27765 [0.24157, 0.31250] | +0.05850 [0.03338, 0.08670] | 三项显著；样本门槛失败 |
| P1a prior residual 512 | +0.24428 [0.21038, 0.27832] | +0.27266 [0.23776, 0.30617] | +0.05556 [0.03166, 0.08385] | 三项显著；样本门槛失败 |

P2 只有同时满足以下条件才进入 temporal/calibration 方法设计：

1. IoU tracking coverage 足够，且 in→out、out→in 各自至少有 500 个 matched transitions；
2. 至少三个模型（必须包含两个 512 模型）的 switch−stable Brier 或 absolute-error 95% CI 完全大于 0；
3. 至少两个 512 模型的 out→in−stable-in L2 CI 完全大于 0，或存在同等强度且预先解释清楚的 localization reliability 证据；
4. 效应不只来自单一 crowd/head-size/target-distance 小分桶。

逐项判定：

1. **门槛 1 失败**：tracking coverage 很高，且切换覆盖 90/96 个 sequences，但 in→out=238、out→in=251，均低于冻结的 500；
2. **门槛 2 通过**：四个模型的 Brier 与 absolute-error CI 均完全大于 0；
3. **门槛 3 通过**：全部三个 512 模型以及 448 baseline 的 out→in L2 CI 均完全大于 0；
4. **门槛 4 初步通过但未作正式分桶 bootstrap**：raw direction-matched effect 出现在所有 crowd bins；head-size 上主要来自 large/medium，small 只有 10/11 个切换样本，不能解释为可靠的反例。

由于四项是 AND 关系，**P2 按预注册规则判定为严格 No-Go，当前不实现 temporal module**。这不等于现象不存在，而是现有 VAT test transitions 的规模不足以支持把它直接包装成新方法主线。

### 11.4 P2 补充解释：存在边界退化，但还不能叫“时序记忆错误”

| Model | Switch AP（positive rate=0.5133） | in→out 当前 in-score | out→in 当前 in-score | stable-out in-score | stable-in in-score |
|---|---:|---:|---:|---:|---:|
| Baseline v0 448 | 0.49235 | 0.72626 | 0.69431 | 0.36960 | 0.82201 |
| Learned SASA+GGSF 512 | 0.49306 | 0.74189 | 0.70744 | 0.36774 | 0.83769 |
| P1a static prior 512 | 0.49238 | 0.74753 | 0.71110 | 0.37446 | 0.84143 |
| P1a prior residual 512 | 0.48678 | 0.72045 | 0.68689 | 0.35566 | 0.82471 |

四个模型在切换帧上的 AP 都接近随机排序，而且 in→out 帧的当前 `in-score` 反而高于 out→in 帧。这是一个一致的 **boundary-lag-like pattern**。但这些模型是逐帧静态模型，本身没有可产生 temporal hysteresis 的记忆状态，因此不能把这个结果直接解释成“模型记住了上一帧”。更可能的竞争解释包括：边界帧视觉证据滞后、VAT in/out 标签切换约定、遮挡/出画过程，以及 IoU track 或标注噪声。

原 `switch−stable` 比较还存在 current-label 比例不同的潜在混淆。为核验这一点，又用 `per_transition.csv` 做了同标签、sequence-bootstrap sanity check：

- in→out 对 stable-out：absolute-error delta 为 +0.35666 到 +0.37415，四模型 CI 均完全大于 0；
- out→in 对 stable-in：absolute-error delta 为 +0.12770 到 +0.13782，四模型 CI 均完全大于 0；
- out→in 对 stable-in 的 L2 delta 为 +0.05556 到 +0.06185，与主结果一致。

因此，退化不是简单由 switch/stable 的正负样本比例造成的；真正未排除的是 **transition 标注/视觉可判定性**。

### 11.5 接下来怎么做

当前不启动任何 GPU 训练。下一步只做一个最多一天的 **P2.1 transition validity audit**：

1. 从 489 个唯一切换点中按方向与 sequence 分层抽取 120 个（in→out/out→in 各 60），生成 `t−2, t−1, t, t+1` contact sheets，并叠加人物 bbox、in/out 标签、可用的 gaze target 和四模型 in-score；
2. 人工检查这 120 个切换点，分类为 clear transition、gradual/ambiguous、annotation inconsistency、track mismatch、occlusion/scene cut；
3. 在查看人工结果前冻结通过条件：track/annotation 明显错误不超过 10%，至少 70% 样本能从邻帧获得比当前帧更明确的判别证据，并在排除 invalid transitions 后保留同方向退化；
4. 只有 P2.1 通过，才做一次小范围、针对 `transition-aware calibrated scene-level gaze following` 的重叠检索。文献 gap 也成立时，才考虑缓存输出上的轻量 calibration/residual 方法；不重跑 DINO backbone；
5. 任一条件失败，立即结束 temporal 路线。P2 只保留为 failure analysis，不写成贡献。

从转投策略看，**P2 当前不能替代原论文主线**。更稳妥的主线仍是把真实增益归因于 hierarchical/multi-layer representation reuse，删除 GGSF 主贡献与 crowded 主叙事；随后只在 VAT 上筛选同分辨率、同参数预算的 single-layer / mean / concat / FPN-like / learned fusion controls，再把唯一胜出的配置跑一次 GazeFollow。P2.1 的目的只是用极低成本判断是否存在值得保留的第二问题，不应挤占这组核心对照实验的单卡时间。

### 11.6 P2.1 下一次执行命令

先运行不读取真实数据的自测：

```bash
cd /home/fb/src/paper/gazelleV1
/home/fb/anaconda3/envs/py310/bin/python AAAIScripts/p21_transition_validity_audit.py self-test
```

预期输出：

```text
Self-test passed: deterministic sampling, IDs, IoU, and contact-sheet drawing.
```

自测通过后，后台生成 120 张 contact sheets（in→out/out→in 各 60）和人工标注表。该步骤只读取图片、annotation 和 P2 CSV，不加载模型/PT，也不使用 GPU：

```bash
cd /home/fb/src/paper/gazelleV1
mkdir -p /home/fb/src/paper/gazelleV1/AAAIResults/P21/transition_validity /home/fb/src/paper/gazelleV1/AAAIResults/logs
chmod +x AAAIScripts/run_p21_transition_validity_audit.sh
nohup bash AAAIScripts/run_p21_transition_validity_audit.sh \
  > /home/fb/src/paper/gazelleV1/AAAIResults/logs/p21_transition_validity.log 2>&1 &
```

查看进度：

```bash
tail -f /home/fb/src/paper/gazelleV1/AAAIResults/logs/p21_transition_validity.log
```

生成结果：

```text
AAAIResults/P21/transition_validity/audit.csv
AAAIResults/P21/transition_validity/manifest.json
AAAIResults/P21/transition_validity/sheets/*.jpg
```

`audit.csv` 中只填写以下四列，其他列不得修改：

| 列 | 允许值 |
|---|---|
| `review_valid_transition` | `yes` / `no` / `uncertain` |
| `review_temporal_context_helpful` | `yes` / `no` / `uncertain` |
| `review_issue_type` | `clear_transition` / `gradual_ambiguous` / `annotation_inconsistency` / `track_mismatch` / `occlusion_scene_cut` / `other` |
| `review_notes` | 自由文本，可留空 |

必须完成全部 120 行。`uncertain` 在有效性门槛中按 invalid 保守处理。生成脚本若发现已有 `audit.csv` 会拒绝覆盖，避免误删人工标注。

判定口径：`review_valid_transition=yes` 表示四帧确实跟踪同一人物，且当前帧的 in/out 标签变化可信；`review_temporal_context_helpful=yes` 表示邻帧让当前帧单独无法明确判断的边界变得更明确；`review_issue_type` 只填最主要的一类问题。不要根据模型分数高低判断标签是否有效。

人工标注完成并同步回服务器后，运行自动汇总（该步骤很短，不需要 `nohup`）：

```bash
cd /home/fb/src/paper/gazelleV1
/home/fb/anaconda3/envs/py310/bin/python -u AAAIScripts/p21_transition_validity_audit.py summarize \
  --transition-csv /home/fb/src/paper/gazelleV1/AAAIResults/P2/transition_reliability/per_transition.csv \
  --audit-csv /home/fb/src/paper/gazelleV1/AAAIResults/P21/transition_validity/audit.csv \
  --output-dir /home/fb/src/paper/gazelleV1/AAAIResults/P21/transition_validity \
  --per-direction 60 \
  --bootstrap-iterations 2000 \
  --seed 3106
```

汇总输出：

```text
AAAIResults/P21/transition_validity/review_summary.json
AAAIResults/P21/transition_validity/filtered_comparisons.csv
```

`review_summary.json` 只有在以下三个冻结门槛全部通过时才输出 `outcome=GO`：invalid rate ≤10%，temporal-context-helpful rate ≥70%，以及清除 invalid transitions 后两个方向的同标签 absolute-error CI 在至少三个模型（含两个 512 模型）上仍完全大于 0。否则自动输出 `NO_GO`。
