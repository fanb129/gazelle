# AAAI 2027 诊断与最小候选实验方案

> 状态：P0 已于 2026-07-14 完成并同步；P0.5 脚本已就绪；P1 尚未启动。
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

只有 Q1/Q2 给出正证据，才运行 `AAAIModules/` 中的最小 person-conditioned hierarchical router。该候选旨在增强 bbox-conditioned layer selection，没有 relational loss，也不建模人与人交互。若 X-TCR 很低或只反映一般定位误差，则不应把 crowded/query specificity 作为主故事。

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
mkdir -p /home/fb/src/paper/gazelleV1/AAAIResults/P0 /home/fb/src/paper/gazelleV1/AAAIResults/P1 /home/fb/src/paper/gazelleV1/AAAIResults/logs
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
3 passed
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
| Fixed global mean | `[待补充]` | `[待补充]` | `[待补充]` | `[待补充]` | `[待补充]` | `[待补充]` |
| Previous-frame shuffled | `[待补充]` | `[待补充]` | `[待补充]` | `[待补充]` | `[待补充]` | `[待补充]` |
| Equal four-layer | `[待补充]` | `[待补充]` | `[待补充]` | `[待补充]` | `[待补充]` | `[待补充]` |
| Last-only | `[待补充]` | `[待补充]` | `[待补充]` | `[待补充]` | `[待补充]` | `[待补充]` |
| Shallow+deep static（L2+L11） | `[待补充]` | `[待补充]` | `[待补充]` | `[待补充]` | `[待补充]` | `[待补充]` |
| GGSF identity | `[待补充]` | `[待补充]` | `[待补充]` | `[待补充]` | `[待补充]` | `[待补充]` |

主要输出：

```text
AAAIResults/P05/vat_{fixed_mean,shuffle,equal,last_only,shallow_deep,ggsf_identity}.*
AAAIResults/P05/learned_vs_{fixed_mean,shuffle,equal,last_only,shallow_deep,ggsf_identity}.report.json
AAAIResults/P05/last_only_vs_shallow_deep.report.json
```

### 8.3 P1 的最小训练顺序

P0.5 后不要把现有全部 P1 命令一次排队。单卡 3090 按信息增益排序：

1. 先完成上述不训练的 P0.5；若 fixed/static 干预相对 last-only/equal 没有优势，立即停止 hierarchy 方法线；
2. 若 P0.5 有正信号，先审计是否已有与候选架构严格匹配的 512 GazeFollow checkpoint；有则复用，没有才顺序训练 **last-only** 与最优 **static shallow–deep** 两个 GF control（各约 1.5 天）；
3. 使用各自 matching GF checkpoint 分别训练 VAT（各约 7 小时），比较 last-only 与 static shallow–deep；
4. 只有 static shallow–deep 在主要定位指标上稳定改善且 AP 不退化，才按 “GF 约 1.5 天 → VAT 约 7 小时” 训练 person-conditioned router；
5. router 必须进一步超过 static shallow–deep，否则论文只保留静态、简单且可解释的融合，不包装 router。

最终方法进入论文的门槛不是“某个 point estimate 更好”，而是：在相同分辨率、初始化、数据和选 epoch 规则下，目标指标的 sequence-bootstrap CI 支持改善，且 AP 无明显回退；router 还必须超过 static fusion，而不能只超过旧 448 baseline。

## 9. P1：最小 person-conditioned router（仅在 P0.5/静态 pilot 通过后）

### 9.1 设计边界

`AAAIModules.PersonHierarchicalGazeLLE`：

- DINOv3 scene backbone 每个 batch 只运行一次；
- 从每层 bbox ROI 和全局 context 计算每个人的四层权重；
- 加权后仍使用原 Gazelle `4C → dim` projection、transformer 和 heatmap/inout heads；
- 输出 raw `layer_weights` 与明确 metadata；
- 不含 relational loss，不观察同帧其他 query，不解决真正的 inter-person interaction。

### 9.2 先建立统一 512 的 GazeFollow controls

根据 P0 的信息增益，优先建立 last-layer 与 shallow+deep（L2+L11）两个 matched controls。它们都使用 unchanged Gazelle，但通过新 runner 保存 `best.pt`，不再默认使用最后一个 epoch。

```bash
nohup /home/fb/anaconda3/envs/py310/bin/python -u AAAIScripts/train_gazelle_control.py \
  --dataset gazefollow \
  --data-path /newhome/fb/dataset/gazefollow_extended \
  --fusion selected_layers \
  --selected-layers last \
  --spatial-prior none \
  --output-dir /home/fb/src/paper/gazelleV1/AAAIResults/P1/control_gf_last_seed3106 \
  --epochs 15 \
  --batch-size 60 \
  --seed 3106 \
  --device cuda:0 \
  > /home/fb/src/paper/gazelleV1/AAAIResults/logs/control_gf_last_seed3106.log 2>&1 &

nohup /home/fb/anaconda3/envs/py310/bin/python -u AAAIScripts/train_gazelle_control.py \
  --dataset gazefollow \
  --data-path /newhome/fb/dataset/gazefollow_extended \
  --fusion selected_layers \
  --selected-layers 2,11 \
  --spatial-prior none \
  --output-dir /home/fb/src/paper/gazelleV1/AAAIResults/P1/control_gf_l2_l11_seed3106 \
  --epochs 15 \
  --batch-size 60 \
  --seed 3106 \
  --device cuda:0 \
  > /home/fb/src/paper/gazelleV1/AAAIResults/logs/control_gf_l2_l11_seed3106.log 2>&1 &
```

两条 GazeFollow 训练各约 1.5 天，单卡上必须逐条运行；前一条完成并确认 `best.pt` 后再启动后一条。

对应 VAT control 必须分别使用 matching GF checkpoint，不能用一个 GF checkpoint 初始化所有架构：

```bash
nohup /home/fb/anaconda3/envs/py310/bin/python -u AAAIScripts/train_gazelle_control.py \
  --dataset vat \
  --data-path /newhome/fb/dataset/videoattentiontarget \
  --fusion selected_layers \
  --selected-layers last \
  --spatial-prior none \
  --init-checkpoint /home/fb/src/paper/gazelleV1/AAAIResults/P1/control_gf_last_seed3106/best.pt \
  --output-dir /home/fb/src/paper/gazelleV1/AAAIResults/P1/control_vat_last_seed3106 \
  --epochs 8 --batch-size 60 --seed 3106 --device cuda:0 \
  > /home/fb/src/paper/gazelleV1/AAAIResults/logs/control_vat_last_seed3106.log 2>&1 &

nohup /home/fb/anaconda3/envs/py310/bin/python -u AAAIScripts/train_gazelle_control.py \
  --dataset vat \
  --data-path /newhome/fb/dataset/videoattentiontarget \
  --fusion selected_layers \
  --selected-layers 2,11 \
  --spatial-prior none \
  --init-checkpoint /home/fb/src/paper/gazelleV1/AAAIResults/P1/control_gf_l2_l11_seed3106/best.pt \
  --output-dir /home/fb/src/paper/gazelleV1/AAAIResults/P1/control_vat_l2_l11_seed3106 \
  --epochs 8 --batch-size 60 --seed 3106 --device cuda:0 \
  > /home/fb/src/paper/gazelleV1/AAAIResults/logs/control_vat_l2_l11_seed3106.log 2>&1 &
```

如果已有严格匹配的 512 GF checkpoint，可先通过 manifest/key-shape 审计复用；不能复用 448 v0 或含不同 fusion 参数的 checkpoint。

### 9.3 再训练 GazeFollow candidate

默认从随机初始化的 task head/router 开始，与 GazeFollow controls 的 protocol 对齐。若未来选择 warm-start，`--init-checkpoint` 只能使用架构兼容的 512、四层 raw-concat Gazelle checkpoint；不能使用含 SASA/GGSF 参数的 checkpoint。

```bash
nohup /home/fb/anaconda3/envs/py310/bin/python -u AAAIScripts/train_person_router.py \
  --dataset gazefollow \
  --data-path /newhome/fb/dataset/gazefollow_extended \
  --output-dir /home/fb/src/paper/gazelleV1/AAAIResults/P1/router_gf_seed3106 \
  --epochs 15 \
  --batch-size 60 \
  --seed 3106 \
  --device cuda:0 \
  > /home/fb/src/paper/gazelleV1/AAAIResults/logs/router_gf_seed3106.log 2>&1 &
```

快速一批次 smoke：

```bash
/home/fb/anaconda3/envs/py310/bin/python -u AAAIScripts/train_person_router.py \
  --dataset gazefollow \
  --data-path /newhome/fb/dataset/gazefollow_extended \
  --output-dir /home/fb/src/paper/gazelleV1/AAAIResults/smoke/router_gf \
  --epochs 1 \
  --batch-size 2 \
  --max-train-batches 1 \
  --max-eval-batches 1 \
  --device cuda:0
```

训练入口保存：

```text
run_manifest.json
history.json
epoch_*.pt
best.pt
```

### 9.4 再用 matching GF router checkpoint 初始化 VAT

```bash
nohup /home/fb/anaconda3/envs/py310/bin/python -u AAAIScripts/train_person_router.py \
  --dataset vat \
  --data-path /newhome/fb/dataset/videoattentiontarget \
  --output-dir /home/fb/src/paper/gazelleV1/AAAIResults/P1/router_vat_seed3106 \
  --init-checkpoint /home/fb/src/paper/gazelleV1/AAAIResults/P1/router_gf_seed3106/best.pt \
  --epochs 8 \
  --batch-size 60 \
  --frame-sample-every 6 \
  --seed 3106 \
  --device cuda:0 \
  > /home/fb/src/paper/gazelleV1/AAAIResults/logs/router_vat_seed3106.log 2>&1 &
```

### 9.5 用同一 taxonomy 评估 candidate

```bash
nohup /home/fb/anaconda3/envs/py310/bin/python -u AAAIScripts/failure_taxonomy.py \
  --dataset vat \
  --data-path /newhome/fb/dataset/videoattentiontarget \
  --json-path /newhome/fb/dataset/videoattentiontarget/test_preprocessed.json \
  --checkpoint /home/fb/src/paper/gazelleV1/AAAIResults/P1/router_vat_seed3106/best.pt \
  --model-source aaai_router \
  --model-label person_router_seed3106 \
  --device cuda:0 \
  --output-prefix /home/fb/src/paper/gazelleV1/AAAIResults/P1/router_vat_full \
  > /home/fb/src/paper/gazelleV1/AAAIResults/logs/router_vat_full.log 2>&1 &
```

结果占位：

| Model（统一 512、统一 protocol） | AUC ↑ | L2 ↓ | AP ↑ | X-TCR ↓ | Margin ↑ | Params | Best epoch |
|---|---:|---:|---:|---:|---:|---:|---:|
| Last-layer control | `[待补充]` | `[待补充]` | `[待补充]` | `[待补充]` | `[待补充]` | `[待补充]` | `[待补充]` |
| Shallow+deep static（L2+L11） | `[待补充]` | `[待补充]` | `[待补充]` | `[待补充]` | `[待补充]` | `[待补充]` | `[待补充]` |
| Equal weight | `[待补充]` | `[待补充]` | `[待补充]` | `[待补充]` | `[待补充]` | `[待补充]` | `[待补充]` |
| SASA | `[待补充]` | `[待补充]` | `[待补充]` | `[待补充]` | `[待补充]` | `[待补充]` | `[待补充]` |
| Person-conditioned router | `[待补充]` | `[待补充]` | `[待补充]` | `[待补充]` | `[待补充]` | `[待补充]` | `[待补充]` |

## 10. Matched-control 计划生成

以下命令只生成计划和命令，不默认启动长训练：

```bash
/home/fb/anaconda3/envs/py310/bin/python AAAIScripts/matched_control_runner.py plan \
  --suite hierarchy \
  --dataset vat \
  --data-path /newhome/fb/dataset/videoattentiontarget \
  --json-path /newhome/fb/dataset/videoattentiontarget/test_preprocessed.json \
  --fixed-spatial none \
  --seeds 3106 \
  --python /home/fb/anaconda3/envs/py310/bin/python \
  --output-dir /home/fb/src/paper/gazelleV1/AAAIResults/P1/hierarchy_controls
```

会生成 `plan.json`、`plan.csv` 和 `commands.txt`。在检查每个 VAT variant 是否具有 matching GazeFollow initialization 之前，**不要添加 `--execute`**。当前一个 GF checkpoint 初始化所有不同 fusion variant 会造成不公平的部分随机初始化。

## 11. 已完成的执行与结果完整性

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
| 完整候选模型服务器 pytest | `[待在 py310 环境补充]` |

## 12. 当前结论与下一次执行入口

P0 的以下小文件已回传并完成检查：

```text
AAAIResults/P0/vat_base_full.report.json
AAAIResults/P0/vat_spot_full.report.json
AAAIResults/P0/vat_base_vs_spot.report.json
AAAIResults/P0/hierarchy_probe/results.json
AAAIResults/P0/sasa_ggsf_audit/results.json
```

P0 文件已经全部同步并纳入本文档。当前 **不要启动第 9 节的长训练命令**；先按第 8.2 节运行 P0.5。结果同步回来后填写 P0.5 表格，再按第 8.3 节的门槛决定是否启动训练。
