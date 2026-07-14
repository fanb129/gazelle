# AAAI 2027 诊断与最小候选实验方案

> 状态：代码已实现；真实服务器结果待运行后补充。  
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

### 5.4 P0-A 结果占位

| 模型 | VAT AUC ↑ | VAT L2 ↓ | In/Out AP ↑ | X-TCR ↓ | Association margin ↑ | Distinct-target collapse ↓ |
|---|---:|---:|---:|---:|---:|---:|
| Baseline v0 448（仅探索） | `[待补充]` | `[待补充]` | `[待补充]` | `[待补充]` | `[待补充]` | `[待补充]` |
| GazeSpot 512（仅探索） | `[待补充]` | `[待补充]` | `[待补充]` | `[待补充]` | `[待补充]` | `[待补充]` |

| 分层变量 | 区间 | 样本数 | Baseline X-TCR | GazeSpot X-TCR | Paired improvement 与 95% CI |
|---|---|---:|---:|---:|---|
| 人数 | 1 / 2 / 3 / 4 / 5+ | `[待补充]` | `[待补充]` | `[待补充]` | `[待补充]` |
| Head size | small / medium / large | `[待补充]` | `[待补充]` | `[待补充]` | `[待补充]` |
| Target separation | close / medium / far | `[待补充]` | `[待补充]` | `[待补充]` | `[待补充]` |

解释占位：

```text
[待补充：X-TCR 是否随 distinct target proximity / crowd 增大？]
[待补充：控制 head size 后，该趋势是否仍存在？]
[待补充：标准 L2 与 X-TCR 是否揭示不同失败样本？]
```

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

结果占位：

| 层/层对 | Token cosine ↓ | Normalized effective rank ↑ | Spatial variance | Linear CKA ↓ | Directional novelty ↑ |
|---|---:|---:|---:|---:|---:|
| L2 | `[待补充]` | `[待补充]` | `[待补充]` | - | - |
| L5 | `[待补充]` | `[待补充]` | `[待补充]` | - | - |
| L8 | `[待补充]` | `[待补充]` | `[待补充]` | - | - |
| L11 | `[待补充]` | `[待补充]` | `[待补充]` | - | - |
| L2–L11 | - | - | - | `[待补充]` | `[待补充]` |

这些只属于 representation proxies；即使层间差异明显，也不能单独证明多层融合改善 gaze。最终仍需 matched task controls。

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

结果占位：

| 模块 | 指标 | Overall | Crowd 1 | Crowd 2–3 | Crowd 4+ | 判断 |
|---|---|---:|---:|---:|---:|---|
| GGSF | mask mean | `[待补充]` | `[待补充]` | `[待补充]` | `[待补充]` | `[待补充：是否接近 1]` |
| GGSF | dynamic range | `[待补充]` | `[待补充]` | `[待补充]` | `[待补充]` | `[待补充]` |
| GGSF | inter-person L1 | `[待补充]` | - | `[待补充]` | `[待补充]` | `[待补充：是否近 0]` |
| SASA | normalized entropy | `[待补充]` | `[待补充]` | `[待补充]` | `[待补充]` | `[待补充：是否近 uniform]` |
| SASA | inter-person L1 | `[待补充]` | - | `[待补充]` | `[待补充]` | `[待补充]` |

注意：当前 SASA 不直接使用 `head_token`；若没有 GGSF 先产生 person-specific feature，同图不同人物的 layer weights 理论上会一致。这是需要验证和如实报告的实现事实。

## 8. P0 Go / No-Go 判据

### 继续 person-conditioned hierarchy 路线

至少满足：

- 多层 feature 不是完全冗余；
- 单层错误具有互补性，后续 matched controls 中至少一个多层方法优于统一分辨率 last-layer；
- SASA 的 image-global/弱人物条件无法充分解释人物间差异；
- X-TCR/heatmap similarity 揭示标准 L2 之外的稳定人物特异性不足；如果没有，则方法故事只保留层级互补性。

### 停止或降级该路线

出现任一情况应暂停：

- X-TCR 在 shared-target 聚类后很低，或退化完全由 small head / target distance/偶然目标接近解释；
- 四层高度冗余，单层/多层 matched controls 无差异；
- 新 router 只改善训练集或单一人为 subset；
- 增益来自 512 vs 448、不同初始化、不同数据或选择不同 epoch。

## 9. P1：最小 person-conditioned router

### 9.1 设计边界

`AAAIModules.PersonHierarchicalGazeLLE`：

- DINOv3 scene backbone 每个 batch 只运行一次；
- 从每层 bbox ROI 和全局 context 计算每个人的四层权重；
- 加权后仍使用原 Gazelle `4C → dim` projection、transformer 和 heatmap/inout heads；
- 输出 raw `layer_weights` 与明确 metadata；
- 不含 relational loss，不观察同帧其他 query，不解决真正的 inter-person interaction。

### 9.2 先建立统一 512 的 GazeFollow controls

至少需要 last-layer 与 four-layer raw concat 两个 matched controls。它们都使用 unchanged Gazelle，但通过新 runner 保存 `best.pt`，不再默认使用最后一个 epoch。

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
  --fusion raw_concat \
  --selected-layers all \
  --spatial-prior none \
  --output-dir /home/fb/src/paper/gazelleV1/AAAIResults/P1/control_gf_raw_seed3106 \
  --epochs 15 \
  --batch-size 60 \
  --seed 3106 \
  --device cuda:0 \
  > /home/fb/src/paper/gazelleV1/AAAIResults/logs/control_gf_raw_seed3106.log 2>&1 &
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
  --fusion raw_concat \
  --selected-layers all \
  --spatial-prior none \
  --init-checkpoint /home/fb/src/paper/gazelleV1/AAAIResults/P1/control_gf_raw_seed3106/best.pt \
  --output-dir /home/fb/src/paper/gazelleV1/AAAIResults/P1/control_vat_raw_seed3106 \
  --epochs 8 --batch-size 60 --seed 3106 --device cuda:0 \
  > /home/fb/src/paper/gazelleV1/AAAIResults/logs/control_vat_raw_seed3106.log 2>&1 &
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
| Four-layer raw concat | `[待补充]` | `[待补充]` | `[待补充]` | `[待补充]` | `[待补充]` | `[待补充]` | `[待补充]` |
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

## 11. 本地已完成的执行结果

以下是开发机真实执行结果，只证明代码路径，不是实验结果：

| 检查 | 结果 |
|---|---|
| `failure_taxonomy.py --self-test` | 通过 |
| `compare_failure_reports.py --self-test` | 通过 |
| hierarchy synthetic smoke | 成功，28 rows |
| SASA/GGSF synthetic smoke | 成功，32 rows |
| router synthetic（PyTorch 环境） | 权重 shape `[3,4]`，逐人和为 1，`models_query_interaction=False` |
| 全部新 Python 文件 `py_compile` | 通过 |
| `git diff --check` | 通过 |
| 真实 VAT/GF checkpoint inference | `[待在 3090.lab 补充]` |
| 完整候选模型服务器 pytest | `[待在 py310 环境补充]` |

## 12. 运行后应回传什么

优先回传以下小文件，不需要先发送 checkpoint：

```text
AAAIResults/P0/vat_base_full.report.json
AAAIResults/P0/vat_spot_full.report.json
AAAIResults/P0/vat_base_vs_spot.report.json
AAAIResults/P0/hierarchy_probe/results.json
AAAIResults/P0/sasa_ggsf_audit/results.json
AAAIResults/P1/*/run_manifest.json
AAAIResults/P1/*/history.json
```

拿到 P0 结果后，先做 Go/No-Go 判断，再决定是否启动约 7 小时的 VAT candidate；不要直接排队所有 controls。
