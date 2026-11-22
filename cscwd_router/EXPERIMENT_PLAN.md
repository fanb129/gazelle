# CSCWD Router：限时实验清单（不新增 dense 训练）

更新：2026-11-22，任务 W2。按用户本轮决定执行：**只新增 Router 模型的训练；全量官方 train + official test，不切 val；逐 epoch test，按预定单一指标选一个 checkpoint。** 旧 dense / support 工件可以用作初始化。官方 GazeLLE checkpoint 只做推理评测与测速，不重训。

本文件是执行前的实验清单，不是已完成的实验报告。新入口尚未实现，本轮没有启动服务器任务。下一项先完成 E0，再给可直接手动执行的完整 nohup 命令。

## 1. 最小工作量与执行次序

| 顺序 | 实验 | 新增训练预算 | 要得到的结果 |
|---|---|---:|---|
| E0 | 准备 DINOv3 的 train/test-selected 入口及 checkpoint 读取 | 无，先做本机工程检查 | 正确的数据 split、单一选模指标、初始化、恢复和报告 |
| E1 | GazeFollow Router K50 主实验 | **5 epochs**；复用已训好的全量 GF support | AUC / Avg L2 / Min L2、完整逐轮 test 历史、选中 checkpoint |
| E2 | VAT Router K50 主实验 | **3 epochs support + 5 epochs sparse decoder**；复用已有 VAT dense 初始化 | AUC / L2 / In-out AP，最终逐帧 test |
| E3 | 官方 GazeLLE 与 E1/E2 的精度、实际效率对照 | **0 epochs**，仅推理 | 官方 checkpoint 的同协议精度；B1/B8 延迟、吞吐、显存 |
| E4 | 多人共享路由与代表性可视化 | **0 epochs**，使用已选中的 Router checkpoint | 人数分层、共享 sparse / 共享 dense 的实际耗时；support 与 token 示例 |

本轮基础训练量为 GF 5 epochs + VAT 8 epochs，共 **13 个新增阶段 epoch**；support 与 decoder 的计算成本不同，不将这个计数解释成 GPU 时长。优先完成 E1/E2，并在各自主模型选好后执行 E3。E4 在同一模型上追加推理。

不新增：GF dense 15 epochs、VAT dense base 8 epochs、matched dense continuation、多 seed、多个模型规模训练。K50 固定为主点；K100→K75→K50 的训练 curriculum 不意味着另开三份训练。

## 2. 固定主模型与选模规则

- `DINOv3 ViT-B/16`，512×512 输入，block 5 后路由，四层 `raw_concat`，关闭 SASA/GGSF。
- backbone 全冻结；只有相应阶段的 Router、decoder 或 in/out head 更新。
- 纯 RGB + 真实头框，单帧，无深度、检测框、分割器或时序模块。
- `seed=3106`；主部署预算 K50。学习率、epoch、预算和并列规则在启动前写入 manifest。
- 新 run 的训练 split 始终是完整官方 train，评测 split 是官方 test，`selection_policy=test_selected`。不构造 holdout 或 validation loader。
- GF 按 **test Min L2 最小**选模型；VAT 按 **test L2 最小**选模型；并列时按 AUC 更高、再按 epoch 更早选择。VAT 的 AP 不再作为额外挑选另一个 checkpoint 的依据。
- 每个模型的全部主表指标来自同一个被选中的 checkpoint。各 epoch 独立最高 AUC、最低 L2、最高 AP 可以留在日志中，不拼成主表一行。
- support 阶段固定 3 epochs，只用于学习 support，取末轮作为 decoder 阶段的初始化；保存逐轮 test/support 记录，但不按另一种 test 选择器额外挑选 support。GF 已有全量 support 可以直接复用。
- sparse decoder 的 5 个训练 epoch 使用 `[1.0, 0.75, 0.50, 0.50, 0.50]`；**每轮 test 始终按最终 K50 计算**，使选模比较的是同一部署模型。

当前候选优化设置沿用旧 Router：support AdamW `lr=1e-3`；decoder `lr=1e-4`；VAT in/out `lr=1e-3`；`weight_decay=0`；decoder batch 16、gradient accumulation 2、clip norm 1.0；FP32 训练/精度评测。它们是启动前配置，不是新结果。

## 3. E1：GazeFollow，只重跑 Router decoder 适配

### 初始化

复用已在全部 GF train 上训练过 3 epochs 的 support 工件：

```text
/home/fb/src/paper/gazelleV1/experiments/coverage_router/gf_fulltrain_support_k025_seed3106/final.pt
SHA256: 3668596eb5e927c6a9e7144f3be23cfecd6231a591ce5678a29769d1595133d6
```

路径与 SHA 来自旧 `gf_fulltrain_official_test_seed3106/pipeline_complete.json`；`run_manifest.json` 表明该阶段不读 test、训练 3 epochs，仅更新 Router。这个 checkpoint 还包含对应 dense decoder，因此不必重训 dense，也不必重复 support。

新 run 是从权重初始化并重置优化器/选模状态，不是从旧 final 权重续训更多 epochs 后冒充同一预算。Router 冻结，仅训练 sparse decoder 5 epochs。

### 数据与输出

- 数据：`/newhome/fb/dataset/gazefollow_extended` 的完整 train/test。
- 每轮 test：4782 个官方图像查询，FP32；沿用原分辨率 AUC 与多标注 L2 口径。
- 大工件目录：`/home/fb/src/paper/gazelleV1/experiments/cscwd_router/gf_router_testselected_k050_seed3106/`。
- 小报告目录：`cscwd_router/results/gf_router_testselected_k050_seed3106/`。
- 保存 `history.jsonl`、`best_test.pt`、`last.pt`、选中 epoch 的完整指标；PT 留服务器。

沿用全量 support 中的 K25 训练不表示最终推理采用 K25；最后的 Top-K 排序预算设为 K50。

## 4. E2：VAT，只训练 support 与 sparse decoder

### 初始化

复用旧 VAT dense base checkpoint，不新增 dense 训练：

```text
/home/fb/src/paper/gazelleV1/experiments/coverage_router/vat_splitclean_dense_seed3106/best_val_selection.pt
SHA256: bc946c6a07b230f7e677c15d52d2b984ccb65060f456fbe22ea997f7d234f050
```

SHA 来自旧 VAT support run 的 `run_manifest.json`。它是 DINOv3 多层、带 in/out head 的兼容初始化。该旧工件曾在 VAT train 内做 holdout 选模；新 run 如实登记这个初始化历史，然后在 **全部 VAT train** 上执行新阶段，不再划 val。旧 holdout 指标不作为本轮 test 结果。

### 新增训练

1. 全量 VAT train 上 support-only 3 epochs，冻结 backbone/decoder/inout，仅学习 Router；优先使用 prefix 特征训练，避免 support 阶段无用的深层执行。
2. 固定末轮 Router，训练 sparse decoder 与 in/out head 5 epochs，逐轮按 K50 评测官方 test。

旧 GF 初始化需要补训 VAT in/out head 和 decoder，不能直接把 GF checkpoint 当成已适配的 VAT 模型。本轮用现成 VAT 初始化节省该 dense base 阶段。

### 采样与输出

- 数据：`/newhome/fb/dataset/videoattentiontarget` 的完整 train/test。
- train 与逐 epoch test 按官方训练脚本每 6 帧采样。选中 checkpoint 后，**逐帧（sample rate 1）评测完整 test**，主表报告这一行。
- checkpoint 按上述逐 epoch test 的 L2 选定，逐帧复评不再反向重选其他 epoch。
- AUC/L2 只针对画内查询；AP 包含画内与画外。
- 大工件目录：`/home/fb/src/paper/gazelleV1/experiments/cscwd_router/vat_router_testselected_k050_seed3106/`，分 `support/` 与 `sparse_decoder/`。
- 小报告目录：`cscwd_router/results/vat_router_testselected_k050_seed3106/`。

K50 的旧 holdout 退化在新结果出来前保留。新数据预算与 test 选模可能改变结果，但没有预设“全量训练一定修复”的结论。

## 5. E3：不训练 dense，但要推理比较对象

### 5.1 精度主表

官方 GazeLLE-B/14 使用以下已有权重，仅做 eval：

```text
GF:  /home/fb/src/paper/gazelleV1/checkpoints/gazelle_dinov2_vitb14.pt
VAT: /home/fb/src/paper/gazelleV1/checkpoints/gazelle_dinov2_vitb14_inout.pt
```

GF 已有服务器 T4 的官方 checkpoint 评测，原始小报告经 Git 回传并核对一致后可直接复用；不为改表格重复运行。VAT 官方 checkpoint 若没有同协议逐帧报告，则只补一次完整推理。

论文主表同时标明来源：官方发布 checkpoint 的实测、本文 `test-selected` 模型的实测、其他论文的 reported results。官方 checkpoint 与本文自训练模型的选模来源分别列出，不声称所有方法采用相同训练/选模历史。

默认实测比较对象为 **GazeLLE ViT-B/14**；ViT-L/14 的公开结果可单列。没有测 L/14 时不泛称“超过所有 GazeLLE 版本”。

### 5.2 模型前向效率

对 GF 与 VAT 各使用其实际主表 checkpoint，运行 **官方 GazeLLE-B/14 vs 本文 DINOv3 Router K50**：

- 同一张独占 RTX 3090、FP16、`inference_mode`、关闭 autocast weight cache；固定真实 RGB/真实头框样本。
- 保留原模型输入尺寸：GazeLLE 448，本文 512；使用相同原始图像与人物查询，预处理各自完成。
- 必做 B1、B8 与正反序：GF/VAT × 2 batch × 2 顺序，共 8 个测量单元。
- warmup 50、每轮 latency 200 次、throughput 500 次、重复 5 轮。每次 latency 同步；throughput 在测量段边界同步。
- 计时包含完整模型 forward 的 backbone、Router、gather/scatter、融合和 decoder；图像读取/预处理/H2D 在计时外，明确报告 model-forward latency。
- 报 median/p95 latency、images/s、peak memory 与每轮数据；加速比由每个顺序内的 GazeLLE / Router latency 比值计算，再汇总正反序。

B4 可省。只测自己的 FPS 或跨论文引用不同 GPU 的毫秒数，不能得到相对 GazeLLE 的实际加速比。官方模型的推理成本是必要比较成本，不属于新增 dense 训练。

## 6. E4：不新增训练的多人证据与图

VAT 中选取有真实多头标注的图像，固定样本清单。按同时查询 1 人、2 人、3 人分层，样本不足时如实报告人数与规模，不制造头框。比较官方 GazeLLE 的共享 dense 前向与 Router 的共享 sparse union；per-person repeat 可作为额外控制，但不替代原生 GazeLLE 基线。

记录每组人数的模型前向时延、每图与每查询吞吐、目标点/热图覆盖率及定位。该子集明确标为多人分析，不能冒充完整 VAT 主表。推理仍是单帧，无时序输入。

从固定样本导出输入与 GT 头框、每人 support、union Top-K、保留 patch、GT 和最终热图，包含定位变差的样例。大 NPY/PT 不进 Git；小索引和 MD 登记路径/SHA。

## 7. 余量实验：按价值追加，不阻塞主表

| 项目 | 新增训练 | 正确解释 |
|---|---:|---|
| 同一 checkpoint 下 K75 / K100 推理 | 0 | 单一 K50-trained checkpoint 的预算响应与同路径控制；不是各预算充分重训的最优 Pareto |
| K25 推理、B4 测速 | 0 | 时间允许才补 |
| scene-only Router 对照 | 3 support + 5 decoder，仅 GF 一组 | 去除人物条件后按相同预算训练，支撑 gaze conditioning 的贡献；仍是 Router 训练 |

如果只能增加一个训练对照，优先 scene-only Router，而不是再训一个大 backbone。纯推理时换成未训练 Router 会引入 decoder 的路由分布变化，不能与同预算重训的对照混称。

本轮按用户要求不新增 matched dense，因此当前新主表定位是**完整系统的外部比较**。旧 matched dense 的结果可以保留为独立旧协议证据；不能与新 `test-selected` 行拼成严格同预算的因果消融，也不能仅凭超过 GazeLLE 就把全部提升归因于 Router。

## 8. 文献参照与目标层级

下列均为论文 reported values，**不是本轮已跑结果**：

| 方法 | GF AUC ↑ | GF Avg L2 ↓ | GF Min L2 ↓ | VAT AUC ↑ | VAT L2 ↓ | VAT AP ↑ |
|---|---:|---:|---:|---:|---:|---:|
| GazeLLE ViT-B | 0.956 | 0.104 | 0.045 | 0.933 | 0.107 | 0.897 |
| GazeLLE ViT-L | 0.958 | 0.099 | 0.041 | 0.937 | 0.103 | 0.903 |
| Multi-scale Object-Aware Gaze Estimation via Geometric Reasoning | 0.961 | 0.094 | 0.038 | 0.948 | 0.095 | 0.923 |

来源：[GazeLLE 原论文，主结果表](https://arxiv.org/html/2412.09586v2)、[Mi et al. 原论文，PDF 第 10 页 Table 1](https://arxiv.org/pdf/2606.29334)。已核对 Mi et al. 使用 DINOv3 ViT-L/16、RTX 5090，并离线使用 YOLO11x 和 SAM2-hiera-large 生成 object masks；本文是 ViT-B/16，输入与辅助处理差异需要标明。[实现设置](https://arxiv.org/html/2606.29334v1#S4.SS1)

目标按顺序：

1. 完成 GF/VAT 的主模型与统一结果，检验是否达到或超过官方 GazeLLE-B 的精度，并测得实际模型前向效率。
2. 若精度相近但更快，按各指标报告真实的精度–效率权衡；若只有部分指标更好，逐项描述。
3. 超过 Mi et al. 作为进取目标。当前旧 GF K50 为 `0.956462 / 0.104375 / 0.045044`，与其公开 `0.961 / 0.094 / 0.038` 仍有差距，不能预写已经或必将胜出。

本轮不重新实现/训练 Mi et al.。若没有同硬件的完整流水线测量，其 reported 参数量、离线 object-mask 处理与 5090 实验不能转成本文相对它的速度优势。可以比较公开精度，本文相对 GazeLLE 的速度则通过 E3 直接实测。

## 9. 实现接续与记录方式

E0 将在 `cscwd_router/code/` 新增最小入口，复用现有 Router/训练函数，旧 `gazelle/` 和共享历史脚本保持原样。需要补齐：

- 明确 train/test-only metadata 与 `best_test.pt`；旧 trainer 的无 holdout 模式已经读取官方 train/test，但 GF 默认按 Avg L2 选模，与本轮约定的 Min L2 不同。
- 加载旧全量 GF support 和旧 VAT base 的权重初始化，重置优化器与选择历史；运行前核对本文件登记的 SHA。
- 支持 support-only prefix、sparse decoder 阶段、真实 in/out 头、逐 epoch test、VAT rate-1 复评及 epoch 恢复。
- 记录源码、数据标注、初始化和输出工件的 SHA，保存逐轮全部指标与逐查询输出，供统计与可视化复用。
- 给出 GPU 号、服务器 Python、全部参数、日志和输出路径齐全的 nohup 命令，用户手动执行。

服务器 Python 为 `/home/fb/anaconda3/envs/py310/bin/python`，仓库 `/home/fb/src/paper/gazelleV1`。本轮是实验设计与本地 Git 提交，不把尚未存在的新入口写成可执行命令。

Git 约定更新：本机完成的授权改动由助手自动创建**本地 commit**，只 stage 本任务明确文件；不混入已有目录迁移。push 和服务器执行不因本条约定自动发生。
